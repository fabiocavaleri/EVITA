import os
import re
import json
import sys
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from framework.oracles import Record
from hdmap.MapParser import MapParser
from config import HD_MAP_PATH

# from .utils.utils import (export_planned_trajectories, 
from utils.utils import (export_planned_trajectories, 
                         extract_from_record, 
                         dynamic_obstacle_from_record, 
                         set_traffic_lights_cycles, 
                         extract_planning_problems, 
                         convert_vehicle_perceptions, 
                         adjust_perceptions, 
                         convert_traffic_light_perception,
                         convert_planned_trajectories)

# from .utils.RecordFileError import RecordFileError
from utils.RecordFileError import RecordFileError

from commonroad.geometry.shape import Rectangle
from commonroad.common.file_reader import XMLFileReader
from commonroad.common.file_writer import XMLFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.common.file_writer import CommonRoadFileWriter

#####

# Set standard vehicle shape
# Also defined in config.py
VEHICLE_WIDTH = 2.11
VEHICLE_LENGTH = 4.933
VEHICLE_SHAPE = Rectangle(VEHICLE_LENGTH, VEHICLE_WIDTH)

ROUTE_PATTERN = re.compile(r"ROUTE_(\d+)")

# Apollo timestep are in nanoseconds; CommonRoad timesteps are tenths of a second
NANOSECONDS_PER_TIMESTEP = 1e8

####################
#     HELPERS      #
####################

def route_index(name: str) -> Optional[int]:
    """
    Number of the vehicle's container (ROUTE_<n>) in a record file or container name, None if absent
    """
    match = ROUTE_PATTERN.search(name)
    
    if match:
        return int(match.group(1))
    else:
        return None
 
def list_record_files(record_files_path: str) -> List[str]:
    """
    Record files of the scenario's vehicles: files (not folders) whose name contains ROUTE_<n>
    """
    return sorted(f for f in os.listdir(record_files_path)
                  if os.path.isfile(os.path.join(record_files_path, f)) and route_index(f) is not None)

def vehicle_id_by_route(record_files: List[str], planning_problem_ids: List[int], scenario) -> Dict[int, int]:
    """
    Vehicle id of each container number
    With planning problems, the vehicle on ROUTE_k gets the k-th planning problem id; otherwise ids start after the largest id in use.
    """
    routes = sorted({route_index(f) for f in record_files})
 
    if planning_problem_ids and len(planning_problem_ids) >= len(routes):
        return {route: planning_problem_ids[route] for route in routes}
 
    first_free_id = max(scenario._id_set, default = 0) + 1
    return {route: first_free_id + route for route in routes}

def to_timestep(timestamp_ns: float, start_ns: float) -> int:
    """
    CommonRoad timestep of an Apollo timestep (nanoseconds), relative to the scenario start.
    Rounded down, as the vehicle histories, so that the same moment gets the same time step in every table.
    """
    return int((timestamp_ns - start_ns) // NANOSECONDS_PER_TIMESTEP)

####################
#     ...      #
####################

def read_records(record_files_path: str, record_files: List[str], id_by_route: Dict[int, int], ids_map: Optional[Dict]) -> Dict[int, Dict]:
    """
    Read every record file and merge the chunks of each vehicle.
    Returns, per vehicle id: history, planned trajectories, collisions, perceptions and violations.
    """
    vehicles = {}
 
    for record_file in record_files:
        vehicle_id = id_by_route[route_index(record_file)]
        record_path = os.path.join(record_files_path, record_file)

        try:
            record = Record(record_path)
        except Exception as exc:
            raise RecordFileError(f"Failed to open record file {record_file}") from exc
 
        (history,
         planned_trajectories,
         collisions,
         vehicle_perception,
         traffic_light_perception,
         stop_sign_violations,
         traffic_light_violations,
         speeding_violations) = extract_from_record(record)
 
        perceptions = convert_vehicle_perceptions(vehicle_perception, ids_map)
        traffic_light_perceptions = convert_traffic_light_perception(traffic_light_perception)

        if ids_map is not None:
            planned_trajectories = convert_planned_trajectories(planned_trajectories, ids_map)
 
        data = vehicles.setdefault(vehicle_id, {"history": {}, "planned_trajectories": {}, "collisions": [],
                                                "perceptions": {}, "traffic_light_perceptions": {},
                                                "stop_sign_violations": [], "traffic_light_violations": [], "speeding_violations": []})
        data["history"].update(history)
        data["planned_trajectories"].update(planned_trajectories)
        data["collisions"].extend(collisions)
        data["perceptions"].update(perceptions)
        data["traffic_light_perceptions"].update(traffic_light_perceptions)
        data["stop_sign_violations"].extend(stop_sign_violations)
        data["traffic_light_violations"].extend(traffic_light_violations)
        data["speeding_violations"].extend(speeding_violations)
 
    return vehicles

def build_obstacles(executed_scenario, vehicles: Dict[int, Dict], scenario_initial_timestep: int) -> Tuple[Dict, List[int]]:
    """
    Add each vehicle's history to the executed scenario as a dynamic obstacle.
    Returns the planned trajectories of each vehicle and the ids of vehicles without history (never started).
    """
    planned_trajectories, no_history = {}, []
 
    for vehicle_id in tqdm(sorted(vehicles), desc = "Processing scenario vehicles"):
        data = vehicles[vehicle_id]
        if not data["history"]:
            no_history.append(vehicle_id)
            continue
 
        dynamic_obstacle, vehicle_planned_trajectories = dynamic_obstacle_from_record(vehicle_id, dict(sorted(data["history"].items())), dict(sorted(data["planned_trajectories"].items())), VEHICLE_SHAPE, scenario_initial_timestep)
 
        planned_trajectories[vehicle_id] = vehicle_planned_trajectories
        executed_scenario.add_objects(dynamic_obstacle)
 
    return planned_trajectories, no_history

def collisions_table(vehicles: Dict[int, Dict], ids_map: Optional[Dict], scenario_initial_timestep: int) -> pd.DataFrame:
    """
    Save first collision of each vehicle with each other obstacle, as reported by Apollo's collision oracle.
    Obstacles that are not scenario vehicles keep Apollo's id.
    """
    rows = []
    # Iterate over each vehicle and its collision data
    for vehicle_id, data in vehicles.items():
        seen = set()
        # Sort collision entries by timestamp and iterate over them
        for entry in sorted(data["collisions"], key = lambda e: e[2]):
            # Convert the other obstacle's id using ids_map if available, otherwise keep the original id
            if ids_map is not None:
                other_obstacle = ids_map.get(entry[1], entry[1])
            else:
                other_obstacle = entry[1]

            # Avoid duplicate entries for the same collision with the same obstacle
            if other_obstacle in seen:
                continue

            seen.add(other_obstacle)
            rows.append([vehicle_id, other_obstacle, to_timestep(entry[2], scenario_initial_timestep), entry[3]])
 
    return pd.DataFrame(rows, columns = ["vehicle_id", "has_collided_with", "collision_timestep", "colliding_vehicle_position"])
 
def perceptions_tables(vehicles: Dict[int, Dict], scenario_initial_timestep: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Obstacles and traffic lights perceived by each vehicle, per timestep.
    """

    # Adjust obstacle perceptions
    obstacle_perceptions = adjust_perceptions(
        {vehicle_id: data["perceptions"] for vehicle_id, data in vehicles.items()},
        scenario_initial_timestep,
        "obstacle_id"
    )

    obstacle_rows = [
        [vehicle_id, int(timestep), obstacle["obstacle_id"], obstacle["obstacle_position"], obstacle["obstacle_orientation"], obstacle["obstacle_velocity"]]
        for vehicle_id, per_timestep in obstacle_perceptions.items()
        for timestep, perceived in per_timestep.items()
        for obstacle in perceived
    ]

    obstacles_df = pd.DataFrame(obstacle_rows, columns = ["vehicle_id", "timestep", "obstacle_id", "obstacle_position", "obstacle_orientation", "obstacle_velocity"])

    # Adjust traffic light perceptions
    light_perceptions = adjust_perceptions(
        {vehicle_id: data["traffic_light_perceptions"] for vehicle_id, data in vehicles.items()},
        scenario_initial_timestep,
        "id"
    )

    light_rows = [
        [vehicle_id, int(timestep), light["id"], light["color"], light["confidence"]]
        for vehicle_id, per_timestep in light_perceptions.items()
        for timestep, perceived in per_timestep.items()
        for light in perceived
    ]

    lights_df = pd.DataFrame(light_rows, columns = ["vehicle_id", "timestep", "tl_id", "tl_color", "perception_confidence"])

    return obstacles_df, lights_df
 
def violations_table(vehicles: Dict[int, Dict], scenario_initial_timestep: int) -> pd.DataFrame:
    """
    Stop-sign, traffic-light and speeding violations; each violation once per vehicle and timestep
    """
    rows = []
    seen = set()
 
    def add(vehicle_id, timestamp_ns, violation):
        key = (vehicle_id, to_timestep(timestamp_ns, scenario_initial_timestep), violation)
        
        if key not in seen:
            seen.add(key)
            rows.append(list(key))
    
    # Iterate over each vehicle and its violations data
    for vehicle_id, data in vehicles.items():
        # Sort violations by timestamp and add them to the rows list, avoiding duplicates
        for _, stop_sign, timestamp_ns in data["stop_sign_violations"]:
            add(vehicle_id, timestamp_ns, stop_sign)
        # Sort violations by timestamp and add them to the rows list, avoiding duplicates
        for _, traffic_light, timestamp_ns in data["traffic_light_violations"]:
            add(vehicle_id, timestamp_ns, traffic_light)
        # Sort violations by timestamp and add them to the rows list, avoiding duplicates
        for _, (_, speeding, timestamp_ns) in data["speeding_violations"]:
            add(vehicle_id, timestamp_ns, speeding)
 
    return pd.DataFrame(rows, columns = ["vehicle_id", "timestep", "violation"]).sort_values(["vehicle_id", "timestep"]).reset_index(drop = True)

####################
#     ...      #
####################

def scenario_converter(record_files_path: str, executed_scenario_folder: str, map_name: str, ids_map: dict = None):
    """
    Convert DoppelTest generated scenario data into CommonRoad format and save the converted scenario and data
    """

    # Open HD map with MapParser
    mp = MapParser(HD_MAP_PATH)

    # Load the base scenario with the map lanelet network
    BASE_DIR = Path(__file__).resolve().parent
    base_scenario_file = os.path.join(BASE_DIR, "converted_maps", f"{map_name}.xml")
    base_scenario, _ = XMLFileReader(base_scenario_file).open(lanelet_assignment = True)    

    # Load .json file with scenario specifications
    scenario_specifications_file = os.path.join(record_files_path, "c.json")
    with open(scenario_specifications_file, "r") as f:
        scenario_specifications = json.load(f)

    # Retrieve traffic lights data configuration stored in c.json
    traffic_light_durations = {
        "red": scenario_specifications["tc_section"]["duration_b"],
        "yellow": scenario_specifications["tc_section"]["duration_y"],
        "green": scenario_specifications["tc_section"]["duration_g"]
    }

    # Update CommonRoad scenario's traffic lights
    scenario = set_traffic_lights_cycles(base_scenario, scenario_specifications["tc_section"]["initial"], traffic_light_durations)

    # Vehicles' record files and ids
    record_files = list_record_files(record_files_path)
    if not record_files:
        raise RecordFileError(f"No record files in {record_files_path}")

    scenario_file_path = os.path.join(executed_scenario_folder, "scenario.xml")
    if os.path.exists(scenario_file_path):
        # EVITA scenario: the generated scenario and its planning problems are already there
        _, planning_problem_set = XMLFileReader(scenario_file_path).open()
        planning_problem_ids = list(planning_problem_set.planning_problem_dict)
        id_by_route = vehicle_id_by_route(record_files, planning_problem_ids, scenario)
    else:
        # DoppelTest scenario: build the planning problems from the scenario specification
        id_by_route = vehicle_id_by_route(record_files, [], scenario)
        with open(base_scenario_file.replace(".xml", "_lane_ids_map.json"), "r") as f:
            map_lane_id_map = json.load(f)

        planning_problem_set = extract_planning_problems(base_scenario, sorted(set(id_by_route.values())), scenario_specifications, map_lane_id_map)
        CommonRoadFileWriter(scenario, planning_problem_set).write_to_file(scenario_file_path)

    # If ids_map is available (EVITA generated scenario) setup vehicles ids map
    if ids_map is not None:
        ids_map = {apollo_id: id_by_route[route_index(container)] for apollo_id, container in ids_map.items() if route_index(container) in id_by_route}

    # Read the record files and convert vehicle's data
    vehicles = read_records(record_files_path, record_files, id_by_route, ids_map)

    vehicles_starts = [min(data["history"]) for data in vehicles.values() if data["history"]]
    
    if not vehicles_starts:
        raise RecordFileError("No vehicle has history states in its record files")
    else:
        scenario_initial_timestep = min(vehicles_starts)

    # Add vehicles to CommonRoad scenario as dynamic obstacles and retrieve their planned trajectories
    executed_scenario = deepcopy(scenario)
    planned_trajectories, no_history = build_obstacles(executed_scenario, vehicles, scenario_initial_timestep)

    # Compute collisions, perceptions and violations tables
    collisions_df = collisions_table(vehicles, ids_map, scenario_initial_timestep)
    obstacles_perception_df, traffic_lights_perception_df = perceptions_tables(vehicles, scenario_initial_timestep)
    violations_df = violations_table(vehicles, scenario_initial_timestep)
    
    # Export
    os.makedirs(executed_scenario_folder, exist_ok = True)

    export_planned_trajectories(planned_trajectories, executed_scenario_folder)
    collisions_df.to_csv(os.path.join(executed_scenario_folder, "apollo_scenario_collisions.csv"), index = False)
    obstacles_perception_df.to_csv(os.path.join(executed_scenario_folder, "vehicles_perception.csv"), index = False)
    traffic_lights_perception_df.to_csv(os.path.join(executed_scenario_folder, "traffic_lights_perceptions.csv"), index = False)
    violations_df.to_csv(os.path.join(executed_scenario_folder, "scenario_violations.csv"), index = False)
 
    with open(os.path.join(executed_scenario_folder, "vehicle_ids.json"), "w") as f:
        json.dump({"vehicle_id_by_route": {f"ROUTE_{route}": vehicle_id for route, vehicle_id in id_by_route.items()},
                   "apollo_ids": {str(apollo_id): vehicle_id for apollo_id, vehicle_id in (ids_map or {}).items()},
                   "no_history": no_history,
                   "scenario_initial_timestep": scenario_initial_timestep}, f, indent = 4)
 
    XMLFileWriter(executed_scenario, None, decimal_precision = 20).write_scenario_to_file(os.path.join(executed_scenario_folder, "executed_scenario.xml"), OverwriteExistingFile.ALWAYS)

if __name__ == "__main__":
    record_files_path = sys.argv[1]
    executed_scenario_folder = sys.argv[2]
    map_name = sys.argv[3]

    ids_map = None
    
    if len(sys.argv) > 4:
        ids_map = json.loads(sys.argv[4])
        # Convert keys to integers
        ids_map = {int(k): v for k, v in ids_map.items()}
    
    scenario_converter(record_files_path, executed_scenario_folder, map_name, ids_map)
