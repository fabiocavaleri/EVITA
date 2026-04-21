import os
import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from framework.oracles import Record
from hdmap.MapParser import MapParser
from config import HD_MAP_PATH

from .utils.utils import (export_planned_trajectories, 
                         extract_from_record, 
                         dynamic_obstacle_from_record, 
                         set_traffic_lights_cycles, 
                         extract_planning_problems, 
                         convert_vehicle_perceptions, 
                         adjust_perceptions, 
                         convert_traffic_light_perception,
                         convert_planned_trajectories)

from .utils.RecordFileError import RecordFileError

from commonroad.geometry.shape import Rectangle
from commonroad.common.file_reader import XMLFileReader
from commonroad.common.file_writer import XMLFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.common.file_writer import CommonRoadFileWriter

#####

# Set standard vehicle shape
# Also defined in config.py
vehicle_width = 2.11
vehicle_length = 4.933
vehicle_heigth = 1.48
shape = Rectangle(vehicle_length, vehicle_width)
            
def scenario_converter(record_files_path: str, executed_scenario_folder: str, map_name: str, ids_map: dict = None):

    mp = MapParser(HD_MAP_PATH)

    # Load the base scenario with the map lanelet network
    BASE_DIR = Path(__file__).resolve().parent
    base_scenario_file = os.path.join(BASE_DIR, "converted_maps", f"{map_name}.xml")
    base_scenario, _ = XMLFileReader(base_scenario_file).open(lanelet_assignment = True)          
    
    # Set folder path containing the scenario .record files
    vehicles_data = {}
    scenario_collisions = {}
    scenario_stop_signs_violations = {}
    scenario_traffic_lights_violations = {}
    scenario_speeding_violations = {}
    vehicles_perceptions = {}
    traffic_light_perceptions = {}
    scenario_planned_trajectories = {}

    # Get all the scenario vehicles' .record files
    scenario_vehicles_record_files = sorted([record_file for record_file in os.listdir(record_files_path) if not record_file.endswith('.json') and not record_file.endswith('.xml')])

    # Load .json file with scenario specifications
    scenario_specs_file = os.path.join(record_files_path, "c.json")
    with open(scenario_specs_file, "r") as f:
        scenario_specs = json.load(f)

    # Retrieve traffic lights data stored in c.json
    traffic_lights_initial_states = scenario_specs["tc_section"]["initial"]
    traffic_light_durations = {
        "red": scenario_specs["tc_section"]["duration_b"],
        "yellow": scenario_specs["tc_section"]["duration_y"],
        "green": scenario_specs["tc_section"]["duration_g"]
    }

    # Update scenario's traffic lights
    scenario = set_traffic_lights_cycles(base_scenario, traffic_lights_initial_states, traffic_light_durations)

    # Get scenario objects id count
    scenario_id_base_counter = len(scenario._id_set)  

    # Build output scenario file path
    scenario_file_path = os.path.join(executed_scenario_folder, "scenario.xml")
    
    if not os.path.exists(scenario_file_path):
        # Open .json file containing Apollo-CommonRoad lanes ids mapping
        map_signal_ids_map_file_path = base_scenario_file.replace(".xml", "_lane_ids_map.json")
        with open(map_signal_ids_map_file_path, "r") as f:
            map_lane_id_map = json.load(f)

        pp_ids = []
        for vehicle_record_file in scenario_vehicles_record_files:
            # Get vehicle numeric identifier
            vehicle = vehicle_record_file.split(".")[0]
            vehicle_id = scenario_id_base_counter + int(vehicle.split('_')[3])
            if vehicle_id not in pp_ids:
                pp_ids.append(vehicle_id)
        
        planning_problem_set = extract_planning_problems(base_scenario, pp_ids, scenario_specs, map_lane_id_map)

        writer = CommonRoadFileWriter(
            scenario,
            planning_problem_set
        )

        writer.write_to_file(scenario_file_path)

    executed_scenario = deepcopy(scenario)

    if ids_map is not None:
        for key, value in ids_map.items():
            new_value = scenario_id_base_counter + int(value.split('_')[3])
            ids_map[key] = new_value

    for vehicle_record_file in scenario_vehicles_record_files:
        # Get vehicle numeric identifier
        vehicle = vehicle_record_file.split(".")[0]
        vehicle_id = scenario_id_base_counter + int(vehicle.split('_')[3])

        # Build path to the vehicle .record file
        vehicle_record_file_path = os.path.join(record_files_path, vehicle_record_file)

        try:
            # Open DoppelTest generated .record file
            record = Record(vehicle_record_file_path)
        except Exception as e:
            print("Error while opening .record file")
            raise RecordFileError("Failed to open .record file") from e
    
        # Extract simulation data from the .record file
        (
            extracted_vehicle_history,
            extracted_planned_trajectories,
            detected_collisions,
            vehicle_perception,
            traffic_light_perception,
            detected_stop_signs_violations,
            detected_traffic_lights_violations,
            detected_speeding_violations
        ) = extract_from_record(record)

        converted_vehicle_perception = convert_vehicle_perceptions(vehicle_perception, ids_map)
        converted_traffic_light_perception = convert_traffic_light_perception(traffic_light_perception)

        converted_planned_trajectories = extracted_planned_trajectories
        if ids_map is not None:
            converted_planned_trajectories = convert_planned_trajectories(extracted_planned_trajectories, ids_map)

        # Save extracted and converted data
        if vehicle_id in vehicles_data:
            vehicles_data[vehicle_id]["history"].update(extracted_vehicle_history)
            vehicles_data[vehicle_id]["planned_trajectories"].update(converted_planned_trajectories)
            
            scenario_collisions[vehicle_id].extend(detected_collisions)
            scenario_stop_signs_violations[vehicle_id].extend(detected_stop_signs_violations)
            scenario_traffic_lights_violations[vehicle_id].extend(detected_traffic_lights_violations)
            scenario_speeding_violations[vehicle_id].extend(detected_speeding_violations)
            traffic_light_perceptions[vehicle_id].update(converted_traffic_light_perception)
            vehicles_perceptions[vehicle_id].update(converted_vehicle_perception)
        else:
            vehicles_data[vehicle_id] = {
                "history": extracted_vehicle_history,
                "planned_trajectories": converted_planned_trajectories,
            }

            scenario_collisions[vehicle_id] = detected_collisions
            scenario_stop_signs_violations[vehicle_id] = detected_stop_signs_violations
            scenario_traffic_lights_violations[vehicle_id] = detected_traffic_lights_violations
            scenario_speeding_violations[vehicle_id] = detected_speeding_violations
            traffic_light_perceptions[vehicle_id] = converted_traffic_light_perception
            vehicles_perceptions[vehicle_id] = converted_vehicle_perception
    
    # Compute scenario start timestep to normalize trajectories timesteps
    scenario_start_timestep = None

    for vehicle_id in vehicles_data.keys():
        # Check if vehicle has history states in the .record file
        if not vehicles_data[vehicle_id].get("history"):
            # print(f"{vehicle_id} history empty")
            continue
        # else:
        #     print(f"{vehicle_id} history ok")

        # Extract vehicle's start timestep
        vehicle_start_timestep = next(
            iter(vehicles_data[vehicle_id]["history"].values())
        )["timestamp"]

        # Update scenario start timestep with the minimum among the vehicles start timesteps
        if scenario_start_timestep is None:
            scenario_start_timestep = vehicle_start_timestep
        elif vehicle_start_timestep < scenario_start_timestep:
            scenario_start_timestep = vehicle_start_timestep
    
    # If all the scenario vehicles have no history re-execute the scenario
    if scenario_start_timestep is None:
        print("Impossible to find scenario start timestep from vehicles history. Scenario will be re-executed")
        raise Exception
    
    # Convert vehicles data to CommonRoad format and add them to the scenario
    for vehicle_id in tqdm(vehicles_data.keys(), desc = "Processing scenario vehicles"):
        tqdm.write(f"Processing vehicle {vehicle_id}")
        extracted_vehicle_history = vehicles_data[vehicle_id]["history"]
        extracted_planned_trajectories = vehicles_data[vehicle_id]["planned_trajectories"]

        # Skip vehicles with no history
        if not extracted_vehicle_history:
            continue

        # Convert vehicle history and planned trajectories to CommonRoad format
        # Create a DynamicObstacle object for the vehicle
        dynamic_obstacle, planned_trajectories = dynamic_obstacle_from_record(vehicle_id, extracted_vehicle_history, extracted_planned_trajectories, shape, scenario_start_timestep)

        # Save vehicle's planned trajectories
        scenario_planned_trajectories[vehicle_id] = planned_trajectories

        # Add dynamic obstacle to the CommonRoad scenario
        executed_scenario.add_objects(dynamic_obstacle)

    #
    # Parse scenario collisions to a DataFrame
    #
    rows = []
    # Get collision data for each scenario vehicle
    for vehicle_id, collision_data in scenario_collisions.items():
        # Iterate over each collision entry
        collided_vehicles = []
        for entry in collision_data:
            if ids_map is not None:
                colliding_vehicle_id = ids_map[entry[1]]
            else:
                colliding_vehicle_id = entry[1]

            # Normalize and convert collision timestep to tenths of seconds
            adjusted_timestep = round((entry[2] - scenario_start_timestep) / 1e8)

            colliding_vehicle_position = entry[3]

            # Build DataFrame row
            element = [vehicle_id, colliding_vehicle_id, adjusted_timestep, colliding_vehicle_position]

            # Avoid duplicate entries for the same collision
            if colliding_vehicle_id not in collided_vehicles:
                rows.append(element)
                collided_vehicles.append(colliding_vehicle_id)
    
    scenario_collisions_df = pd.DataFrame(
        rows,
        columns = ["vehicle_id", "has_collided_with", "collision_timestep", "colliding_vehicle_position"]
    )
    
    #
    # Adjust vehicle perceptions and convert them to DataFrame
    #
    adjusted_vehicles_perceptions = adjust_perceptions(vehicles_perceptions, scenario_start_timestep, "obstacle_id")

    rows = []
    # Get vehicle obstacles perceptions data for each scenario vehicle
    for vehicle_id, vehicle_perceptions in adjusted_vehicles_perceptions.items():
        # Iterate vehicle perceptions at each timestep
        for timestep, perceptions in vehicle_perceptions.items():
            # Iterate over each perceived obstacle
            for obstacle in perceptions:
                # Build DataFrame row
                element = [vehicle_id, int(timestep), obstacle["obstacle_id"], obstacle["obstacle_position"], obstacle["obstacle_orientation"], obstacle["obstacle_velocity"]]
                rows.append(element)

    scenario_vehicles_perceptions_df = pd.DataFrame(
        rows,
        columns = ["vehicle_id", "timestep", "obstacle_id", "obstacle_position", "obstacle_orientation", "obstacle_velocity"]
    )

    #
    # Adjust vehicle traffic light perceptions and convert them to DataFrame
    #
    adjusted_traffic_lights_perceptions = adjust_perceptions(traffic_light_perceptions, scenario_start_timestep, "id")

    rows = []
    # Get vehicle traffic lights perceptions data for each scenario vehicle
    for vehicle_id, vehicle_perceptions in adjusted_traffic_lights_perceptions.items():
        # Iterate vehicle perceptions at each timestep
        for timestep, perceptions in vehicle_perceptions.items():
            # Iterate over each perceived traffic light
            for tl in perceptions:
                # Build DataFrame row
                element = [vehicle_id, int(timestep), tl["id"], tl["color"], tl["confidence"]]
                rows.append(element)

    scenario_traffic_lights_perceptions_df = pd.DataFrame(
        rows,
        columns = ["vehicle_id", "timestep", "tl_id", "tl_color", "perception_confidence"]
    )

    #
    # Convert stop signs and traffic lights violations to DataFrame
    #

    rows = []
    violation_timesteps = []
    # Get stop signs violations for each scenario vehicle
    for vehicle_id, violations in scenario_stop_signs_violations.items():
        for _, stop_sign, timestep in violations:
            converted_timestep = int(round((timestep - scenario_start_timestep) / 1e8))
            if converted_timestep not in violation_timesteps:
                # Build DataFrame row
                element = [vehicle_id, converted_timestep, stop_sign]
                rows.append(element)
                violation_timesteps.append(converted_timestep)

    violation_timesteps = []
    # Get traffic lights violations for each scenario vehicle
    for vehicle_id, violations in scenario_traffic_lights_violations.items():
        for _, traffic_light, timestep in violations:
            converted_timestep = int(round((timestep - scenario_start_timestep) / 1e8))
            if converted_timestep not in violation_timesteps:
                # Build DataFrame row
                element = [vehicle_id, converted_timestep, traffic_light]
                rows.append(element)
                violation_timesteps.append(converted_timestep)
    
    violation_timesteps = []
    # Get speeding violations for each scenario vehicle
    for vehicle_id, violations in scenario_speeding_violations.items():
        for _, (_, speeding_violation, timestep) in violations:
            converted_timestep = int(round((timestep - scenario_start_timestep) / 1e8))
            if converted_timestep not in violation_timesteps:
                # Build DataFrame row
                element = [vehicle_id, converted_timestep, speeding_violation]
                rows.append(element)
                violation_timesteps.append(converted_timestep)

    scenario_violations_df = pd.DataFrame(
        rows,
        columns = ["vehicle_id", "timestep", "violation"]
    )

    scenario_violations_df = scenario_violations_df.sort_values(
        by = ["vehicle_id", "timestep"]
    ).reset_index(drop = True)

    #
    # Save data
    #

    # Export planned trajectories in .json format (dict structure)
    export_planned_trajectories(scenario_planned_trajectories, executed_scenario_folder)

    # Save scenario collisions DataFrame to .csv file
    output_path = os.path.join(executed_scenario_folder, "apollo_scenario_collisions.csv")
    scenario_collisions_df.to_csv(output_path, index = False)

    # Save scenario obstacles perceptions DataFrame to .csv file
    output_path = os.path.join(executed_scenario_folder, "vehicles_perception.csv")
    scenario_vehicles_perceptions_df.to_csv(output_path, index = False)

    # Save scenario traffic lights perceptions DataFrame to .csv file
    output_path = os.path.join(executed_scenario_folder, "traffic_lights_perceptions.csv")
    scenario_traffic_lights_perceptions_df.to_csv(output_path, index = False)

    # Save scenario violations DataFrame to .csv file
    output_path = os.path.join(executed_scenario_folder, "scenario_violations.csv")
    scenario_violations_df.to_csv(output_path, index = False)
    
    # Export converted executed scenario
    XMLFileWriter(executed_scenario, None, decimal_precision = 20).write_scenario_to_file(f"{executed_scenario_folder}/executed_scenario.xml", OverwriteExistingFile.ALWAYS)

if __name__ == "__main__":
    record_files_path = sys.argv[1]
    executed_scenario_folder = sys.argv[2]
    map_name = sys.argv[3]

    if len(sys.argv) > 4:
        ids_map = json.loads(sys.argv[4])
        # Convert keys to integers
        ids_map = {int(k): v for k, v in ids_map.items()}
        scenario_converter(record_files_path, executed_scenario_folder, map_name, ids_map)
    else:
        scenario_converter(record_files_path, executed_scenario_folder, map_name)
