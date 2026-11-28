"""
Converts DoppelTest's own Apollo runs into EVITA run folders, evaluated with EVITA's evaluator, 
so that analysis.sh compares them with EVITA's runs on Apollo.

    python scripts/convert_doppeltest_runs.py <folder> [<folder> ...] [--map <converted map.xml>] [--force] [--workers N]

Every folder is searched for DoppelTest run folders (those with Generation_*/Scenario_* folders).
Each scenario (DoppelTest's record files and scenario JSON, c.json) gets a sequential id (00001, ...;
doppeltest_scenarios.csv lists each id's generation and scenario) and, in the run folder:
1. a trace from its record files.
   DoppelTest's Apollo ids are random and not saved by DoppelTest: they come from ids_map.json when the run saved one, 
   otherwise they are recovered by matching the obstacles each vehicle perceived with the other vehicles' own positions;
2. the CommonRoad scenario rebuilt from c.json (starts, goals and routes on the converted map), for EVITA's validation;
3. validity as for EVITA's scenarios: a scenario that cannot be converted, fails EVITA's Apollo validation, or fails
   the health check (a vehicle that did not run, lost its planning or never moved) is invalid: its folder goes to
   debug/<id>/ with reason.txt; a valid one is evaluated into <id>/logs/simulation.db;
4. for the run: scenario_outcomes.csv, performance.csv, metadata.csv, the report (behavior.csv, maps, ...), and
   conversion_completed.json.

Converted runs are skipped unless --force (which converts them again).
--workers N converts N scenarios of a run at a time, in separate processes (each loads the maps once); the main
process records the outcomes and moves invalid folders in id order, so the results do not depend on N.
DoppelTest's own folders (Generation_*, its records) are kept; the analysis ignores folders without a simulation.db.
"""
import os
import re
import sys
import csv
import glob
import json
import shutil
import logging
import argparse
import numpy as np

from datetime import datetime
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOPPELTEST = os.path.join(ROOT, "EVITA_DoppelTest")
sys.path.insert(0, ROOT)
sys.path.append(DOPPELTEST)

from commonroad.common.util import Interval
from commonroad.planning.goal import GoalRegion
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import CustomState, InitialState
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

from evita import simulators
from evita.apollo import trace as tr
from evita.config import ValidationConfig
from evita.validator import ScenarioValidator
from evita.execution import STATUS_COLLISION
from evita.apollo.health import health_problems
from evita.apollo.map_data import ApolloMapData
from evita.data_analysis import generate_report
from evita.outcomes import OUTCOMES_FILE, record_outcome
from evita.apollo.scenario_converter import APOLLO_VEHICLE
from evita.evaluators.scenario_evaluator import ScenarioEvaluator

import record_export as rx
from hdmap.MapParser import MapParser

from EVITA_DoppelTest.config import (APOLLO_VEHICLE_LENGTH, APOLLO_VEHICLE_WIDTH, APOLLO_VEHICLE_back_edge_to_center,
                                     HD_MAP_PATH, MAX_ADC_COUNT, SCENARIO_UPPER_LIMIT)

GENERATOR = "DoppelTest"
SIMULATOR = "apollo"
COMPLETED_MARKER = "conversion_completed.json"
MAPPING_FILE = "doppeltest_scenarios.csv"
REPORT_FILES = ["behavior.csv", "interaction_map.npy", "collision_map.npy", "collision_map_impact_location_and_riv.npy",
                "collision_map_only_impact_location.npy", "requirements_violations.csv"]

# Obstacle-to-vehicle matching (Apollo ids): a perceived obstacle is a vehicle if its positions stay this close (median)
ID_MATCH_DISTANCE = 4.0
logger = logging.getLogger("convert_doppeltest_runs")


# ---------- Finding runs and scenarios ----------

def find_runs(folders):
    """
    DoppelTest run folders under the given folders: those with Generation_* folders
    """
    runs = set()
    for folder in folders:
        for generation in glob.glob(os.path.join(os.path.abspath(folder), "**", "Generation_*"), recursive = True):
            if os.path.isdir(generation) and "/debug/" not in generation:
                runs.add(os.path.dirname(generation))

    return sorted(runs)

def scenarios_of(run):
    """
    (generation, scenario, folder) of every scenario of the run with a scenario JSON, in order
    """
    found = []
    for folder in sorted(glob.glob(os.path.join(run, "Generation_*", "Scenario_*"))):
        if os.path.exists(os.path.join(folder, "c.json")):
            found.append((os.path.basename(os.path.dirname(folder)), os.path.basename(folder), folder))

    return found

def containers_with_records(records_dir, scenario_name):
    """
    Containers (...ROUTE_k) with record files for the scenario, in order of k
    """

    pattern = re.compile(r"^(\w*ROUTE_(\d+))\." + re.escape(scenario_name) + r"(\.\d+)?$")
    found = {}
    for name in os.listdir(records_dir):
        match = pattern.match(name)
        if match:
            found[int(match.group(2))] = match.group(1)
    
    return [found[k] for k in sorted(found)]

# ---------- Trace ----------

def recover_apollo_ids(trace, vehicles):
    """
    Each vehicle's Apollo id, from the obstacles the other vehicles perceived: an obstacle whose positions stay within
    ID_MATCH_DISTANCE (median) of a vehicle's own localization is that vehicle (votes over all observers)
    """
    # Extract each vehicle's positions (t, x, y) from its localization rows
    positions = {}
    for vehicle in trace["vehicles"]:
        rows = np.array(vehicle["localization"]["rows"], dtype = float) if vehicle["localization"]["rows"] else None
        if rows is not None:
            columns = vehicle["localization"]["columns"]
            positions[vehicle["container"]] = (rows[:, columns.index("t")], rows[:, columns.index("x")], rows[:, columns.index("y")])
    
    # Count the number of times each obstacle is perceived by each vehicle
    votes = {}
    for observer in trace["vehicles"]:
        samples = {}
        for message in observer.get("obstacles", [])[::5]:
            for obstacle in message["obstacles"]:
                samples.setdefault(obstacle["id"], []).append((message["t"], obstacle["x"], obstacle["y"]))

        for obstacle_id, points in samples.items():
            points = np.array(points, dtype = float)
            best, best_distance = None, ID_MATCH_DISTANCE
            for container, (t, x, y) in positions.items():
                if container == observer["container"] or len(t) < 2:
                    continue

                distance = float(np.median(np.hypot(np.interp(points[:, 0], t, x) - points[:, 1], np.interp(points[:, 0], t, y) - points[:, 2])))
                if distance < best_distance:
                    best, best_distance = container, distance
            
            if best is not None:
                votes.setdefault(best, {}).setdefault(obstacle_id, 0)
                votes[best][obstacle_id] += 1
    
    for vehicle in vehicles:
        candidates = votes.get(vehicle["container"], {})
        vehicle["apollo_id"] = max(candidates, key = candidates.get) if candidates else None
    
    return vehicles

def build_scenario_trace(folder, generation, scenario_name, json_scenario, trace_path):
    """
    Write the scenario's trace from DoppelTest's record files (Apollo ids from ids_map.json, else recovered)
    """
    containers = containers_with_records(folder, scenario_name)
    if not containers:
        raise RuntimeError("no record files")
    
    adcs = json_scenario.get("ad_section", {}).get("adcs", [])
    vehicles = []
    for container in containers:
        index = int(container.rsplit("_", 1)[-1])
        adc = adcs[index] if index < len(adcs) else {}
        vehicles.append({"container": container, "apollo_id": None, "json_index": index, "evita_id": None,
                         "routing": adc.get("routing", []), "start_s": adc.get("start_s"), "dest_s": adc.get("dest_s"),
                         "start_t": adc.get("start_t")})
    
    # Build the scenario section
    section = {"generation": generation, "scenario": scenario_name, "status": "executed",
               "reason": "converted from DoppelTest's record files", "hd_map": HD_MAP_PATH,
               "scenario_upper_limit_s": SCENARIO_UPPER_LIMIT, "max_adc_count": MAX_ADC_COUNT,
               "vehicle": {"length": APOLLO_VEHICLE_LENGTH, "width": APOLLO_VEHICLE_WIDTH,
                           "back_edge_to_center": APOLLO_VEHICLE_back_edge_to_center},
               "json": json_scenario}
    
    # Build the scenario trace
    trace = rx.build_trace(folder, scenario_name, section, vehicles)
    ids_map = os.path.join(folder, "ids_map.json")
    if os.path.exists(ids_map):
        by_container = {container: int(nid) for nid, container in json.load(open(ids_map)).items()}
        for vehicle in trace["vehicles"]:
            vehicle["apollo_id"] = by_container.get(vehicle["container"])
    else:
        recover_apollo_ids(trace, trace["vehicles"])
    
    rx.write_trace(trace, trace_path)

# ---------- CommonRoad scenario from DoppelTest's JSON ----------

def vehicle_center(mp, lane, s):
    """
    Position and heading of the vehicle's center when Apollo's reference point is at s on the lane
    """
    point, heading = mp.get_coordinate_and_heading(lane, s)
    offset = APOLLO_VEHICLE.center_offset

    return np.array([point.x + offset * np.cos(heading), point.y + offset * np.sin(heading)]), float(heading)

def rebuild_scenario(base_scenario, base_pps, json_scenario, map_data, config):
    """
    The scenario as CommonRoad planning problems (ids 30000 + index, as in the trace), for EVITA's validation
    """
    mp = MapParser.get_instance()
    planning_problems = PlanningProblemSet()

    for index, adc in enumerate(json_scenario["ad_section"]["adcs"]):
        lanes = [lane for k, lane in enumerate(adc["routing"]) if k == 0 or lane != adc["routing"][k - 1]]
        start, start_heading = vehicle_center(mp, lanes[0], adc["start_s"])
        goal, goal_heading = vehicle_center(mp, lanes[-1], adc["dest_s"])

        initial = InitialState(position = start, orientation = start_heading, velocity = 0.0,
                               time_step = int(round(adc["start_t"] / base_scenario.dt)), yaw_rate = 0.0, slip_angle = 0.0, acceleration = 0.0)
        goal_region = GoalRegion([CustomState(time_step = Interval(0, config.max_time_step),
                                              position = Rectangle(config.length, config.width, goal, goal_heading))])
        problem = PlanningProblem(30000 + index, initial, goal_region)
        problem.path = [map_data.lanelet_id(lane) for lane in lanes]
        planning_problems.add_planning_problem(problem)

    return base_scenario, planning_problems

# ---------- One run ----------

def clear_conversion(run):
    """
    Remove previous conversion of the run (its scenario folders, outcomes, performance, metadata, report)
    """
    for folder in glob.glob(os.path.join(run, "[0-9][0-9][0-9][0-9][0-9]")) + [os.path.join(run, "debug")]:
        shutil.rmtree(folder, ignore_errors = True)

    for name in [OUTCOMES_FILE, "performance.csv", "metadata.csv", MAPPING_FILE, COMPLETED_MARKER] + REPORT_FILES:
        if os.path.exists(os.path.join(run, name)):
            os.remove(os.path.join(run, name))

# Per process: the HD map (DoppelTest's map parser), the converted map, the configuration and EVITA's validator
_WORKER = {}

def init_worker(map_file):
    """
    Load what every scenario's conversion needs, once per process (the main process too, with --workers 1)
    """
    logging.basicConfig(level = logging.ERROR)
    MapParser(HD_MAP_PATH)
    config = simulators.generation_configuration(simulators.get_simulator(simulators.APOLLO))
    _WORKER.update(map_file = map_file, map_data = ApolloMapData.load(map_file), config = config, validator = ScenarioValidator(ValidationConfig(), config, apollo = True))

def convert_scenario(job):
    """
    One scenario: trace, rebuilt scenario, validation, health check, evaluation into its folder.
    Returns its id, the reason it is invalid ("" if valid) and its number of vehicles; the main process records the
    outcome and moves invalid folders to debug/.
    """
    number, generation, scenario_name, source, run = job
    map_file, map_data, config, validator = _WORKER["map_file"], _WORKER["map_data"], _WORKER["config"], _WORKER["validator"]
    folder = os.path.join(run, str(number).rjust(5, "0"))
    os.makedirs(folder, exist_ok = True)
    reason, vehicles = "", 0

    try:
        # Open original DoppelTest scenario and planning problems
        json_scenario = json.load(open(os.path.join(source, "c.json")))
        vehicles = len(json_scenario.get("ad_section", {}).get("adcs", []))
        shutil.copy2(os.path.join(source, "c.json"), os.path.join(folder, "c.json"))

        # Rebuild CommonRoad scenario and planning problems
        base, base_pps = CommonRoadFileReader(map_file).open()
        scenario, planning_problems = rebuild_scenario(base, base_pps, json_scenario, map_data, config)
        CommonRoadFileWriter(scenario, planning_problems).write_to_file(os.path.join(folder, "scenario.xml"), OverwriteExistingFile.ALWAYS)

        valid, message = validator.is_scenario_valid(scenario, planning_problems)
        if not valid:
            reason = f"Refused by EVITA's Apollo validation: {message}"
        
        # Build the scenario trace
        trace_path = os.path.join(folder, "trace.json.gz")
        build_scenario_trace(source, generation, scenario_name, json_scenario, trace_path)
        execution = tr.to_execution_result(tr.read_trace(trace_path), scenario, map_data, folder, scenario_id = number)

        if not reason:
            collisions = {a.id: a.last_timestep for a in execution.agents.values() if a.status == STATUS_COLLISION}
            unhealthy = health_problems(execution.native, collisions)

            if unhealthy:
                reason = "Apollo's run is unhealthy: " + "; ".join(unhealthy)
        
        if not reason:
            ScenarioEvaluator(folder).evaluate(execution, None)
    except Exception as exc:
        reason = reason or f"Cannot convert the scenario: {type(exc).__name__}: {exc}"
        
    return number, reason, vehicles

def convert_run(run, map_file, workers):
    """
    Convert all scenarios of the run (in parallel with --workers N), record their outcomes, move invalid folders to debug/
    """
    # Extract all run scenarios
    scenarios = scenarios_of(run)
    if not scenarios:
        print("  no scenario with c.json, skipped")
        return None
    
    started = datetime.now()

    # Write the mapping of scenario ids to DoppelTest's generation and scenario names
    jobs = [(number, generation, scenario_name, source, run) for number, (generation, scenario_name, source) in enumerate(scenarios, start = 1)]
    with open(os.path.join(run, MAPPING_FILE), "w", newline = "") as mapping_file:
        mapping = csv.writer(mapping_file)
        mapping.writerow(["scenario", "generation", "doppeltest_scenario"])
        for number, generation, scenario_name, _, _ in jobs:
            mapping.writerow([number, generation, scenario_name])

    # The scenarios, in parallel with --workers N (each worker writes only its own scenario's folder)
    results = {}
    def progress(done):
        if done % 25 == 0 or done == len(jobs):
            print(f"  {done}/{len(jobs)} scenarios converted", flush = True)

    if workers <= 1:
        if not _WORKER:
            init_worker(map_file)
        for job in jobs:
            number, reason, vehicles = convert_scenario(job)
            results[number] = (reason, vehicles)
            progress(len(results))
    else:
        with ProcessPoolExecutor(max_workers = workers, initializer = init_worker, initargs = (map_file,)) as pool:
            for future in as_completed([pool.submit(convert_scenario, job) for job in jobs]):
                number, reason, vehicles = future.result()
                results[number] = (reason, vehicles)
                progress(len(results))

    # Outcomes in id order, invalid folders to debug/ (main process only)
    counts, causes = {"valid": 0, "invalid": 0}, {}
    for number in sorted(results):
        reason, vehicles = results[number]
        outcome = SimpleNamespace(generation_id = number, is_valid = not reason, validation_message = reason)
        record_outcome(run, outcome, SimpleNamespace(planning_problem_dict = dict.fromkeys(range(vehicles))), True, SIMULATOR)
        folder = os.path.join(run, str(number).rjust(5, "0"))
        if reason:
            counts["invalid"] += 1
            causes[reason.split(":")[0]] = causes.get(reason.split(":")[0], 0) + 1
            os.makedirs(os.path.join(run, "debug"), exist_ok = True)

            with open(os.path.join(folder, "reason.txt"), "w") as f:
                f.write(reason + "\n")
            
            shutil.move(folder, os.path.join(run, "debug", os.path.basename(folder)))
        else:
            counts["valid"] += 1

    # Write the performance data to a CSV file
    with open(os.path.join(run, "performance.csv"), "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time", "skipped_scenarios"])
        writer.writerow([len(scenarios), counts["valid"], counts["invalid"], "", 0])

    parameters = {}
    if os.path.exists(os.path.join(run, "run_parameters.json")):
        parameters = json.load(open(os.path.join(run, "run_parameters.json")))
    
    # Write the metadata to a CSV file
    with open(os.path.join(run, "metadata.csv"), "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["generator", "scenario_name", "time_budget", "timestamp_id", "simulator"])
        hours = parameters.get("run_for_hour")
        writer.writerow([GENERATOR, parameters.get("map", os.path.splitext(os.path.basename(map_file))[0]),
                         float(hours) * 3600 if hours is not None else "", os.path.basename(run), SIMULATOR])
        
    if counts["valid"]:
        generate_report(run)
    
    with open(os.path.join(run, COMPLETED_MARKER), "w") as f:
        json.dump({"converted": datetime.now().isoformat(timespec = "seconds"), "seconds": round((datetime.now() - started).total_seconds(), 1),
                   "workers": workers, "scenarios": len(scenarios), **counts, "invalid_causes": causes}, f, indent = 2)
    
    return len(scenarios), counts, causes

def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Convert DoppelTest's Apollo runs into EVITA run folders")
    parser.add_argument("folders", nargs = "+", help = "Folders searched for DoppelTest run folders (with Generation_* folders)")
    default_map = os.path.join(DOPPELTEST, "data", "converted_maps", os.path.basename(os.path.dirname(HD_MAP_PATH)), os.path.basename(os.path.dirname(HD_MAP_PATH)) + ".xml")
    parser.add_argument("--map", default = default_map, help = "Converted map of Apollo's HD map (default: HD_MAP_PATH's)")
    parser.add_argument("--force", action = "store_true", help = "Convert runs again (their previous conversion is removed)")
    parser.add_argument("--workers", type = int, default = 5, help = "Scenarios converted in parallel (default 5); each worker needs memory for one trace")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    
    logging.basicConfig(level = logging.ERROR)

    runs = find_runs(args.folders)
    print(f"{len(runs)} DoppelTest run(s) found; {args.workers} worker(s)")
    
    if not runs:
        return 1

    for run in runs:
        print(f"\n== {os.path.relpath(run, ROOT) if run.startswith(ROOT) else run}")

        if os.path.exists(os.path.join(run, COMPLETED_MARKER)) and not args.force:
            print("  already converted (use --force to convert it again)")
            continue

        clear_conversion(run)

        result = convert_run(run, args.map, args.workers)
        if result:
            total, counts, causes = result
            print(f"  {total} scenarios: {counts['valid']} valid (evaluated), {counts['invalid']} invalid (in debug/)" +
                  "".join(f"\n      {n:4d}  {cause}" for cause, n in sorted(causes.items(), key = lambda item: -item[1])))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))