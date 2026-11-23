"""
Live Apollo test of the new pipeline, without EVITA's search and without the evaluation:
generation -> execution -> conversion, for a few scenarios.

    python scripts/apollo_live_run.py --map EVITA_DoppelTest/conversion_scripts/converted_maps/borregas_ave.xml \
                                      [--scenarios 3] [--vehicles 2] [--seed 1] [--out tmp/apollo_live] [--dry-run]

The Apollo containers must be running (as for EVITA: EVITA_DoppelTest/apollo_runner_startup.py), unless --dry-run.

For each scenario:
1. Generation: EVITA's Apollo generator (scenario_mutation.add_evita_random_planning_problem, validated by the
   ScenarioValidator in Apollo mode, retried as in EVITA's sampling), on the given map, with Apollo's vehicle model
   (cr_vehicle_id 101); DoppelTest traffic lights (evita.apollo.traffic_control); the DoppelTest JSON
   (evita.apollo.scenario_converter). A scenario the converter cannot describe is regenerated.
   Written: scenario.xml (with the planning problems and traffic lights), c.json.
2. Execution (not with --dry-run): EVITA_DoppelTest/apollo_scenario_runner.py with .venv38's Python, with a timeout.
   Written: runner.log, trace.json.gz. Exit codes: 0 executed, 2 rejected, 3 unexpected error, 10 records problem.
3. Conversion (if executed): evita.apollo.trace. Written: executed.xml, report.txt.
   Reported per vehicle: whether its Apollo ran, where it started vs where the planning problem put it (along the
   lane and sideways: both should be about 0 with the new conversion), first motion vs its start time, mission
   complete, last time step, collision entries; for the scenario, at how many time steps the received traffic lights
   differ from the specified ones (runner clock drift).
A summary (summary.csv and on screen) lists every scenario.

Run in .venv310 (from any folder).
"""

import os
import sys
import random
import argparse
import subprocess
import contextlib
from copy import deepcopy

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOPPELTEST = os.path.join(PROJECT_ROOT, "EVITA_DoppelTest")
RUNNER = os.path.join(DOPPELTEST, "apollo_scenario_runner.py")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from cr_scenario_handler.utils.vehicle_models import APOLLO_LINCOLN_MKZ, vehicle_parameters

from evita.config import ScenarioGeneratorConfig, ValidationConfig
from evita.validator import ScenarioValidator
from evita.generators import scenario_mutation
from evita.generators.utils import get_base_scenario_from_file
from evita.apollo.map_data import ApolloMapData
from evita.apollo import traffic_control as tc
from evita.apollo import scenario_converter as converter
from evita.apollo import trace as tr

SCENARIO_DURATION_STEPS = 300           # DoppelTest's SCENARIO_UPPER_LIMIT (30 s) in time steps of 0.1 s
MAX_VEHICLES = 5                        # DoppelTest's MAX_ADC_COUNT


# ---------- 1. Generation ----------

def make_config(max_vehicles: int, vehicle_model: int) -> ScenarioGeneratorConfig:
    """
    EVITA's generation configuration for Apollo (as main_ga_evita.py), with the vehicle model's dimensions
    """
    config = ScenarioGeneratorConfig(SCENARIO_DURATION_STEPS, max_vehicles)
    config.cr_vehicle_id = vehicle_model
    parameters = vehicle_parameters(vehicle_model)
    config.length, config.width = parameters.l, parameters.w
    return config


def generate(base_scenario, base_planning_problems, config, map_file, map_data, vehicles, rng):
    """
    One valid scenario (EVITA's Apollo generator and validator), with DoppelTest traffic lights and its JSON.
    Returns (scenario, planning problems, JSON, attempts).
    """
    validator = ScenarioValidator(ValidationConfig(), config, apollo = True)
    lights = [light.traffic_light_id for light in base_scenario.lanelet_network.traffic_lights]
    target = vehicles or rng.randint(config.min_amount_pp, config.max_amount_pp)
    attempts = 0

    while target >= config.min_amount_pp:
        for _ in range(ValidationConfig.retry_generation):
            attempts += 1
            scenario = deepcopy(base_scenario)
            planning_problems = deepcopy(base_planning_problems)
            for _ in range(target):
                planning_problems = scenario_mutation.add_evita_random_planning_problem(config, scenario, planning_problems, map_file)

            control = tc.random_traffic_control(map_data, lights, tc.max_duration_green(config.max_time_step, scenario.dt), rng)
            control.to_commonroad(scenario.lanelet_network, scenario.dt)

            if not validator.is_scenario_valid(scenario, planning_problems)[0]:
                continue
            try:
                scenario_json = converter.build_scenario_json(scenario, planning_problems, map_data)
            except converter.ScenarioSpecError:
                continue
            return scenario, planning_problems, scenario_json, attempts
        if vehicles:
            break
        target -= 1
    raise RuntimeError(f"No valid scenario after {attempts} attempts")


# ---------- 2. Execution ----------

def execute(python38, scenario_json_path, records_dir, index, trace_path, log_path, timeout):
    command = [python38, RUNNER, "--scenario-json", scenario_json_path, "--records-dir", records_dir,
               "--scenario", str(index), "--trace", trace_path]
    environment = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python")
    with open(log_path, "w") as log:
        try:
            result = subprocess.run(command, cwd = DOPPELTEST, stdout = log, stderr = subprocess.STDOUT, env = environment, timeout = timeout)
            return result.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout} s\n")
            return "timeout"


# ---------- 3. Conversion ----------

def convert(trace_path, scenario, planning_problems, map_data, executed_path, report_path):
    """
    Executed scenario and a report; returns one row per vehicle and the number of time steps with differing lights
    """
    trace = tr.read_trace(trace_path)
    conversion = tr.convert_trace(trace, scenario, map_data)
    CommonRoadFileWriter(conversion.executed_scenario, planning_problems).write_to_file(executed_path, OverwriteExistingFile.ALWAYS)

    offset = trace.vehicle.center_offset
    starts = {pp.planning_problem_id: (np.array(pp.initial_state.position), pp.initial_state.time_step)
              for pp in planning_problems.planning_problem_dict.values()}
    rows = []
    for vehicle in trace.vehicles:
        executed = conversion.vehicles.get(vehicle.vehicle_id)
        row = {"vehicle": vehicle.vehicle_id, "container": vehicle.container, "ran": len(vehicle.localization) > 0,
               "plans": len(vehicle.planning), "collision_entries": len(vehicle.oracles.get("collisions", []))}
        if row["ran"] and vehicle.vehicle_id in starts:
            heading = vehicle.column("heading")[0]
            direction = np.array([np.cos(heading), np.sin(heading)])
            center = np.array([vehicle.column("x")[0], vehicle.column("y")[0]]) + offset * direction
            difference = starts[vehicle.vehicle_id][0] - center
            row["start_along_m"] = round(float(difference @ direction), 2)
            row["start_sideways_m"] = round(float(abs(direction[0] * difference[1] - direction[1] * difference[0])), 2)
            row["start_time_step"] = starts[vehicle.vehicle_id][1]
        if executed is not None:
            row.update({"first_motion_step": executed.first_motion_step, "mission_complete_step": executed.mission_complete_step,
                        "last_step": executed.last_time_step})
        rows.append(row)

    # Traffic lights as received vs as specified
    specified = tc.TrafficControl.from_commonroad(scenario.lanelet_network, scenario.dt)
    received = {light.traffic_light_id: light for light in conversion.executed_scenario.lanelet_network.traffic_lights}
    last = max((r.get("last_step") or 0) for r in rows)
    differing = sum(1 for k in range(last + 1)
                    if any(tc.COLOR_OF_STATE.get(received[i].get_state_at_time_step(k)) != color
                           for i, color in specified.colors_at(k * scenario.dt).items()))

    with open(report_path, "w") as report:
        report.write(pd.DataFrame(rows).to_string(index = False) + "\n")
        report.write(f"\nTraffic lights: received colors differ from the specified ones at {differing} of {last + 1} time steps\n")
        for warning in conversion.warnings:
            report.write(f"warning: {warning}\n")
    return rows, differing, last + 1, conversion.warnings


# ---------- Main ----------

def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Live Apollo test: generation, execution, conversion (no evaluation)")
    parser.add_argument("--map", required = True, help = "converted CommonRoad map (its id files and feasible paths next to it)")
    parser.add_argument("--scenarios", type = int, default = 3, help = "number of scenarios")
    parser.add_argument("--vehicles", type = int, help = "vehicles per scenario (default: random, as EVITA)")
    parser.add_argument("--max-vehicles", type = int, default = MAX_VEHICLES, help = "largest random number of vehicles")
    parser.add_argument("--vehicle-model", type = int, default = APOLLO_LINCOLN_MKZ, help = "cr_vehicle_id (default 101, Apollo's)")
    parser.add_argument("--seed", type = int, default = 1)
    parser.add_argument("--out", default = os.path.join(PROJECT_ROOT, "tmp", "apollo_live"))
    parser.add_argument("--python38", default = os.path.join(PROJECT_ROOT, ".venv38", "bin", "python"))
    parser.add_argument("--timeout", type = int, default = 600, help = "seconds per scenario execution")
    parser.add_argument("--dry-run", action = "store_true", help = "generation only (no Apollo)")
    args = parser.parse_args(argv)
    args.map, args.out, args.python38 = os.path.abspath(args.map), os.path.abspath(args.out), os.path.abspath(args.python38)

    # Checks before starting
    map_data = ApolloMapData.load(args.map)
    base_scenario, base_planning_problems = get_base_scenario_from_file(args.map, remove_obstacles = True, remove_planning_problems = True)

    if not args.dry_run and not os.path.exists(args.python38):
        print(f"Missing {args.python38} (give .venv38's Python with --python38)", file = sys.stderr)
        return 1

    # The conversion's warnings are in each report.txt: not repeated on screen
    import logging
    logging.getLogger("evita.apollo.trace").setLevel(logging.ERROR)

    rng = random.Random(args.seed)
    random.seed(args.seed)          # EVITA's generator uses the random module
    config = make_config(args.max_vehicles, args.vehicle_model)
    os.makedirs(args.out, exist_ok = True)
    print(f"{args.scenarios} scenarios on {args.map}; vehicle model {args.vehicle_model} ({config.length} x {config.width} m); output {args.out}")

    summary, all_vehicles = [], []
    for index in range(args.scenarios):
        folder = os.path.join(args.out, f"Scenario_{index}")
        os.makedirs(folder, exist_ok = True)
        row = {"scenario": f"Scenario_{index}"}

        scenario, planning_problems, scenario_json, attempts = generate(base_scenario, base_planning_problems, config, args.map, map_data, args.vehicles, rng)
        scenario_xml, json_path = os.path.join(folder, "scenario.xml"), os.path.join(folder, "c.json")
        CommonRoadFileWriter(scenario, planning_problems).write_to_file(scenario_xml, OverwriteExistingFile.ALWAYS)
        converter.write_scenario_json(scenario_json, json_path)
        row.update({"vehicles": len(scenario_json["ad_section"]["adcs"]), "generation_attempts": attempts,
                    "duration_g_s": scenario_json["tc_section"]["duration_g"]})
        print(f"  Scenario_{index}: {row['vehicles']} vehicles (after {attempts} attempts), green {row['duration_g_s']} s", end = "", flush = True)

        if args.dry_run:
            print()
            summary.append(row)
            continue

        trace_path = os.path.join(folder, "trace.json.gz")
        code = execute(args.python38, json_path, os.path.join(args.out, "records"), index, trace_path, os.path.join(folder, "runner.log"), args.timeout)
        row["runner_exit"] = code
        if code != 0:
            print(f"; runner exit {code} (see {folder}/runner.log)")
            summary.append(row)
            continue

        try:
            vehicles, differing, steps, warnings = convert(trace_path, scenario, planning_problems, map_data,
                                                           os.path.join(folder, "executed.xml"), os.path.join(folder, "report.txt"))
        except Exception as exc:
            print(f"; conversion failed: {type(exc).__name__}: {exc}")
            row["conversion"] = f"{type(exc).__name__}: {exc}"
            summary.append(row)
            continue

        for v in vehicles:
            v["scenario"] = row["scenario"]
        all_vehicles += vehicles
        ran = [v for v in vehicles if v["ran"]]
        row.update({"conversion": "ok", "vehicles_not_run": len(vehicles) - len(ran), "light_steps_differing": differing,
                    "steps": steps, "max_start_sideways_m": max((v.get("start_sideways_m", 0) for v in ran), default = None),
                    "max_start_along_m": max((abs(v.get("start_along_m", 0)) for v in ran), default = None),
                    "missions_complete": sum(v.get("mission_complete_step") is not None for v in ran),
                    "collision_entries": sum(v["collision_entries"] for v in vehicles), "warnings": len(warnings)})
        print(f"; executed: {len(ran)}/{len(vehicles)} ran, {row['missions_complete']} completed, start error up to "
              f"{row['max_start_along_m']} m along / {row['max_start_sideways_m']} m sideways, lights differ at {differing}/{steps} steps")
        summary.append(row)

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index = False)
    if all_vehicles:
        pd.DataFrame(all_vehicles).to_csv(os.path.join(args.out, "vehicles.csv"), index = False)
    print(f"\nSummary in {args.out}/summary.csv" + (f", vehicles in {args.out}/vehicles.csv" if all_vehicles else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
