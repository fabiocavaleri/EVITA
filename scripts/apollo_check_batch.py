"""
Run the step 4 checks (trace, convert, spec) on every scenario of an earlier Apollo run.

    python scripts/apollo_check_batch.py --records-root <...>/records/<run> --map <converted map.xml> [options]

Run from any folder (paths are taken from this file's location). For each <records root>/Generation_g/Scenario_i:
    1. export the trace from its record files (EVITA_DoppelTest/export_existing_records.py, with .venv38's Python)
    2. scripts/apollo_check.py trace / convert / spec, each report written to a text file
The old pipeline's outputs are read from the same path under evita-records (records -> evita-records):
ids_map.json, executed_scenario.xml (for the comparison) and the generated scenario (scenario/scenario.xml, or the
only other .xml file of the folder, or --scenario-file).

Outputs, in --out (default /tmp/apollo_check):
    <Generation_g>_<Scenario_i>/  trace.json.gz, export.log, trace.txt, convert.txt, spec.txt, executed_new.xml
    vehicles.csv                  one row per vehicle: what was recorded, conversion facts, comparisons
    scenarios.csv                 one row per scenario: outcome of each step, problems found
and an overview on screen.

Run in .venv310.
"""

import os
import re
import sys
import glob
import argparse
import contextlib
import subprocess
from types import SimpleNamespace

import logging

import numpy as np
import pandas as pd

# Paths from this file's location, so the script works from any folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOPPELTEST = os.path.join(PROJECT_ROOT, "EVITA_DoppelTest")
EXPORT_SCRIPT = os.path.join(DOPPELTEST, "export_existing_records.py")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import apollo_check
from commonroad.common.file_reader import CommonRoadFileReader
from evita.apollo.map_data import ApolloMapData
from evita.apollo import trace as tr



def natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def scenario_folders(records_root: str):
    """
    (generation, scenario, records folder) of every scenario, in natural order
    """
    found = []
    for generation in sorted(os.listdir(records_root), key = natural_key):
        generation_path = os.path.join(records_root, generation)
        if not (generation.startswith("Generation_") and os.path.isdir(generation_path)):
            continue
        for scenario in sorted(os.listdir(generation_path), key = natural_key):
            path = os.path.join(generation_path, scenario)
            if scenario.startswith("Scenario_") and os.path.isdir(path):
                found.append((generation, scenario, path))
    return found


def generated_scenario(evita_dir: str, scenario_file: str = None):
    """
    The generated scenario of a scenario's evita-records folder: --scenario-file if given; otherwise, in this order,
    scenario/scenario.xml, scenario.xml, the only other .xml file of the folder itself, the only one in its subfolders
    (executed_scenario.xml never counts)
    """
    if scenario_file:
        path = os.path.join(evita_dir, scenario_file)
        return path if os.path.exists(path) else None
    for name in (os.path.join("scenario", "scenario.xml"), "scenario.xml"):
        path = os.path.join(evita_dir, name)
        if os.path.exists(path):
            return path
    for pattern in ("*.xml", os.path.join("**", "*.xml")):
        candidates = [f for f in glob.glob(os.path.join(evita_dir, pattern), recursive = True)
                      if os.path.basename(f) != "executed_scenario.xml"]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return None
    return None


def xml_files(evita_dir: str) -> str:
    """
    The .xml files of a scenario's folder (to report when the generated scenario cannot be chosen)
    """
    return " ".join(os.path.relpath(f, evita_dir) for f in glob.glob(os.path.join(evita_dir, "**", "*.xml"), recursive = True))


def same_run(records_dir: str, evita_dir: str):
    """
    Whether the records and the evita-records folder belong to the same run: the scenario JSON saved with the records
    (c.json) and the one saved by EVITA (evita_scenario.json) must describe the same vehicles. None if one is missing.
    """
    import json
    paths = (os.path.join(records_dir, "c.json"), os.path.join(evita_dir, "evita_scenario.json"))
    if not all(os.path.exists(p) for p in paths):
        return None
    first, second = (json.load(open(p)) for p in paths)
    return first.get("ad_section") == second.get("ad_section")


def run_report(function, args, report_path):
    """
    Run one apollo_check command with its output written to report_path; returns the error message or ""
    """
    with open(report_path, "w") as report, contextlib.redirect_stdout(report):
        try:
            function(args)
            return ""
        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")
            return f"{type(exc).__name__}: {exc}"


def vehicle_rows(trace_path, generation, scenario, scenario_xml, map_file, old_executed):
    """
    One row per vehicle with what was recorded, the conversion's facts and the comparison with the old executed scenario
    """
    trace = tr.read_trace(trace_path)
    if trace.status != tr.EXECUTED:
        return []

    t0 = tr.time_zero(trace)
    conversion, old = None, None
    base, _, original_ids = apollo_check.prepare_conversion(trace, scenario_xml, map_file, old_executed)
    planned_starts = []
    if scenario_xml:
        _, planning_problems = CommonRoadFileReader(scenario_xml).open()
        planned_starts = [np.array(pp.initial_state.position) for pp in planning_problems.planning_problem_dict.values()]
    offset = trace.vehicle.center_offset
    try:
        conversion = tr.convert_trace(trace, base, ApolloMapData.load(map_file))
    except Exception:
        conversion = None
    if old_executed:
        old, _ = CommonRoadFileReader(old_executed).open()

    rows = []
    for vehicle in trace.vehicles:
        times = vehicle.column("t")
        plans = vehicle.planning
        main = pd.Series([p["main_decision"]["decision"] for p in plans]).value_counts() if plans else pd.Series(dtype = int)
        row = {
            "generation": generation, "scenario": scenario, "container": vehicle.container,
            "evita_id": vehicle.evita_id, "apollo_id": vehicle.apollo_id,
            "localization_samples": len(times),
            "localization_s": round((times[-1] - times[0]) / 1e9, 2) if len(times) else 0,
            "localization_max_gap_s": round(float(np.max(np.diff(times))) / 1e9, 3) if len(times) > 1 else None,
            # Where the largest gap starts, from time step 0 (negative: while the runner prepares the scenario)
            "localization_max_gap_at_s": round(float(times[int(np.argmax(np.diff(times)))] - t0) / 1e9, 2) if len(times) > 1 else None,
            "localization_start_s": round((times[0] - t0) / 1e9, 2) if len(times) else None,
            "plans": len(plans),
            "plan_interval_median_ms": round(float(np.median(np.diff([p["t"] for p in plans]))) / 1e6, 1) if len(plans) > 1 else None,
            "plans_stop_share": round(main.get("stop", 0) / len(plans), 2) if plans else None,
            "plans_not_ready": int(main.get("not_ready", 0)),
            "mission_complete_plans": int(main.get("mission_complete", 0)),
            "estop_plans": sum(p["estop"]["is_estop"] for p in plans),
            "replan_plans": sum(p["is_replan"] for p in plans),
            "object_kinds": " ".join(sorted({apollo_check.object_kind(o["id"]) for p in plans for o in p["object_decisions"]})),
            "light_messages": len(vehicle.traffic_lights),
            "light_interval_median_ms": round(float(np.median(np.diff([m["t"] for m in vehicle.traffic_lights]))) / 1e6, 1) if len(vehicle.traffic_lights) > 1 else None,
            "obstacle_ids_seen": " ".join(str(i) for i in sorted({o["id"] for m in vehicle.obstacles for o in m["obstacles"]})),
            "collision_entries": len(vehicle.oracles.get("collisions", [])),
            "problem": "no localization (Apollo did not run)" if not len(times) else ("no planning" if not plans else "")
        }

        # Where Apollo actually started the vehicle (its center at the first localization) vs where the planning problem
        # put it, along Apollo's heading and sideways. The old conversion snapped starts to centerline vertices: errors
        # along the lane (up to ~27 m on Borregas Ave). A sideways offset of a lane width or more means another lane
        # (e.g. a scenario and a JSON from different map versions)
        if len(times) and vehicle.json_index < len(planned_starts):
            heading = vehicle.column("heading")[0]
            direction = np.array([np.cos(heading), np.sin(heading)])
            center = np.array([vehicle.column("x")[0], vehicle.column("y")[0]]) + offset * direction
            difference = planned_starts[vehicle.json_index] - center
            row["start_vs_planned_along_m"] = round(float(difference @ direction), 2)
            row["start_vs_planned_sideways_m"] = round(float(abs(direction[0] * difference[1] - direction[1] * difference[0])), 2)

        executed = conversion.vehicles.get(vehicle.vehicle_id) if conversion else None
        if executed is not None:
            row.update({"first_motion_step": executed.first_motion_step, "mission_complete_step": executed.mission_complete_step,
                        "last_step": executed.last_time_step, "disengagement_steps": len(executed.disengagement_steps)})

        if old is not None and executed is not None and executed.obstacle is not None:
            old_obstacle = apollo_check.old_obstacle_of(old, original_ids.get(vehicle.vehicle_id, vehicle.vehicle_id), vehicle)
            if old_obstacle is not None:
                old_states = [old_obstacle.initial_state] + (old_obstacle.prediction.trajectory.state_list if old_obstacle.prediction else [])
                new_points = np.array([s.position for s in executed.driven_states])
                old_path = np.array([s.position for s in old_states])
                distances = apollo_check.path_distance(new_points, old_path)
                row.update({"old_path_distance_median_m": round(float(np.median(distances)), 3) if len(distances) else None,
                            "old_path_distance_max_m": round(float(np.max(distances)), 3) if len(distances) else None,
                            "start_shift_m": round(float(np.linalg.norm(new_points[0] - old_path[0])), 3)})
        rows.append(row)
    return rows


def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Step 4 checks on every scenario of an Apollo run")
    parser.add_argument("--records-root", required = True, help = "folder with the Generation_g folders of the records")
    parser.add_argument("--evita-records-root", help = "same for evita-records (default: records -> evita-records)")
    parser.add_argument("--map", required = True, help = "converted CommonRoad map used by the run")
    parser.add_argument("--out", default = "/tmp/apollo_check", help = "output folder")
    parser.add_argument("--python38", default = os.path.join(PROJECT_ROOT, ".venv38", "bin", "python"), help = "Python of .venv38")
    parser.add_argument("--hd-map", help = "Apollo HD map of the run (default: EVITA_DoppelTest's config.HD_MAP_PATH)")
    parser.add_argument("--scenario-file", help = "path of the generated scenario within each scenario's evita-records folder "
                                                  "(default: scenario/scenario.xml, or the only other .xml file there)")
    parser.add_argument("--random-lights", type = int, default = 1, help = "seed of the DoppelTest lights for spec")
    parser.add_argument("--limit", type = int, help = "check only the first N scenarios")
    args = parser.parse_args(argv)

    # Absolute paths: the export runs in EVITA_DoppelTest/, so relative paths would point elsewhere there
    for name in ("records_root", "evita_records_root", "map", "out", "python38", "hd_map"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))

    records_root = os.path.abspath(args.records_root)
    evita_root = os.path.abspath(args.evita_records_root) if args.evita_records_root else records_root.replace("/records/", "/evita-records/")
    if evita_root == records_root:
        parser.error("cannot derive the evita-records folder: give --evita-records-root")

    # The conversion's warnings are in each convert.txt: not repeated on screen
    logging.getLogger("evita.apollo").setLevel(logging.ERROR)

    # Stop at once if a tool is missing, instead of failing on every scenario
    missing = [path for path in (EXPORT_SCRIPT, args.python38) if not os.path.exists(path)]
    if missing:
        print("Missing: " + ", ".join(missing), file = sys.stderr)
        print("Copy export_existing_records.py into EVITA_DoppelTest/, or give .venv38's Python with --python38", file = sys.stderr)
        return 1
    if not os.path.isdir(records_root):
        print(f"No such folder: {records_root}", file = sys.stderr)
        return 1

    scenarios = scenario_folders(records_root)[:args.limit]
    print(f"{len(scenarios)} scenarios in {records_root}")
    if scenarios:
        first = os.path.join(evita_root, scenarios[0][0], scenarios[0][1])
        found = generated_scenario(first, args.scenario_file)
        print(f"Generated scenario of the first one: {found or 'NOT FOUND'} (in {first})")
        if found is None:
            print(f"  .xml files there: {xml_files(first) or 'none'}")
            print("  spec needs it (convert then uses the converted map and the old executed scenario's ids): "
                  "if it has another name, give its path within that folder with --scenario-file")
    os.makedirs(args.out, exist_ok = True)

    all_vehicles, summary = [], []
    for generation, scenario, records_dir in scenarios:
        name = f"{generation}_{scenario}"
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok = True)
        evita_dir = os.path.join(evita_root, generation, scenario)
        ids_map = os.path.join(evita_dir, "ids_map.json")
        scenario_xml = generated_scenario(evita_dir, args.scenario_file)
        old_executed = os.path.join(evita_dir, "executed_scenario.xml")
        trace_path = os.path.join(out, "trace.json.gz")
        row = {"generation": generation, "scenario": scenario,
               "has_ids_map": os.path.exists(ids_map), "has_scenario_xml": scenario_xml is not None,
               "scenario_xml": os.path.relpath(scenario_xml, evita_dir) if scenario_xml else f"not found; .xml files: {xml_files(evita_dir) or 'none'}",
               "has_old_executed": os.path.exists(old_executed), "same_run": same_run(records_dir, evita_dir)}

        # 1. Export (Python 3.8)
        command = [args.python38, EXPORT_SCRIPT, "--records-dir", records_dir, "--trace", trace_path]
        if row["has_ids_map"]:
            command += ["--ids-map", ids_map]
        if args.hd_map:
            command += ["--hd-map", os.path.abspath(args.hd_map)]
        environment = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python")
        result = subprocess.run(command, cwd = DOPPELTEST, capture_output = True, text = True, env = environment)
        with open(os.path.join(out, "export.log"), "w") as log:
            log.write(result.stdout + result.stderr)
        row["export"] = "ok" if result.returncode == 0 else f"exit {result.returncode}"
        if result.returncode != 0:
            summary.append(row)
            print(f"  {name}: export failed (exit {result.returncode}, see {out}/export.log)")
            continue

        # 2. Reports
        row["trace"] = run_report(apollo_check.command_trace, SimpleNamespace(trace = trace_path), os.path.join(out, "trace.txt")) or "ok"
        row["convert"] = run_report(apollo_check.command_convert, SimpleNamespace(
            trace = trace_path, scenario = scenario_xml, map = args.map, out = os.path.join(out, "executed_new.xml"),
            ids_from_scenario = True, compare = old_executed if row["has_old_executed"] else None), os.path.join(out, "convert.txt")) or "ok"
        if row["has_scenario_xml"]:
            row["spec"] = run_report(apollo_check.command_spec, SimpleNamespace(
                scenario = scenario_xml, map = args.map, out = os.path.join(out, "c_new.json"), out_scenario = None,
                random_lights = args.random_lights, compare = os.path.join(records_dir, "c.json") if os.path.exists(os.path.join(records_dir, "c.json")) else None),
                os.path.join(out, "spec.txt")) or "ok"
        else:
            row["spec"] = "no scenario.xml"

        # 3. Numbers per vehicle
        try:
            vehicles = vehicle_rows(trace_path, generation, scenario, scenario_xml if row["has_scenario_xml"] else None,
                                    args.map, old_executed if row["has_old_executed"] else None)
        except Exception as exc:
            vehicles = []
            row["numbers"] = f"{type(exc).__name__}: {exc}"
        all_vehicles += vehicles
        row["vehicles"] = len(vehicles)
        row["vehicles_not_run"] = sum(v["problem"].startswith("no localization") for v in vehicles)
        row["vehicles_without_plans"] = sum(v["problem"] == "no planning" for v in vehicles)
        row["collision_entries"] = sum(v["collision_entries"] for v in vehicles)
        summary.append(row)
        problems = [f"{k}: {v}" for k, v in row.items() if k in ("trace", "convert", "spec") and v not in ("ok", "no scenario.xml")]
        flags = ["RECORDS AND RESULTS FROM DIFFERENT RUNS (c.json differs from evita_scenario.json)"] if row["same_run"] is False else []
        if row["vehicles_not_run"]:
            flags.append(f"{row['vehicles_not_run']} vehicle(s) did not run")
        if row["vehicles_without_plans"]:
            flags.append(f"{row['vehicles_without_plans']} without plans")
        print(f"  {name}: {len(vehicles)} vehicles" + (f"; {', '.join(flags)}" if flags else "") + (f"; {'; '.join(problems)}" if problems else ""))

    vehicles_df, scenarios_df = pd.DataFrame(all_vehicles), pd.DataFrame(summary)
    vehicles_df.to_csv(os.path.join(args.out, "vehicles.csv"), index = False)
    scenarios_df.to_csv(os.path.join(args.out, "scenarios.csv"), index = False)

    # Overview
    print(f"\nOverview ({len(scenarios_df)} scenarios, {len(vehicles_df)} vehicles; details in {args.out}/scenarios.csv and vehicles.csv)")
    if len(scenarios_df) and "same_run" in scenarios_df:
        different = (scenarios_df["same_run"] == False).sum()
        unknown = scenarios_df["same_run"].isna().sum()
        print(f"  same run (c.json = evita_scenario.json): {(scenarios_df['same_run'] == True).sum()} yes, {different} NO, {unknown} not checked"
              + ("  <- the comparisons of those scenarios are meaningless" if different else ""))
    if len(scenarios_df):
        for step in ("export", "trace", "convert", "spec"):
            if step in scenarios_df:
                print(f"  {step:8s}: {(scenarios_df[step] == 'ok').sum()} ok, {(~scenarios_df[step].isin(['ok', 'no scenario.xml']) & scenarios_df[step].notna()).sum()} failed")
    if len(vehicles_df):
        ran = vehicles_df[vehicles_df["localization_samples"] > 0]
        print(f"  vehicles whose Apollo did not run: {(vehicles_df['localization_samples'] == 0).sum()} of {len(vehicles_df)}, "
              f"in {vehicles_df[vehicles_df['localization_samples'] == 0][['generation', 'scenario']].drop_duplicates().shape[0]} scenarios")
        if len(ran):
            print(f"  localization start from time step 0: median {ran['localization_start_s'].median():.2f} s, max {ran['localization_start_s'].abs().max():.2f} s")
            print(f"  plan interval: median {ran['plan_interval_median_ms'].median():.0f} ms | light interval: median {vehicles_df['light_interval_median_ms'].median():.1f} ms")
            print(f"  plans mostly stop (> 90%): {(ran['plans_stop_share'] > 0.9).sum()} vehicles | estop: {(ran['estop_plans'] > 0).sum()} vehicles | collisions: {(vehicles_df['collision_entries'] > 0).sum()} vehicles")
            if "old_path_distance_median_m" in ran:
                compared = ran.dropna(subset = ["old_path_distance_median_m"])
                print(f"  vs old executed scenario ({len(compared)} vehicles): path distance median {compared['old_path_distance_median_m'].median():.3f} m, "
                      f"max {compared['old_path_distance_max_m'].max():.3f} m; start shift median {compared['start_shift_m'].median():.2f} m")
            if "start_vs_planned_sideways_m" in ran and ran["start_vs_planned_sideways_m"].notna().any():
                along, sideways = ran["start_vs_planned_along_m"].dropna().abs(), ran["start_vs_planned_sideways_m"].dropna()
                print(f"  Apollo's start vs the planning problem's ({len(sideways)} vehicles): along the lane median {along.median():.2f} m, "
                      f"max {along.max():.2f} m (old vertex snapping); sideways median {sideways.median():.2f} m, max {sideways.max():.2f} m; "
                      f"sideways > 1.5 m: {(sideways > 1.5).sum()} vehicles (another lane?)")
            print(f"  object kinds in decisions: {sorted({k for kinds in ran['object_kinds'].dropna() for k in kinds.split()})}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
