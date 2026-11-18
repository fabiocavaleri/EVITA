"""
False-positive analysis of EVITA's interaction detection (removal experiment).

Question: when EVITA reports that agent a interacts at time step t with vehicles N, was the replanning caused by N?

For each sampled interaction, two scenarios are simulated:
  - control:   the original scenario, unchanged;
  - treatment: the same scenario without the vehicles in N.
Agent a's replanning at t (condition C1: a planned position moves by at least TH_ED between its plans at t and t + 1)
is measured in both runs:
  - control replans, treatment does not  -> true positive  (the replanning was caused by N)
  - control replans, treatment replans    -> false positive (the replanning happens without N)
  - control does not replan               -> not reproducible (excluded from the rate)

Three tasks, run in order:

    python false_positive_analysis.py sample
    python false_positive_analysis.py execute [--workers 3]
    python false_positive_analysis.py analyze

Working folder (OUTPUT_FOLDER):
    sample.csv                    the accepted cases (one per scenario and agent)
    discarded.csv                 candidates not accepted, with the reason
    cases/<case>/control.xml      the original scenario
    cases/<case>/treatment.xml    the scenario without the interacting vehicles
    cases/<case>/runs/<variant>/  executor output (logs/<vehicle>/logs.csv, logs/simulation.db)
    cases/<case>/<variant>.json   outcome of the run (status, reason, duration)
    cases.csv, summary.json       results of the analysis
"""
import os
import re
import sys
import ast
import json
import math
import time
import shutil
import sqlite3
import argparse
import traceback
import contextlib
import numpy as np
import pandas as pd
import multiprocessing

from tqdm import tqdm
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========== Settings ==========

# Source of the detected interactions (from the analysis of EVITA's runs)
INTERACTIONS_CSV = os.path.join("results", "Main-Experiment", "EVITA", "interactions_collisions_analysis", "aggregated_interactions.csv")

# Working folder of this experiment
OUTPUT_FOLDER = os.path.join("results", "false_positive_analysis")

# Number of accepted cases, and seed of the sampling
SAMPLE_SIZE = 200
SEED = 42

# Minimum number of vehicles left after removing the interacting ones: with fewer than 2, C2 cannot hold
MIN_REMAINING_VEHICLES = 2

# Th_ED (metres): must be the value used in the experiments being analysed
TH_ED = 2.5

# Half-width of the window around t (time steps of 0.1 s): 10 = +-1 s
TIME_WINDOW = 10

# Two consecutive plans closer than this everywhere (metres) are the same plan (FrenetiX did not replan)
IDENTICAL_TOLERANCE = 1e-3

VARIANTS = ["control", "treatment"]

# ========== Helpers ==========


def case_folder(case_id: str) -> str:
    return os.path.join(OUTPUT_FOLDER, "cases", case_id)


def run_folder(case_id: str, variant: str) -> str:
    return os.path.join(case_folder(case_id), "runs", variant)


def outcome_file(case_id: str, variant: str) -> str:
    return os.path.join(case_folder(case_id), f"{variant}.json")

def parse_ids(text) -> list:
    """
    Vehicle ids from the CSV's is_interacting_with column ("[10002, 10005]").
    """
    value = ast.literal_eval(text) if isinstance(text, str) else text
    return sorted(int(v) for v in value)


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """
    95 % Wilson confidence interval of a proportion; (None, None) if n = 0.
    """
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)

# ========== Task 1: sampling ==========

def sample(force: bool = False) -> int:
    """
    Sample `SAMPLE_SIZE` cases, one per (scenario, agent), drawn at random without replacement.
    A candidate is discarded if its scenario cannot be read, if the agent or an interacting vehicle is missing, or if fewer than
    MIN_REMAINING_VEHICLES vehicles remain after the removal.
    """
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.common.file_writer import CommonRoadFileWriter
    from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
    from commonroad.planning.planning_problem import PlanningProblemSet

    sample_file = os.path.join(OUTPUT_FOLDER, "sample.csv")
    if os.path.exists(sample_file) and not force:
        print(f"{sample_file} exists: use --force to sample again (this deletes the cases and their runs)")
        return 1
    
    # Remove existing cases and runs, then create the folder for the new cases
    if os.path.isdir(os.path.join(OUTPUT_FOLDER, "cases")):
        shutil.rmtree(os.path.join(OUTPUT_FOLDER, "cases"))
    os.makedirs(os.path.join(OUTPUT_FOLDER, "cases"), exist_ok = True)

    interactions = pd.read_csv(INTERACTIONS_CSV)

    # One candidate per (scenario, agent): its first detected interaction; then a random order, fixed by the seed
    interactions = interactions.sort_values(["scenario_folder_path", "agent_id", "timestep"])
    candidates = interactions.drop_duplicates(subset = ["scenario_folder_path", "agent_id"], keep = "first")
    candidates = candidates.sample(frac = 1, random_state = SEED).reset_index(drop = True)

    # Initialize the lists of accepted and discarded cases
    accepted, discarded = [], []

    for _, row in tqdm(candidates.iterrows(), total = len(candidates), desc = "Sampling"):
        # Stop when the sample is full
        if len(accepted) >= SAMPLE_SIZE:
            break

        # Extract the scenario and agent, and the interacting vehicles to remove
        agent_id = int(row["agent_id"])
        timestep = int(row["timestep"])
        removed_ids = parse_ids(row["is_interacting_with"])
        scenario_file = os.path.join(row["scenario_folder_path"], "scenario", f"{row['scenario']}.xml")

        # Extract map name and repetition from the scenario folder path
        match = re.search(r"SCENARIO_MAP_(.*?)_REPETITION_(\d+)", row["scenario_folder_path"])
        map_name, repetition = (match.group(1), int(match.group(2))) if match else ("unknown", -1)
        identity = {"scenario_folder_path": row["scenario_folder_path"], "agent_id": agent_id, "timestep": timestep}

        # Open CommonRoad scenario
        try:
            scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
        except Exception as error:
            discarded.append({**identity, "reason": f"cannot read the scenario: {error}"})
            continue

        # Extract vehicle ids
        vehicle_ids = set(planning_problem_set.planning_problem_dict)

        # Skip candidate if the agent or an interacting vehicle is missing, or if too few vehicles remain after the removal
        if agent_id not in vehicle_ids:
            discarded.append({**identity, "reason": "agent not in the scenario"})
            continue

        if not removed_ids or not set(removed_ids) <= vehicle_ids or agent_id in removed_ids:
            discarded.append({**identity, "reason": f"interacting vehicles {removed_ids} not all in the scenario"})
            continue

        remaining = len(vehicle_ids) - len(removed_ids)
        if remaining < MIN_REMAINING_VEHICLES:
            discarded.append({**identity, "reason": f"only {remaining} vehicle(s) left after the removal"})
            continue

        # The candidate is accepted: create its case folder and write the control and treatment scenarios
        case_id = f"case_{len(accepted) + 1:03d}"
        os.makedirs(case_folder(case_id), exist_ok = True)
        shutil.copyfile(scenario_file, os.path.join(case_folder(case_id), "control.xml"))

        kept = [pp for vehicle_id, pp in planning_problem_set.planning_problem_dict.items() if vehicle_id not in removed_ids]
        CommonRoadFileWriter(scenario = scenario, planning_problem_set = PlanningProblemSet(kept)).write_to_file(os.path.join(case_folder(case_id), "treatment.xml"), OverwriteExistingFile.ALWAYS)

        accepted.append({"case_id": case_id, "map": map_name, "repetition": repetition, "scenario": row["scenario"], **identity, "removed_ids": json.dumps(removed_ids), "n_vehicles": len(vehicle_ids), "n_remaining": remaining})

    # Save the accepted and discarded cases to CSV files
    pd.DataFrame(accepted).to_csv(sample_file, index = False)
    pd.DataFrame(discarded, columns = ["scenario_folder_path", "agent_id", "timestep", "reason"]).to_csv(os.path.join(OUTPUT_FOLDER, "discarded.csv"), index = False)

    print(f"Accepted {len(accepted)} cases, discarded {len(discarded)} candidates (see discarded.csv)")

    if len(accepted) < SAMPLE_SIZE:
        print(f"WARNING: only {len(accepted)} of {SAMPLE_SIZE} cases: not enough candidates")
        return 1
    
    return 0

# ========== Task 2: execution ==========

@contextlib.contextmanager
def _redirect_output_to(path: str):
    """
    Send this process's stdout and stderr (file descriptors 1 and 2, inherited by FrenetiX's own processes) to a file.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        for fd in (saved_stdout, saved_stderr, log_fd):
            os.close(fd)

class _NullCounter:
    """
    Progress bar that shows nothing.
    """
    count = 0
    total = 0

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

class _NullProgressBarManager:
    """
    Stand-in for enlighten's manager: the simulation's progress bars stay silent in the workers.
    """
    def counter(self, *args, **kwargs):
        return _NullCounter()

    def stop(self):
        pass

def _execute_one(case_id: str, variant: str) -> dict:
    """
    Simulate and evaluate one scenario.
    Returns the outcome.
    """
    started = time.perf_counter()
    outcome = {"case_id": case_id, "variant": variant, "status": "failed", "reason": ""}
    simulation = None

    log_file = os.path.join(case_folder(case_id), f"{variant}.log")

    with _redirect_output_to(log_file):
        try:
            from commonroad.common.file_reader import CommonRoadFileReader
            from evita.evaluators.frenetix_evaluator import FrenetixScenarioEvaluator
            from evita.executors.frenetix_executor import FrenetixScenarioExecution

            runs_folder = os.path.join(case_folder(case_id), "runs")
            # Remove existing run folder, as the executor fails if it exists
            if os.path.exists(run_folder(case_id, variant)):
                shutil.rmtree(run_folder(case_id, variant))
            os.makedirs(runs_folder, exist_ok = True)

            scenario, planning_problem_set = CommonRoadFileReader(os.path.join(case_folder(case_id), f"{variant}.xml")).open()
            setattr(scenario, "is_valid", True)
            # The executor names its folder after generation_id
            setattr(scenario, "generation_id", variant)

            # The scenario file was saved by the original run after the road extension: executed as is, not extended again
            simulation = FrenetixScenarioExecution(runs_folder, driving_agent_name = "frenetix", debug = False).execute(scenario, planning_problem_set, _NullProgressBarManager())
            if simulation is None:
                outcome["reason"] = "simulation failed (see the executor's log)"
            else:
                FrenetixScenarioEvaluator(runs_folder).evaluate(simulation, TH_ED)
                outcome["status"] = "done"
        except Exception:
            traceback.print_exc()   # into the log file
            outcome["reason"] = traceback.format_exc().strip().splitlines()[-1]
        finally:
            if simulation is not None:
                try:
                    simulation.sim_logger.con.close()
                    if simulation.config.simulation.multiprocessing:
                        simulation.close_processes()
                except Exception:
                    pass

    outcome["duration_s"] = round(time.perf_counter() - started, 1)
    return outcome

def execute(workers: int, retry_failed: bool = False) -> int:
    """
    Run the control and treatment scenario of every case, in parallel processes.
    Completed runs are skipped, so an interrupted execution continues where it stopped.
    """
    sample_file = os.path.join(OUTPUT_FOLDER, "sample.csv")

    # Skip if the sample does not exist, as the cases are not known
    if not os.path.exists(sample_file):
        print("No sample.csv: run `sample` first")
        return 1

    cases = pd.read_csv(sample_file)

    # Create a parallel task for each case and variant, skipping completed runs (status = done) and failed ones (if retry_failed is False)
    tasks = []
    for case_id in cases["case_id"]:
        for variant in VARIANTS:
            path = outcome_file(case_id, variant)
            if os.path.exists(path):
                with open(path) as handle:
                    status = json.load(handle).get("status")
                if status == "done" or (status == "failed" and not retry_failed):
                    continue
            tasks.append((case_id, variant))

    print(f"{len(cases)} cases, {len(tasks)} runs to execute with {workers} workers")

    if not tasks:
        return 0

    failed = 0
    with ProcessPoolExecutor(max_workers = workers, mp_context = multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_execute_one, case_id, variant): (case_id, variant) for case_id, variant in tasks}

        # Iterate over the completed futures
        for completed, future in enumerate(as_completed(futures), start = 1):
            case_id, variant = futures[future]
            try:
                outcome = future.result()
            except Exception as error:   # the worker process died
                outcome = {"case_id": case_id, "variant": variant, "status": "failed", "reason": f"worker error: {error!r}", "duration_s": None}

            # Only the main process writes the outcomes
            with open(outcome_file(case_id, variant), "w") as handle:
                json.dump(outcome, handle, indent = 2)

            message = f"[{completed}/{len(tasks)}] {case_id}/{variant}: {outcome['status']} in {outcome['duration_s']} s"
            
            if outcome["status"] != "done":
                failed += 1
                message += f" ({outcome['reason']}; log: {os.path.join(case_folder(case_id), variant + '.log')})"
            
            print(message, flush = True)

    print(f"Execution finished: {len(tasks) - failed} done, {failed} failed")
    return 1 if failed else 0

# ========== Task 3: analysis ==========

def replanning_deviations(logs_folder: str, agent_id: int) -> dict:
    """
    For each time step t with plans at t and t + 1 compute the largest distance between the two plans over the time steps.
    """
    from evita.evaluators.frenetix_evaluator import FrenetixScenarioEvaluator
    from evita.generators.utils import crop_trajectories

    # Check if the agent's logs exist: if not, the run failed and there are no plans
    if not os.path.exists(os.path.join(logs_folder, str(agent_id), "logs.csv")):
        return {}

    # Use the evaluator's private method to extract the planned trajectories from the logs
    stand_in = SimpleNamespace(agent_id_list = [agent_id], config = SimpleNamespace(simulation = SimpleNamespace(log_path = logs_folder)))
    table = FrenetixScenarioEvaluator._planned_trajectories(stand_in)
    plans = {int(t): trajectory for t, trajectory in zip(table["timestep"], table["trajectory"])}

    deviations = {}
    # Iterate over planning timesteps
    for t in sorted(plans):
        # Skip timestep if replanning at t + 1 is missing
        if t + 1 not in plans:
            continue

        # Crop consecutive planned trajectories
        cropped_new, cropped_old = crop_trajectories(plans[t], plans[t + 1])
        if cropped_new is None:
            continue

        # Compute max deviation between the two planned trajectories
        deviations[t] = max(float(np.linalg.norm(np.asarray(a.position, dtype = float) - np.asarray(b.position, dtype = float))) for a, b in zip(cropped_new.state_list, cropped_old.state_list))
    
    return deviations

def interactions_reported(db_file: str, agent_id: int) -> dict:
    """
    Time step -> vehicles the agent interacts with, as EVITA's evaluation reports it (rows with is_interacting = 1).
    """
    # Skip if simulation.db is missing
    if not os.path.exists(db_file):
        return {}

    connection = sqlite3.connect(db_file)
    
    # Extract the rows with is_interacting = 1 for the agent, and parse the vehicle ids from the is_interacting_with column
    try:
        rows = connection.execute("SELECT timestep, is_interacting_with FROM scenario_evaluation WHERE agent_id = ? AND is_interacting = 1.0", (agent_id,)).fetchall()
    finally:
        connection.close()

    reported = {}
    # Iterate over the rows and parse the vehicle ids
    for t, with_text in rows:
        try:
            reported[int(t)] = parse_ids(with_text)
        except (ValueError, SyntaxError):
            reported[int(t)] = []
    
    return reported

def measure_run(logs_folder: str, agent_id: int, timestep: int) -> dict:
    """
    Agent's replanning (C1) and reported interactions at t and in the window around t, in one run.
    """
    deviations = replanning_deviations(logs_folder, agent_id)
    reported = interactions_reported(os.path.join(logs_folder, "simulation.db"), agent_id)
    window = range(timestep - TIME_WINDOW, timestep + TIME_WINDOW + 1)

    window_deviations = [deviations[t] for t in window if t in deviations]
    window_reported = sorted({v for t in window if t in reported for v in reported[t]})

    return {"has_plans": bool(deviations),
            "deviation_at_t": deviations.get(timestep),
            "C1_at_t": deviations.get(timestep, 0.0) >= TH_ED,
            "max_deviation_window": max(window_deviations) if window_deviations else None,
            "C1_window": any(d >= TH_ED for d in window_deviations),
            "interacting_at_t": timestep in reported,
            "interacting_with_at_t": json.dumps(reported.get(timestep, [])),
            "interacting_window": any(t in reported for t in window),
            "interacting_with_window": json.dumps(window_reported)}

def classify(control_c1: bool, treatment_c1: bool) -> str:
    if not control_c1:
        return "not_reproducible"

    return "false_positive" if treatment_c1 else "true_positive"

def rates(table: pd.DataFrame, column: str) -> dict:
    """
    Counts of each class and the false-positive rate FP / (TP + FP), with its 95 % Wilson interval.
    """
    counts = table[column].value_counts().to_dict()
    tp, fp = counts.get("true_positive", 0), counts.get("false_positive", 0)
    low, high = wilson_interval(fp, tp + fp)

    return {"cases": int(len(table)), "true_positive": tp, "false_positive": fp,
            "not_reproducible": counts.get("not_reproducible", 0),
            "false_positive_rate": fp / (tp + fp) if tp + fp else None, "ci95_low": low, "ci95_high": high}

def analyze() -> int:
    # Skip if the sample does not exist, as the cases are not known
    sample_file = os.path.join(OUTPUT_FOLDER, "sample.csv")
    if not os.path.exists(sample_file):
        print("No sample.csv: run `sample` first")
        return 1

    cases = pd.read_csv(sample_file)

    # Initialize counters
    rows, missing = [], 0

    # Iterate over the cases, measuring the control and treatment runs and classifying them
    for _, case in tqdm(cases.iterrows(), total = len(cases), desc = "Analyzing"):
        case_id, agent_id, timestep = case["case_id"], int(case["agent_id"]), int(case["timestep"])

        row = {"case_id": case_id, "map": case["map"], "agent_id": agent_id, "timestep": timestep,
               "removed_ids": case["removed_ids"], "n_vehicles": case["n_vehicles"], "n_remaining": case["n_remaining"]}

        # The original run, as a reference: the detection there is what is being tested
        original = measure_run(os.path.join(case["scenario_folder_path"], "logs"), agent_id, timestep)
        row["original_deviation_at_t"] = original["deviation_at_t"]

        complete = True
        for variant in VARIANTS:
            # Check if the run was completed, and measure it
            outcome = {}
            if os.path.exists(outcome_file(case_id, variant)):
                with open(outcome_file(case_id, variant)) as handle:
                    outcome = json.load(handle)
            
            if outcome.get("status") != "done":
                complete = False
                row[f"{variant}_status"] = outcome.get("status", "not run")
                continue

            row[f"{variant}_status"] = "done"

            # Measure the run's replanning and reported interactions
            measures = measure_run(os.path.join(run_folder(case_id, variant), "logs"), agent_id, timestep)

            row.update({f"{variant}_{name}": value for name, value in measures.items()})

        if not complete:
            missing += 1
            row["class_at_t"] = row["class_window"] = "not_executed"
        else:
            row["class_at_t"] = classify(row["control_C1_at_t"], row["treatment_C1_at_t"])
            row["class_window"] = classify(row["control_C1_window"], row["treatment_C1_window"])
        
        rows.append(row)

    # Save the results to a CSV file and compute the summary statistics
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTPUT_FOLDER, "cases.csv"), index = False)

    executed = table[table["class_at_t"] != "not_executed"]
    summary = {"settings": {"th_ed_m": TH_ED, "time_window_steps": TIME_WINDOW, "min_remaining_vehicles": MIN_REMAINING_VEHICLES,
                            "sample_size": SAMPLE_SIZE, "seed": SEED},
               "cases_not_executed": missing,
               "vehicles_in_sample": {str(int(k)): int(v) for k, v in table["n_vehicles"].value_counts().sort_index().items()},
               "at_t": {"overall": rates(executed, "class_at_t")},
               "window": {"overall": rates(executed, "class_window")}}

    for map_name, map_table in executed.groupby("map"):
        summary["at_t"][map_name] = rates(map_table, "class_at_t")
        summary["window"][map_name] = rates(map_table, "class_window")

    # Among false positives: how many are still reported as interacting, now with a remaining vehicle
    false_positives = executed[executed["class_at_t"] == "false_positive"]
    if len(false_positives):
        summary["false_positives_still_interacting_at_t"] = int(false_positives["treatment_interacting_at_t"].sum())
    else:
        summary["false_positives_still_interacting_at_t"] = 0

    with open(os.path.join(OUTPUT_FOLDER, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent = 2)

    overall = summary["at_t"]["overall"]
    if overall["false_positive_rate"] is not None:
        print(f"False positives at t: {overall['false_positive']} of {overall['true_positive'] + overall['false_positive']} "
              f"({overall['false_positive_rate']:.1%}, 95 % CI {overall['ci95_low']:.1%}-{overall['ci95_high']:.1%}); "
              f"not reproducible: {overall['not_reproducible']}; not executed: {missing}")
    
    print(f"Results written to {OUTPUT_FOLDER}/cases.csv and summary.json")

    return 0

# ========== Command line ==========

def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description = "False-positive analysis of EVITA's interaction detection.")
    commands = parser.add_subparsers(dest = "command", required = True)

    sample_parser = commands.add_parser("sample", help = "Sample the cases and write their control and treatment scenarios")
    sample_parser.add_argument("--force", action = "store_true", help = "Sample again, deleting existing cases and runs")

    execute_parser = commands.add_parser("execute", help = "Simulate and evaluate the control and treatment scenarios")
    execute_parser.add_argument("--workers", type = int, default = 5, help = "Number of parallel simulations")
    execute_parser.add_argument("--retry-failed", action = "store_true", help = "Rerun failed runs")

    commands.add_parser("analyze", help = "Classify the cases and compute the false-positive rate")

    args = parser.parse_args(argv)

    if args.command == "sample":
        return sample(args.force)
    if args.command == "execute":
        return execute(args.workers, args.retry_failed)
    return analyze()

if __name__ == "__main__":
    sys.exit(main())