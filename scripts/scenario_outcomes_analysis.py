"""
Counts of every run's scenarios by outcome and cause, for both simulators, from the runs' scenario_outcomes.csv
(written by EVITA for every generated scenario, see evita/outcomes.py).

    python scripts/scenario_outcomes_analysis.py <folder> [<folder> ...] [--out <summary.csv>]

Every folder is searched for run folders (those with a scenario_outcomes.csv). Each scenario gets one category:
    usable                  executed and evaluated
    refused by validation   not executed: the validator's message (numbers normalised, so that messages group)
    skipped (budget over)   the budget ran out before its execution
    non-interesting         Apollo: a vehicle never moved while its planner was active, only stopping (for a light, a
                            sign, another vehicle): valid behaviour that exercises nothing (the paper's non-interesting
                            scenarios); decided from the scenario's trace
    simulator failure       Apollo: rejected by DoppelTest, runner failure or timeout, unconvertible scenario or trace,
                            a vehicle that did not run, lost its planning, received no plan, or never moved because its
                            planner was not working; FrenetiX: execution error, an agent that cannot start
Output: <run>/scenario_outcomes_classified.csv for every run; the summary per generator and simulator (printed, and
written to --out, default scenario_outcomes_summary.csv in the current folder), with the detailed causes.
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

USABLE, REFUSED, SKIPPED, NON_INTERESTING, FAILURE = "usable", "refused by validation", "skipped (budget over)", "non-interesting", "simulator failure"
CATEGORIES = [USABLE, REFUSED, SKIPPED, NON_INTERESTING, FAILURE]

TIMEOUT_MESSAGE = "Timeout is over"
EXECUTION_ERROR_MESSAGE = "Error during the simulation execution"

# Share of a vehicle's plans that are "not_ready" above which its planner is considered not working
NOT_READY_SHARE = 0.9


def normalise(message: str) -> str:
    """
    A message with its numbers and ids replaced, so that messages of the same kind group together
    """
    message = re.sub(r"\(apollo_dev_ROUTE_\d+\)", "(<container>)", message)
    message = re.sub(r"-?\d+(\.\d+)?", "N", message)
    return message.strip().rstrip(".")


def planner_state(vehicle) -> str:
    """
    For a vehicle that never moved: "failing" (no plans, not_ready almost always, or a planning error), else "active"
    """
    plans = vehicle.get("planning", [])
    if not plans:
        return "failing"
    decisions = [p.get("main_decision", {}) for p in plans]
    not_ready = sum(1 for d in decisions if d.get("decision") == "not_ready")
    errors = any("PLANNING_ERROR" in json.dumps(d.get("details", {})) for d in decisions)
    if not_ready > NOT_READY_SHARE * len(plans) or errors:
        return "failing"
    return "active"


def never_moved_vehicles(message: str):
    return [int(v) for v in re.findall(r"vehicle (\d+) \([^)]*\) never moved", message)]


def classify(row, run_folder: str):
    """
    (category, cause) of one scenario row
    """
    message = row["message"]
    if row["valid"] == "1":
        return USABLE, "usable"
    if message == TIMEOUT_MESSAGE:
        return SKIPPED, "budget over"
    if row["executed"] != "1":
        return REFUSED, normalise(message)
    # DoppelTest's own runs (scripts/convert_doppeltest_runs.py): executed by DoppelTest, refused by EVITA's validation
    if message.startswith("Refused by EVITA's Apollo validation:"):
        return REFUSED, normalise(message.split(":", 1)[1])

    # Executed, then unusable
    if message == EXECUTION_ERROR_MESSAGE:
        return FAILURE, "FrenetiX: execution error"
    if "could not compute a physically possible trajectory" in message or "could not compute a phyisically possible trajectory" in message:
        return FAILURE, "FrenetiX: agent cannot start"
    for prefix, cause in [("Apollo rejected the scenario", "Apollo: rejected by DoppelTest"),
                          ("Apollo cannot run the scenario", "Apollo: scenario cannot be described for Apollo"),
                          ("Apollo's runner timed out", "Apollo: runner timeout"),
                          ("Apollo's runner failed", "Apollo: runner failure"),
                          ("Apollo's trace cannot be converted", "Apollo: trace cannot be converted")]:
        if message.startswith(prefix):
            return FAILURE, cause
    if message.startswith("Apollo's run is unhealthy"):
        problems = message.split(":", 1)[1]
        failures = [cause for text, cause in [("did not run", "Apollo: a vehicle did not run"), ("lost its planning", "Apollo: planning lost"),
                                              ("received no plan", "Apollo: no plan at all")] if text in problems]
        stuck = never_moved_vehicles(message)
        if stuck:
            kinds = never_moved_kinds(run_folder, row["scenario"], stuck)
            if "failing" in kinds:
                failures.append("Apollo: never moved, planner not working")
            elif not failures:
                return NON_INTERESTING, "Apollo: never moved, planner active (stopping)" if "active" in kinds else "Apollo: never moved (no trace)"
        return FAILURE, " + ".join(sorted(set(failures))) if failures else "Apollo: unhealthy (other)"
    return FAILURE, normalise(message) or "unknown"


def never_moved_kinds(run_folder: str, scenario_id: str, vehicle_ids):
    """
    Planner state of each vehicle that never moved, from the scenario's final trace (in debug/ for unusable scenarios)
    """
    for folder in (os.path.join(run_folder, "debug", str(scenario_id).rjust(5, "0")), os.path.join(run_folder, str(scenario_id).rjust(5, "0"))):
        path = os.path.join(folder, "trace.json.gz")
        if os.path.exists(path):
            with gzip.open(path, "rt") as f:
                trace = json.load(f)
            # Vehicle ids as in evita/apollo/trace.py: EVITA's id, or 30000 + the vehicle's index (DoppelTest's own runs)
            vehicles = {(v.get("evita_id") if v.get("evita_id") is not None else 30000 + v.get("json_index", 0)): v for v in trace.get("vehicles", [])}
            return [planner_state(vehicles[v]) if v in vehicles else "unknown" for v in vehicle_ids]
    return ["unknown" for _ in vehicle_ids]


def run_label(run_folder: str):
    """
    (generator, simulator) of a run, from its metadata.csv
    """
    generator, simulator = "unknown", "unknown"
    path = os.path.join(run_folder, "metadata.csv")
    if os.path.exists(path):
        with open(path, newline = "") as f:
            rows = list(csv.DictReader(f))
        if rows:
            generator = rows[0].get("generator", generator).split(".")[-1]
            simulator = rows[0].get("simulator", "frenetix") or "frenetix"
    return generator, simulator


def analyse_run(run_folder: str):
    with open(os.path.join(run_folder, "scenario_outcomes.csv"), newline = "") as f:
        rows = list(csv.DictReader(f))
    classified = []
    for row in rows:
        category, cause = classify(row, run_folder)
        classified.append({**row, "category": category, "cause": cause})
    with open(os.path.join(run_folder, "scenario_outcomes_classified.csv"), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = list(classified[0].keys()) if classified else ["scenario"])
        writer.writeheader()
        writer.writerows(classified)
    return classified


def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Counts of the runs' scenarios by outcome and cause")
    parser.add_argument("folders", nargs = "+", help = "folders searched for run folders (with scenario_outcomes.csv)")
    parser.add_argument("--out", default = "scenario_outcomes_summary.csv", help = "summary CSV to write")
    args = parser.parse_args(argv)

    runs = sorted({os.path.dirname(p) for folder in args.folders for p in glob.glob(os.path.join(folder, "**", "scenario_outcomes.csv"), recursive = True)})
    if not runs:
        print("No run folder with a scenario_outcomes.csv found")
        return 1

    categories = defaultdict(Counter); causes = defaultdict(Counter); run_count = Counter()
    for run in runs:
        group = run_label(run)
        run_count[group] += 1
        for row in analyse_run(run):
            categories[group][row["category"]] += 1
            causes[group][(row["category"], row["cause"])] += 1

    with open(args.out, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["generator", "simulator", "runs", "generated", "executed"] + CATEGORIES)
        for group in sorted(categories):
            counts = categories[group]; generated = sum(counts.values())
            executed = generated - counts[REFUSED] - counts[SKIPPED]
            writer.writerow([*group, run_count[group], generated, executed] + [counts[c] for c in CATEGORIES])
        writer.writerow([])
        writer.writerow(["generator", "simulator", "category", "cause", "scenarios"])
        for group in sorted(causes):
            for (category, cause), n in sorted(causes[group].items(), key = lambda item: (CATEGORIES.index(item[0][0]), -item[1])):
                writer.writerow([*group, category, cause, n])

    for group in sorted(categories):
        counts = categories[group]; generated = sum(counts.values())
        executed = generated - counts[REFUSED] - counts[SKIPPED]
        print(f"\n== {group[0]} on {group[1]}: {run_count[group]} run(s), {generated} scenarios generated, {executed} executed")
        for category in CATEGORIES:
            share = f"{100 * counts[category] / executed:.0f}% of executed" if executed and category not in (REFUSED, SKIPPED) else ""
            print(f"   {counts[category]:5d}  {category:24s} {share}")
            for (c, cause), n in sorted(causes[group].items(), key = lambda item: -item[1]):
                if c == category and cause != category and cause != "usable":
                    print(f"            {n:5d}  {cause}")
    print(f"\nSummary written to {args.out}; each run's classified rows in <run>/scenario_outcomes_classified.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
