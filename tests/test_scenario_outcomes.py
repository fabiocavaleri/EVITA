"""
Tests of the scenario outcomes (evita/outcomes.py) and their analysis (scripts/scenario_outcomes_analysis.py), for both
simulators. Run from the project root: python3 tests/test_scenario_outcomes.py
"""
import os, sys, csv, gzip, json, tempfile
from types import SimpleNamespace
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from evita.outcomes import OUTCOMES_FILE, record_outcome
import scenario_outcomes_analysis as analysis

results = []
def check(name, ok):
    results.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name)

def scenario(generation_id, valid, message = ""):
    s = SimpleNamespace(generation_id = generation_id, is_valid = valid)
    if message:
        s.validation_message = message
    return s
def problems(n):
    return SimpleNamespace(planning_problem_dict = {i: None for i in range(n)})

# ---------- record_outcome ----------
run = tempfile.mkdtemp()
record_outcome(run, scenario(1, True), problems(2), True, "apollo")
record_outcome(run, scenario(2, False, "Goal area of agent 10001 is 4.5 m ahead\n along its route (min 20)"), problems(3), False, "apollo")
with open(os.path.join(run, OUTCOMES_FILE)) as f:
    rows = list(csv.reader(f))
check("record_outcome: one header, one row per scenario, an empty message when usable, whitespace collapsed",
      rows[0] == ["scenario", "simulator", "vehicles", "executed", "valid", "message"] and rows[1] == ["1", "apollo", "2", "1", "1", ""]
      and rows[2][:5] == ["2", "apollo", "3", "0", "0"] and rows[2][5] == "Goal area of agent 10001 is 4.5 m ahead along its route (min 20)" and len(rows) == 3)

# ---------- Classification: an Apollo run with every kind of outcome ----------
def vehicle(evita_id, decisions):
    return {"evita_id": evita_id, "planning": [{"main_decision": d} for d in decisions]}
def write_trace(run_folder, scenario_id, vehicles, debug = True):
    folder = os.path.join(run_folder, "debug" if debug else "", str(scenario_id).rjust(5, "0"))
    os.makedirs(folder, exist_ok = True)
    with gzip.open(os.path.join(folder, "trace.json.gz"), "wt") as f:
        json.dump({"vehicles": vehicles}, f)

apollo_run = tempfile.mkdtemp()
with open(os.path.join(apollo_run, "metadata.csv"), "w") as f:
    f.write("generator,scenario_name,time_budget,timestamp_id,simulator\nevita.generators.random_generator.RandomScenarioGenerator,x,1,1,apollo\n")
outcomes = [
    (1, True, True, ""),
    (2, False, False, "Goal area of agent 10001 is 4.5 m ahead along its route (min 20)"),
    (3, False, False, "Goal area of agent 10002 is 13.9 m ahead along its route (min 20)"),
    (4, False, False, "Timeout is over"),
    (5, True, False, "Apollo rejected the scenario: DoppelTest dropped 1 of 4 vehicles (start within 5 m of another vehicle)"),
    (6, True, False, "Apollo's run is unhealthy: vehicle 10001 (apollo_dev_ROUTE_0) did not run (no localization)"),
    (7, True, False, "Apollo's run is unhealthy: vehicle 10003 (apollo_dev_ROUTE_2) never moved"),
    (8, True, False, "Apollo's run is unhealthy: vehicle 10004 (apollo_dev_ROUTE_3) never moved"),
    (9, True, False, "Apollo's run is unhealthy: vehicle 10002 (apollo_dev_ROUTE_1) lost its planning: no plan during the last 21.6 s of its run, mission not complete"),
    (10, True, False, "Apollo's runner timed out after 600 s"),
]
for sid, executed, valid, message in outcomes:
    record_outcome(apollo_run, scenario(sid, valid, message), problems(2), executed, "apollo")
write_trace(apollo_run, 7, [vehicle(10003, [{"decision": "stop", "details": {"reason_code": "STOP_REASON_SIGNAL"}}] * 50)])     # waiting at a red light
write_trace(apollo_run, 8, [vehicle(10004, [{"decision": "not_ready", "details": {"reason": "routing not ready"}}] * 50)])     # planner stuck
classified = {int(r["scenario"]): (r["category"], r["cause"]) for r in analysis.analyse_run(apollo_run)}
check("Apollo: usable, refused by validation (numbers normalised), skipped", classified[1] == ("usable", "usable")
      and classified[2] == classified[3] == ("refused by validation", "Goal area of agent N is N m ahead along its route (min N)")
      and classified[4][0] == "skipped (budget over)")
check("Apollo: rejection, a vehicle that did not run, planning lost, runner timeout are simulator failures",
      [classified[i] for i in (5, 6, 9, 10)] == [("simulator failure", "Apollo: rejected by DoppelTest"), ("simulator failure", "Apollo: a vehicle did not run"),
                                                  ("simulator failure", "Apollo: planning lost"), ("simulator failure", "Apollo: runner timeout")])
check(f"Apollo: never moved with an active planner (red light) is non-interesting; with a planner stuck not_ready a failure ({classified[7]}, {classified[8]})",
      classified[7] == ("non-interesting", "Apollo: never moved, planner active (stopping)")
      and classified[8] == ("simulator failure", "Apollo: never moved, planner not working"))

# ---------- A FrenetiX run ----------
frenetix_run = tempfile.mkdtemp()
with open(os.path.join(frenetix_run, "metadata.csv"), "w") as f:
    f.write("generator,scenario_name,time_budget,timestamp_id,simulator\nevita.generators.random_generator.RandomScenarioGenerator,x,1,1,frenetix\n")
for sid, executed, valid, message in [(1, True, True, ""), (2, True, False, "Error during the simulation execution"),
                                      (3, True, False, "Agent 4 could not compute a phyisically possible trajectory since the beginning."),
                                      (4, False, False, "Initial State of 4 inside of Goal Region")]:
    record_outcome(frenetix_run, scenario(sid, valid, message), problems(3), executed, "frenetix")
classified = {int(r["scenario"]): (r["category"], r["cause"]) for r in analysis.analyse_run(frenetix_run)}
check("FrenetiX: execution error and an agent that cannot start are simulator failures; validation refusals grouped",
      classified[2] == ("simulator failure", "FrenetiX: execution error") and classified[3] == ("simulator failure", "FrenetiX: agent cannot start")
      and classified[4] == ("refused by validation", "Initial State of N inside of Goal Region") and classified[1][0] == "usable")

# ---------- Summary over both runs ----------
parent = tempfile.mkdtemp(); summary = os.path.join(parent, "summary.csv")
os.rename(apollo_run, os.path.join(parent, "apollo_run")); os.rename(frenetix_run, os.path.join(parent, "frenetix_run"))
code = analysis.main([parent, "--out", summary])
with open(summary) as f:
    lines = list(csv.reader(f))
first_table = lines[1:lines.index([])]
rows = {(r[0], r[1]): r for r in first_table}
apollo, frenetix = rows.get(("RandomScenarioGenerator", "apollo")), rows.get(("RandomScenarioGenerator", "frenetix"))
check(f"summary per generator and simulator: Apollo {apollo[3:10] if apollo else None}, FrenetiX {frenetix[3:10] if frenetix else None}",
      code == 0 and apollo[3:10] == ["10", "7", "1", "2", "1", "1", "5"] and frenetix[3:10] == ["4", "3", "1", "1", "0", "0", "2"])

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
