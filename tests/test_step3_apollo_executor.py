"""
Tests of step 3, parts C and D: the Apollo executor (evita/executors/apollo_executor.py) and the health check
(evita/apollo/health.py), with stand-ins for .venv38's Python, Apollo's start-up script and the runner. The stand-in
runner writes real-format traces from the scenario's JSON (vehicles drive their routes, lights from DoppelTest's
traffic manager); a list of modes, one per run, plays every case.
Run from the project root: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tests/test_step3_apollo_executor.py
"""
import os, sys, json, glob, random, logging, tempfile, textwrap
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
logging.basicConfig(level = logging.CRITICAL)

from evita import simulators
from evita.apollo.map_data import ApolloMapData
from evita.apollo import traffic_control as tc
from evita.executors.apollo_executor import ApolloScenarioExecution, ApolloSetupError
from evita.generators.utils import get_base_scenario_from_file
import apollo_live_run

MAP = os.path.abspath("EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml")
HD_MAP = os.path.abspath("EVITA_DoppelTest/data/maps/borregas_ave/base_map.bin")
results = []
def check(name, ok):
    results.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name)

# ---------- Stand-ins ----------
work = tempfile.mkdtemp()
RUNNER = os.path.join(work, "fake_runner.py")
open(RUNNER, "w").write(textwrap.dedent(f'''
    import sys, os, json, math, time, types, argparse
    sys.path.insert(0, {os.path.abspath("EVITA_DoppelTest")!r})
    for n, a in [("cyber_record", {{}}), ("cyber_record.record", {{"Record": object}})]:
        m = types.ModuleType(n); m.__dict__.update(a); sys.modules[n] = m
    import record_export as rx
    from hdmap.MapParser import MapParser
    from framework.scenario.tc_config import TCSection
    from framework.scenario.TrafficControlManager import TrafficControlManager
    from modules.localization.proto.localization_pb2 import LocalizationEstimate
    from modules.planning.proto.planning_pb2 import ADCTrajectory
    p = argparse.ArgumentParser(); [p.add_argument(a) for a in ("--scenario-json", "--records-dir", "--scenario", "--trace")]
    args = p.parse_args()
    # The mode of this run: the next one of FAKE_MODES (a counter file keeps track of the runs)
    modes = os.environ.get("FAKE_MODES", "ok").split(","); counter = os.environ["FAKE_COUNTER"]
    n = int(open(counter).read()) if os.path.exists(counter) else 0; open(counter, "w").write(str(n + 1))
    mode = modes[min(n, len(modes) - 1)]
    if mode == "crash": sys.exit(3)
    if mode == "records": sys.exit(10)
    if mode == "timeout": time.sleep(30); sys.exit(0)
    sc = json.load(open(args.scenario_json))
    section = {{"scenario": f"Scenario_{{args.scenario}}", "status": "executed", "reason": "",
               "vehicle": {{"length": 4.933, "width": 2.11, "back_edge_to_center": 1.043}}, "json": sc}}
    if mode == "reject":
        section.update(status = "rejected", reason = "DoppelTest dropped 1 of 2 vehicles (start within 5 m of another vehicle)")
        rx.write_trace({{"format_version": 1, "scenario": section, "vehicles": []}}, args.trace); sys.exit(2)
    MapParser({HD_MAP!r}); mp = MapParser.get_instance()
    T0 = 1_700_000_000.0; manager = TrafficControlManager(TCSection(**sc["tc_section"]))
    records = os.path.join(args.records_dir, f"Scenario_{{args.scenario}}"); os.makedirs(records, exist_ok = True)
    vehicles = []
    for k, adc in enumerate(sc["ad_section"]["adcs"]):
        open(os.path.join(records, f"apollo_dev_ROUTE_{{k}}.Scenario_{{args.scenario}}"), "w").write("record")
        # The route Apollo drives: a lane written twice (single-lane routes) is one lane
        lanes = [l for i, l in enumerate(adc["routing"]) if i == 0 or l != adc["routing"][i - 1]]; lengths = [mp.get_lane_length(l) for l in lanes]
        def pose(t, adc = adc, lanes = lanes, lengths = lengths):
            s = adc["start_s"] + 6.0 * max(t - adc["start_t"], 0); i = 0
            while i < len(lanes) - 1 and s > lengths[i]: s -= lengths[i]; i += 1
            s = min(s, lengths[i] - 0.05)
            point, heading = mp.get_coordinate_and_heading(lanes[i], s); return point.x, point.y, heading
        locs, plans = [], []
        if not (mode == "no_localization" and k == 0):
            for j in range(0, 3000, 2):
                t = j * 0.01; x, y, h = pose(t); m = LocalizationEstimate(); m.header.timestamp_sec = T0 + t
                m.pose.position.x, m.pose.position.y, m.pose.heading = x, y, h
                v = 6.0 if t >= adc["start_t"] else 0.0; m.pose.linear_velocity.x, m.pose.linear_velocity.y = v * math.cos(h), v * math.sin(h)
                locs.append(rx.export_localization(m))
            last_plan = 10.0 if (k == 0 and mode in ("planning_stops", "planning_stops_after_mission")) else 30.0
            for j in range(int(last_plan * 10)):
                t = 0.05 + j * 0.1; pl = ADCTrajectory(); pl.header.timestamp_sec = T0 + t
                for q in range(3):
                    x, y, h = pose(t + 0.2 * q); pt = pl.trajectory_point.add(); pt.relative_time, pt.path_point.x, pt.path_point.y, pt.path_point.theta, pt.v = 0.2 * q, x, y, h, 6.0
                done = mode == "planning_stops_after_mission" and k == 0 and t > 9.0
                (pl.decision.main_decision.mission_complete if done else pl.decision.main_decision.cruise).SetInParent()
                pl.engage_advice.advice = 3; plans.append(rx.export_planning(pl, int((T0 + t + 0.003) * 1e9)))
        lights = []
        for j in range(300):
            msg = manager.get_traffic_configuration(j / 10); msg.header.timestamp_sec = T0 + j * 0.1; lights.append(rx.export_traffic_lights(msg))
        vehicles.append({{"container": f"apollo_dev_ROUTE_{{k}}", "apollo_id": 900 + k, "json_index": k, "evita_id": sc["evita_vehicle_ids"][k],
                         "routing": lanes, "start_s": adc["start_s"], "dest_s": adc["dest_s"], "start_t": adc["start_t"], "record_files": [],
                         "localization": {{"columns": rx.LOCALIZATION_COLUMNS, "rows": locs}}, "planning": plans, "obstacles": [], "traffic_lights": lights,
                         "oracles": {{"collisions": [], "stop_sign_violations": [], "traffic_light_violations": [], "speeding_violations": []}}}})
    rx.write_trace({{"format_version": 1, "scenario": section, "vehicles": vehicles}}, args.trace)
'''))
STARTUP_MARKER = os.path.join(work, "startup_calls")
PYTHON38 = os.path.join(work, "python38")
open(PYTHON38, "w").write(f'''#!/bin/bash
case "$1" in
  *apollo_runner_startup.py) echo started >> {STARTUP_MARKER}; exit 0;;
  *apollo_scenario_runner.py) shift; exec {sys.executable} {RUNNER} "$@";;
esac
exit 1
''')
os.chmod(PYTHON38, 0o755)

def executor(**options):
    options.setdefault("python38", PYTHON38); options.setdefault("timeout", 10); options.setdefault("map_file", MAP)
    return ApolloScenarioExecution(tempfile.mkdtemp(), **options)

def run(modes, scenario, planning_problems, **options):
    """Execute with the given sequence of runner modes; returns the result and the number of runs"""
    counter = os.path.join(tempfile.mkdtemp(), "count")
    os.environ["FAKE_MODES"], os.environ["FAKE_COUNTER"] = ",".join(modes), counter
    e = options.pop("executor", None) or executor(start_containers = False, **options)
    result = e.execute(scenario, planning_problems)
    return result, int(open(counter).read()) if os.path.exists(counter) else 0

# ---------- A scenario from EVITA's Apollo generator ----------
random.seed(5)
map_data = ApolloMapData.load(MAP)
config = simulators.generation_configuration(simulators.get_simulator("apollo"))
base, base_pps = get_base_scenario_from_file(MAP, remove_obstacles = True, remove_planning_problems = True)
scenario, planning_problems, _, _ = apollo_live_run.generate(base, base_pps, config, MAP, map_data, 2, random.Random(5))
scenario.generation_id = 7

# ---------- Start of a run ----------
e = executor()
check("start-up: Apollo's containers started once with apollo_runner_startup.py (log in the run folder)",
      open(STARTUP_MARKER).read().count("started") == 1 and os.path.exists(os.path.join(e.output_folder, "apollo_startup.log")))
# The same converted map under another name: it loads, but it is not the HD map Apollo runs (borregas_ave)
import shutil
other = os.path.join(tempfile.mkdtemp(), "other_map")
os.makedirs(other)
for path in glob.glob(os.path.join(os.path.dirname(MAP), "borregas_ave*")):
    shutil.copy(path, os.path.join(other, os.path.basename(path).replace("borregas_ave", "other_map")))
try:
    executor(map_file = os.path.join(other, "other_map.xml"), start_containers = False); check("HD map mismatch refused", False)
except ApolloSetupError as exc:
    check("start-up: a converted map that is not Apollo's HD map is refused", "not the HD map" in str(exc))
try:
    executor(python38 = "/nowhere/python", start_containers = False); check("missing .venv38 refused", False)
except ApolloSetupError as exc:
    check("start-up: a missing .venv38 Python is refused with a clear message", "/nowhere/python" in str(exc))

# ---------- Healthy run, files, records ----------
result, runs = run(["ok"], scenario, planning_problems)
folder = result.output_folder
check(f"healthy run: usable result (no failure), {len(result.agents)} vehicles, one run", result.failure is None and len(result.agents) == 2 and runs == 1)
check("files in <run>/<5-digit id>/: scenario.xml, c.json, runner.log, trace.json.gz, records kept by default",
      os.path.basename(folder) == "00007" and all(os.path.exists(os.path.join(folder, f)) for f in ("scenario.xml", "c.json", "runner.log", "trace.json.gz"))
      and glob.glob(os.path.join(folder, "records", "**", "apollo_dev_ROUTE_0*"), recursive = True))
check("records in records/Scenario_<id>/ (no generation level), named after the scenario",
      os.path.exists(os.path.join(folder, "records", "Scenario_7", "apollo_dev_ROUTE_0.Scenario_7")) and not glob.glob(os.path.join(folder, "records", "Generation_*")))
result, _ = run(["ok"], scenario, planning_problems, delete_records = True)
check("--delete-records: records deleted once the trace is written (trace kept)",
      not os.path.exists(os.path.join(result.output_folder, "records")) and os.path.exists(os.path.join(result.output_folder, "trace.json.gz")))

# ---------- Lights ----------
original = tc.TrafficControl.from_commonroad(scenario.lanelet_network, scenario.dt)
kept, _ = run(["ok"], scenario, planning_problems, random_lights = False)
check("EVITA's search (random_lights off): the scenario's own DoppelTest lights are executed",
      tc.TrafficControl.from_doppeltest(json.load(open(os.path.join(kept.output_folder, "c.json")))["tc_section"], map_data) == original)
drawn = []
for _ in range(4):
    r, _ = run(["ok"], scenario, planning_problems, random_lights = True)
    drawn.append(json.dumps(json.load(open(os.path.join(r.output_folder, "c.json")))["tc_section"], sort_keys = True))
check(f"Random / MaxDanger / MaxInteractions (random_lights on): new lights at every execution ({len(set(drawn))} different in 4); the search's scenario unchanged",
      len(set(drawn)) > 1 and tc.TrafficControl.from_commonroad(scenario.lanelet_network, scenario.dt) == original)

# ---------- Health check and retry ----------
result, runs = run(["reject"], scenario, planning_problems)
check(f"rejected by DoppelTest: invalid at once, no retry ({runs} run): {result.failure}", runs == 1 and result.failure and "rejected" in result.failure and "5 m" in result.failure)
result, runs = run(["crash", "ok"], scenario, planning_problems)
check(f"runner failure then success: retried once, usable ({runs} runs); failed attempt kept in failed_attempt_1/",
      runs == 2 and result.failure is None and os.path.exists(os.path.join(result.output_folder, "failed_attempt_1", "runner.log")))
result, runs = run(["crash", "records"], scenario, planning_problems)
check(f"runner failure twice: invalid after one retry ({runs} runs): {result.failure}", runs == 2 and result.failure and "record files" in result.failure)
result, runs = run(["no_localization", "no_localization"], scenario, planning_problems)
check(f"a vehicle that did not run, twice: invalid ({runs} runs): {result.failure}", runs == 2 and result.failure and "did not run" in result.failure)
result, runs = run(["no_localization", "ok"], scenario, planning_problems)
check(f"a vehicle that did not run, then a healthy retry: usable ({runs} runs)", runs == 2 and result.failure is None)
result, runs = run(["planning_stops", "planning_stops"], scenario, planning_problems)
check(f"planning lost 20 s before the end, mission not complete: invalid: {result.failure}", result.failure and "lost its planning" in result.failure)
result, runs = run(["planning_stops_after_mission"], scenario, planning_problems)
check(f"planning ending after the mission is complete: healthy ({runs} run)", runs == 1 and result.failure is None)
result, runs = run(["timeout", "timeout"], scenario, planning_problems, timeout = 3)
check(f"runner timeout: retried, then invalid: {result.failure}", runs == 2 and result.failure and "timed out" in result.failure)

# ---------- Entry points: the failure makes the scenario invalid, with its reason ----------
from evita.generators.NSGAIIproblem import NSGAIIproblem
from evita.generators.nsgaII_operators import CustomTimeTermination
os.environ["EVITA_PYTHON38"] = PYTHON38
problem = NSGAIIproblem(MAP, "apollo", 2, CustomTimeTermination(1, 0), 2.5, output_folder_name = os.path.join(tempfile.mkdtemp(), "run"), simulator = "apollo")
problem._load_executor_and_evaluator()
check("NSGAIIproblem with apollo: Apollo's executor (lights kept: EVITA evolves them) and the common evaluator",
      type(problem.scenario_executor).__name__ == "ApolloScenarioExecution" and problem.scenario_executor.random_lights is False
      and type(problem.scenario_evaluator).__name__ == "ScenarioEvaluator")
os.environ["FAKE_MODES"], os.environ["FAKE_COUNTER"] = "reject", os.path.join(tempfile.mkdtemp(), "count")
search_scenario = scenario
search_scenario.is_valid = True
execution = problem._execute(search_scenario, planning_problems)
check(f"EVITA's search: an unusable execution makes the scenario invalid, with the reason as its message",
      search_scenario.is_valid is False and "rejected" in search_scenario.validation_message)
moved = os.path.join(problem.debug_folder, "00007")
check("EVITA's search: the unusable scenario's folder is moved to <run>/debug/ (runner log and scenario kept), none left in the run folder",
      os.path.exists(os.path.join(moved, "runner.log")) and os.path.exists(os.path.join(moved, "scenario.xml"))
      and not os.path.exists(os.path.join(problem.output_folder, "00007")))
os.environ["FAKE_MODES"], os.environ["FAKE_COUNTER"] = "ok", os.path.join(tempfile.mkdtemp(), "count")
search_scenario.is_valid = True; search_scenario.validation_message = ""
execution = problem._execute(search_scenario, planning_problems)
table = problem.scenario_evaluator.evaluate(execution, 2.5)
check(f"EVITA's search: a healthy execution evaluated by the common evaluator ({len(table)} rows, database written)",
      search_scenario.is_valid and len(table) > 0 and os.path.exists(os.path.join(execution.output_folder, "logs", "simulation.db")))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
