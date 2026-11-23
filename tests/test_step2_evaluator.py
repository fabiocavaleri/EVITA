"""
Tests of step 2, part B: the common evaluator with Apollo's interaction rule, collisions and database.
    python3 tests/test_step2_evaluator.py                      synthetic cases of the rule, database
    python3 tests/test_step2_evaluator.py <folder> [N]         also every trace.json.gz under <folder> (at most N):
        evaluated end to end; interacting time steps = an independent recount from the raw trace; every collision on
        both vehicles at a geometrically confirmed step; database readable by data_analysis
Run from the project root, with PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python.
"""
import os, re, sys, glob, gzip, json, logging, tempfile
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
logging.basicConfig(level = logging.ERROR)

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.trajectory import Trajectory

from evita.execution import AgentOutcome, ExecutionResult, ObjectDecision, Plan, STATUS_COLLISION, STATUS_TIMELIMIT
from evita.evaluators.scenario_evaluator import AGENT_ID_OFFSET_IN_DATABASE, ScenarioEvaluator
from evita.evaluators.scenario_evaluator import apollo_base_table
from evita.generators.utils import augment_with_decision_interactions
from evita.apollo.map_data import ApolloMapData
from evita.apollo import trace as tr
from evita import data_analysis

MAP = "EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml"
results = []
def check(name, ok):
    results.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name)


# ---------- Synthetic cases of the rule ----------

def straight_trajectory(step, y = 0.0):
    return Trajectory(step, [CustomState(time_step = step + j, position = np.array([j * 1.0, y]), orientation = 0.0, velocity = 10.0) for j in range(10)])

def decision(kind, raw, what = "stop", reason = None, object_id = None):
    return ObjectDecision(decision = what, object_kind = kind, object_id = object_id, raw_object_id = raw, reason = reason)

def plans_from(script):
    """script: {time step: (decisions, main decision, main reason)}; one plan per listed time step"""
    return [Plan(time_step = t, trajectory = straight_trajectory(t), object_decisions = d, main_decision = m, main_reason = r)
            for t, (d, m, r) in sorted(script.items())]

def evaluate_rule(script, status = STATUS_TIMELIMIT, last = 40):
    scenario, _ = CommonRoadFileReader(MAP).open()
    shape = Rectangle(4.9, 2.1)
    states = [CustomState(time_step = k, position = np.array([k * 1.0, 0.0]), orientation = 0.0, velocity = 10.0) for k in range(1, 41)]
    init = InitialState(time_step = 0, position = np.array([0.0, 0.0]), orientation = 0.0, velocity = 10.0, yaw_rate = 0.0, slip_angle = 0.0, acceleration = 0.0)
    scenario.add_objects(DynamicObstacle(10001, ObstacleType.CAR, shape, init, TrajectoryPrediction(Trajectory(1, states), shape)))
    agent = AgentOutcome(id = 10001, initial_time_step = 0, status = status, last_timestep = last, driven_states = [init] + states,
                         plans = plans_from(script), history_length = 41)
    execution = ExecutionResult(scenario_id = "synthetic", output_folder = "", executed_scenario = scenario, agents = {10001: agent}, simulator = "apollo")
    table = augment_with_decision_interactions(execution, apollo_base_table(execution)).set_index("level_1")
    return table

def evaluate_rule_without_trajectory():
    """Two plans: the first without trajectory, the second with a decision"""
    table_input = {2: ([], None, None), 3: ([decision(V, "900_0", "follow", object_id = 10002)], None, None)}
    original = plans_from
    def without_first(script):
        plans = original(script)
        plans[0].trajectory = None
        return plans
    globals()["plans_from"] = without_first
    try:
        return evaluate_rule(table_input)
    finally:
        globals()["plans_from"] = original

def steps_with(table, column = "is_interacting"):
    if column == "is_interacting":
        return [int(t) for t in table.index[table["is_interacting"] == 1.0]]
    return {int(t): v for t, v in table[column].items() if v is not None and v == v}

V, TL, SS = "vehicle", "traffic_light", "stop_sign"
OBSTACLE, SIGNAL, STOP_SIGN, CREEPER, DESTINATION = ("STOP_REASON_OBSTACLE", "STOP_REASON_SIGNAL", "STOP_REASON_STOP_SIGN",
                                                     "STOP_REASON_CREEPER", "STOP_REASON_DESTINATION")

t = evaluate_rule({1: ([], None, None), 2: ([decision(V, "900_0", "follow", object_id = 10002)], None, None),
                   3: ([decision(V, "900_0", "follow", object_id = 10002)], None, None),
                   4: ([decision(V, "900_0", "nudge", object_id = 10002)], None, None)})
check("vehicle: appearance (step 2) and change follow -> nudge (step 4) are events; an unchanged decision (step 3) is not",
      steps_with(t) == [2, 4] and steps_with(t, "is_interacting_with") == {2: [10002], 4: [10002]})

t = evaluate_rule({1: ([decision(TL, "TL_signal_1", reason = SIGNAL, object_id = 8001)], None, None), 2: ([], None, None),
                   3: ([decision(TL, "TL_signal_1", reason = SIGNAL, object_id = 8001)], None, None)})
check("every appearance counts: a decision that blinks off for one plan counts again", steps_with(t) == [1, 3]
      and steps_with(t, "traffic_light_interaction") == {1: [8001], 3: [8001]})

t = evaluate_rule({1: ([decision(V, "900_0", "ignore", object_id = 10002), decision("destination", "DEST", reason = DESTINATION),
                        decision("reference_end", "REF_x", reason = DESTINATION), decision("path_end", "PATH_x", reason = "STOP_REASON_REFERENCE_END"),
                        decision(V, "901_0", reason = "STOP_REASON_CLEAR_ZONE", object_id = 10003), decision("unknown", "XYZ")], None, None)})
check("excluded: ignore, DEST, REF, PATH, a vehicle stop for CLEAR_ZONE, unknown objects", steps_with(t) == [])

t = evaluate_rule({1: ([decision(TL, "CREEP_signal_11", reason = CREEPER, object_id = 8011)], None, None),
                   2: ([decision(SS, "CREEP_stopsign_0", reason = CREEPER)], None, None)})
check("CREEP: CREEP_signal_k counts as that traffic light, CREEP_stopsign_k as that stop sign",
      steps_with(t, "traffic_light_interaction") == {1: [8011]} and steps_with(t, "stop_sign_interaction") == {2: ["CREEP_stopsign_0"]})

t = evaluate_rule({1: ([], "stop", SIGNAL), 2: ([], "stop", SIGNAL), 3: ([decision(TL, "TL_signal_2", reason = SIGNAL, object_id = 8002)], "stop", SIGNAL),
                   4: ([], "cruise", None), 5: ([decision(V, "900_0", reason = OBSTACLE, object_id = 10002)], "stop", OBSTACLE),
                   6: ([], "stop", DESTINATION)})
check("main stop: an event when no decision for such an object is in the plan (step 1), not repeated (2); with the per-object "
      "decision present, only that one counts (3, 5); excluded reasons never (6)", steps_with(t) == [1, 3, 5]
      and steps_with(t, "traffic_light_interaction") == {3: [8002]} and steps_with(t, "is_interacting_with") == {5: [10002]})

t = evaluate_rule({5: ([decision(V, "900_0", "yield", object_id = 10002)], None, None), 20: ([decision(V, "900_0", "follow", object_id = 10002)], None, None),
                   30: ([decision(TL, "TL_signal_1", reason = SIGNAL, object_id = 8001)], None, None)}, status = STATUS_COLLISION, last = 25)
check("nothing is recorded from a collision on (collision at step 25: the event at step 30 is ignored)", steps_with(t) == [5, 20])

t = evaluate_rule({3: ([decision(SS, "SS_stopsign_0", reason = STOP_SIGN)], None, None), 6: ([decision(V, "900_0", "follow", object_id = 10002)], None, None)})
check("a plan without trajectory (Apollo not ready) as previous plan: interaction counted, difference unknown",
      steps_with(evaluate_rule_without_trajectory()) == [3])
check("diff_long / diff_lat: none for the first plan, computed for later events",
      pd.isna(t.loc[3, "diff_long"]) and not pd.isna(t.loc[6, "diff_long"]))


# ---------- Database ----------

scenario, _ = CommonRoadFileReader(MAP).open()
shape = Rectangle(4.9, 2.1)
for agent_id, y in [(10001, 0.0), (10002, 2.0)]:
    states = [CustomState(time_step = k, position = np.array([k * 0.5, y]), orientation = 0.0, velocity = 5.0) for k in range(1, 21)]
    scenario.add_objects(DynamicObstacle(agent_id, ObstacleType.CAR, shape, InitialState(time_step = 0, position = np.array([0.0, y]), orientation = 0.0, velocity = 5.0,
                                                                                         yaw_rate = 0.0, slip_angle = 0.0, acceleration = 0.0), TrajectoryPrediction(Trajectory(1, states), shape)))
agents = {i: AgentOutcome(id = i, initial_time_step = 0, status = STATUS_COLLISION, last_timestep = 0 + 1, history_length = 21,
                          driven_states = [scenario.obstacle_by_id(i).initial_state] + scenario.obstacle_by_id(i).prediction.trajectory.state_list,
                          plans = plans_from({3: ([decision(V, "900_0", "nudge", object_id = 10002 if i == 10001 else 10001)], None, None)}))
          for i in (10001, 10002)}
agents[10003] = AgentOutcome(id = 10003, initial_time_step = 0, status = "ERROR", last_timestep = 0)    # never ran
folder = tempfile.mkdtemp()
execution = ExecutionResult(scenario_id = "db_test", output_folder = folder, executed_scenario = scenario, agents = agents, simulator = "apollo")
table = ScenarioEvaluator(folder).evaluate(execution, 2.5)
database = os.path.join(folder, "logs", "simulation.db")
check("database at <output folder>/logs/simulation.db, accepted by data_analysis._is_evaluated", data_analysis._is_evaluated(database))
ids = data_analysis._query(database, "SELECT DISTINCT agent_id FROM scenario_evaluation ORDER BY agent_id")
meta_ids = data_analysis._query(database, "SELECT agent_ids FROM meta")
check(f"agent ids: scenario_evaluation = vehicle id + {AGENT_ID_OFFSET_IN_DATABASE} (as FrenetiX's), meta = real ids (all agents)",
      [r[0] for r in ids] == [10011, 10012] and json.loads(meta_ids[0][0]) == [10001, 10002, 10003])
collisions = data_analysis._query(database, "SELECT MIN(timestep), agent_id, has_collided_with FROM scenario_evaluation WHERE has_collided = 1.0 GROUP BY agent_id")
check(f"the report's collision query finds both vehicles ({len(collisions)} rows)", len(collisions) == 2)
check("rows: one per agent and time step of its trajectory; none for the agent that never ran", len(table) == 42 and 10003 not in table.index.get_level_values(0))


# ---------- Real traces ----------

def independent_interacting_steps(raw_trace, execution, conversion):
    """Recount of the interacting time steps per vehicle straight from the raw trace (separate implementation of the rule)"""
    vehicle_rule = {"stop", "nudge", "yield", "follow", "overtake", "avoid", "side_pass"}
    vehicle_stop = {"STOP_REASON_OBSTACLE", "STOP_REASON_HEAD_VEHICLE", "STOP_REASON_SIDEPASS_SAFETY", "STOP_REASON_EMERGENCY", None}
    light_stop = {"STOP_REASON_SIGNAL", "STOP_REASON_YELLOW_SIGNAL", "STOP_REASON_CREEPER"}
    sign_stop = {"STOP_REASON_STOP_SIGN", "STOP_REASON_YIELD_SIGN", "STOP_REASON_CREEPER"}
    main_kind = {"STOP_REASON_OBSTACLE": "v", "STOP_REASON_HEAD_VEHICLE": "v", "STOP_REASON_SIDEPASS_SAFETY": "v", "STOP_REASON_EMERGENCY": "v",
                 "STOP_REASON_SIGNAL": "tl", "STOP_REASON_YELLOW_SIGNAL": "tl", "STOP_REASON_STOP_SIGN": "ss", "STOP_REASON_YIELD_SIGN": "ss",
                 "STOP_REASON_CREEPER": "ss", "STOP_REASON_PEDESTRIAN": "p", "STOP_REASON_CROSSWALK": "p"}
    def kind_of(raw, what, reason):
        if what == "ignore":
            return None
        if re.fullmatch(r"-?\d+(_\d+)*", raw):
            return "v" if (what != "stop" or reason in vehicle_stop) and what in vehicle_rule else None
        if raw.startswith("TL_") or raw.startswith("CREEP_signal"):
            return "tl" if what == "stop" and reason in light_stop else None
        if raw.startswith("SS_") or raw.startswith("CREEP_stopsign"):
            return "ss" if what == "stop" and reason in sign_stop else None
        return None
    t0 = tr.time_zero(tr.read_trace(raw_trace))
    data = json.load(gzip.open(raw_trace, "rt"))
    counts = {}
    for vehicle, vehicle_id in zip(data["vehicles"], [v.vehicle_id for v in tr.read_trace(raw_trace).vehicles]):
        agent = execution.agents[vehicle_id]
        obstacle_steps = set(execution.executed_scenario.obstacle_by_id(vehicle_id).prediction.trajectory.state_list[i].time_step
                             for i in range(len(execution.executed_scenario.obstacle_by_id(vehicle_id).prediction.trajectory.state_list))) \
            if any(o.obstacle_id == vehicle_id for o in execution.executed_scenario.dynamic_obstacles) and execution.executed_scenario.obstacle_by_id(vehicle_id).prediction else set()
        if obstacle_steps:
            obstacle_steps.add(execution.executed_scenario.obstacle_by_id(vehicle_id).initial_state.time_step)
        previous, previous_main, steps = {}, None, set()
        for message in vehicle["planning"]:
            step = int(round((message["t"] - t0) / 1e9 / conversion.dt))
            if agent.status == STATUS_COLLISION and step + 1 >= agent.last_timestep:
                break
            current = {}
            for item in message["object_decisions"]:
                for d in item["decisions"]:
                    k = kind_of(item["id"], d["decision"], d["details"].get("reason_code"))
                    if k:
                        current.setdefault(item["id"], [k, set()])[1].add(d["decision"])
            events = [k for raw, (k, ds) in current.items() if previous.get(raw) != frozenset(ds)]
            m = message["main_decision"]
            mk = main_kind.get(m["details"].get("reason_code")) if m["decision"] == "stop" else None
            key = ("stop", m["details"].get("reason_code")) if mk else None
            if mk and key != previous_main and not any(k == mk for k, _ in current.values()):
                events.append(mk)
            if events and step in obstacle_steps:
                steps.add(step)
            previous = {raw: frozenset(ds) for raw, (k, ds) in current.items()}
            previous_main = key
        counts[vehicle_id] = len(steps)
    return counts

if len(sys.argv) > 1:
    traces = sorted(glob.glob(os.path.join(sys.argv[1], "**", "trace.json.gz"), recursive = True))[:int(sys.argv[2]) if len(sys.argv) > 2 else None]
    map_data = ApolloMapData.load(MAP)
    evaluated = matched = collisions_ok = collision_vehicles = databases_ok = 0
    mismatches = []
    for path in traces:
        folder = os.path.dirname(path)
        generated = CommonRoadFileReader(os.path.join(folder, "scenario.xml") if os.path.exists(os.path.join(folder, "scenario.xml")) else MAP).open()[0]
        output = tempfile.mkdtemp()
        raw = tr.read_trace(path)
        if raw.status != tr.EXECUTED:
            continue
        execution = tr.to_execution_result(raw, generated, map_data, output, scenario_id = os.path.basename(folder))
        table = ScenarioEvaluator(output).evaluate(execution, 2.5)
        evaluated += 1
        expected = independent_interacting_steps(path, execution, execution.native)
        for agent_id, n in expected.items():
            got = int(table.loc[agent_id]["is_interacting"].astype(float).sum()) if agent_id in table.index.get_level_values(0) else 0
            matched += got == n
            if got != n:
                mismatches.append((os.path.basename(folder), agent_id, got, n))
        for agent_id, agent in execution.agents.items():
            if agent.status == STATUS_COLLISION:
                collision_vehicles += 1
                rows = table.loc[agent_id] if agent_id in table.index.get_level_values(0) else None
                collisions_ok += rows is not None and (rows["has_collided"].astype(float) == 1.0).sum() == 1
        databases_ok += data_analysis._is_evaluated(os.path.join(output, "logs", "simulation.db"))
    total = sum(1 for _ in mismatches) + matched
    check(f"real traces: {evaluated} evaluated end to end, {databases_ok} databases accepted by data_analysis", evaluated > 0 and databases_ok == evaluated)
    check(f"real traces: interacting time steps = independent recount for {matched} of {total} vehicles" + (f" (first mismatches {mismatches[:3]})" if mismatches else ""), not mismatches)
    check(f"real traces: every collided vehicle has exactly one collision row ({collisions_ok} of {collision_vehicles})", collisions_ok == collision_vehicles)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
