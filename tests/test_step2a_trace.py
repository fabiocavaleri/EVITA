"""
Tests of step 2a: evita/apollo/trace.py, on a synthetic trace written with record_export's own functions
(vehicles driving real Borregas Ave lanes, traffic lights from DoppelTest's TrafficControlManager, runner clock 3% slow).
Run from the project root: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tests/test_step2a_trace.py
"""
import os, sys, types, math, random, tempfile, logging
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "EVITA_DoppelTest")
logging.basicConfig(level = logging.ERROR)
for name, attrs in [("cyber_record", {}), ("cyber_record.record", {"Record": object})]:
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m
import record_export as rx
from hdmap.MapParser import MapParser
from framework.scenario.tc_config import TCSection
from framework.scenario.TrafficControlManager import TrafficControlManager
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.planning.proto.planning_pb2 import ADCTrajectory
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from evita.apollo.map_data import ApolloMapData
from evita.apollo import traffic_control as tcm, trace as tr
from evita.apollo.scenario_converter import APOLLO_VEHICLE

MapParser("EVITA_DoppelTest/data/maps/borregas_ave/base_map.bin"); mp = MapParser.get_instance()
CONVERTED = "EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml"
md = ApolloMapData.load(CONVERTED); generated, _ = CommonRoadFileReader(CONVERTED).open(); dt = generated.dt
off = APOLLO_VEHICLE.center_offset
results = []
def check(name, ok):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name)

T0 = 1_700_000_000.0          # wall clock (s) when the runner publishes its first lights
DRIFT = 1.03                  # one runner tick of 0.1 s lasts 0.103 s
tc = tcm.random_traffic_control(md, [l.traffic_light_id for l in generated.lanelet_network.traffic_lights], 8, random.Random(2))
tc.to_commonroad(generated.lanelet_network, dt)
manager = TrafficControlManager(TCSection(**tc.to_doppeltest(md)))

lanes = sorted(mp.get_lanes(), key = lambda l: -mp.get_lane_length(l))
def drive(lane, s0, speed, start, duration):
    """Reference point at s0 + speed * t on the lane, for t in [start, start + duration]"""
    def pose(t):
        s = min(s0 + speed * max(t - start, 0), mp.get_lane_length(lane) - 0.1)
        p, h = mp.get_coordinate_and_heading(lane, s); return p.x, p.y, h, (speed if start <= t < start + duration else 0.0)
    return pose

def localization(t, pose):
    x, y, h, v = pose(t); m = LocalizationEstimate(); m.header.timestamp_sec = T0 + t
    m.pose.position.x, m.pose.position.y, m.pose.heading = x, y, h
    m.pose.linear_velocity.x, m.pose.linear_velocity.y = v * math.cos(h), v * math.sin(h); return m

def plan(t, pose, decision, estop = False, advice = 3, empty = False):
    m = ADCTrajectory(); m.header.timestamp_sec = T0 + t
    for k in range(0 if empty else 5):
        x, y, h, v = pose(t + 0.2 * k); p = m.trajectory_point.add()
        p.relative_time, p.path_point.x, p.path_point.y, p.path_point.theta, p.v = 0.2 * k, x, y, h, v
    getattr(m.decision.main_decision, decision).SetInParent()
    m.estop.is_estop = estop; m.engage_advice.advice = advice; return m

def lights(k):
    return manager.get_traffic_configuration(k / 10), T0 + k * 0.1 * DRIFT

def vehicle(container, apollo_id, evita_id, json_index, pose, duration, plan_times):
    locs = [rx.export_localization(localization(t, pose)) for t in np.arange(0, duration, 0.01)]
    plans = []
    for i, t in enumerate(plan_times):
        decision = "mission_complete" if t > duration - 1 else "cruise"
        advice = 2 if i < 3 else (4 if 30 <= i < 32 else (1 if i == 45 else 3))   # READY, KEEP, ...; PREPARE_DISENGAGE at 30-31; DISALLOW_ENGAGE at 45
        empty = i in (1, 40, len(plan_times) - 1)                                     # no trajectory: before moving, while driving, after mission complete
        plans.append(rx.export_planning(plan(t, pose, decision, estop = (i == 20), advice = advice, empty = empty), int((T0 + t) * 1e9) + 3_000_000))
    tls = []
    for k in range(int(duration / (0.1 * DRIFT)) + 1):
        msg, published = lights(k); msg.header.timestamp_sec = published; tls.append(rx.export_traffic_lights(msg))
    return {"container": container, "apollo_id": apollo_id, "json_index": json_index, "evita_id": evita_id, "routing": [], "start_s": 0, "dest_s": 0, "start_t": 0,
            "localization": {"columns": rx.LOCALIZATION_COLUMNS, "rows": locs}, "planning": plans, "obstacles": [], "traffic_lights": tls, "oracles": {"collisions": []}}

rng = random.Random(5)
plan_times_a = np.cumsum([rng.uniform(0.045, 0.14) for _ in range(150)]); plan_times_a = plan_times_a[plan_times_a < 12]
pose_a, pose_b = drive(lanes[0], 5, 8.0, 1.0, 9.0), drive(lanes[3], 3, 5.0, 2.5, 6.0)
va = vehicle("ROUTE_0", 901, 20001, 0, pose_a, 12.0, plan_times_a)
vb = vehicle("ROUTE_1", 902, 20002, 1, pose_b, 10.0, np.arange(0.1, 10, 0.1))
va["oracles"]["collisions"] = [["collision", 902, int((T0 + 7.0) * 1e9), [0, 0]], ["collision", 555, int((T0 + 8.05) * 1e9), [0, 0]]]
scenario = {"status": "executed", "reason": "", "vehicle": {"length": APOLLO_VEHICLE.length, "width": APOLLO_VEHICLE.width, "back_edge_to_center": APOLLO_VEHICLE.back_edge_to_center}}
d = tempfile.mkdtemp(); path = os.path.join(d, "trace.json.gz")
rx.write_trace({"format_version": 1, "scenario": scenario, "vehicles": [va, vb]}, path)

trace = tr.read_trace(path)
conversion = tr.convert_trace(trace, generated, md)
check("time zero is the first traffic light message", conversion.t0 == int(round(T0 * 1e9)))
a, b = conversion.vehicles[20001], conversion.vehicles[20002]
check("vehicles by EVITA id, with Apollo ids", (a.apollo_id, b.apollo_id) == (901, 902))

# Driven positions: CommonRoad center = Apollo reference point + offset along the heading
errors = []
for state in a.driven_states:
    x, y, h, _ = pose_a(state.time_step * dt)
    errors.append(math.hypot(state.position[0] - (x + off * math.cos(h)), state.position[1] - (y + off * math.sin(h))))
check(f"driven centers within 1 cm of the reference point + {off:.3f} m (max {max(errors)*100:.2f} cm)", max(errors) < 0.01)
check("time steps 0..119 for a 12 s recording", (a.initial_time_step, a.last_time_step) == (0, 119))
speeds = [s.velocity for s in a.driven_states]
check("speed 8 m/s while driving, 0 before", abs(speeds[50] - 8.0) < 1e-6 and abs(speeds[5]) < 1e-9)
check("first motion at 1.0 s", a.first_motion_step == 10)
check("obstacle size is Apollo's", (a.obstacle.obstacle_shape.length, a.obstacle.obstacle_shape.width) == (APOLLO_VEHICLE.length, APOLLO_VEHICLE.width))

# Plans: all kept (several per time step), centers shifted, facts
check(f"every plan kept ({len(plan_times_a)} plans, irregular 45-140 ms)", len(a.plans) == len(plan_times_a))
p = a.plans[10]; x, y, h, _ = pose_a((p.t / 1e9) - T0)
check("plan time and center", abs(p.trajectory[0, 0] - ((p.t / 1e9) - T0)) < 1e-6 and math.hypot(p.trajectory[0, 1] - (x + off * math.cos(h)), p.trajectory[0, 2] - (y + off * math.sin(h))) < 1e-6)
check("mission complete at the first mission_complete plan", a.mission_complete_step == int(round(a.plans[[pl.main_decision["decision"] for pl in a.plans].index("mission_complete")].time_step)))
check("emergency stop step", a.emergency_steps == {int(round(a.plans[20].time_step))})
check("disengagements while driving: no trajectory (plan 40) or take-over advice (30, 31); not before moving (1), after the "
      "mission (last plan) or for DISALLOW_ENGAGE (45)", a.disengagement_steps == {int(round(a.plans[i].time_step)) for i in (30, 31, 40)})
check("collisions: steps and partners (902 -> 20002, unknown -> None)", a.collision_steps == [70, 80] and a.collision_partners == [20002, None])

# Traffic lights: as received (runner clock 3% slow), not as specified
light = conversion.executed_scenario.lanelet_network.traffic_lights[0]; lid = light.traffic_light_id
received = [tcm.COLOR_OF_STATE[light.get_state_at_time_step(k)] for k in range(120)]
expected = []
for k in range(120):
    tick = max(j for j in range(200) if j * 0.1 * DRIFT <= k * 0.1 + 1e-3)
    expected.append(tc.colors_at(tick / 10)[lid])
specified = [tc.colors_at(k / 10)[lid] for k in range(120)]
check("executed lights = recorded messages at every time step", received == expected)
print(f"     (with the 3% slow runner clock, the specified timeline differs at {sum(r != s for r, s in zip(received, specified))} of 120 time steps)")
check("generated scenario untouched", tcm.TrafficControl.from_commonroad(generated.lanelet_network, dt) == tc and not generated.dynamic_obstacles)

# The executed scenario writes and reads back as CommonRoad XML
from commonroad.planning.planning_problem import PlanningProblemSet
f = os.path.join(d, "executed.xml")
CommonRoadFileWriter(conversion.executed_scenario, PlanningProblemSet()).write_to_file(f, OverwriteExistingFile.ALWAYS)
back, _ = CommonRoadFileReader(f).open()
check("executed scenario XML round trip", sorted(o.obstacle_id for o in back.dynamic_obstacles) == [20001, 20002])

# Clock check: localization 10 s away from the first traffic light message
late = dict(vb); late["localization"] = {"columns": rx.LOCALIZATION_COLUMNS, "rows": [[r[0] + 10_000_000_000] + r[1:] for r in vb["localization"]["rows"]]}
rx.write_trace({"format_version": 1, "scenario": scenario, "vehicles": [va, late]}, path)
check("clock mismatch reported", any("same clock" in w for w in tr.convert_trace(tr.read_trace(path), generated, md).warnings))

# Rejected trace
rx.write_trace({"format_version": 1, "scenario": {"status": "rejected", "reason": "6 vehicles"}, "vehicles": []}, path)
try: tr.convert_trace(tr.read_trace(path), generated, md); check("rejected trace refused", False)
except tr.TraceError as e: check("rejected trace refused", "rejected" in str(e))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
