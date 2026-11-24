"""
Tests of step 2a, part 1: record_export, apollo_scenario_runner, map_data, traffic_control.
Run from the project root: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tests/test_step2a_part1.py
"""
import sys, types, os, json, gzip, tempfile, random
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "EVITA_DoppelTest")

def stub_module(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m
stub_module("cyber_record"); stub_module("cyber_record.record", Record=object)

class FakeContainer:
    def __init__(self, root, name): self.container_name = name
class FakeScenarioRunner:
    def __init__(self, containers): self.containers = containers
    def set_scenario(self, s): self.s = s
    def init_scenario(self): pass
    def run_scenario(self, g, s, save):
        out = os.path.join(runner_module.RuntimeConfig.RECORDS_DIR, g, s); os.makedirs(out, exist_ok=True)
        runners = []
        for i, (c, adc) in enumerate(zip(self.containers, self.s.ad_section.adcs)):
            open(os.path.join(out, f"{c.container_name}.{s}"), "w").close()
            runners.append((types.SimpleNamespace(nid=900+i, container=c), adc))
        return runners
stub_module("apollo.ApolloContainer", ApolloContainer=FakeContainer)
stub_module("framework.scenario.ScenarioRunner", ScenarioRunner=FakeScenarioRunner)

import config
HD_MAP = os.path.abspath("EVITA_DoppelTest/data/maps/borregas_ave/base_map.bin")
CONVERTED = "EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml"
config.HD_MAP_PATH = HD_MAP
import apollo_scenario_runner as runner_module, record_export as rx
runner_module.HD_MAP_PATH = HD_MAP
from hdmap.MapParser import MapParser
MapParser(HD_MAP); mp = MapParser.get_instance()
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from modules.planning.proto.planning_pb2 import ADCTrajectory
from modules.planning.proto.decision_pb2 import StopReasonCode, ObjectNudge
from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacles, PerceptionObstacle
from modules.perception.proto.traffic_light_detection_pb2 import TrafficLightDetection, TrafficLight
from framework.scenario.tc_config import TCSection
from framework.scenario.TrafficControlManager import TrafficControlManager
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from evita.apollo.map_data import ApolloMapData, MapDataError, SAME_COLOR
from evita.apollo import traffic_control as tcm

results = []
def check(name, ok):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name)

# ---------- record_export ----------
def loc(t, x, heading):
    m = LocalizationEstimate(); m.header.timestamp_sec = t; m.pose.position.x = x; m.pose.position.y = 2.0; m.pose.heading = heading; return m
def plan(t):
    m = ADCTrajectory(); m.header.timestamp_sec = t
    for k in range(3):
        p = m.trajectory_point.add(); p.relative_time = 0.1*k; p.path_point.x = 10+k; p.v = 5
    m.decision.main_decision.stop.reason_code = StopReasonCode.Value("STOP_REASON_OBSTACLE")
    od = m.decision.object_decision.decision.add(); od.id = "123"; od.perception_id = 123
    od.object_decision.add().follow.distance_s = -8.0
    n = od.object_decision.add().nudge; n.type = ObjectNudge.Type.Value("LEFT_NUDGE"); n.distance_l = 0.5
    return m
def obstacles(t, x):
    m = PerceptionObstacles(); m.header.timestamp_sec = t
    o = m.perception_obstacle.add(); o.id = 123; o.position.x = x; o.position.y = 2; o.type = PerceptionObstacle.Type.Value("VEHICLE")
    for dx, dy in [(-2,-1),(2,-1),(2,1),(-2,1)]:
        p = o.polygon_point.add(); p.x, p.y = x+dx, 2+dy
    return m
def lights(t):
    m = TrafficLightDetection(); m.header.timestamp_sec = t; l = m.traffic_light.add(); l.id = "signal_3"; l.color = TrafficLight.RED; return m

p = rx.export_planning(plan(100.05), 100_060_000_000)
check("planning: main decision name and details kept apart", p["main_decision"] == {"decision": "stop", "details": {"reason_code": "STOP_REASON_OBSTACLE"}})
check("planning: object decisions keep the nudge's own type", p["object_decisions"][0]["decisions"] == [{"decision": "follow", "details": {"distance_s": -8.0}}, {"decision": "nudge", "details": {"type": "LEFT_NUDGE", "distance_l": 0.5}}])
check("planning: header time and record time", (p["t"], p["t_record"]) == (100_050_000_000, 100_060_000_000))

class FakeRecord:
    data = {}
    def __init__(self, path): self.path = path
    def read_messages(self): return iter(FakeRecord.data[os.path.basename(self.path)])
rx.Record = FakeRecord
d = tempfile.mkdtemp()
FakeRecord.data = {
  "ROUTE_0.Scenario_3.00000": [(rx.TOPIC_TRAFFIC_LIGHTS, lights(100.0), 100_000_000_000), (rx.TOPIC_LOCALIZATION, loc(100.02, 1.0, 0.0), 100_020_000_000),
                               (rx.TOPIC_LOCALIZATION, loc(100.01, 0.9, 0.0), 100_011_000_000), (rx.TOPIC_OBSTACLES, obstacles(100.03, 2.0), 100_030_000_000)],
  "ROUTE_0.Scenario_3.00001": [(rx.TOPIC_LOCALIZATION, loc(100.02, 1.0, 0.0), 100_021_000_000), (rx.TOPIC_PLANNING, plan(100.05), 100_060_000_000)],
  "ROUTE_1.Scenario_3": [(rx.TOPIC_LOCALIZATION, loc(100.0, 50.0, 3.14), 100_000_000_000)],
  "ROUTE_0.Scenario_30": []}
for f in FakeRecord.data: open(os.path.join(d, f), "w").close()
check("record files: chunks in order, Scenario_30 excluded", [os.path.basename(f) for f in rx.vehicle_record_files(d, "ROUTE_0", "Scenario_3")] == ["ROUTE_0.Scenario_3.00000", "ROUTE_0.Scenario_3.00001"])
trace = rx.build_trace(d, "Scenario_3", {}, [{"container": "ROUTE_0", "evita_id": 20000}, {"container": "ROUTE_1", "evita_id": 20001}])
v0 = trace["vehicles"][0]
check("localization sorted, repeated time removed", [r[0] for r in v0["localization"]["rows"]] == [100_010_000_000, 100_020_000_000])
check("every message kind exported", (len(v0["planning"]), len(v0["obstacles"]), len(v0["traffic_lights"])) == (1, 1, 1))
check("collision oracle fed", len(v0["oracles"]["collisions"]) > 0)
path = os.path.join(d, "out", "trace.json.gz"); rx.write_trace(trace, path)
check("trace written atomically and readable", json.load(gzip.open(path, "rt"))["vehicles"][1]["evita_id"] == 20001 and not os.path.exists(path + ".part"))
try: rx.build_trace(d, "Scenario_3", {}, [{"container": "ROUTE_7"}]); check("missing records raise", False)
except rx.RecordExportError: check("missing records raise", True)

# ---------- apollo_scenario_runner ----------
class OneMessageRecord:
    def __init__(self, path): pass
    def read_messages(self): return iter([(rx.TOPIC_LOCALIZATION, loc(100.0, 1.0, 0.0), 100_000_000_000)])
rx.Record = OneMessageRecord
lanes = sorted(mp.get_lanes(), key=lambda l: -mp.get_lane_length(l)); a, b = lanes[0], lanes[5]
def run(name, adcs, ids=True):
    sc = {"ad_section": {"adcs": adcs}, "pd_section": {"pds": []}, "tc_section": {"initial": {}, "final": {}, "duration_g": 5, "duration_y": 3, "duration_b": 2}}
    if ids: sc["evita_vehicle_ids"] = [20000 + i for i in range(len(adcs))]
    path = os.path.join(d, name + ".json"); json.dump(sc, open(path, "w")); trace_path = os.path.join(d, name + ".json.gz")
    args = runner_module.parse_arguments(["--scenario-json", path, "--records-dir", os.path.join(d, "records"), "--scenario", name, "--trace", trace_path])
    runner_module.RuntimeConfig.RECORDS_DIR = args.records_dir
    return runner_module.main(args), json.load(gzip.open(trace_path, "rt"))
ok = [{"routing": [a], "start_s": 5.0, "dest_s": 60.0, "start_t": 0.0}, {"routing": [b], "start_s": 5.0, "dest_s": 20.0, "start_t": 1.5}]
code, t = run("ok", ok)
check("runner: normal scenario executed with ids", code == 0 and [(v["container"], v["evita_id"]) for v in t["vehicles"]] == [("ROUTE_0", 20000), ("ROUTE_1", 20001)])
check("runner: vehicle dimensions in the trace", set(t["scenario"]["vehicle"]) == {"length", "width", "back_edge_to_center"})
code, t = run("close", [ok[0], {"routing": [a], "start_s": 8.0, "dest_s": 60.0, "start_t": 0.0}, ok[1]])
check("runner: dropped vehicle -> rejected", code == 2 and t["scenario"]["status"] == "rejected" and t["vehicles"] == [])
code, t = run("too_many", [{"routing": [a], "start_s": 5.0 + 12*i, "dest_s": 90.0, "start_t": 0.0} for i in range(6)])
check("runner: 6 vehicles -> rejected", code == 2)
code, t = run("no_ids", ok, ids=False)
check("runner: JSON without EVITA ids", code == 0 and [v["evita_id"] for v in t["vehicles"]] == [None, None])
path = os.path.join(d, "crash.json"); json.dump({"ad_section": {"adcs": ok}, "pd_section": {"pds": []}, "tc_section": {"initial": {}, "final": {}, "duration_g": 5, "duration_y": 3, "duration_b": 2}}, open(path, "w"))
crash_args = ["--scenario-json", path, "--records-dir", os.path.join(d, "records"), "--scenario", "crash", "--trace", os.path.join(d, "crash.json.gz")]
original_run = FakeScenarioRunner.run_scenario
def failing_run(self, g, s, save): raise RuntimeError("container ROUTE_1 is not running")
FakeScenarioRunner.run_scenario = failing_run
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    crash_code = runner_module.run(crash_args)
    missing_code = runner_module.run(["--trace", "x"])
FakeScenarioRunner.run_scenario = original_run
with contextlib.redirect_stdout(io.StringIO()):
    help_code = runner_module.run(["--help"])
check("runner: Apollo failure -> exit 3, missing argument -> 1, --help -> 0", (crash_code, missing_code, help_code) == (3, 1, 0))

# ---------- map_data ----------
md = ApolloMapData.load(CONVERTED); sc, pps = CommonRoadFileReader(CONVERTED).open()
md.check(sc.lanelet_network)
check("map_data: lookups", (md.lanelet_id("lane_0"), md.lane_id(1), md.signal_id(md.traffic_light_id("signal_0"))) == (1, "lane_0", "signal_0"))
check("map_data: relations symmetric", all((x, r) in md.related_traffic_lights(y) for x in md.relations_by_traffic_light for y, r in md.related_traffic_lights(x)))
from commonroad.scenario.lanelet import Lanelet
net = CommonRoadFileReader(CONVERTED).open()[0].lanelet_network; net.remove_lanelet(7)
check("map_data: lanelet missing from the scenario -> warning only", any("[7]" in w for w in md.check(net)))
net = CommonRoadFileReader(CONVERTED).open()[0].lanelet_network
# Lanelet 20 replaced by one with lanelet 21's geometry (as with shifted ids). Several lanelets of Borregas Ave share
# their centerline (Apollo's lanes overlap there), so the two are checked to differ first
import numpy as np
a_lane, b_lane = net.find_lanelet_by_id(20), net.find_lanelet_by_id(21)
differ = a_lane.center_vertices.shape != b_lane.center_vertices.shape or not np.allclose(a_lane.center_vertices, b_lane.center_vertices)
net.remove_lanelet(20)
net.add_lanelet(Lanelet(b_lane.left_vertices, b_lane.center_vertices, b_lane.right_vertices, 20))
try: md.check(net); check("map_data: lanelet with another map's geometry -> error", False)
except MapDataError as e: check("map_data: lanelet with another map's geometry -> error", differ and "[20]" in str(e))
light = md.traffic_light_id("signal_3")
net = CommonRoadFileReader(CONVERTED).open()[0].lanelet_network; net.remove_traffic_light(light)
try: md.check(net); check("map_data: traffic light missing -> error", False)
except MapDataError as e: check("map_data: traffic light missing -> error", str(light) in str(e))

# ---------- traffic_control ----------
dt = sc.dt; ids = [l.traffic_light_id for l in sc.lanelet_network.traffic_lights]; rng = random.Random(1)
names = {TrafficLight.GREEN: "GREEN", TrafficLight.YELLOW: "YELLOW", TrafficLight.RED: "RED"}
controls = [tcm.random_traffic_control(md, ids, tcm.max_duration_green(300, dt), rng) for _ in range(100)]
controls.append(tcm.TrafficControl(controls[0].initial, dict(controls[0].initial), 7))
mismatches = 0
for tc in controls:
    n = CommonRoadFileReader(CONVERTED).open()[0].lanelet_network; tc.to_commonroad(n, dt); by_id = {l.traffic_light_id: l for l in n.traffic_lights}
    manager = TrafficControlManager(TCSection(**tc.to_doppeltest(md)))
    for k in range(301):
        expected = {md.traffic_light_id(l.id): names[l.color] for l in manager.get_traffic_configuration(k / 10).traffic_light}
        mismatches += expected != tc.colors_at(k / 10) or expected != {i: tcm.COLOR_OF_STATE[l.get_state_at_time_step(k)] for i, l in by_id.items()}
check("traffic_control: identical to DoppelTest at every tick (101 controls x 301 ticks)", mismatches == 0)
signals = list(mp.get_signals()); same = 0
for seed in range(300):
    random.seed(seed); pref = random.sample(signals, random.randint(0, 6))
    random.seed(seed); dt_conf = TCSection.generate_config(preference = list(pref))
    random.seed(seed); ours = tcm.random_configuration(md, [md.traffic_light_id(s) for s in signals], [md.traffic_light_id(s) for s in pref], random_generator = random)
    same += {md.traffic_light_id(s): c for s, c in dt_conf.items()} == ours
check("traffic_control: generation identical to DoppelTest's generate_config (300 seeds)", same == 300)
tc = controls[3]; tc.to_commonroad(sc.lanelet_network, dt)
f = os.path.join(d, "x.xml"); CommonRoadFileWriter(sc, pps).write_to_file(f, OverwriteExistingFile.ALWAYS)
check("traffic_control: XML round trip", tcm.TrafficControl.from_commonroad(CommonRoadFileReader(f).open()[0].lanelet_network, dt) == tc)
check("traffic_control: JSON round trip", tcm.TrafficControl.from_doppeltest(tc.to_doppeltest(md), md) == tc)
m = tc
for _ in range(50): m = tcm.mutate_configurations(tcm.mutate_green_duration(m, 15, rng), md, rng)
check("traffic_control: mutations keep yellow and all-red", (m.duration_y, m.duration_b) == (3, 2) and 5 <= m.duration_g <= 15)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
