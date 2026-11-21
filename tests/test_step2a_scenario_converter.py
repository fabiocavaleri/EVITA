"""
Tests of step 2a: evita/apollo/scenario_converter.py, against DoppelTest's HD map and from_json.
Run from the project root: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tests/test_step2a_scenario_converter.py
"""
import os, sys, math, random, json, tempfile, logging
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "EVITA_DoppelTest")
logging.basicConfig(level = logging.ERROR)
from hdmap.MapParser import MapParser
from framework.scenario import Scenario as ApolloScenario
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.state import InitialState, CustomState
from commonroad.geometry.shape import Rectangle
from commonroad.common.util import Interval
from evita.apollo.map_data import ApolloMapData
from evita.apollo import scenario_converter as ss, traffic_control as tcm

MapParser("EVITA_DoppelTest/data/maps/borregas_ave/base_map.bin"); mp = MapParser.get_instance()
CONVERTED = "EVITA_DoppelTest/conversion_scripts/converted_maps/borregas_ave_fix.xml"
md = ApolloMapData.load(CONVERTED); sc, _ = CommonRoadFileReader(CONVERTED).open(); net = sc.lanelet_network
off = ss.APOLLO_VEHICLE.center_offset
results = []
def check(name, ok):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name)
def point_on(lanelet, s):
    p = ss.centerline(lanelet).interpolate(s); return np.array([p.x, p.y])
def problem(pid, lanelet, s, route = None, goal_lanelet = None, goal_s = 3.0, t = 0):
    init = InitialState(position = point_on(lanelet, s), orientation = 0, velocity = 0, time_step = t, yaw_rate = 0, slip_angle = 0, acceleration = 0)
    goal = GoalRegion([CustomState(time_step = Interval(0, 300), position = Rectangle(4, 3, point_on(goal_lanelet or lanelet, goal_s), 0))])
    pp = PlanningProblem(pid, init, goal)
    if route is not None:
        pp.path = route          # what EVITAPlanningProblem adds
    return pp

# Heading: identical to DoppelTest's MapParser (used by ApolloRunner to place vehicles)
diff = max(abs(math.atan2(*ss.apollo_heading(ss.centerline(l), s)[::-1]) - mp.get_coordinate_and_heading(md.lane_id(l.lanelet_id), s)[1])
           for l in net.lanelets for s in np.linspace(0, ss.centerline(l).length, 40))
check("heading identical to DoppelTest's MapParser", diff < 1e-3)

# Positions: center of Apollo's car (from start_s / dest_s via the HD map) vs the CommonRoad center
rng = random.Random(3); start_err, dest_err = [], []
for _ in range(400):
    lanelet = rng.choice(net.lanelets)
    if not lanelet.successor or ss.centerline(lanelet).length < 8:
        continue
    nxt = net.find_lanelet_by_id(lanelet.successor[0])
    s, gs = rng.uniform(off + 0.5, ss.centerline(lanelet).length - 2), rng.uniform(off + 0.2, ss.centerline(nxt).length)
    pp = problem(20000, lanelet, s, [lanelet.lanelet_id, nxt.lanelet_id], nxt, gs)
    adc = ss.build_adc(net, pp, md, sc.dt)
    for key, lane, target, errors in [("start_s", adc["routing"][0], pp.initial_state.position, start_err), ("dest_s", adc["routing"][-1], point_on(nxt, gs), dest_err)]:
        ref, heading = mp.get_coordinate_and_heading(lane, adc[key])
        errors.append(np.linalg.norm(np.array([ref.x + off * math.cos(heading), ref.y + off * math.sin(heading)]) - target))
print(f"     {len(start_err)} vehicles: start median {np.median(start_err)*100:.2f} cm, max {max(start_err)*100:.1f} cm; goal median {np.median(dest_err)*100:.2f} cm, max {max(dest_err)*100:.1f} cm")
check("start and goal: median within 1 cm, 90% within 5 cm, all within 0.5 m",
      np.median(start_err) < 0.01 and np.median(dest_err) < 0.01 and np.mean(np.array(start_err) < 0.05) > 0.9 and max(start_err + dest_err) < 0.5)

# Routes
chain = next([l.lanelet_id, l.successor[0], net.find_lanelet_by_id(l.successor[0]).successor[0]]
             for l in net.lanelets if l.successor and net.find_lanelet_by_id(l.successor[0]).successor)
a, b, c = (net.find_lanelet_by_id(i) for i in chain)
check("route given is kept", ss.vehicle_route(net, problem(1, a, 6.0, chain, c)) == chain)
check("route found without one (shortest chain of successors)", ss.vehicle_route(net, problem(2, a, 6.0, None, c)) == chain)
# Goal just after a split, at a point inside two overlapping branches: the route must go to the branch the goal is
# centered on, not to the branch with the shortest route
cases = []
for split in (l for l in net.lanelets if len(l.successor) >= 2):
    branches = [net.find_lanelet_by_id(i) for i in split.successor]
    for target in branches:
        for goal_s in np.arange(1.0, 15.0, 0.5):
            inside = set(net.find_lanelet_by_position([list(point_on(target, goal_s))])[0])
            others = (inside & set(split.successor)) - {target.lanelet_id}
            # Only where the branches can be told apart: another branch contains the goal, but its centerline is
            # more than 5 cm from it (where centerlines coincide, the goal is ambiguous and either branch is right)
            if others and min(ss.position_along(net.find_lanelet_by_id(o), point_on(target, goal_s))[1] for o in others) > 0.05:
                cases.append((split, target, goal_s)); break
right = sum(ss.vehicle_route(net, problem(9, split, 2.0, None, target, goal_s))[-1] == target.lanelet_id for split, target, goal_s in cases)
check(f"goal inside two overlapping branches: route to the branch it is centered on ({right}/{len(cases)} cases)", len(cases) > 0 and right == len(cases))
for name, pp in [("route with a gap refused", problem(3, a, 6.0, [chain[0], chain[2]], c)),
                 ("goal behind the start refused", problem(4, a, 10.0, [chain[0]], a, 4.0))]:
    try: ss.build_adc(net, pp, md, sc.dt); check(name, False)
    except ss.ScenarioSpecError: check(name, True)
pp = problem(5, a, 6.0, chain, c); pp.initial_state.position = pp.initial_state.position + np.array([0, 8.0])
try: ss.build_adc(net, pp, md, sc.dt); check("start off its lanelet refused", False)
except ss.ScenarioSpecError: check("start off its lanelet refused", True)
import dataclasses
no_lane = dataclasses.replace(md, lane_by_lanelet = {k: v for k, v in md.lane_by_lanelet.items() if k != chain[1]})
try: ss.build_adc(net, problem(7, a, 6.0, chain, c), no_lane, sc.dt); check("route lanelet without Apollo lane -> ScenarioSpecError", False)
except ss.ScenarioSpecError: check("route lanelet without Apollo lane -> ScenarioSpecError", True)
logging.getLogger("evita.apollo.scenario_converter").setLevel(logging.CRITICAL)
check("start too close to the lane's beginning moved to 0", ss.build_adc(net, problem(6, a, 0.6, chain, c), md, sc.dt)["start_s"] == 0.0)

# Whole scenario through DoppelTest's from_json
tc = tcm.random_traffic_control(md, [l.traffic_light_id for l in net.traffic_lights], 15, random.Random(1)); tc.to_commonroad(net, sc.dt)
spec = ss.build_scenario_json(sc, PlanningProblemSet([problem(20001, a, 6.0, chain, c), problem(20002, b, 15.0, chain[1:], c, 3.0, 12)]), md)
check("vehicle ids and start times", spec["evita_vehicle_ids"] == [20001, 20002] and [x["start_t"] for x in spec["ad_section"]["adcs"]] == [0.0, 1.2])
path = os.path.join(tempfile.mkdtemp(), "c.json"); ss.write_scenario_json(spec, path)
check("DoppelTest's from_json keeps every vehicle", len(ApolloScenario.from_json(path).ad_section.adcs) == 2)
check("traffic control in the JSON reads back unchanged", tcm.TrafficControl.from_doppeltest(spec["tc_section"], md) == tc)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
