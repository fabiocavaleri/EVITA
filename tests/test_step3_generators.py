"""
Tests of step 3, part E: the generators under Apollo (Apollo's generation settings, Apollo's validation) and the
Apollo-specific start-distance validation (DoppelTest's rule, replacing EVITA's same-lane check under Apollo).
Run from the project root: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tests/test_step3_generators.py
"""
import os, sys, random, logging, tempfile, contextlib
from types import SimpleNamespace
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "EVITA_DoppelTest")
logging.basicConfig(level = logging.CRITICAL)
import numpy as np

from hdmap.MapParser import MapParser
from apollo.utils import PositionEstimate
from framework.scenario import Scenario as ApolloScenario
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.state import CustomState, InitialState

from evita import simulators
from evita.config import ScenarioGeneratorConfig, ValidationConfig
from evita.generators.utils import get_base_scenario_from_file
from evita.apollo.map_data import ApolloMapData
from evita.apollo import scenario_converter as converter
from evita.apollo import traffic_control as tc

MAP = "EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml"
results = []
def check(name, ok):
    results.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name)

MapParser("EVITA_DoppelTest/data/maps/borregas_ave/base_map.bin")
map_data = ApolloMapData.load(MAP)
frenetix, apollo = simulators.get_simulator("frenetix"), simulators.get_simulator("apollo")
apollo_config = simulators.generation_configuration(apollo)
base, base_pps = get_base_scenario_from_file(MAP, remove_obstacles = True, remove_planning_problems = True)
network = base.lanelet_network

def vehicle(vehicle_id, lanelet, s, velocity = 0.0):
    """A planning problem centered on a lanelet's centerline at s, oriented along it"""
    line = converter.centerline(lanelet)
    point, ahead = line.interpolate(s), line.interpolate(min(s + 0.5, line.length))
    heading = float(np.arctan2(ahead.y - point.y, ahead.x - point.x))
    goal_point = line.interpolate(min(s + 20, line.length - 1))
    pp = PlanningProblem(vehicle_id, InitialState(position = np.array([point.x, point.y]), orientation = heading, velocity = velocity, time_step = 0,
                                                  yaw_rate = 0.0, slip_angle = 0.0, acceleration = 0.0),
                         GoalRegion([CustomState(time_step = Interval(0, 300), position = Rectangle(4, 3, np.array([goal_point.x, goal_point.y]), heading))]))
    pp.path = [lanelet.lanelet_id]
    return pp

apollo_validator = simulators.scenario_validator(apollo, ValidationConfig(), apollo_config)
frenetix_validator = simulators.scenario_validator(frenetix, ValidationConfig(), simulators.generation_configuration(frenetix))

# ---------- The Apollo start-distance validation ----------
straight = max((l for l in network.lanelets if l.adj_left_same_direction and l.adj_left),
               key = lambda l: converter.centerline(l).length)
neighbour = network.find_lanelet_by_id(straight.adj_left)
a, b = vehicle(1, straight, 20.0), vehicle(2, neighbour, 20.0)
ok, message = apollo_validator._validate_apollo_start_distances([a, b])
check(f"side by side on neighbouring lanes: invalid for Apollo ({message})", not ok and "apart" in message)
check("side by side on neighbouring lanes: EVITA's same-lane check alone would accept them", frenetix_validator._validate_init_distances(base, [a, b])[0])
ok3, _ = apollo_validator._validate_apollo_start_distances([vehicle(1, straight, 20.0), vehicle(2, straight, 27.9)])   # gap 7.9 - 4.93 = 2.97 m
ok6, _ = apollo_validator._validate_apollo_start_distances([vehicle(1, straight, 20.0), vehicle(2, straight, 30.5)])   # gap 5.57 m
check("same lane: a 3 m gap invalid, a 5.6 m gap valid", not ok3 and ok6)
fast = [vehicle(1, straight, 20.0, velocity = 10.0), vehicle(2, straight, 30.5, velocity = 10.0)]
check("same lane at 10 m/s with a 5.6 m gap: Apollo's rule accepts (Apollo starts at rest), EVITA's 2 s headway would refuse",
      apollo_validator._validate_apollo_start_distances(fast)[0] and not frenetix_validator._validate_init_distances(base, fast)[0])

# Agreement with DoppelTest's own rule, on random pairs placed on the map
rng = random.Random(3); agree = total = 0
lanelets = [l for l in network.lanelets if converter.centerline(l).length > 12]
for _ in range(400):
    l1, l2 = rng.choice(lanelets), rng.choice(lanelets)
    p1 = vehicle(1, l1, rng.uniform(6, converter.centerline(l1).length - 6)); p2 = vehicle(2, l2, rng.uniform(6, converter.centerline(l2).length - 6))
    try:
        adcs = [converter.build_adc(base.lanelet_network, pp, map_data, base.dt) for pp in (p1, p2)]
    except converter.ScenarioSpecError:
        continue
    evita_ok = apollo_validator._validate_apollo_start_distances([p1, p2])[0]
    doppeltest_ok = not PositionEstimate(adcs[0]["routing"][0], adcs[0]["start_s"]).is_too_close(PositionEstimate(adcs[1]["routing"][0], adcs[1]["start_s"]))
    total += 1; agree += evita_ok == doppeltest_ok
check(f"EVITA's Apollo rule agrees with DoppelTest's is_too_close on {agree} of {total} random pairs on Borregas Ave", total > 300 and agree >= total - 2)

# ---------- Route length under Apollo: EVITA's rule (goal at least min_dist_goal ahead) on Apollo's route ----------
def with_goal(vehicle_id, lanelet, s, goal_lanelet, goal_s):
    """A planning problem starting at s on a lanelet, its goal area centered at goal_s on another (or the same) lanelet"""
    pp = vehicle(vehicle_id, lanelet, s)
    line = converter.centerline(goal_lanelet); point = line.interpolate(goal_s); ahead = line.interpolate(min(goal_s + 0.5, line.length))
    heading = float(np.arctan2(ahead.y - point.y, ahead.x - point.x))
    pp.goal = GoalRegion([CustomState(time_step = Interval(0, 300), position = Rectangle(4, 3, np.array([point.x, point.y]), heading))])
    pp.path = None
    return pp
min_goal = ValidationConfig.min_dist_goal
near, far = with_goal(1, straight, 10.0, straight, 20.0), with_goal(1, straight, 10.0, straight, 10.0 + min_goal + 10)
ok_near, message_near = apollo_validator._validate_apollo_reachable_goal(base, near)
check(f"Apollo: a goal 10 m ahead on the start lane refused (EVITA's min_dist_goal = {min_goal} m): {message_near}", not ok_near and "ahead along its route" in message_near)
check(f"Apollo: a goal {min_goal + 10} m ahead accepted", apollo_validator._validate_apollo_reachable_goal(base, far)[0])
unreachable = next(l for l in network.lanelets if l.lanelet_id != straight.lanelet_id and not converter.shortest_successor_route(network, [straight.lanelet_id], [l.lanelet_id]))
ok_none, message_none = apollo_validator._validate_apollo_reachable_goal(base, with_goal(1, straight, 10.0, unreachable, 5.0))
check(f"Apollo: a goal no chain of successors reaches refused ({message_none})", not ok_none and "No route" in message_none)

# ---------- Generation settings for the orchestrator's generators ----------
check("generation_overrides: none for FrenetiX; Apollo's vehicles, vehicle model and size",
      simulators.generation_overrides(frenetix) == {} and simulators.generation_overrides(apollo)
      == {"cr_vehicle_id": 101, "length": 4.933, "width": 2.11, "max_amount_pp": 5})

from evita.generators.random_generator import RandomScenarioGenerator
from evita.generators.basic_generator import EAOnePlusOneGenerator_MaxDanger, EAOnePlusOneGenerator_MaxInteractions
settings = lambda c: {k: v for k, v in vars(c).items() if not callable(v)}
default = settings(ScenarioGeneratorConfig())
random_frenetix = RandomScenarioGenerator(MAP, debug = False, debug_folder = tempfile.mkdtemp(), simulator = "frenetix")
check("RandomScenarioGenerator with FrenetiX: default configuration, FrenetiX's validator",
      settings(random_frenetix.scenario_generator_configuration) == default and random_frenetix.scenario_validator.apollo is False)
random_apollo = RandomScenarioGenerator(MAP, debug = False, debug_folder = tempfile.mkdtemp(), simulator = "apollo", **simulators.generation_overrides(apollo))
check("RandomScenarioGenerator with Apollo: Apollo's configuration (5 vehicles, model 101) and validator",
      settings(random_apollo.scenario_generator_configuration) == settings(apollo_config) and random_apollo.scenario_validator.apollo is True)
for cls in (EAOnePlusOneGenerator_MaxDanger, EAOnePlusOneGenerator_MaxInteractions):
    g = cls(MAP, debug = False, debug_folder = tempfile.mkdtemp(), simulator = "apollo", **simulators.generation_overrides(apollo))
    check(f"{cls.__name__} with Apollo: Apollo's configuration", settings(g.wrapped_generator.scenario_generator_configuration) == settings(apollo_config))

random.seed(11)
dropped = vehicles = generated = 0; max_vehicles = 0
for _ in range(10):
    produced = random_apollo.generate()
    if not produced:
        continue
    scenario, pps = produced[0]; generated += 1
    max_vehicles = max(max_vehicles, len(pps.planning_problem_dict))
    # The executor draws the lights at every execution (Random does not evolve them): the same here
    lights = [light.traffic_light_id for light in scenario.lanelet_network.traffic_lights]
    tc.random_traffic_control(map_data, lights, tc.max_duration_green(apollo_config.max_time_step, scenario.dt)).to_commonroad(scenario.lanelet_network, scenario.dt)
    scenario_json = converter.build_scenario_json(scenario, pps, map_data)
    path = os.path.join(tempfile.mkdtemp(), "c.json"); converter.write_scenario_json(scenario_json, path)
    vehicles += len(scenario_json["ad_section"]["adcs"]); dropped += len(scenario_json["ad_section"]["adcs"]) - len(ApolloScenario.from_json(path).ad_section.adcs)
check(f"Random under Apollo: {generated} scenarios, at most {max_vehicles} vehicles, all convertible; DoppelTest drops {dropped} of {vehicles} vehicles",
      generated >= 8 and max_vehicles <= 5 and dropped == 0)

# ---------- EVITA's operators ----------
from evita.generators.nsgaII_operators import CustomTimeTermination, EVITAScenarioMutation, EVITAScenarioSampling, ScenarioCrossover, ScenarioMutation
term = CustomTimeTermination(1, 0)
sampling = EVITAScenarioSampling(base, base_pps, apollo_config, term, MAP)
check("EVITAScenarioSampling: Apollo validator with Apollo's vehicle size",
      sampling.scenario_validator.apollo and sampling.scenario_validator.generation_config.length == 4.933)
check("EVITAScenarioMutation and ScenarioCrossover(apollo_validation=True): Apollo validator; defaults: FrenetiX's",
      EVITAScenarioMutation(MAP, apollo_config, term).scenario_validator.apollo and ScenarioCrossover(apollo_config, term, apollo_validation = True).scenario_validator.apollo
      and not ScenarioMutation(MAP, ScenarioGeneratorConfig(), term).scenario_validator.apollo and not ScenarioCrossover(ScenarioGeneratorConfig(), term).scenario_validator.apollo)

class Termination:
    def __init__(self):
        self.progress_bar_manager = SimpleNamespace(counter = lambda **kwargs: SimpleNamespace(update = lambda *a: None, close = lambda **k: None))
    def generating(self): return contextlib.nullcontext()
    def paused(self): return contextlib.nullcontext()
    def update_main_progress_bar(self): pass
random.seed(4)
population = EVITAScenarioSampling(base, base_pps, apollo_config, Termination(), MAP)._do(None, 10)
dropped = vehicles = 0
for scenario, pps in population:
    scenario_json = converter.build_scenario_json(scenario, pps, map_data)
    path = os.path.join(tempfile.mkdtemp(), "c.json"); converter.write_scenario_json(scenario_json, path)
    vehicles += len(scenario_json["ad_section"]["adcs"]); dropped += len(scenario_json["ad_section"]["adcs"]) - len(ApolloScenario.from_json(path).ad_section.adcs)
check(f"EVITA's Apollo sampling: DoppelTest drops {dropped} of {vehicles} vehicles (1 of 38 before the Apollo rule)", dropped == 0)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
