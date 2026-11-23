"""
Tests of step 2a, generation: EVITA's Apollo sampling (EVITAScenarioSampling) and traffic light mutations
(scenario_mutation.mutate_tl_initial_configuration, mutate_green_light_duration) with DoppelTest's traffic control.
Run from the project root: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tests/test_step2a_generation.py
"""
import os, sys, random, contextlib, tempfile, logging
from copy import deepcopy
from types import SimpleNamespace
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "EVITA_DoppelTest")
logging.basicConfig(level = logging.ERROR)

from hdmap.MapParser import MapParser
from framework.scenario import Scenario as ApolloScenario
from framework.scenario.tc_config import TCSection
from framework.scenario.TrafficControlManager import TrafficControlManager
from modules.perception.proto.traffic_light_detection_pb2 import TrafficLight
from cr_scenario_handler.utils.vehicle_models import APOLLO_LINCOLN_MKZ, vehicle_parameters

from evita.config import ScenarioGeneratorConfig, ValidationConfig
from evita.validator import ScenarioValidator
from evita.generators import scenario_mutation
from evita.generators.nsgaII_operators import EVITAScenarioSampling
from evita.generators.utils import get_base_scenario_from_file
from evita.apollo.map_data import ApolloMapData, SAME_COLOR
from evita.apollo import traffic_control as tc
from evita.apollo import scenario_converter as converter

MAP = "EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml"
HD_MAP = "EVITA_DoppelTest/data/maps/borregas_ave/base_map.bin"
results = []
def check(name, ok):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name)


class Termination:
    """Stand-in for EVITA's termination criterion (budget and progress bars are not needed here)"""
    def __init__(self):
        self.progress_bar_manager = SimpleNamespace(counter = lambda **kwargs: SimpleNamespace(update = lambda *a: None, close = lambda **k: None))
    def generating(self): return contextlib.nullcontext()
    def paused(self): return contextlib.nullcontext()
    def update_main_progress_bar(self): pass


def relations_respected(configuration, map_data):
    for light, color in configuration.items():
        for other, relation in map_data.related_traffic_lights(light):
            if relation == SAME_COLOR and configuration[other] != color:
                return False
            if relation != SAME_COLOR and color == configuration[other] == tc.GREEN:
                return False
    return True


random.seed(3)
MapParser(HD_MAP)
map_data = ApolloMapData.load(MAP)
config = ScenarioGeneratorConfig(300, 5)
config.cr_vehicle_id = APOLLO_LINCOLN_MKZ
parameters = vehicle_parameters(APOLLO_LINCOLN_MKZ)
config.length, config.width = parameters.l, parameters.w
base_scenario, base_planning_problems = get_base_scenario_from_file(MAP, remove_obstacles = True, remove_planning_problems = True)
dt = base_scenario.dt
max_green = tc.max_duration_green(config.max_time_step, dt)

# ---------- Sampling ----------
sampling = EVITAScenarioSampling(base_scenario, base_planning_problems, config, Termination(), MAP)
X = sampling._do(None, 10)
base_control = tc.TrafficControl.from_commonroad(sampling.base_scenario.lanelet_network, dt)
check(f"base scenario: DoppelTest traffic control (green {base_control.duration_g} s, yellow {base_control.duration_y} s, all red {base_control.duration_b} s)",
      (base_control.duration_y, base_control.duration_b) == (3, 2) and tc.MIN_DURATION_GREEN <= base_control.duration_g <= max_green)
check("base scenario: configurations respect the signal relations", relations_respected(base_control.initial, map_data) and relations_respected(base_control.final, map_data))
check("initial population shares the base scenario's traffic control",
      all(tc.TrafficControl.from_commonroad(scenario.lanelet_network, dt) == base_control for scenario, _ in X))

validator = ScenarioValidator(ValidationConfig(), config, apollo = True)
check("every individual valid (Apollo-mode validator)", all(validator.is_scenario_valid(s, p)[0] for s, p in X))
spec_errors, dropped, vehicles = 0, 0, 0
directory = tempfile.mkdtemp()
for k, (scenario, planning_problems) in enumerate(X):
    try:
        scenario_json = converter.build_scenario_json(scenario, planning_problems, map_data)
    except converter.ScenarioSpecError:
        spec_errors += 1
        continue
    path = os.path.join(directory, f"c{k}.json"); converter.write_scenario_json(scenario_json, path)
    requested = len(scenario_json["ad_section"]["adcs"]); vehicles += requested
    dropped += requested - len(ApolloScenario.from_json(path).ad_section.adcs)
check(f"every individual converts to DoppelTest's JSON ({len(X)} individuals, {vehicles} vehicles)", spec_errors == 0)
print(f"     DoppelTest's from_json dropped {dropped} of {vehicles} vehicles (5 m rule; such scenarios are rejected by the runner)")

# ---------- Mutations ----------
configurations_changed = durations_changed = 0
always_valid = True
for scenario, _ in X[:4]:
    current = deepcopy(scenario)
    for _ in range(25):
        before = tc.TrafficControl.from_commonroad(current.lanelet_network, dt)
        current = scenario_mutation.mutate_tl_initial_configuration(config, current, MAP)
        after_configuration = tc.TrafficControl.from_commonroad(current.lanelet_network, dt)
        configurations_changed += (after_configuration.initial, after_configuration.final) != (before.initial, before.final)
        always_valid &= (after_configuration.duration_g, after_configuration.duration_y, after_configuration.duration_b) == (before.duration_g, before.duration_y, before.duration_b)
        always_valid &= relations_respected(after_configuration.initial, map_data) and relations_respected(after_configuration.final, map_data)

        current = scenario_mutation.mutate_green_light_duration(config, current, MAP)
        after_green = tc.TrafficControl.from_commonroad(current.lanelet_network, dt)
        durations_changed += after_green.duration_g != after_configuration.duration_g
        always_valid &= (after_green.initial, after_green.final) == (after_configuration.initial, after_configuration.final)
        always_valid &= (after_green.duration_y, after_green.duration_b) == (3, 2) and tc.MIN_DURATION_GREEN <= after_green.duration_g <= max_green
check("mutations (4 scenarios x 25 rounds): lights always decode; configurations keep the durations, green keeps the "
      "configurations; yellow 3 s, all red 2 s, green 5-15 s; relations respected", always_valid)
check(f"mutations change something (configurations {configurations_changed}/100, green {durations_changed}/100)",
      configurations_changed > 50 and durations_changed > 50)

names = {TrafficLight.GREEN: "GREEN", TrafficLight.YELLOW: "YELLOW", TrafficLight.RED: "RED"}
final = tc.TrafficControl.from_commonroad(current.lanelet_network, dt)
manager = TrafficControlManager(TCSection(**final.to_doppeltest(map_data)))
lights = {light.traffic_light_id: light for light in current.lanelet_network.traffic_lights}
mismatches = sum({map_data.traffic_light_id(l.id): names[l.color] for l in manager.get_traffic_configuration(k / 10).traffic_light}
                 != {i: tc.COLOR_OF_STATE[light.get_state_at_time_step(k)] for i, light in lights.items()} for k in range(301))
check("after mutations, the scenario's lights equal DoppelTest's traffic manager at every 0.1 s tick", mismatches == 0)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
