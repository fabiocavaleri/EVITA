"""
Tests of step 3, parts A and B: the simulator switch (evita/simulators.py, --simulator in both entry points) and the
executors' preparation hook (prepare). With --simulator frenetix (the default) everything must be as before.
Run from the project root: python3 tests/test_step3_simulators.py
"""
import os, sys, logging, tempfile
sys.path.insert(0, os.getcwd())
logging.basicConfig(level = logging.ERROR)

from evita import simulators
from evita.config import ScenarioGeneratorConfig, ValidationConfig
from evita.interfaces import AbstractScenarioExecutor
from evita.executors.frenetix_executor import FrenetixScenarioExecution
from evita.utils import LANELET_EXTENSION, extend_scenario_after_goal_areas
from evita.generators.utils import get_base_scenario_from_file
from EVITA_DoppelTest.config import MAX_ADC_COUNT, SCENARIO_UPPER_LIMIT

MAP = "EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml"
results = []
def check(name, ok):
    results.append(bool(ok)); print(("PASS " if ok else "FAIL ") + name)

# ---------- A. Registry ----------
frenetix, apollo = simulators.get_simulator("frenetix"), simulators.get_simulator("apollo")
def settings(config):
    """A configuration's data attributes (bound methods never compare equal)"""
    return {k: v for k, v in vars(config).items() if not callable(v)}
default, from_registry = settings(ScenarioGeneratorConfig()), settings(simulators.generation_configuration(frenetix))
check("FrenetiX: generation configuration identical to ScenarioGeneratorConfig() (every attribute)", default == from_registry)
config = simulators.generation_configuration(apollo)
import yaml
dt = simulators.frenetix_time_step()
check(f"FrenetiX's time step read from its planning configuration ({dt} s)",
      dt == float(yaml.safe_load(open(simulators.FRENETIX_PLANNING_CONFIGURATION))["dt"]) and dt > 0)
check(f"Apollo: DoppelTest's limits ({MAX_ADC_COUNT} vehicles, {SCENARIO_UPPER_LIMIT} s / {dt} s = {config.max_time_step} steps), vehicle model 101 "
      f"({config.length} x {config.width} m)", (config.max_amount_pp, config.max_time_step, config.cr_vehicle_id, config.length, config.width)
      == (MAX_ADC_COUNT, int(round(SCENARIO_UPPER_LIMIT / dt)), 101, 4.933, 2.11))
others = {k: v for k, v in settings(config).items() if k not in ("max_amount_pp", "max_time_step", "cr_vehicle_id", "length", "width")}
check("Apollo: every other setting as FrenetiX's", all(default[k] == v for k, v in others.items() if k in default))
validator_f = simulators.scenario_validator(frenetix, ValidationConfig(), from_registry)
validator_a = simulators.scenario_validator(apollo, ValidationConfig(), config)
check("validator: Apollo mode only for Apollo", (validator_f.apollo, validator_a.apollo) == (False, True))
try:
    simulators.get_simulator("carla"); check("unknown simulator refused", False)
except ValueError:
    check("unknown simulator refused", True)

# ---------- B. Preparation ----------
scenario, _ = get_base_scenario_from_file(MAP, remove_obstacles = True, remove_planning_problems = True)
executor = FrenetixScenarioExecution(tempfile.mkdtemp(), driving_agent_name = "frenetix", map_file = MAP, delete_records = False)
prepared, expected = executor.prepare(scenario), extend_scenario_after_goal_areas(scenario, extend_by_meters = LANELET_EXTENSION)
import numpy as np
def centerlines(s):
    return {l.lanelet_id: l.center_vertices for l in s.lanelet_network.lanelets}
a, b, original = centerlines(prepared), centerlines(expected), centerlines(scenario)
lengthened = [i for i in original if len(a[i]) != len(original[i])]
check(f"FrenetiX prepare = the road extension (same lanelets and centerlines as extend_scenario_after_goal_areas; {len(lengthened)} lanelets lengthened)",
      a.keys() == b.keys() and all(np.array_equal(a[i], b[i]) for i in a) and len(lengthened) > 0)
class Plain(AbstractScenarioExecutor):
    def __init__(self, output_folder, **kwargs): pass
    def execute(self, *args, **kwargs): return None
check("default prepare (e.g. Apollo's): the scenario unchanged", Plain("").prepare(scenario) is scenario)

# ---------- Entry points ----------
from evita.generators import nsgaII_search_no_generator as search
argv = sys.argv
sys.argv = ["x", "--population-size", "2", "--time-budget", "1", "--scenario-seed-file", MAP]
default_args = search.parse_arguments()
sys.argv = ["x", "--population-size", "2", "--time-budget", "1", "--scenario-seed-file", MAP, "--simulator", "apollo", "--delete-records"]
apollo_args = search.parse_arguments()
sys.argv = argv
check("EVITA's search: --simulator frenetix and records kept by default; --simulator apollo --delete-records",
      (default_args.simulator, default_args.delete_records, apollo_args.simulator, apollo_args.delete_records) == ("frenetix", False, "apollo", True))

from evita.generators.NSGAIIproblem import NSGAIIproblem
from evita.generators.nsgaII_operators import CustomTimeTermination
problem = NSGAIIproblem(MAP, "frenetix", 2, CustomTimeTermination(1, 0), 2.5, output_folder_name = os.path.join(tempfile.mkdtemp(), "run"))
problem._load_executor_and_evaluator()
check("NSGAIIproblem by default: FrenetiX's configuration, validator, executor and evaluator",
      settings(problem.scenario_generator_configuration) == default and problem.scenario_validator.apollo is False
      and type(problem.scenario_executor).__name__ == "FrenetixScenarioExecution" and type(problem.scenario_evaluator).__name__ == "FrenetixScenarioEvaluator")
problem_a = NSGAIIproblem(MAP, "apollo", 2, CustomTimeTermination(1, 0), 2.5, output_folder_name = os.path.join(tempfile.mkdtemp(), "run"), simulator = "apollo")
check("NSGAIIproblem with simulator apollo: Apollo's configuration and validator",
      problem_a.scenario_generator_configuration.cr_vehicle_id == 101 and problem_a.scenario_validator.apollo is True and problem_a.simulator.name == "apollo")

from click.testing import CliRunner
from evita.__main__ import cli
result = CliRunner().invoke(cli, ["run", "--scenario-generator-path", "evita/generators", "--scenario-generator-module", "evita.generators.doppeltest_generator",
                                  "--scenario-generator-class", "DoppelTestScenarioGenerator", "--map-file", MAP, "--driving-agent-name", "apollo",
                                  "--time-budget", "1", "--simulator", "apollo"])
check("orchestrator: DoppelTestScenarioGenerator refused with --simulator apollo (pointer to DoppelTest's main_ga.py)",
      result.exit_code != 0 and "main_ga.py" in result.output)
help_text = CliRunner().invoke(cli, ["run", "--help"]).output
check("orchestrator: --simulator and --delete-records options", "--simulator" in help_text and "--delete-records" in help_text)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
