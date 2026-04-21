# Implementation of basic search algorithms, including EAOnePlusOne

import logging
from typing import Optional, List

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

# from .utils import get_basic_scenario_from_file, parse_scenario_generator_configuration_from_kwargs
from . import utils as scenario_generation_utils
from .NSGAIIproblem import NSGAIIproblem
from .nsgaII_operators import ScenarioSampling, ScenarioMutation, ScenarioCrossover
from ..config import ScenarioGeneratorConfig
from ..interfaces import AbstractScenarioGenerator
from ..utils import EVITA_MAIN_LOGGER_NAME


# TODO Add a SEED!
# TODO Add a logger!
class NSGAIIGenerator(AbstractScenarioGenerator):

    def __init__(self, population, generations, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(NSGAIIGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
        self.population = population
        self.generations = generations

        # Load the scenario object from the seed file, and leave only the lanelet network
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(
            commonroad_scenario_seed_file, remove_obstacles=True, remove_planning_problems=True)

        self.scenario_generator_configuration: ScenarioGeneratorConfig
        self.scenario_generator_configuration = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs(
            **kwargs)

    def generate(self) -> List:
        termination = get_termination("n_gen", self.generations)
        problem = NSGAIIproblem()
        sampling = ScenarioSampling(self.base_scenario, self.base_planning_problem_set,
                                    self.scenario_generator_configuration)

        mutation = ScenarioMutation()
        crossover = ScenarioCrossover(),
        algorithm = NSGA2(pop_size=self.population, offspring_population_size=self.population, sampling=sampling,
                          mutation=mutation,
                          crossover=crossover
                          )
        algorithm.setup(problem, termination=termination, save_history=True, verbose=True)
        algorithm.run()
        result = algorithm.result()

