import logging
import random

from copy import deepcopy

from typing import Optional, List, Any, Dict

import os
from ..orchestrator import debug_plot_to_file
from ..interfaces import AbstractScenarioGenerator, AbstractFitnessFunction

from . import utils as scenario_generation_utils
from . import scenario_mutation 
from ..utils import EVITA_MAIN_LOGGER_NAME

from ..validator import ValidationConfig, ScenarioValidator

# TODO Add a SEED!
class RandomScenarioGenerator(AbstractScenarioGenerator):
    
    def __init__(self, commonroad_scenario_seed_file: Optional[str]=None, **kwargs) -> None:
        super(RandomScenarioGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        assert commonroad_scenario_seed_file, "Random Scenario Generation requires a Seed File!"

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME+ "." + __name__ + "." + self.__class__.__name__)

        # Load the scenario object from the seed file, and leave only the lanelet network
        # TODO Should we consider the given planning problems here?
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles=True, remove_planning_problems=True)
        
        self.scenario_generator_configuration = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs(**kwargs)

        # Instantiate the scenario validator
        self.scenario_validator_configuration = ValidationConfig()
        # Make sure that the agent is not too close to the goal area in a separate lane otherwise it will not be able to turn
        # If the speed is not zero!
        self.scenario_validator_configuration.min_dist_goal = 50
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)

        self.generation = 0
        self.add_random_planning_problem =  scenario_mutation.add_random_planning_problem

        # DEBUG
        self.debug = kwargs["debug"] if "debug" in kwargs else False
        self.debug_folder = kwargs["debug_folder"] if "debug_folder" in kwargs else None
        self.debug_id = 0

    def generate(self, **kwargs) -> List:
        """
        Return new scenarios (in CommonRoad) format
        """
        # generate the initial individual
        scenario, planning_problem_set = self._generate_random_individual()
        # scenario, planning_problem_set = self.scenario_generator.generate_random_scenario_and_planning_problem_set()
        # self.logger.debug(f"Generated {len(planning_problem_set.planning_problem_dict)} random planning problems")
        return [(scenario, planning_problem_set)]

    def _generate_random_individual(self):
        """
        First set a target number of AVs. Then try as much as possible to generate a valid scenario with them. If this fails, decrease the target number of AV and repeat.
        # TODO REFACTOR: ENSURE THE SAME GENERATION CODE IS USED BY ALL THE GENERATORS!
        """
        target_number_of_planning_problems = random.randint(self.scenario_generator_configuration.min_amount_pp, self.scenario_generator_configuration.max_amount_pp)

        while target_number_of_planning_problems > self.scenario_generator_configuration.min_amount_pp:
            self.logger.debug(f"Try to generate a scenario with {target_number_of_planning_problems}")
            for _ in range(0, self.scenario_validator_configuration.retry_generation):
                try:
                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)            

                    for _ in range(0, target_number_of_planning_problems):
                        _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)
                    
                    scenario_tuples = [(_scenario, _planning_problem_set)]

                    self.scenario_validator.validate_all(scenario_tuples)
                    
                    if self.debug:
                        # Plot the scenario
                        for scenario, planning_problem_set in scenario_tuples:
                            debug_plot_file = os.path.join(self.debug_folder, f"plot_{(str(self.debug_id).rjust(5, '0'))}.png")
                            debug_plot_to_file(scenario, planning_problem_set, debug_plot_file)

                    self.debug_id = self.debug_id + 1

                    if _scenario.is_valid:
                        return _scenario, _planning_problem_set
                except Exception as exc:
                    self.logger.error(f"An exception prevented the scenario to be generated. {exc}")
                
            target_number_of_planning_problems = target_number_of_planning_problems -1
        
        raise Exception("Cannot generate any scenario!")


    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        """ 
        Random generation is not driven by feedback, so there's nothing to evaluate.
        """
        pass        
