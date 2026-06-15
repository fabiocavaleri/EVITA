import logging
import random

from copy import deepcopy

from typing import Optional, List, Dict

import os
from ..orchestrator import debug_plot_to_file
from ..interfaces import AbstractScenarioGenerator

from . import utils as scenario_generation_utils
from . import scenario_mutation 
from ..utils import EVITA_MAIN_LOGGER_NAME

from ..validator import ValidationConfig, ScenarioValidator

class RandomScenarioGenerator(AbstractScenarioGenerator):
    def __init__(self, commonroad_scenario_seed_file: Optional[str]=None, **kwargs) -> None:
        """
        Generate random scenarios by adding random planning problems to a given seed scenario
        The seed scenario is expected to contain only the lanelet network, and no obstacles or planning problems
        The generator will then add random planning problems to the scenario, and validate the generated scenario using the scenario validator
        If the generated scenario is not valid, it will try again until it reaches the maximum number of retries
        If it cannot generate a valid scenario with the target number of planning problems, it will decrease the target number of planning problems and try again
        """
        super(RandomScenarioGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        assert commonroad_scenario_seed_file, "Random Scenario Generation requires a Seed File!"

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME+ "." + __name__ + "." + self.__class__.__name__)

        # Load the scenario object from the seed file, and leave only the lanelet network
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles=True, remove_planning_problems=True)
        
        # Instantiate the scenario generator configuration
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
        # Generate the initial individual
        scenario, planning_problem_set = self._generate_random_individual()

        return [(scenario, planning_problem_set)]

    def _generate_random_individual(self):
        """
        First set a target number of AVs
        Then try as much as possible to generate a valid scenario with them
        If this fails, decrease the target number of AV and repeat
        """
        # Randomly set a target number of planning problems to generate
        target_number_of_planning_problems = random.randint(self.scenario_generator_configuration.min_amount_pp, self.scenario_generator_configuration.max_amount_pp)

        # Loop until we can generate a valid scenario with the target number of planning problems, or until we reach the minimum number of planning problems
        while target_number_of_planning_problems > self.scenario_generator_configuration.min_amount_pp:
            self.logger.debug(f"Try to generate a scenario with {target_number_of_planning_problems}")
            # Try to generate a valid scenario with the target number of planning problems, retrying if necessary
            for _ in range(0, self.scenario_validator_configuration.retry_generation):
                try:
                    # Copy the base scenario and planning problem set
                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)            

                    # Add random planning problems to the scenario until we reach the target number of planning problems
                    for _ in range(0, target_number_of_planning_problems):
                        _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)
                    
                    # Validate the generated scenario
                    scenario_tuples = [(_scenario, _planning_problem_set)]
                    self.scenario_validator.validate_all(scenario_tuples)
                    
                    # Plot the scenario if debug is enabled
                    if self.debug:
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
