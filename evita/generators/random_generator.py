"""
Random scenario generation (Random baseline): random vehicles on the seed road network, validated before being
returned.
"""
import copy
import logging
import os
import random

from typing import Dict, List, Optional, Tuple

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from . import scenario_mutation
from . import utils as scenario_generation_utils
from ..interfaces import AbstractScenarioGenerator
from ..orchestrator import debug_plot_to_file
from ..utils import EVITA_MAIN_LOGGER_NAME, Budget
from ..validator import ScenarioValidator, ValidationConfig

class RandomScenarioGenerator(AbstractScenarioGenerator):
    """
    Generates each scenario independently: choose a target number of vehicles, place them at random, and validate;
    retry up to retry_generation times, then lower the target by one, down to min_amount_pp.

    Validation uses the same configuration as for the other generators, and does not count against the time
    budget (the budget is paused while validating, as for the other generators).
    """
    def __init__(self, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(RandomScenarioGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        if not commonroad_scenario_seed_file:
            raise ValueError("Random Scenario Generation requires a Seed File!")

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME+ "." + __name__ + "." + self.__class__.__name__)

        # Load the scenario object from the seed file, and leave only the lanelet network
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles = True, remove_planning_problems = True)
        
        # Instantiate the scenario generator configuration
        self.scenario_generator_configuration = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs(**kwargs)

        # Instantiate the scenario validator
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, self.scenario_generator_configuration)

        self.generation = 0
        self.add_random_planning_problem =  scenario_mutation.add_random_planning_problem

        # Debugging: plot every candidate into debug_folder
        self.debug = kwargs.get("debug", False)
        self.debug_folder = kwargs.get("debug_folder")
        self.debug_id = 0

    def generate(self, budget: Optional[Budget] = None, **kwargs) -> Optional[List]:
        """
        One new valid random scenario, or None if none can be generated. If `budget` is given, it is paused while
        candidates are validated.
        """
        try:
            scenario, planning_problem_set = self._generate_random_individual(budget)
        except RuntimeError as exc:
            self.logger.error(f"Cannot generate any scenario! {exc}")
            return None

        return [(scenario, planning_problem_set)]

    def _generate_random_individual(self, budget: Optional[Budget] = None) -> Tuple[Scenario, PlanningProblemSet]:
        """
        Try up to retry_generation random candidates with the target number of vehicles; if none is valid, lower the
        target by one and try again, down to min_amount_pp.
        """
        # Randomly set a target number of planning problems to generate
        config = self.scenario_generator_configuration
        target = random.randint(config.min_amount_pp, config.max_amount_pp)

        # Loop until we can generate a valid scenario with the target number of planning problems, or until we reach the minimum number of planning problems
        while target >= config.min_amount_pp:
            self.logger.info(f"Trying to generate a scenario with {target} vehicles...")

            # Try to generate a valid scenario with the target number of planning problems, retrying if necessary
            for _ in range(self.scenario_validator_configuration.retry_generation):
                try:
                    # Copy the base scenario and planning problem set
                    _scenario = copy.copy(self.base_scenario)
                    _planning_problem_set = copy.copy(self.base_planning_problem_set)            

                    # Add random planning problems to the scenario until we reach the target number of planning problems
                    for _ in range(target):
                        _planning_problem_set = self.add_random_planning_problem(config, _scenario, _planning_problem_set)

                    # Validation does not count against the budget
                    if budget is not None:
                        budget.pause()
                    try:
                        self.scenario_validator.validate(_scenario, _planning_problem_set)
                    finally:
                        if budget is not None:
                            budget.resume()
                    
                    # Plot the scenario if debug is enabled
                    if self.debug and self.debug_folder:
                        debug_plot_file = os.path.join(self.debug_folder, f"plot_{str(self.debug_id).rjust(5, '0')}.png")
                        debug_plot_to_file(_scenario, _planning_problem_set, debug_plot_file)
                    self.debug_id += 1

                    if _scenario.is_valid:
                        # The returned scenario gets its own copy of the road network
                        valid_scenario = copy.deepcopy(self.base_scenario)
                        valid_scenario.is_valid = _scenario.is_valid
                        valid_scenario.validation_message = _scenario.validation_message
                        
                        return valid_scenario, _planning_problem_set
                except Exception as exc:
                    self.logger.error(f"An exception prevented the scenario to be generated. {exc}")

            target -= 1

        raise RuntimeError("Cannot generate any scenario!")

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        """ 
        Random generation is not driven by feedback, so there's nothing to evaluate.
        """
        pass        
