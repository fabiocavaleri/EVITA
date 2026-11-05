"""
(1+1) evolutionary algorithm for scenario generation, and its two configurations used as baselines:
MaxInteractions (number of interactions) and MaxDanger (TIAV: maximum danger).
"""
import logging
import random

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from . import scenario_mutation
from . import utils as scenario_generation_utils
from ..config import ScenarioGeneratorConfig
from ..fitness import MaxDangerFitnessFunction, MaxInteractionsFitnessFunction
from ..interfaces import AbstractFitnessFunction, AbstractScenarioGenerator
from ..utils import EVITA_MAIN_LOGGER_NAME

# Attempts to build an individual with the target number of vehicles before giving up
MAX_INDIVIDUAL_ATTEMPTS = 100

class EAOnePlusOneGenerator(AbstractScenarioGenerator):
    """
    Class that implements a simple EA 1+1 algorithm for scenario generation
    It maintains a single individual (scenario + planning problem set) and mutates it at each generatio
    If the mutated individual has better fitness than the current one, it replaces it. Otherwise, it is discarded.

    This class is not supposed to be used directly from the orchestrator
    """
    
    def __init__(self, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(EAOnePlusOneGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        # Check that the seed file and fitness function are provided
        if not commonroad_scenario_seed_file:
            raise ValueError("EA 1+1 Scenario Generation requires a Seed File!")
        if not isinstance(kwargs.get("fitness"), AbstractFitnessFunction):
            raise ValueError(f"EA 1+1 Scenario Generation requires a fitness function, an instance of {AbstractFitnessFunction.__name__}, but got {type(kwargs.get('fitness'))}")

        # Get the logger
        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)        
        
        # Initialize the fitness function
        self.fitness: AbstractFitnessFunction = kwargs["fitness"]
        self.logger.info(f"Using fitness function {self.fitness}")

        # Load the scenario object from the seed file, and leave only the lanelet network
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles = True, remove_planning_problems = True)
        
        # Initialize the scenario generator configuration
        self.scenario_generator_configuration: ScenarioGeneratorConfig = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs(**kwargs)

        # Initialize the generation counter, the best individual and its fitness value
        self.generation = 0
        self.best_individual: Optional[Tuple[Scenario, PlanningProblemSet]] = None
        # Maximization problem. We start we a very low fitness value
        self.best_fitness_value = float('-inf')

        # Mutations of a vehicle, each applied with probability 1 / (number of operators)
        self.planning_problem_mutation_operators =  [scenario_mutation.mutate_initial_position, scenario_mutation.mutate_initial_velocity, scenario_mutation.mutate_goal_area_position]

        # Define the functions and probabilities for adding and removing planning problems
        self.add_random_planning_problem =  scenario_mutation.add_random_planning_problem
        self.remove_random_planning_problem = scenario_mutation.remove_random_planning_problem

    def generate(self, **kwargs) -> List:
        """
        One new individual: a mutation of the best individual, or a random individual at the start.
        """
        # If we have a best individual, we mutate it to generate a new individual.
        # Otherwise, we generate a new random individual
        if self.best_individual is not None:
            scenario, planning_problem_set = self._mutate(*self.best_individual)
        else: 
            # Generate the initial individual
            scenario, planning_problem_set = self._generate_random_individual()

        # Check that the generated individual is valid in terms of number of planning problems
        assert self.scenario_generator_configuration.min_amount_pp <= len(planning_problem_set.planning_problem_dict) <= self.scenario_generator_configuration.max_amount_pp, "Wrong number of AVs!"

        return [(scenario, planning_problem_set)]

    def _generate_random_individual(self):
        """
        The road network with a random number of vehicles in [min_amount_pp, max_amount_pp].
        """
        # Deepcopy the base scenario and planning problem set
        scenario = deepcopy(self.base_scenario)
        planning_problem_set = deepcopy(self.base_planning_problem_set)

        # Introduce a random number of planning problems
        config = self.scenario_generator_configuration
        target = random.randint(config.min_amount_pp, config.max_amount_pp)

        # Adding a vehicle can fail (no start found): keep adding until the target is reached
        for _ in range(MAX_INDIVIDUAL_ATTEMPTS):
            if len(planning_problem_set.planning_problem_dict) >= target:
                break

            planning_problem_set = self.add_random_planning_problem(config, scenario, planning_problem_set)

        if len(planning_problem_set.planning_problem_dict) < config.min_amount_pp:
            raise RuntimeError(f"Cannot place {config.min_amount_pp} vehicles on the road network")

        return scenario, planning_problem_set

    def _mutate(self, scenario: Scenario, planning_problem_set: PlanningProblemSet) -> Tuple[Scenario, PlanningProblemSet]:
        """
        Copy of the individual with each vehicle's planning problem mutated (each operator applied with probability 1 / number of operators).
        Then possibly one vehicle removed and one added, within [min_amount_pp, max_amount_pp].
        """
        config = self.scenario_generator_configuration

        # Deepcopy the scenario and planning problem set
        _scenario = deepcopy(scenario)
        _planning_problem_set = deepcopy(planning_problem_set)

        # Try to mutate the existing planning problems.       
        for key, planning_problem in list(_planning_problem_set.planning_problem_dict.items()):
            mutated_planning_problem = planning_problem
            # Iterate over the mutation operators and apply them with a certain probability
            for operator in self.planning_problem_mutation_operators:
                if random.random() <= 1 / len(self.planning_problem_mutation_operators):
                    self.logger.debug(f"Mutating {mutated_planning_problem} with {operator.__name__}")
                    mutated_planning_problem = operator(config, mutated_planning_problem, _scenario)
            
            # Update the planning problem set with the mutated planning problem
            _planning_problem_set.planning_problem_dict[key] = mutated_planning_problem
             
        # Possibly remove an existing planning problem
        if len(_planning_problem_set.planning_problem_dict) > config.min_amount_pp and random.random() <= (1 / len(_planning_problem_set.planning_problem_dict.keys())):
             self.logger.debug(f"Mutating {mutated_planning_problem} with {self.remove_random_planning_problem.__name__}")
             _planning_problem_set = self.remove_random_planning_problem(config, _scenario, _planning_problem_set)

        # Possibly add a new planning problem
        if len(_planning_problem_set.planning_problem_dict) < config.max_amount_pp and random.random() <= (1 / len(_planning_problem_set.planning_problem_dict.keys())):
             self.logger.debug(f"Mutating {mutated_planning_problem} with {self.add_random_planning_problem.__name__}")
             _planning_problem_set = self.add_random_planning_problem(config, _scenario, _planning_problem_set)
        
        return _scenario, _planning_problem_set

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        """
        Compute the fitness of the (single) executed individual
        It becomes the best one if it is valid and its fitness is at least as high as the best so far.
        """
        # Extract the scenario, planning problem set, agent data and evaluation data from the simulation data
        simulation_data = list_of_simulation_data[0]
        scenario, planning_problem_set = simulation_data["individual"]

        if not getattr(scenario, "is_valid", False):
            return
        
        # Increment generation count only for valid individuals                
        self.generation += 1

        # Invoke the given fitness function and return a fitness evaluation that can be a value or a list of value depending on the
        # used fitness function. In this case, we expect a single value
        fitness_value = self.fitness.compute(simulation_data["agent-data"], simulation_data["evaluation-data"])
        self.logger.info(f"Current fitness value {fitness_value}")

        # This algorithm assumes Maximization of the fitness function
        if fitness_value >= self.best_fitness_value:
            self.best_individual = scenario, planning_problem_set
            self.best_fitness_value = fitness_value
            self.logger.info(f"Found a better individual at generation {self.generation} - Fitness value {fitness_value}")
       
class EAOnePlusOneGenerator_MaxInteractions(AbstractScenarioGenerator):
    """
    (1+1) evolutionary algorithm maximizing the number of interactions.
    """

    def __init__(self, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(EAOnePlusOneGenerator_MaxInteractions, self).__init__(commonroad_scenario_seed_file, **kwargs)
        self.wrapped_generator = EAOnePlusOneGenerator(commonroad_scenario_seed_file = commonroad_scenario_seed_file, fitness = MaxInteractionsFitnessFunction(), **kwargs)

    def generate(self, **kwargs) -> List:
        return self.wrapped_generator.generate()

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        return self.wrapped_generator.evaluate(list_of_simulation_data)
    
class EAOnePlusOneGenerator_MaxDanger(AbstractScenarioGenerator):
    """
    (1+1) evolutionary algorithm maximizing the danger level.
    """
    def __init__(self, commonroad_scenario_seed_file: Optional[str] = None, **kwargs) -> None:
        super(EAOnePlusOneGenerator_MaxDanger, self).__init__(commonroad_scenario_seed_file, **kwargs)
        self.wrapped_generator = EAOnePlusOneGenerator(commonroad_scenario_seed_file = commonroad_scenario_seed_file, fitness = MaxDangerFitnessFunction(), **kwargs)

    def generate(self, **kwargs) -> List:
        return self.wrapped_generator.generate()

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        return self.wrapped_generator.evaluate(list_of_simulation_data)