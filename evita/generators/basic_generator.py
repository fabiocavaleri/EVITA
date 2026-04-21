# Implementation of basic search algorithms, including EAOnePlusOne

import logging
import random

from typing import Optional, List, Any, Dict, Tuple
from copy import deepcopy

from ..interfaces import AbstractScenarioGenerator, AbstractFitnessFunction
from ..fitness import MaxDangerFitnessFunction, MaxInteractionsFitnessFunction

from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

from cr_scenario_handler.simulation.simulation import Simulation

# from .utils import get_basic_scenario_from_file, parse_scenario_generator_configuration_from_kwargs
from . import utils as scenario_generation_utils
from . import scenario_mutation 
# from .scenario_generator import ScenarioGenerator
from ..utils import EVITA_MAIN_LOGGER_NAME
from ..config import ScenarioGeneratorConfig

# TODO Add a SEED!
class EAOnePlusOneGenerator(AbstractScenarioGenerator):
    """ This calss is not supposed to be used directly from the orchestrator """
    
    def __init__(self, commonroad_scenario_seed_file: Optional[str]=None, **kwargs) -> None:
        super(EAOnePlusOneGenerator, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        assert commonroad_scenario_seed_file, "EA 1+1 Scenario Generation requires a Seed File!"
        
        assert "fitness" in kwargs, "EA 1+1 Scenario Generation requires a Fitness Function!"
        assert issubclass(type(kwargs["fitness"]), AbstractFitnessFunction), \
            f"EA 1+1 Fitness Function an instance of {AbstractFitnessFunction.__name__}, but got {type(kwargs['fitness'])}"

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)        
        
        self.fitness: AbstractFitnessFunction
        self.fitness = kwargs["fitness"]

        self.logger.info(f"Using fitness function {self.fitness}")

        # Load the scenario object from the seed file, and leave only the lanelet network
        # TODO Should we consider the given planning problems here?
        self.base_scenario, self.base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles=True, remove_planning_problems=True)
        
        # TODO Can we include the fitness function here?!
        self.scenario_generator_configuration: ScenarioGeneratorConfig
        self.scenario_generator_configuration = scenario_generation_utils.parse_scenario_generator_configuration_from_kwargs(**kwargs)

        self.generation = 0
        self.best_individual: Tuple[Scenario, PlanningProblemSet] = None
        # Maximization problem. We start we a very low fitness value
        self.best_fitness_value = float('-inf')

        # Container of enabled mutation operators for this generator
        self.planning_problem_mutation_operators =  [scenario_mutation.mutate_initial_position, scenario_mutation.mutate_initial_velocity, scenario_mutation.mutate_goal_area_position]

        self.add_random_planning_problem =  scenario_mutation.add_random_planning_problem
        self.add_random_planning_problem_prob = 0.3
        
        self.remove_random_planning_problem = scenario_mutation.remove_random_planning_problem
        self.remove_random_planning_problem_prob = 0.3

    def generate(self, **kwargs) -> List:
        """
        Return new scenarios (in CommonRoad) format
        """
        if self.best_individual:
            # mutate the best individual
            scenario, planning_problem_set = self._mutate(*self.best_individual)
        else: 
            # generate the initial individual
            scenario, planning_problem_set = self._generate_random_individual()

        # Better safe than sorry
        assert self.scenario_generator_configuration.min_amount_pp <= len(planning_problem_set.planning_problem_dict) <= self.scenario_generator_configuration.max_amount_pp, "Wrong number of AVs!"

        return [(scenario, planning_problem_set)]

    def _generate_random_individual(self):
        _scenario = deepcopy(self.base_scenario)
        _planning_problem_set = deepcopy(self.base_planning_problem_set)

        # introduce a random number of planning problems
        for _ in range(0, random.randint(self.scenario_generator_configuration.min_amount_pp, self.scenario_generator_configuration.max_amount_pp)):
            _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)

        return _scenario, _planning_problem_set

    def _mutate(self, scenario: Scenario, planning_problem_set: PlanningProblemSet) -> Tuple[Scenario, PlanningProblemSet]:
        # With some probability add a planning problem.
        _scenario = deepcopy(scenario)
        _planning_problem_set = deepcopy(planning_problem_set)

        # Try to mutate the existing planning problems.       
        for key, planning_problem in _planning_problem_set.planning_problem_dict.items():

            mutated_planning_problem = planning_problem

            for planning_problem_mutation_operator in self.planning_problem_mutation_operators:
                if random.random() <= 1 / len(self.planning_problem_mutation_operators):
                    print(f"Mutating {mutated_planning_problem} wiht {planning_problem_mutation_operator.__name__}")
                    mutated_planning_problem = planning_problem_mutation_operator(self.scenario_generator_configuration, mutated_planning_problem, _scenario)
            
            _planning_problem_set.planning_problem_dict[key] = mutated_planning_problem

        # Possibly add a new planning problem
        if len(_planning_problem_set.planning_problem_dict) < self.scenario_generator_configuration.max_amount_pp and \
            random.random() <= self.add_random_planning_problem_prob:
             print(f"Mutating {mutated_planning_problem} wiht {self.add_random_planning_problem.__name__}")
             _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)
        # Possibly remove an existing planning problem - note there's the risk of removing the one we just added
        if len(_planning_problem_set.planning_problem_dict) > self.scenario_generator_configuration.min_amount_pp and random.random() and \
              random.random() <= self.remove_random_planning_problem_prob:
             print(f"Mutating {mutated_planning_problem} wiht {self.remove_random_planning_problem.__name__}")
             _planning_problem_set = self.remove_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)

        
        return _scenario, _planning_problem_set

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        """
        This algorithm expects only one individual to evaluate
        """

        simulation_data = list_of_simulation_data[0]
        scenario, planning_problem_set = simulation_data["individual"]
        
        if scenario.is_valid:
            # Increment generation count only for valid individuals                
            self.generation = 1

            # Extract data about agents and metrics
            agent_data = simulation_data["agent-data"]
            evaluation_data = simulation_data["evaluation-data"]
            
            # Invoke the given fitness function and return a fitness evaluation that can be a value or a list of value depending on the
            # used fitness function. In this case, we expect a single value
            fitness_value = self.fitness.compute(agent_data, evaluation_data)

            self.logger.info(f"Current fitness value {fitness_value}")
            # This algorithm assumes Maximization of the fitness function
            if fitness_value >= self.best_fitness_value:
                self.best_individual = scenario, planning_problem_set
                self.best_fitness_value = fitness_value
                self.logger.info(f"Found a better individual at generation {self.generation} - Fitness value {fitness_value}")

        
class EAOnePlusOneGenerator_MaxInteractions(AbstractScenarioGenerator):
    
    def __init__(self, commonroad_scenario_seed_file: Optional[str]=None, **kwargs) -> None:
        super(EAOnePlusOneGenerator_MaxInteractions, self).__init__(commonroad_scenario_seed_file, **kwargs)

        fitness: MaxInteractionsFitnessFunction
        fitness =  MaxInteractionsFitnessFunction()

        self.wrapped_generator: AbstractScenarioGenerator
        self.wrapped_generator = EAOnePlusOneGenerator(commonroad_scenario_seed_file=commonroad_scenario_seed_file, fitness=fitness)

    def generate(self, **kwargs) -> List:
        return self.wrapped_generator.generate()

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        return self.wrapped_generator.evaluate(list_of_simulation_data)
    
class EAOnePlusOneGenerator_MaxDanger(AbstractScenarioGenerator):
    def __init__(self, commonroad_scenario_seed_file: Optional[str]=None, **kwargs) -> None:
        super(EAOnePlusOneGenerator_MaxDanger, self).__init__(commonroad_scenario_seed_file, **kwargs)
        
        fitness: MaxDangerFitnessFunction
        fitness = MaxDangerFitnessFunction()

        self.wrapped_generator: AbstractScenarioGenerator
        self.wrapped_generator = EAOnePlusOneGenerator(commonroad_scenario_seed_file=commonroad_scenario_seed_file, fitness=fitness)

    def generate(self, **kwargs) -> List:
        return self.wrapped_generator.generate()

    def evaluate(self, list_of_simulation_data: List[Dict]) -> None:
        return self.wrapped_generator.evaluate(list_of_simulation_data)