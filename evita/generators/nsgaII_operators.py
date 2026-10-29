import os
import random
import logging
import contextlib
import enlighten
from copy import deepcopy

import time
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.termination import Termination

from evita.config import ValidationConfig, ScenarioGeneratorConfig
from evita.utils import Budget, EVITA_MAIN_LOGGER_NAME
from evita.generators import scenario_mutation
from evita.validator import ScenarioValidator, EVITAScenarioValidator
from evita.generators import utils as scenario_generation_utils

import matplotlib.pyplot as plt
from commonroad.planning.planning_problem import PlanningProblemSet
from conversion_scripts.utils.conversion_utils import generate_traffic_lights

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

class ScenarioSampling(Sampling):
    """
    Sampling operator to generate random scenarios and planning problem sets
    """
    def __init__(self, base_scenario, base_planning_problem_set, scenario_generator_configuration, termination_criterion):
        """
        Initialize the ScenarioSampling operator with:
        - Scenario
        - Planning problem set
        - Scenario generator configuration
        - Scenario validator configuration
        - Scenario mutation functions
        """
        super().__init__()
        self.base_scenario = base_scenario
        self.base_planning_problem_set = base_planning_problem_set
        self.scenario_generator_configuration = scenario_generator_configuration
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)
        self.add_random_planning_problem = scenario_mutation.add_random_planning_problem

        self.scenario_counter = 0

        self.termination_criterion = termination_criterion
        self.logger = module_logger

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate `n_samples` random scenarios and planning problem sets
        """
        X = np.full((n_samples, 2), None, dtype = object)

        generation_bar = self.termination_criterion.progress_bar_manager.counter(total = n_samples, desc = "Scenario Generation", unit = "scenarios", color = "green", leave = False)

        self.logger.info(" ====== Generating the Initial Population ====== ")

        with self.termination_criterion.generating():
            for i in range(n_samples):
                X[i, 0], X[i, 1] = self._generate_random_individual()
    
                generation_bar.update()
                self.termination_criterion.update_main_progress_bar()
 
        generation_bar.close(clear = True)
            
        return X

    def _is_valid(self, scenario, planning_problem_set) -> bool:
        """
        Check if the given scenario and planning problem set are valid using the scenario validator.
        """
        with self.termination_criterion.paused():
            return self.scenario_validator.is_scenario_valid(scenario, planning_problem_set)[0]

    def _generate_random_individual(self):
        """
        First set a target number of AVs
        Then try as much as possible to generate a valid scenario with them
        If this fails, decrease the target number of AV and repeat
        """
        # Randomly choose the number of AVs (planning problems) to generate
        target_number_of_planning_problems = random.randint(self.scenario_generator_configuration.min_amount_pp, self.scenario_generator_configuration.max_amount_pp)

        # Loop until a valid scenario is generated or the target number of planning problems is below the minimum allowed
        while target_number_of_planning_problems >= self.scenario_generator_configuration.min_amount_pp:
            self.logger.debug(f"Try to generate a scenario with {target_number_of_planning_problems}")
            
            # Set a loop to manage generation retries in case of failure
            for _ in range(self.scenario_validator_configuration.retry_generation):
                try:
                    # Copy base scenario and planning problem
                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)

                    # Generate a random planning problem for the target number of AVs
                    for _ in range(target_number_of_planning_problems):
                        _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)
                    
                    # Check if scenario is valid with the generated planning problems
                    if self._is_valid(_scenario, _planning_problem_set):
                        self.scenario_counter += 1
                        
                        self.logger.debug(f"Scenario {self.scenario_counter} generated with {target_number_of_planning_problems}")
                        setattr(_scenario, "debug_id", self.scenario_counter)

                        return _scenario, _planning_problem_set
                except Exception as exc:
                    self.logger.debug(f"An exception occurred while generating the scenario. {exc}")
                    pass

            # Update the target number of planning problems to generate
            target_number_of_planning_problems = target_number_of_planning_problems - 1

        raise Exception("Cannot generate any scenario!")

class EVITAScenarioSampling(ScenarioSampling):
    """
    Sampling operator to generate random scenarios and planning problem sets for EVITA
    """
    def __init__(self, base_scenario, base_planning_problem_set, scenario_generator_configuration, map_file_path):
        # Use the parent ScenarioSampling class to initialize the common attributes
        super().__init__(base_scenario, base_planning_problem_set, scenario_generator_configuration)
        # Update the scenario validator to the EVITA one
        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)
        # Update the add_random_planning_problem function to the EVITA one
        self.add_random_planning_problem = scenario_mutation.add_evita_random_planning_problem

        # Save the map file path
        self.map_file_path = map_file_path

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate `n_samples` random scenarios and planning problem sets for EVITA
        """
        # Generate the traffic lights configuration for the scenario
        self.base_scenario = generate_traffic_lights(self.base_scenario, self.scenario_generator_configuration.max_time_step, self.map_file_path)

        X = np.full((n_samples, 2), None, dtype = object)

        print("\n")

        # Generate `n_samples` scenarios and planning problems
        for i in range(n_samples):
            individual = self._generate_random_individual()
            X[i, 0] = individual[0]
            X[i, 1] = individual[1]

        return X

    def _generate_random_individual(self):
        """
        First set a target number of AVs. 
        Then try as much as possible to generate a valid scenario with them. 
        If this fails, decrease the target number of AV and repeat.
        """

        # Randomly choose the number of AVs (planning problems) to generate
        target_number_of_planning_problems = random.randint(self.scenario_generator_configuration.min_amount_pp, self.scenario_generator_configuration.max_amount_pp)

        # Loop until a valid scenario is generated or the target number of planning problems is below the minimum allowed
        while target_number_of_planning_problems >= self.scenario_generator_configuration.min_amount_pp:
            print(f"Try to generate a scenario with {target_number_of_planning_problems}")
            
            # Set a loop to manage generation retries in case of failure
            for _ in range(0, self.scenario_validator_configuration.retry_generation):
                try:
                    # Copy base scenario and planning problem
                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)

                    # Generate `target_number_of_planning_problems` planning problems for the scenario
                    for _ in range(0, target_number_of_planning_problems):
                        _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set, self.map_file_path)
                    
                    # Check if the generated scenario and planning problems are valid
                    if self.scenario_validator.is_scenario_valid(_scenario, _planning_problem_set)[0]:
                        print(f"\nScenario {self.scenario_counter} generated with {target_number_of_planning_problems}")
                        setattr(_scenario, "debug_id", self.scenario_counter)

                        # Print the main characteristics of the generated planning problems
                        for pp in _planning_problem_set._planning_problem_dict.values():
                            print(f"Planning Problem: {str(pp.planning_problem_id):<10} "
                                  f"Initial Lane: {str(_scenario.lanelet_network.find_lanelet_by_position([pp.initial_state.position])):<20} "
                                  f"Goal Lane: {str(_scenario.lanelet_network.find_lanelet_by_position([pp.goal.state_list[-1].position.center])):<20} "
                                  f"Route: {pp.path}")

                        print(f"\n{'=' * 150}\n")

                        self.scenario_counter += 1

                        return _scenario, _planning_problem_set
                except Exception as exc:
                    print(f"An exception occurred while generating the scenario. {exc}")
                    pass

class ScenarioMutation(Mutation):
    """
    Mutation operator to mutate scenarios and planning problem sets
    """
    def __init__(self, map_file_path: str, scenario_generator_configuration, termination_criterion):
        """
         Initialize the ScenarioMutation operator with:
        - Scenario generator configuration
        - Scenario validator configuration
        - Scenario mutation functions and their probabilities of application
        """
        super().__init__()
        self.scenario_generator_configuration = scenario_generator_configuration

        self.planning_problem_mutation_operators = [scenario_mutation.mutate_initial_position, scenario_mutation.mutate_initial_velocity, scenario_mutation.mutate_goal_area_position]

        self.add_random_planning_problem = scenario_mutation.add_random_planning_problem
        self.add_random_planning_problem_prob = 0.3
        
        self.remove_random_planning_problem = scenario_mutation.remove_random_planning_problem
        self.remove_random_planning_problem_prob = 0.3
        
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)

        self.map_file_path = map_file_path

        self.termination_criterion = termination_criterion
        self.logger = module_logger

    def _do(self, problem, X, **kwargs):
        """
         Mutate the scenarios and planning problem sets in X by applying the mutation operators with a certain probability
         The mutation operators are applied to each planning problem of the scenario and to the scenario itself
         After mutation, the validity of the scenario is checked and the mutated scenario is kept only if it's valid
         Otherwise, the original scenario is kept
        """
        with self.termination_criterion.generating():
            # Iterate over each individual (scenario, planning problem set)
            for i in range(len(X)):
                # Copy the scenario and planning problem set to mutate them
                _scenario = deepcopy(X[i, 0])
                _planning_problem_set = deepcopy(X[i, 1])

                self.logger.debug(f"{'-' * 25} Mutation of scenario {_scenario.generation_id} {'-' * 25}")

                # Does the scenario have the is_valid flag?
                # If so, delete it
                if hasattr(_scenario, "is_valid"):
                    delattr(_scenario, "is_valid")

                # Try to mutate the existing planning problems
                for key, planning_problem in _planning_problem_set.planning_problem_dict.items():
                    mutated_planning_problem = planning_problem

                    # Iterate over each planning problem mutation operator
                    for planning_problem_mutation_operator in self.planning_problem_mutation_operators:
                        # Apply the mutation operator with a probability of 1 / number of mutation operators
                        if random.random() <= 1 / len(self.planning_problem_mutation_operators):
                            print(f"Mutating Planning Problem {mutated_planning_problem.planning_problem_id} with {planning_problem_mutation_operator.__name__}")
                            mutated_planning_problem = planning_problem_mutation_operator(self.scenario_generator_configuration, mutated_planning_problem, _scenario)

                    # Save mutated planning problem
                    _planning_problem_set.planning_problem_dict[key] = mutated_planning_problem
                
                # Try to mutate the scenario's traffic lights if the scenario is urban
                if hasattr(self, "traffic_lights_mutation_operators"):
                    mutated_scenario = _scenario
                    # Iterate over each traffic light mutation operator
                    for traffic_light_mutation_operator in self.traffic_lights_mutation_operators:
                        if random.random() <= 1 / len(self.traffic_lights_mutation_operators):
                            # Apply the mutation operator with a probability of 1 / number of mutation operators
                            self.logger.debug(f"Mutating scenario {getattr(mutated_scenario, 'generation_id', '?')} with {traffic_light_mutation_operator.__name__}")
                            mutated_scenario = traffic_light_mutation_operator(self.scenario_generator_configuration, _scenario, self.map_file_path)

                    _scenario = mutated_scenario
                    
                # Possibly remove an existing planning problem - note there's the risk of removing the one we just added
                # - If we haven't reached the minimum amount of planning problems yet
                # - With a certain probability to avoid always removing a planning problem at each mutation
                if len(_planning_problem_set.planning_problem_dict) > self.scenario_generator_configuration.min_amount_pp and \
                        random.random() <= self.remove_random_planning_problem_prob:
                    self.logger.debug(f"Mutating scenario with {self.remove_random_planning_problem.__name__}")
                    _planning_problem_set = self.remove_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)

                # Possibly add a new planning problem:
                # - If we haven't reached the maximum amount of planning problems yet
                # - With a certain probability to avoid always adding a planning problem at each mutation
                if len(_planning_problem_set.planning_problem_dict) < self.scenario_generator_configuration.max_amount_pp and \
                        random.random() <= self.add_random_planning_problem_prob:
                    self.logger.debug(f"Mutating scenario with {self.add_random_planning_problem.__name__}")
                    _planning_problem_set = self.add_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)

                # Check mutated scenario validity
                with self.termination_criterion.paused():
                    self.scenario_validator.validate(_scenario, _planning_problem_set)

                if _scenario.is_valid:
                    X[i, 0] = _scenario
                    X[i, 1] = _planning_problem_set
                else:
                    self.logger.debug(f"Mutated scenario is not valid: {getattr(_scenario, 'validation_message', '')}")
        return X

class EVITAScenarioMutation(ScenarioMutation):
    """
    Mutation operator to mutate scenarios and planning problem sets for EVITA
    """
    def __init__(self, map_file_path: str):
        # Use the parent ScenarioMutation class to initialize the common attributes and mutation operators
        super().__init__(map_file_path)

        # Update the mutation operators to the EVITA ones
        self.traffic_lights_mutation_operators = [scenario_mutation.mutate_tl_initial_configuration,
                                                  scenario_mutation.mutate_green_light_duration]
        
        self.planning_problem_mutation_operators.remove(scenario_mutation.mutate_goal_area_position)
        self.planning_problem_mutation_operators.append(scenario_mutation.mutate_evita_goal_area_position)
        
        
        self.add_random_planning_problem = scenario_mutation.add_evita_random_planning_problem
        
        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)

class ScenarioCrossover(Crossover):
    """
    Crossover operator to crossover scenarios and planning problem sets
    """
    def __init__(self, scenario_generator_configuration, termination_criterion):
        # Define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)

        self.scenario_generator_configuration = scenario_generator_configuration

        self.termination_criterion = termination_criterion
        self.logger = module_logger

    def _do(self, problem, X, **kwargs):
        # The input has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        with self.termination_criterion.generating():
            for k in range(n_matings):
                parent_1_scenarios, parent_2_scenarios = X[0, k, 0], X[1, k, 0]
                parent_1_pps, parent_2_pps = X[0, k, 1], X[1, k, 1]

                # Get the planning problems from both parents and find the shortest length
                parent_1_planning_problems = list(parent_1_pps.planning_problem_dict.values())
                parent_2_planning_problems = list(parent_2_pps.planning_problem_dict.values())
                shortest = min(len(parent_1_planning_problems), len(parent_2_planning_problems))

                # If the shortest length is at least 2, perform crossover by selecting a random cut index and combining the planning problems from both parents
                if shortest >= 2:
                    cut_index = random.randint(1, shortest - 1)

                    # Create the offspring planning problem sets by combining the planning problems from both parents
                    offspring_1_planning_problems = parent_1_planning_problems[:cut_index] + parent_2_planning_problems[cut_index:]
                    offspring_2_planning_problems = parent_2_planning_problems[:cut_index] + parent_1_planning_problems[cut_index:]
                
                    # Renumber the planning problems in the offspring to have unique IDs starting from the base ID
                    base_id = self.scenario_generator_configuration.av_base_id
                    offspring_1_planning_problems = self._renumber_planning_problems(offspring_1_planning_problems, base_id)
                    offspring_2_planning_problems = self._renumber_planning_problems(offspring_2_planning_problems, base_id)

                    # Create new PlanningProblemSet objects for the offspring planning problems
                    offspring_1_scenarios = deepcopy(parent_1_scenarios)
                    offspring_2_scenarios = deepcopy(parent_2_scenarios)
                    planningProblemSet_offspring_1 = PlanningProblemSet(offspring_1_planning_problems)
                    planningProblemSet_offspring_2 = PlanningProblemSet(offspring_2_planning_problems)

                    # Validate the offspring scenarios and planning problem sets
                    with self.termination_criterion.paused():
                        self.scenario_validator.validate_all([(offspring_1_scenarios, planningProblemSet_offspring_1), (offspring_2_scenarios, planningProblemSet_offspring_2)])

                    # If both offspring are valid, assign them to the output array Y
                    if offspring_1_scenarios.is_valid and offspring_2_scenarios.is_valid:
                        Y[0, k, 0] = offspring_1_scenarios
                        Y[1, k, 0] = offspring_2_scenarios
                        
                        Y[0, k, 1] = planningProblemSet_offspring_1
                        Y[1, k, 1] = planningProblemSet_offspring_2
                        continue

                # Otherwise (or if shortest < 2): return the parents
                Y[0, k, 0] = parent_1_scenarios
                Y[1, k, 0] = parent_2_scenarios
                
                Y[0, k, 1] = parent_1_pps
                Y[1, k, 1] = parent_2_pps
                
        return Y
    
    def _renumber_planning_problems(self, planning_problems, base_id: int) -> list:
        """
        Renumber the planning problems in the list to have unique IDs starting from base_id + 1
        """
        renumbered = []

        for index, planning_problem in enumerate(planning_problems):
            planning_problem_copy = deepcopy(planning_problem)
            planning_problem_copy._planning_problem_id = base_id + 1 + index
            renumbered.append(planning_problem_copy)
        
        return renumbered

class EVITAScenarioCrossover(ScenarioCrossover):
    """
    Crossover operator to crossover scenarios and planning problem sets for EVITA
    """
    def __init__(self):
        # Use the parent ScenarioCrossover class to initialize the common attributes and crossover logic
        super().__init__()
        # Update the scenario validator to the EVITA one
        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)

class ScenarioDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0] and a.X[1] == b.X[1]

# Generator to list all the simulation folders in this run
from ..utils import simulation_folders

from ..data_analysis import generate_report
import os
import csv

'''
def checkpoint(checkpoint_id: int, output_folder: str, all_scenarios:int, valid_scenarios:int, invalid_scenarios:int, total_generation_time:int, apollo: bool) -> None:
    """
    Link all the scenario folders generated so far + compute the results for the generation (aka csv files) to the checkpoint folder.
    """
    # Build the checkpoint folder path and create it
    checkpoint_folder = os.path.join(output_folder, "checkpoints", f"{str(checkpoint_id).rjust(4,'0')}").replace("/records/", "/evita-records/")
    os.makedirs(checkpoint_folder, exist_ok=False)
    
    print(f"Creating checkpoint at {checkpoint_folder}")

    # Create a scenarios.csv file containining the id of all the scenarios generated so far
    scenario_file = os.path.join(checkpoint_folder, "scenarios.csv")
    with open(scenario_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        # Write the header of the csv file depending on whether we are in Apollo mode or not
        if not apollo:
            csv_writer.writerow(["generation_id", "scenario_id", "scenario_file", "planned_trajectories_file"])
        else:
            csv_writer.writerow(["scenario_id", "scenario_file", "planned_trajectories_file"])
        
        # Iterate over all the simulation folders and write their information to the csv file
        for _, simulation_folder in simulation_folders(output_folder):
            if apollo:
                scenarios_folders = os.listdir(os.path.join(output_folder, simulation_folder))
                for scenario_id in scenarios_folders:
                    scenario_file = os.path.join(output_folder, simulation_folder, scenario_id, "scenario.xml").replace("/records/", "/evita-records/")
                    planned_trajectories_file = os.path.join(output_folder, simulation_folder, scenario_id, "planned_trajectories.json").replace("/records/", "/evita-records/")
                    csv_writer.writerow([str(simulation_folder), str(scenario_id), str(scenario_file), str(planned_trajectories_file)])
            else:
                scenario_id = simulation_folder
                simulation_db = os.path.join(output_folder, simulation_folder, "logs", "simulation.db")
                scenario_file = os.path.join(output_folder, simulation_folder, "scenario", f"scenario_{simulation_folder}.xml")
                csv_writer.writerow([str(scenario_id), str(scenario_file), str(simulation_db)])

    performance_file = os.path.join(checkpoint_folder, "performance.csv")
    with open(performance_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    # Generate the usual report in the checkpoint folder using whatever data is available at this point
    if not apollo:
        generate_report(output_folder, checkpoint_folder)
'''

class CustomTimeTermination(Termination):
    """
    Termination criterion based on a maximum time budget for the scenario generation
    It also manages the checkpointing of the generation process by creating a checkpoint every `checkpoint_every` minutes.
    """
    def __init__(self, max_time_minutes: int, checkpoint_every: int, progress_bar_manager = None) -> None:
        super().__init__()

        self.budget = Budget(max_time_minutes * 60)
 
        # Progress bars: one manager shared by all the bars, and the main bar (created by start())
        if progress_bar_manager is not None:
            self.progress_bar_manager = progress_bar_manager
        else:
            self.progress_bar_manager = enlighten.get_manager()

        self.main_progress_bar = None
 
        # Performance counters
        self.all_scenarios = 0
        self.valid_scenarios = 0
        self.invalid_scenarios = 0
        self.total_generation_time = 0.0
 
        # Checkpoints (in seconds of elapsed budget time)
        self.checkpoint_every_seconds = checkpoint_every * 60
        self.checkpoints = set()
 
        # Bookkeeping for nested pauses and for excluding paused time from the generation time
        self._pause_depth = 0
        self._pause_started_at = None
        self._paused_total = 0.0
 
        # Kept for backward compatibility with existing callers
        self.problem_ref = None

    # ========================
    # Time budget
    # ========================
 
    def is_started(self) -> bool:
        return self.budget.get_start_time() is not None
 
    def start(self) -> None:
        """
        Start the time budget and show the main progress bar.
        """
        if self.is_started():
            return

        self.budget.start()
        self.main_progress_bar = self.progress_bar_manager.counter(total = int(self.budget.get_time_budget()), desc = "Time Budget", unit = "seconds", color = "red", leave = False)
        self.update_main_progress_bar()
 
    def is_over(self) -> bool:
        return self.is_started() and self.budget.is_over()
 
    def remaining_time(self) -> float:
        if self.is_started():
            return self.budget.get_remaining_real_time()
        else: 
            return self.budget.get_time_budget()

    @contextlib.contextmanager
    def paused(self):
        """
        Exclude the enclosed block from the time budget
        """
        # If this is the outermost pause, record the start time and pause the budget
        outermost = (self._pause_depth == 0)
        
        if outermost:
            self._pause_started_at = time.perf_counter()
            if self.is_started():
                self.budget.pause()

        self._pause_depth += 1

        # Yield control to the enclosed block, and then resume the budget and update the paused time when exiting the block
        try:
            yield
        finally:
            self._pause_depth -= 1

            # If this is the outermost pause, resume the budget and update the paused time
            if outermost:
                if self.is_started():
                    self.budget.resume()
                
                self._paused_total += time.perf_counter() - self._pause_started_at
                self._pause_started_at = None
 
    @property
    def max_time_seconds(self) -> float:
        return self.budget.get_time_budget()
 
    @property
    def elapsed_time_seconds(self) -> float:
        return self.budget.get_elapsed() if self.is_started() else 0.0

    # ========================
    # Monitoring
    # ========================
 
    @contextlib.contextmanager
    def generating(self):
        """
        Account the enclosed block as scenario generation time
        Time spent in paused blocks (e.g. validation) inside it is not counted.
        """
        wall_start, paused_start = time.perf_counter(), self._paused_total

        # Yield control to the enclosed block, and then update the total generation time when exiting the block
        try:
            yield
        finally:
            self.total_generation_time += (time.perf_counter() - wall_start) - (self._paused_total - paused_start)
            self.update_main_progress_bar()
 
    def count(self, scenario) -> None:
        """
        Count a processed scenario
        scenarios skipped because the budget was over are counted as valid
        """
        self.all_scenarios += 1

        if getattr(scenario, "is_valid", False) or getattr(scenario, "validation_message", None) == "Timeout is over":
            self.valid_scenarios += 1
        else:
            self.invalid_scenarios += 1
 
    # ========================
    # Progress bar
    # ========================
 
    def update_main_progress_bar(self) -> None:
        """
        Advance the main bar to the current elapsed (non-paused) budget time
        """
        if self.main_progress_bar is None:
            return

        target = min(self.budget.get_elapsed(), self.main_progress_bar.total)
        delta = target - self.main_progress_bar.count
        
        if delta > 0:
            self.main_progress_bar.update(delta)

    # ========================
    # pymoo interface
    # ========================
 
    def _update(self, algorithm) -> float:
        """
        Called by pymoo after each generation. Returns the fraction of the budget used: pymoo stops at 1.0
        """
        # Safety net, in case start() was not called before algorithm.run()
        self.start()
 
        self.update_main_progress_bar()
        self._checkpoint_if_due(algorithm)
 
        used = self.budget.get_time_budget() - self.budget.get_remaining_real_time()
        return max(0.0, used / self.budget.get_time_budget())

    def _checkpoint_if_due(self, algorithm) -> None:
        """
        Check if a checkpoint is due, and if so, create it
        """
        if not self.checkpoint_every_seconds or self.checkpoint_every_seconds <= 0:
            return

        checkpoint_id = int(self.budget.get_elapsed() / self.checkpoint_every_seconds)

        if checkpoint_id > 0 and checkpoint_id not in self.checkpoints:
            from evita.orchestrator import checkpoint
 
            with self.paused():
                try:
                    checkpoint(checkpoint_id, algorithm.problem.output_folder, self.all_scenarios, self.valid_scenarios, self.invalid_scenarios, self.total_generation_time)
                    self.checkpoints.add(checkpoint_id)
                except Exception as e:
                    module_logger.critical(f"Failed to checkpoint execution at checkpoint {checkpoint_id}")
                    raise e
 