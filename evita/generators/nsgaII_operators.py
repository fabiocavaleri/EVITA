"""
NSGA-II operators of EVITA (paper, Section 4): sampling of the initial population, mutation and crossover of
scenarios, duplicate elimination, and the termination criterion that manages the time budget.

EVITAScenarioSampling and EVITAScenarioMutation adapt the operators to the Apollo pipeline; they are unchanged.
"""
import copy
import time
import random
import logging
import enlighten
import traceback
import numpy as np
import contextlib

from copy import deepcopy

from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.termination import Termination

from commonroad.planning.planning_problem import PlanningProblemSet

from conversion_scripts.utils.conversion_utils import generate_traffic_lights

from evita.config import ValidationConfig
from evita.generators import scenario_mutation
from evita.utils import EVITA_MAIN_LOGGER_NAME, Budget
from evita.timing import TASK_TIMER
from evita.validator import ScenarioValidator

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

# Validation message of scenarios not executed because the budget ran out (as in the orchestrator)
TIMEOUT_MESSAGE = "Timeout is over"

# ========== Sampling ==========

class ScenarioSampling(Sampling):
    """
    Initial population: random scenarios on the seed road network.
    Each has a random number of vehicles in [min_amount_pp, max_amount_pp]
    Candidates are validated (not counted against the budget) and retried up to retry_generation times, then with one vehicle fewer
    """
    def __init__(self, base_scenario, base_planning_problem_set, scenario_generator_configuration, termination_criterion, map_file_path = None):
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
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, scenario_generator_configuration)
        self.add_random_planning_problem = scenario_mutation.add_random_planning_problem

        self.scenario_counter = 0
        self.termination_criterion = termination_criterion
        self.logger = module_logger

        # Save the map file path
        self.map_file_path = map_file_path

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
        A valid random individual: try up to retry_generation candidates with a random target number of vehicles,
        then lower the target by one, down to min_amount_pp.
        """
        config = self.scenario_generator_configuration
        # Randomly choose the number of AVs (planning problems) to generate
        target = random.randint(config.min_amount_pp, config.max_amount_pp)

        # Loop until a valid scenario is generated or the target number of planning problems is below the minimum allowed
        while target >= config.min_amount_pp:
            self.logger.debug(f"Try to generate a scenario with {target} vehicles...")
            
            # Set a loop to manage generation retries in case of failure
            for _ in range(self.scenario_validator_configuration.retry_generation):
                try:
                    # Copy base scenario and planning problem
                    _scenario = deepcopy(self.base_scenario)
                    _planning_problem_set = deepcopy(self.base_planning_problem_set)

                    # Generate a random planning problem for the target number of AVs
                    for _ in range(target):
                        _planning_problem_set = self.add_random_planning_problem(config, _scenario, _planning_problem_set, self.map_file_path)
                    
                    # Check if scenario is valid with the generated planning problems
                    if self._is_valid(_scenario, _planning_problem_set):
                        self.scenario_counter += 1
    
                        self.logger.debug(f"Scenario {self.scenario_counter} generated with {target} vehicles")
                        setattr(_scenario, "debug_id", self.scenario_counter)

                        return _scenario, _planning_problem_set
                except Exception as exc:
                    self.logger.debug(f"An exception occurred while generating the scenario. {exc}")

            self.logger.debug(f"Retrying to generate a scenario with {target} vehicles...")

            # Update the target number of planning problems to generate
            target -= 1

        raise Exception("Cannot generate any scenario!")

class EVITAScenarioSampling(ScenarioSampling):
    """
    Sampling operator to generate random scenarios and planning problem sets for EVITA
    """
    def __init__(self, base_scenario, base_planning_problem_set, scenario_generator_configuration, termination_criterion, map_file_path):
        # Use the parent ScenarioSampling class to initialize the common attributes
        super().__init__(base_scenario, base_planning_problem_set, scenario_generator_configuration, termination_criterion)

        # Update the scenario validator to the EVITA one
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, apollo = True)
        
        # Update the add_random_planning_problem function to the EVITA one
        self.add_random_planning_problem = scenario_mutation.add_evita_random_planning_problem

        # Save the map file path
        self.map_file_path = map_file_path

    def _do(self, problem, n_samples, **kwargs):
        """
        Generate `n_samples` random scenarios and planning problem sets for EVITA
        """
        # Generate the traffic lights configuration for the scenario
        with self.termination_criterion.generating():
            self.base_scenario = generate_traffic_lights(self.base_scenario, self.scenario_generator_configuration.max_time_step, self.map_file_path)

        return super()._do(problem, n_samples, **kwargs)

# ========== Mutation ==========

class ScenarioMutation(Mutation):
    """
    Mutation operator to mutate scenarios and planning problem sets
    """
    def __init__(self, map_file_path: str, scenario_generator_configuration, termination_criterion):
        """
        Mutation of each individual:
        - Each vehicle's planning problem is selected with probability 1 / (number of vehicles), and each of its attributes (initial position, initial velocity, goal area) is then mutated with probability 1/3;
        - Traffic lights (Apollo only) are mutated with probability 1/2 per operator;
        - One vehicle is removed with probability 1 / (number of vehicles) (if above min_amount_pp), and one is added with probability 1 / (number of vehicles) (if below max_amount_pp).
        The mutant is validated (not counted against the budget) and replaces the individual only if valid.
        """
        super().__init__()
        self.scenario_generator_configuration = scenario_generator_configuration

        self.planning_problem_mutation_operators = [scenario_mutation.mutate_initial_position, scenario_mutation.mutate_initial_velocity, scenario_mutation.mutate_goal_area_position]

        self.add_random_planning_problem = scenario_mutation.add_random_planning_problem
        self.remove_random_planning_problem = scenario_mutation.remove_random_planning_problem
        
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, scenario_generator_configuration)

        self.map_file_path = map_file_path
        self.termination_criterion = termination_criterion
        self.logger = module_logger

    def _do(self, problem, X, **kwargs):
        """
        Mutate the scenarios and planning problem sets in X by applying the mutation operators with a certain probability
        """
        config = self.scenario_generator_configuration

        with self.termination_criterion.generating():
            # Iterate over each individual (scenario, planning problem set)
            for i in range(len(X)):
                # Copy the scenario and planning problem set to mutate them
                _scenario = deepcopy(X[i, 0])
                _planning_problem_set = deepcopy(X[i, 1])

                self.logger.debug(f"{'-' * 25} Mutation of scenario {getattr(_scenario, 'generation_id', '?')} {'-' * 25}")

                # The mutant must be validated again
                for attribute in ("is_valid", "validation_message"):
                    if hasattr(_scenario, attribute):
                        delattr(_scenario, attribute)

                # Planning problems: each selected with probability 1 / |AVs|, each attribute then with probability 1 / 3
                planning_problems = _planning_problem_set.planning_problem_dict
                for key, planning_problem in list(planning_problems.items()):
                    if random.random() > 1 / len(planning_problems):
                        continue

                    mutated_planning_problem = planning_problem

                    # Iterate over each planning problem mutation operator
                    for operator in self.planning_problem_mutation_operators:
                        # Apply the mutation operator with a probability of 1 / number of mutation operators
                        if random.random() <= 1 / len(self.planning_problem_mutation_operators):
                            self.logger.debug(f"Mutating Planning Problem {mutated_planning_problem.planning_problem_id} with {operator.__name__}")
                            mutated_planning_problem = operator(self.scenario_generator_configuration, mutated_planning_problem, _scenario, self.map_file_path)

                    # Save mutated planning problem
                    planning_problems[key] = mutated_planning_problem
                
                # Try to mutate the scenario's traffic lights if the scenario is urban
                if hasattr(self, "traffic_lights_mutation_operators"):
                    mutated_scenario = _scenario
                    # Iterate over each traffic light mutation operator
                    for operator in self.traffic_lights_mutation_operators:
                        if random.random() <= 1 / len(self.traffic_lights_mutation_operators):
                            # Apply the mutation operator with a probability of 1 / number of mutation operators
                            self.logger.debug(f"Mutating scenario {getattr(mutated_scenario, 'generation_id', '?')} with {operator.__name__}")
                            mutated_scenario = operator(self.scenario_generator_configuration, _scenario, self.map_file_path)

                    _scenario = mutated_scenario
                    
                # Possibly remove an existing planning problem
                # - If we haven't reached the minimum amount of planning problems yet
                # - With a certain probability to avoid always removing a planning problem at each mutation
                if len(_planning_problem_set.planning_problem_dict) > self.scenario_generator_configuration.min_amount_pp and random.random() <= (1 / len(planning_problems)):
                    self.logger.debug(f"Mutating scenario with {self.remove_random_planning_problem.__name__}")
                    _planning_problem_set = self.remove_random_planning_problem(self.scenario_generator_configuration, _scenario, _planning_problem_set)

                # Possibly add a new planning problem:
                # - If we haven't reached the maximum amount of planning problems yet
                # - With a certain probability to avoid always adding a planning problem at each mutation
                if len(_planning_problem_set.planning_problem_dict) < self.scenario_generator_configuration.max_amount_pp and random.random() <= (1 / len(planning_problems)):
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
    def __init__(self, map_file_path: str, scenario_generator_configuration, termination_criterion):
        # Use the parent ScenarioMutation class to initialize the common attributes and mutation operators
        super().__init__(map_file_path, scenario_generator_configuration, termination_criterion)

        # Update the mutation operators to the EVITA ones
        self.traffic_lights_mutation_operators = [scenario_mutation.mutate_tl_initial_configuration, scenario_mutation.mutate_green_light_duration]
        
        self.planning_problem_mutation_operators.remove(scenario_mutation.mutate_goal_area_position)
        self.planning_problem_mutation_operators.append(scenario_mutation.mutate_evita_goal_area_position)
        
        self.add_random_planning_problem = scenario_mutation.add_evita_random_planning_problem

# ========== Crossover  ==========

class ScenarioCrossover(Crossover):
    """
    One-point crossover of the parents' lists of planning problems at a random position; the offspring replace the parents only if both are valid (validation not counted against the budget).
    Planning problems are renumbered from av_base_id + 1, so ids stay unique.
    """
    def __init__(self, scenario_generator_configuration, termination_criterion):
        # Define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.scenario_generator_configuration = scenario_generator_configuration

        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, scenario_generator_configuration)

        self.termination_criterion = termination_criterion
        self.logger = module_logger

    def _do(self, problem, X, **kwargs):
        # The input has the following shape (n_parents, n_matings, n_var)
        _, n_matings, _ = X.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype = object)

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
                    base_id = self.scenario_generator_configuration.av_base_id

                    # Create the offspring planning problem sets by combining the planning problems from both parents
                    offspring_1_planning_problems = PlanningProblemSet(self._renumber_planning_problems(parent_1_planning_problems[:cut_index] + parent_2_planning_problems[cut_index:], base_id))
                    offspring_2_planning_problems = PlanningProblemSet(self._renumber_planning_problems(parent_2_planning_problems[:cut_index] + parent_1_planning_problems[cut_index:], base_id))
                
                    # Create new PlanningProblemSet objects for the offspring planning problems
                    offspring_1_scenarios = deepcopy(parent_1_scenarios)
                    offspring_2_scenarios = deepcopy(parent_2_scenarios)

                    # Remove 'is_valid' and 'validation_message' attribute from the offspring scenarios if it exists
                    for offspring in (offspring_1_scenarios, offspring_2_scenarios):
                        for attribute in ("is_valid", "validation_message"):
                            if hasattr(offspring, attribute):
                                delattr(offspring, attribute)

                    # Validate the offspring scenarios and planning problem sets
                    with self.termination_criterion.paused():
                        self.scenario_validator.validate_all([(offspring_1_scenarios, offspring_1_planning_problems), (offspring_2_scenarios, offspring_2_planning_problems)])

                    # If both offspring are valid, assign them to the output array Y
                    if offspring_1_scenarios.is_valid and offspring_2_scenarios.is_valid:
                        Y[0, k, 0] = offspring_1_scenarios
                        Y[1, k, 0] = offspring_2_scenarios
                        
                        Y[0, k, 1] = offspring_1_planning_problems
                        Y[1, k, 1] = offspring_2_planning_problems
                        continue

                # Otherwise (or if shortest < 2): return the parents
                Y[0, k, 0] = parent_1_scenarios
                Y[1, k, 0] = parent_2_scenarios
                
                Y[0, k, 1] = parent_1_pps
                Y[1, k, 1] = parent_2_pps
                
        return Y
    
    @staticmethod
    def _renumber_planning_problems(planning_problems, base_id: int) -> list:
        """
        Copies of the planning problems with ids base_id + 1, base_id + 2, ...
        """
        renumbered = []

        for index, planning_problem in enumerate(planning_problems):
            planning_problem_copy = deepcopy(planning_problem)
            planning_problem_copy._planning_problem_id = base_id + 1 + index
            renumbered.append(planning_problem_copy)
        
        return renumbered

class ScenarioDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0] and a.X[1] == b.X[1]

# ========== Termination: time budget, counters, checkpoints ==========

class CustomTimeTermination(Termination):
    """
    Termination criterion based on a time budget (minutes), with the same rules as the orchestrator:
    generation, execution and evaluation count; blocks run inside paused() (validation, plots, checkpoints) do not.
    It also keeps the performance counters, the main progress bar, and creates a checkpoint every `checkpoint_every` minutes of budget time.
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
        self.skipped_scenarios = 0
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
            with TASK_TIMER.measure("generation"):
                yield
        finally:
            self.total_generation_time += (time.perf_counter() - wall_start) - (self._paused_total - paused_start)
            self.update_main_progress_bar()
 
    def count(self, scenario) -> None:
        """
        Count a processed scenario valid (executed and evaluated), skipped (not executed because the budget ran out) or invalid
        """
        self.all_scenarios += 1
        if getattr(scenario, "is_valid", False):
            self.valid_scenarios += 1
        elif getattr(scenario, "validation_message", None) == TIMEOUT_MESSAGE:
            self.skipped_scenarios += 1
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

        if checkpoint_id <= 0 or checkpoint_id in self.checkpoints:
            return

        from evita.orchestrator import checkpoint

        self.checkpoints.add(checkpoint_id)
        with self.paused():
            try:
                with TASK_TIMER.measure("checkpoints"):
                    checkpoint(checkpoint_id, algorithm.problem.output_folder, self.all_scenarios, self.valid_scenarios, self.invalid_scenarios, self.total_generation_time, self.skipped_scenarios, n_bins = args.n_bins)
            except Exception:
                module_logger.critical(f"Failed to checkpoint execution at checkpoint {checkpoint_id}")
                module_logger.critical(traceback.format_exc())
 