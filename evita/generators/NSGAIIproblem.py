"""
EVITA's search problem for NSGA-II: an individual is a scenario with its planning problems; its three objectives are computed from its execution
with the simulator chosen by --simulator (evita/simulators.py: FrenetiX or Apollo).
"""
import os
import sys
import time
import logging
import traceback
import numpy as np

from pymoo.core.problem import ElementwiseProblem

from commonroad.planning.goal import Interval

from evita import simulators

from ..timing import TASK_TIMER
from ..execution import ExecutionResult
from ..generators.utils import compute_interaction_map
from ..config import ValidationConfig
from ..utils import EVITA_MAIN_LOGGER_NAME, RESULTS_FOLDER
from ..orchestrator import (EXECUTION_ERROR_MESSAGE, TIMEOUT_MESSAGE, _load_class, backup_debug, debug_plot_to_file, ensure_generation_id)

class NSGAIIproblem(ElementwiseProblem):
    """
    EVITA's search problem
    Decision variables: the scenario and its planning problem set
    Objectives, are all minimized by pymoo:
      1. Minus the number of interactions (Maximize);
      2. The number of vehicles (Minimize);
      3. Minus the distance between the scenario's interaction map and the closest map of the previous generations (maximize)
      
    Scenarios that cannot be evaluated get the worst value of every objective.
    """

    def __init__(self, map_file, driving_agent_name, population_size, termination_criterion, trajectory_difference_threshold, n_bins = None, **kwargs):
        """
        Initialize the NSGA-II problem
        """

        self.debug = kwargs.pop("debug", False)
        self.force_max_duration = kwargs.pop("force_max_duration", None)

        # Simulator executing the scenarios (evita/simulators.py), and Apollo's switch for its .record files
        self.simulator = simulators.get_simulator(kwargs.pop("simulator", simulators.FRENETIX))
        self.delete_records = kwargs.pop("delete_records", False)

        kwargs.setdefault("requires_kwargs", True)
        super().__init__(n_var = 2, n_obj = 3, n_ieq_constr = 0, **kwargs)

        self.worst_fitness = [sys.maxsize, sys.maxsize, sys.maxsize]

        # Names recorded in the run's metadata
        self.scenario_generator_module = "evita.generators.nsgaii"
        self.scenario_generator_class = "NSGAIIproblem"

        self.map_file = map_file
        self.driving_agent_name = driving_agent_name  # "frenetix"
        self.pop_size = population_size
        self.termination_criterion = termination_criterion

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

        # Create the folder for this execution here:
        # Generate the Unique ID of this Execution.
        self.timestamp_id = time.time() * 100000000 // 1000000

        # Create the unique folder that will host the results of this execution using the test generator data and a timestamp as id
        if "output_folder_name" in kwargs and kwargs["output_folder_name"]:
            self.output_folder_name = kwargs["output_folder_name"]
        else:
            self.output_folder_name = "_".join([str(self.scenario_generator_module), str(self.scenario_generator_class), str(self.timestamp_id)])
    
        self.output_folder = os.path.join(RESULTS_FOLDER, self.output_folder_name)
        self.debug_folder = os.path.join(self.output_folder, "debug")
        
        # This is the folder of the entire run, each simulation will have its own folder
        os.makedirs(self.output_folder, exist_ok = True)
        os.makedirs(self.debug_folder, exist_ok = True)

        self.logger.info(f"Outputting results to {self.output_folder}")
        if self.debug:
            self.logger.info(f"Outputting debug data to {self.debug_folder}")

        self.scenario_generator_configuration = simulators.generation_configuration(self.simulator)
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = simulators.scenario_validator(self.simulator, self.scenario_validator_configuration, self.scenario_generator_configuration)

        self.valid_scenarios = 0
        self.invalid_scenarios = 0
        
        if n_bins is None:
            self.n_bins = self.scenario_generator_configuration.n_bins
        else:
            self.n_bins = n_bins

        self.trajectory_difference_threshold = trajectory_difference_threshold
        
        self.interaction_maps_by_generation = []

        # Executor and evaluator, created at the first evaluation
        self.scenario_executor = None
        self.scenario_evaluator = None

        # Progress bar of the generation being evaluated
        self.generation_bar = None
        self.generation_bar_id = None

    def _load_executor_and_evaluator(self) -> None:
        """
        Create the simulator's scenario executor and evaluator (evita/simulators.py).
        """
        se_class = _load_class(self.simulator.executor_path, self.simulator.executor_module, self.simulator.executor_class, self.logger)
        # EVITA's search evolves the traffic lights (Apollo): the executor keeps them
        self.scenario_executor = se_class(self.output_folder, driving_agent_name = self.driving_agent_name, debug = self.debug, map_file = self.map_file, delete_records = self.delete_records, random_lights = False)

        sev_class = _load_class(".", self.simulator.evaluator_module, self.simulator.evaluator_class, self.logger)
        self.scenario_evaluator = sev_class(self.output_folder)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Called by pymoo for each individual: validate the scenario, execute and evaluate it if valid and the budget allows, and set its objectives in out["F"].
        """
        # Extract the scenario and the planning problem set from the decision variables
        scenario = x[0]
        planning_problem_set = x[1]
        self.current_generation = kwargs["algorithm"].n_iter

        # Ensure that the scenario has an unique number
        ensure_generation_id([(scenario, planning_problem_set)])

        # The executor and evaluator are created once
        if self.scenario_executor is None or self.scenario_evaluator is None:
            self._load_executor_and_evaluator()

        # Update the progress bar for the current generation
        if self.current_generation != self.generation_bar_id:
            if self.generation_bar is not None:
                self.generation_bar.close(clear = True)

            self.generation_bar = self.termination_criterion.progress_bar_manager.counter(total = self.pop_size, desc = f"Generation {self.current_generation}", unit = "scenarios", color = "blue", leave = False)
            self.generation_bar_id = self.current_generation
            self.logger.info(f" ====== Evaluating Generation {self.current_generation} ====== ")

        self.termination_criterion.update_main_progress_bar()

        # Generation and position of this individual within it
        self.current_individual = self.generation_bar.count

        # Set the default fitness to the worst value, in case the scenario is invalid or cannot be evaluated
        fitness = self.worst_fitness

        # ===== 1. Validation =====
        with self.termination_criterion.paused():
            self.scenario_validator.validate(scenario, planning_problem_set)
 
            if self.debug:
                with TASK_TIMER.measure("debug_plots"):
                    debug_plot_file = os.path.join(self.debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}.png")
                    debug_plot_to_file(scenario, planning_problem_set, debug_plot_file)

        # ===== 2. Budget check: no new scenario is executed once the budget is over =====
        if self.termination_criterion.is_over():
            self.logger.critical("Timeout is over. Force the current scenario to be invalid")
            scenario.is_valid = False
            scenario.validation_message = TIMEOUT_MESSAGE
        elif scenario.is_valid:
            # ===== 3. Execution (counted in the budget) =====
            execution = self._execute(scenario, planning_problem_set)

            try:
                if scenario.is_valid:
                    fitness = self._evaluate_and_score(scenario, execution)
                elif scenario.validation_message == "Error during the simulation execution":
                    backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)
            finally:
                self._release(execution)

        # ===== Fitness, counters, and progress bars =====
        out["F"] = fitness

        # Skipped scenarios (budget over) are counted separately by the termination criterion
        if scenario.is_valid:
            self.valid_scenarios += 1
        elif getattr(scenario, "validation_message", None) != TIMEOUT_MESSAGE:
            self.invalid_scenarios += 1
 
        self.termination_criterion.count(scenario)
        self.generation_bar.update()
        self.termination_criterion.update_main_progress_bar()
        self.logger.info(f"Scenario {scenario.generation_id}: fitness {fitness}")

    def _execute(self, scenario, planning_problem_set):
        """
        Simulate a copy of the scenario with road added after the goal areas
        The original scenario is kept for the search
        Returns the simulation, or None if it failed (the scenario is then invalid)
        """
        # Force max planning problem duration if required
        if self.force_max_duration:
            self.logger.critical(f"Forcing max duration to {self.force_max_duration} for scenario {scenario.generation_id}")
            for planning_problem in planning_problem_set.planning_problem_dict.values():
                for goal_state in planning_problem.goal.state_list:
                    if "time_step" in goal_state.attributes:
                        goal_state.time_step = Interval(0, self.force_max_duration)

        try:
            self.logger.info(f"Executing scenario: {scenario.generation_id} - {self.termination_criterion.remaining_time()}")
            for pp in planning_problem_set.planning_problem_dict.values():
                self.logger.debug(f"- {pp.planning_problem_id} - {pp.initial_state.position} - {pp.initial_state.velocity} - planned start at {pp.initial_state.time_step}")
 
            # Make sure we enlong the lanelets nearby goal areas or the planner will fail to reach them
            with TASK_TIMER.measure("execution_preparation", key = scenario.generation_id):
                extended_scenario = self.scenario_executor.prepare(scenario)
 
            # Time Shift so at least one agent starts at time_step = 0
            # time_shifted_planning_problem_set = time_shift(scenario, planning_problem_set)
            time_shifted_planning_problem_set = planning_problem_set
 
            # Execute the extended scenario; the original one is the one used for evolution.
            # The executor shares the progress bar manager and advances the main bar at every simulation step
            with TASK_TIMER.measure("simulation", key = scenario.generation_id):
                execution = self.scenario_executor.execute(extended_scenario, 
                                                            time_shifted_planning_problem_set, 
                                                            self.termination_criterion.progress_bar_manager,
                                                            on_step = self.termination_criterion.update_main_progress_bar)
        except Exception:
            # A failure of one simulation must not end the run: the scenario is invalid, and the search continues
            self.logger.critical(f"A critical error prevented {scenario.generation_id} to complete. Mark it as Error and set the validity flag to False")
            self.logger.critical(traceback.format_exc())
            scenario.is_valid = False
            scenario.validation_message = EXECUTION_ERROR_MESSAGE
            return None
        
        if execution is None:
            scenario.is_valid = False
            scenario.validation_message = EXECUTION_ERROR_MESSAGE
            self.logger.critical(f"Scenario {scenario.generation_id} failed during simulation execution. Continuing with next scenario.")

            return None

        # The executor's health check (Apollo): the execution cannot be used, for the reason it gives
        if execution.failure:
            self.logger.critical(f"Scenario {scenario.generation_id} cannot be used: {execution.failure}")
            scenario.is_valid = False
            scenario.validation_message = execution.failure
            backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)
            return execution

        # Invalid if an agent fails at time_step 0 or never moves from its initial time step
        if execution.not_started:
            agent_id = execution.not_started[0]
            self.logger.critical(f"The scenario is invalid. Agent {agent_id} did not even start")
            scenario.is_valid = False
            scenario.validation_message = f"Agent {agent_id} could not compute a physically possible trajectory since the beginning."

        return execution

    def _evaluate_and_score(self, scenario, execution):
        """
        Evaluate the executed scenario and compute its fitness.
        A failure makes the scenario invalid (worst fitness), and its folder is moved to the debug folder.
        """
        try:
            criticality_dataframe = self._evaluate_simulation(scenario, execution)

            with TASK_TIMER.measure("fitness", key = scenario.generation_id):
                return self._compute_fitness(execution, criticality_dataframe, self.current_generation)
        except Exception:
            msg = f" {scenario.generation_id} to be evaluated. Mark it as invalid"
            self.logger.critical("=" * (len(msg) + 2))
            self.logger.critical(f" {msg} ")
            self.logger.critical("=" * (len(msg) + 2))

            # If an error is triggered during the evaluation, get rid of the scenario
            self.logger.critical(traceback.format_exc())

            scenario.is_valid = False

            # Try to move scenario folder away
            backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)

            return self.worst_fitness

    def _release(self, execution) -> None:
        """
        Free the simulator's resources held by the execution (only for ExecutionResult: the Apollo subclass passes its
        own outputs folder instead, and has nothing to release).
        """
        if isinstance(execution, ExecutionResult):
            self.scenario_executor.release(execution)

    def _evaluate_simulation(self, scenario, execution):
        """
        Compute the criticality metrics, then release the simulation's resources
        """
        self.logger.info(f"Evaluating scenario: {scenario.generation_id} - {self.termination_criterion.remaining_time()}")
 
        with TASK_TIMER.measure("evaluation", key = scenario.generation_id):
            criticality_dataframe = self.scenario_evaluator.evaluate(execution, self.trajectory_difference_threshold, on_step = self.termination_criterion.update_main_progress_bar)
 
        self.logger.info(f"Evaluation completed - {self.termination_criterion.remaining_time()}")

        return criticality_dataframe
        
    def _compute_fitness(self, execution, criticality_dataframe, generation):
        """
        Compute the three objectives (pymoo minimizes):
        - minus the number of interactions
        - the number of cars
        - minus the distance of this interaction map from the closest map of the previous generations
        """
        # Compute the interaction map for the scenario
        interaction_map = compute_interaction_map(criticality_dataframe, self.n_bins)

        fitness_num_interactions = criticality_dataframe["is_interacting"].sum()
        fitness_num_cars = len(execution.agents)
 
        # Variety of interactions: distance from the closest interaction map found in the PREVIOUS generations
        previous_maps = [past_map for past_generation, past_map in self.interaction_maps_by_generation if past_generation < generation]
        min_dist = min((np.linalg.norm(interaction_map - past_map) for past_map in previous_maps), default = sys.maxsize)
 
        self.interaction_maps_by_generation.append((generation, interaction_map))
 
        return [-fitness_num_interactions, fitness_num_cars, -min_dist]