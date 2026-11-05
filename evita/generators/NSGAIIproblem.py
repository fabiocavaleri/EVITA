"""
EVITA's search problem for NSGA-II: an individual is a scenario with its planning problems; its three objectives are computed from its simulation with FrenetiX.

EVITADoppelTestNSGAIIproblem (at the end of the file) adapts the problem to the Apollo pipeline.
"""
import os
import sys
import json
import time
import logging
import traceback
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy

from pymoo.core.problem import ElementwiseProblem

from commonroad.common.file_reader import XMLFileReader
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.planning.goal import Interval
from commonroad.visualization.mp_renderer import MPRenderer

from cr_scenario_handler.utils.agent_status import AgentStatus

from EVITA_DoppelTest.conversion_scripts.utils.RecordFileError import RecordFileError
from EVITA_DoppelTest.conversion_scripts.utils.conversion_utils import build_json_scenario
from EVITA_DoppelTest.conversion_scripts.utils.interactions_utils import augment_with_collisions, augment_with_interactions, planned_trajectories_to_df

from ..config import ScenarioGeneratorConfig, ValidationConfig
from ..generators.utils import compute_interaction_map
from ..orchestrator import (EXECUTION_ERROR_MESSAGE, SCENARIO_EVALUATOR_CLASS, SCENARIO_EVALUATOR_MODULE, SCENARIO_EXECUTOR_CLASS,
                            SCENARIO_EXECUTOR_MODULE, TIMEOUT_MESSAGE, _load_class, backup_debug, debug_plot_to_file, ensure_generation_id)
from ..utils import EVITA_MAIN_LOGGER_NAME, FRENETIX_ROOT, LANELET_EXTENSION, RESULTS_FOLDER, extend_scenario_after_goal_areas
from ..timing import TASK_TIMER
from ..validator import ScenarioValidator

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

    def __init__(self, map_file, driving_agent_name, population_size, termination_criterion, trajectory_difference_threshold = 1.0, n_bins = None, **kwargs):
        """
        Initialize the NSGA-II problem
        """

        self.debug = kwargs.pop("debug", False)
        self.force_max_duration = kwargs.pop("force_max_duration", None)

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

        self.scenario_generator_configuration = ScenarioGeneratorConfig()
        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, self.scenario_generator_configuration)

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
        Create the FrenetiX scenario executor and evaluator.
        """
        se_class = _load_class(FRENETIX_ROOT, SCENARIO_EXECUTOR_MODULE, SCENARIO_EXECUTOR_CLASS, self.logger)
        self.scenario_executor = se_class(self.output_folder, driving_agent_name = self.driving_agent_name, debug = self.debug)

        sev_class = _load_class(".", SCENARIO_EVALUATOR_MODULE, SCENARIO_EVALUATOR_CLASS, self.logger)
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
            simulation = self._execute(scenario, planning_problem_set)

            if scenario.is_valid:
                try:
                    # ===== 4. Evaluation =====
                    criticality_dataframe = self._evaluate_simulation(scenario, simulation)

                    # ===== 5. Fitness =====
                    with TASK_TIMER.measure("fitness", key = scenario.generation_id):
                        fitness = self._compute_fitness(simulation, criticality_dataframe, self.current_generation)
                except Exception:
                    msg = f" {scenario.generation_id} to be evaluated. Mark it as invalid"
                    self.logger.critical("=" * (len(msg) + 2))
                    self.logger.critical(f" {msg} ")
                    self.logger.critical("=" * (len(msg) + 2))

                    # If an error is triggered during the evaluation, get rid of the scenario
                    self.logger.critical(traceback.format_exc())

                    scenario.is_valid = False
                    fitness = self.worst_fitness

                    # Try to move scenario folder away
                    backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)
            elif scenario.validation_message == "Error during the simulation execution":
                # If the scenario is invalid because of an issue during the simulation, move the folder away
                backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)

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
                extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters = LANELET_EXTENSION)
 
            # Time Shift so at least one agent starts at time_step = 0
            # time_shifted_planning_problem_set = time_shift(scenario, planning_problem_set)
            time_shifted_planning_problem_set = planning_problem_set
 
            # Execute the extended scenario; the original one is the one used for evolution.
            # The executor shares the progress bar manager and advances the main bar at every simulation step
            with TASK_TIMER.measure("simulation", key = scenario.generation_id):
                simulation = self.scenario_executor.execute(extended_scenario, 
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
        
        if simulation is None:
            scenario.is_valid = False
            scenario.validation_message = EXECUTION_ERROR_MESSAGE
            self.logger.critical(f"Scenario {scenario.generation_id} failed during simulation execution. Continuing with next scenario.")
            
            return None

        for agent in simulation.agents:
            # Invalid if an agent fails at time_step 0 or never moves from its initial time step
            if (agent.agent_state.status == AgentStatus.ERROR and agent.agent_state.last_timestep == 0) or \
                    (agent.planning_problem.initial_state.time_step == agent.agent_state.last_timestep):
                self.logger.critical(f"The scenario is invalid. Agent {agent.id} did not even start")
                scenario.is_valid = False
                scenario.validation_message = f"Agent {agent.id} could not compute a physically possible trajectory since the beginning."
                break

        return simulation

    def _evaluate_simulation(self, scenario, simulation):
        """
        Compute the criticality metrics, then release the simulation's resources
        """
        self.logger.info(f"Evaluating scenario: {scenario.generation_id} - {self.termination_criterion.remaining_time()}")
 
        try:
            with TASK_TIMER.measure("evaluation", key = scenario.generation_id):
                criticality_dataframe = self.scenario_evaluator.evaluate(simulation, self.trajectory_difference_threshold, on_step = self.termination_criterion.update_main_progress_bar)
        finally:
            # Perform cleanup after evaluation
            simulation.sim_logger.con.close()

            if simulation.config.simulation.multiprocessing:
                simulation.close_processes()
 
        self.logger.info(f"Evaluation completed - {self.termination_criterion.remaining_time()}")

        return criticality_dataframe
        
    def _compute_fitness(self, simulation, criticality_dataframe, generation):
        """
        Compute the three objectives (pymoo minimizes):
        - minus the number of interactions
        - the number of cars
        - minus the distance of this interaction map from the closest map of the previous generations
        """
        # Compute the interaction map for the scenario
        interaction_map = compute_interaction_map(criticality_dataframe, self.n_bins)

        fitness_num_interactions = criticality_dataframe["is_interacting"].sum()
        fitness_num_cars = len(simulation.agents)
 
        # Variety of interactions: distance from the closest interaction map found in the PREVIOUS generations
        previous_maps = [past_map for past_generation, past_map in self.interaction_maps_by_generation if past_generation < generation]
        min_dist = min((np.linalg.norm(interaction_map - past_map) for past_map in previous_maps), default = sys.maxsize)
 
        self.interaction_maps_by_generation.append((generation, interaction_map))
 
        return [-fitness_num_interactions, fitness_num_cars, -min_dist]

# ========== Apollo pipeline ==========

class EVITADoppelTestNSGAIIproblem(NSGAIIproblem):
    """
    This class defines the problem to be solved by the NSGA-II algorithm for the EVITA DoppelTest
    The problem is defined as follows:
    - Decision variables: the scenario and the planning problem set
    - Objectives:
        1. Maximize the number of interactions in the scenario
        2. Maximize the variety of the interaction in the scenario'
        3. Minimize the number of cars in the scenario
    """
    def __init__(self, map_file_path, driving_agent_name, pop_size, termination_criterion, records_file_path, trajectory_difference_threshold = 1.0, n_bins = None, **kwargs):
        """
        Use parent constructor to initialize the NSGAIIproblem and then add the specific attributes for EVITA + DoppelTest
        """
        super().__init__(map_file_path, driving_agent_name, pop_size, termination_criterion, trajectory_difference_threshold, n_bins, **kwargs)

        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration, apollo = True)

        self.records_file_path = records_file_path

        # Python 3.8 environment and script that run a scenario with Apollo
        self.python_38 = "../.venv38/bin/python"
        self.scenario_running_script = "./apollo_scenario_runner.py"

    def _load_executor_and_evaluator(self):
        """
        Apollo runs in a separate process (apollo_scenario_runner.py): no Frenetix executor or evaluator is needed
        """
        self.scenario_executor = None
        self.scenario_evaluator = None

    def _scenario_output_folder(self) -> str:
        """
        Folder of the current scenario's Apollo outputs: <evita-records>/Generation_<g>/Scenario_<i>
        """
        return os.path.join(self.records_file_path.replace("/records/", "/evita-records/"), f"Generation_{self.current_generation}", f"Scenario_{self.current_individual}")

    def plot_pps(self, _scenario, planning_problems, output_folder: str):
        """
        This function plots the planning problems of a scenario and saves the plots to file
        """
        # Iterate over all scenario's planning problems
        for pp in planning_problems:
            # Initialize the plot
            fig, ax = plt.subplots()
            rnd = MPRenderer(ax = ax)

            # Define the drawing parameters for the lanelet network and the traffic signs
            rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
            rnd.draw_params.lanelet_network.traffic_sign.scale_factor = 0.7
            rnd.draw_params.lanelet_network.traffic_light.scale_factor = 0.7
            rnd.draw_params.lanelet_network.lanelet.stop_line_color = "black"

            # Plot the road and the signals
            _scenario.lanelet_network.draw(rnd)

            # Define the drawing parameters for the initial state and the goal area
            rnd.draw_params.shape.facecolor = 'lightgreen'
            rnd.draw_params.shape.edgecolor = 'green'
            rnd.draw_params.shape.opacity = 0.6

            # Extract and plot the initial state and the goal area of the planning problem
            pp.goal.state_list[-1].position.draw(rnd)
            rnd.render()

            ax.scatter(
                pp.initial_state.position[0],
                pp.initial_state.position[1],
                s = 12,
                marker = 'o',
                linewidths = 1,
                c = 'red',
                zorder = 10
            )

            plt.axis('off')
            plt.savefig(
                f"{output_folder}/pp_{pp.planning_problem_id}.png",
                dpi = 500,
                bbox_inches = "tight", # trims extra whitespace around the figure
                pad_inches = 0,        # no padding
                transparent = False,   # keep background white (or True if you want transparent)
            )
            plt.close()

    def _execute(self, scenario, planning_problem_set):
        """
        Run the scenario with Apollo. Returns the folder with Apollo's outputs, or None if the execution failed
        (scenario.is_valid and scenario.validation_message are then updated, as for Frenetix)
        """
        output_folder_path = self._scenario_output_folder()
        self.logger.info(f"Executing scenario: {scenario.generation_id} with Apollo - {self.termination_criterion.remaining_time()}")

        try:
            os.makedirs(output_folder_path, exist_ok = True)
            os.makedirs(os.path.join(output_folder_path, "scenario"), exist_ok = True)

            # Scenario in Apollo's .json format, and in CommonRoad format
            json_scenario = build_json_scenario(scenario, deepcopy(planning_problem_set), self.map_file)
            evita_scenario_file_path = os.path.join(output_folder_path, "scenario", "evita_scenario.json")
            with open(evita_scenario_file_path, "w") as f:
                json.dump(json_scenario, f, indent = 2)

            CommonRoadFileWriter(scenario, planning_problem_set).write_to_file(os.path.join(output_folder_path, "scenario", "scenario.xml"))

            input("PAUSE: Press Enter to continue with the Apollo execution...")

            # Run the scenario with Apollo; the script also converts the result to CommonRoad and saves collisions, interactions, and perception data
            self.logger.info(f"Apollo records will be saved at: {os.path.join(self.records_file_path, f'Generation_{self.current_generation}', f'Scenario_{self.current_individual}')}")
            try:
                subprocess.run([self.python_38, self.scenario_running_script, evita_scenario_file_path,
                                str(self.current_generation), str(self.current_individual), self.records_file_path], check = True)
            except subprocess.CalledProcessError as e:
                if e.returncode == 10 or not os.path.exists(os.path.join(output_folder_path, "executed_scenario.xml")):
                    self.logger.critical(f"Scenario {scenario.generation_id} failed during the execution with Apollo (record file issue)")
                    scenario.is_valid = False
                    scenario.validation_message = "Error during the simulation execution"
                    return None

            # Plots of the planning problems: outputs, not testing, so excluded from the budget
            with self.termination_criterion.paused():
                planning_problems_plots_folder = os.path.join(output_folder_path, "scenario", "planning_problems_plots")
                os.makedirs(planning_problems_plots_folder, exist_ok = True)
                self.plot_pps(scenario, planning_problem_set.planning_problem_dict.values(), planning_problems_plots_folder)

            return output_folder_path
        except (ValueError, RecordFileError) as e:
            self.logger.critical(f"Scenario {scenario.generation_id} could not be executed with Apollo: {e}")
            scenario.is_valid = False
            scenario.validation_message = "Error during the simulation execution"
            return None

    def _evaluate_simulation(self, scenario, simulation):
        """
        Compute collisions and interactions from Apollo's outputs (simulation is the folder returned by _execute)
        """
        output_folder_path = simulation
        self.logger.info(f"Evaluating scenario: {scenario.generation_id} - {self.termination_criterion.remaining_time()}")

        # Executed scenario, collisions, planned trajectories and perceptions written by the Apollo runner
        executed_scenario, _ = XMLFileReader(os.path.join(output_folder_path, "executed_scenario.xml")).open(lanelet_assignment = True)
        scenario_collisions = pd.read_csv(os.path.join(output_folder_path, "apollo_scenario_collisions.csv"))
        scenario_vehicles_perceptions = pd.read_csv(os.path.join(output_folder_path, "vehicles_perception.csv"))
        scenario_traffic_lights_perceptions = pd.read_csv(os.path.join(output_folder_path, "traffic_lights_perceptions.csv"))
        with open(os.path.join(output_folder_path, "planned_trajectories.json"), "r") as f:
            scenario_planned_trajectories_df = planned_trajectories_to_df(json.load(f))

        # Collisions (and the collisions that could not be matched to obstacles)
        collisions_df, issue_rows = augment_with_collisions(executed_scenario, scenario_collisions)
        collisions_df.to_csv(os.path.join(output_folder_path, "fitness_functions", "collisions.csv"), index = False)
        if len(issue_rows) > 0:
            pd.DataFrame(issue_rows, columns = scenario_collisions.columns.tolist() + ["detected_obstacles"]).to_csv(os.path.join(output_folder_path, "fitness_functions", "invalid_collisions.csv"), index = False)

        # Interactions and interaction map
        interactions_df = augment_with_interactions(executed_scenario, scenario_planned_trajectories_df, scenario_vehicles_perceptions, scenario_traffic_lights_perceptions, scenario_collisions, self.trajectory_difference_threshold)
        if interactions_df is None or interactions_map is None:
            raise ValueError(f"No interactions could be computed for scenario {scenario.generation_id}")

        interactions_df.to_csv(os.path.join(output_folder_path, "fitness_functions", "interactions.csv"), index = False)
        pd.DataFrame(interactions_map).to_csv(os.path.join(output_folder_path, "fitness_functions", "interactions_map.csv"), index = False, header = False)

        

        self.logger.info(f"Evaluation completed - {self.termination_criterion.remaining_time()}")

        return {"executed_scenario": executed_scenario, "interactions_df": interactions_df, "interactions_map": interactions_map}

    def _compute_fitness(self, simulation, evaluation, generation):
        """
        Same three objectives as NSGAIIproblem (pymoo minimizes), from Apollo's outputs:
        - minus the number of interactions
        - the number of vehicles
        - minus the distance of this interaction map from the closest map of the previous generations
        """
        interaction_map = np.asarray(evaluation["interactions_map"], dtype = float)

        fitness_num_interactions = evaluation["interactions_df"]["is_interacting"].sum()
        fitness_num_cars = len(evaluation["executed_scenario"].dynamic_obstacles)

        # Variety of interactions: distance from the closest interaction map found in the PREVIOUS generations
        previous_maps = [past_map for past_generation, past_map in self.interaction_maps_by_generation if past_generation < generation]
        min_dist = min((np.linalg.norm(interaction_map - past_map) for past_map in previous_maps), default = sys.maxsize)

        self.interaction_maps_by_generation.append((generation, interaction_map))

        return [-fitness_num_interactions, fitness_num_cars, -min_dist]