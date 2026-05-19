import os
import sys
import time
import shutil
import logging
import traceback
import importlib
import numpy as np

from pymoo.core.problem import ElementwiseProblem

from ..config import ValidationConfig
from ..generators.utils import compute_interaction_map
from ..interfaces import AbstractScenarioExecutor, AbstractScenarioEvaluator
from ..utils import RESULTS_FOLDER, ROOT_DIR, extend_scenario_after_goal_areas, LANELET_EXTENSION, EVITA_MAIN_LOGGER_NAME, time_shift
from ..validator import ScenarioValidator, EVITAScenarioValidator

import json
import subprocess
import pandas as pd
from copy import deepcopy
from typing import List, Tuple

from EVITA_DoppelTest.conversion_scripts.utils.RecordFileError import RecordFileError

import matplotlib.pyplot as plt
from commonroad.visualization.mp_renderer import MPRenderer

from EVITA_DoppelTest.conversion_scripts.utils.conversion_utils import build_json_scenario
from EVITA_DoppelTest.conversion_scripts.utils.interactions_utils import augment_with_interactions, planned_trajectories_to_df, augment_with_collisions

from commonroad.scenario.scenario import Scenario
from commonroad.common.file_reader import XMLFileReader
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.planning.planning_problem import PlanningProblemSet

from cr_scenario_handler.utils.agent_status import AgentStatus


TOTAL_GENERATED_SCENARIOS = 0
def ensure_generation_id(scenario_tuples: List[Tuple[Scenario, PlanningProblemSet]]) -> None:
    global TOTAL_GENERATED_SCENARIOS
    # Debug Plot
    for scenario, planning_problem_set in scenario_tuples:
        TOTAL_GENERATED_SCENARIOS = TOTAL_GENERATED_SCENARIOS + 1
        # Tag the scenario with a generation ID
        setattr(scenario, "generation_id", TOTAL_GENERATED_SCENARIOS)

def backup_debug(_output_folder, _debug_folder, _scenario, _logger):
    # We need to clean up this folder to avoid the data analysis to pick broken data up!
    _logger.critical(f"Moving problematic folder {_output_folder} of invalid scenario {_scenario.generation_id} into debug folder {_debug_folder} ")
    exeuction_folder_name = str(_scenario.generation_id).rjust(5, '0')
    execution_folder = os.path.join(_output_folder, exeuction_folder_name)
    if os.path.exists(execution_folder):
        os.makedirs(_debug_folder, exist_ok=True)
        shutil.move(execution_folder, _debug_folder)
    else:
        _logger.critical(f"Folder {execution_folder} does not exist")

class NSGAIIproblem(ElementwiseProblem):
    def __init__(self, map_file, driving_agent_name, pop_size, termination_criterion, **kwargs):
        super().__init__(n_var=2, n_obj=3, n_ieq_constr=0, **kwargs)
        
        self.scenario_generator_module = "evita.generators.nsgaii"
        self.scenario_generator_class = "NSGAIIproblem"

        self.map_file = map_file  # "./base-scenarios/showcase-scenario.xml"
        self.driving_agent_name = driving_agent_name  # "frenetix"
        self.pop_size = pop_size
        self.termination_criterion = termination_criterion
        self.individual_number = 0
        self.interaction_maps = list()

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

        # Create the folder for this execution here:
        # Generate the Unique ID of this Execution.
        # Create the unique folder that will host the results of this execution using the test generator data and a timestamp as id
        self.timestamp_id = time.time() * 100000000 // 1000000
        self.output_folder_name = kwargs["output_folder_name"] if "output_folder_name" in kwargs and kwargs["output_folder_name"] else "_".join([str(self.scenario_generator_module), str(self.scenario_generator_class), str(self.timestamp_id)])
        
        self.output_folder = os.path.join(RESULTS_FOLDER, self.output_folder_name)
        
        
        # This is the folder of the entire run, each simulation will have its own folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.logger.info(f"Outputting results to {self.output_folder}")
        self.debug_folder = os.path.join(self.output_folder, "debug")
        os.makedirs(self.debug_folder, exist_ok=True)
        self.logger.info(f"Outputting debug data to {self.debug_folder}")

        self.scenario_validator_configuration = ValidationConfig()
        self.scenario_validator = ScenarioValidator(self.scenario_validator_configuration)
        self.valid_scenarios = 0
        self.invalid_scenarios = 0


    def _evaluate(self, x, out, *args, **kwargs):
        
        scenario = x[0]
        planning_problem_set = x[1]
        
        # Preprocess the scenario
        
        # ensure that the scenario has an unique number
        ensure_generation_id([(scenario, planning_problem_set)])
        
        # simulate
        # This requires a list of tuples
        self.scenario_validator.validate_all([(scenario, planning_problem_set)])
    # Validate the generated scenarios. This will add the attribute: is_valid to the scenario object
    # scenario_validator.validate_all(scenario_tuples)

        print(f"SCENARIO {scenario.debug_id} is valid {scenario.is_valid}")
              
        if scenario.is_valid:
            # Do not simulate invalid scenarios
            simulation_data = self.simulate(scenario, planning_problem_set)
        
        # Evaluate only valid scenario, simulation might invalidate them
        if scenario.is_valid:
            agent_data = simulation_data["agent-data"]
            evaluation_data = simulation_data["evaluation-data"]
            fitness_num_interactions = evaluation_data["is_interacting"].sum()
            fitness_num_cars = len(agent_data)
            interaction_map = simulation_data["interaction-map"]
            min_dist = sys.maxsize
            for i in range(1, (self.individual_number - (self.individual_number % self.pop_size)) + 1):
                dist = np.linalg.norm(interaction_map - self.interaction_maps.__getitem__(i - 1))
                min_dist = min(dist, min_dist)
            self.individual_number += 1
            self.interaction_maps.append(interaction_map)
            # maximize number of interactions
            # minimise the number of cars
            # maximize minimum distance
            out["F"] = [-fitness_num_interactions, fitness_num_cars, -min_dist]
            self.valid_scenarios = self.valid_scenarios + 1
        else:
            out["F"] = [sys.maxsize, sys.maxsize, sys.maxsize] # large values for a non-valid solution
            self.invalid_scenarios = self.invalid_scenarios + 1
        # out["F"] = [2, 3]
        print(out["F"])


    # This one execute one scenario only, no need to return a list with one single element
    def simulate(self, scenario, planning_problem_set):

        # Load the Scenario Executor - Hardcoded for the moment
        scenario_executor_path = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")
        scenario_executor_module = "evita.executors.frenetix_executor"
        scenario_executor_class = "FrenetixScenarioExecution"

        sys.path.append(scenario_executor_path)

        se_module = importlib.import_module(scenario_executor_module)
        se_class = getattr(se_module, scenario_executor_class)

        # Load the Scenario Evaluator - Hardcoded for the moment
        scenario_evaluator_path = "."
        scenario_evaluator_module = "evita.evaluators.frenetix_evaluator"
        scenario_evaluator_class = "FrenetixScenarioEvaluator"

        self.logger.debug(f"Add scenario evaluator to path {scenario_evaluator_path}")
        sys.path.append(scenario_evaluator_path)
        
        self.logger.debug(f"Try to import {scenario_evaluator_class} from {scenario_evaluator_module}")
        sev_module = importlib.import_module(scenario_evaluator_module)
        sev_class = getattr(sev_module, scenario_evaluator_class)

        scenario_executor: AbstractScenarioExecutor
        scenario_executor = se_class(self.output_folder, driving_agent_name=self.driving_agent_name)

        scenario_evaluator: AbstractScenarioEvaluator
        scenario_evaluator = sev_class(self.output_folder)

        exeuction_folder_name = str(scenario.generation_id).rjust(5, '0')
        execution_folder = os.path.join(self.output_folder, exeuction_folder_name)
        # debug_folder = os.path.join(self.output_folder, "debug")
        # os.makedirs(debug_folder, exist_ok=True)


        # Instantiate the scenario validator
        # scenario_validator_configuration = ValidationConfig()
        # scenario_validator = ScenarioValidator(scenario_validator_configuration)
        
        # Start the generation  + execution loop
        # scenario_validator.validate(scenario, planning_problem_set)

        # NOTE: This is useful for debug, in theory is the scenario generator who decides how long are the simulations or the timeout of each AV
        # TODO Enfore max duration of the scenario, i.e., until when the planners can reach their goal area
        # for planning_problem in planning_problem_set.planning_problem_dict.values():
        #     for goal_state in planning_problem.goal.state_list:
        #         if "time_step" in goal_state.attributes:
        #             goal_state.time_step = Interval(0, DEBUG_MAX_DURATION)

        if scenario.is_valid:
            # In some cases, the scenario is valid, but there are no
            try:
                self.logger.info(f"Executing scenario: {scenario.generation_id}")
                [ self.logger.debug(f"- {pp.planning_problem_id} - {pp.initial_state.position} - {pp.initial_state.velocity} - planned start at {pp.initial_state.time_step}") for pp in planning_problem_set.planning_problem_dict.values()]

                # TODO: It seems that sometime the ID of the lanelet for the planning problem is not correcly set/found?
                # TODO Extension can be also done by attaching a new lanelet if necessary, but that requires computing the right origentation and such..
                # Extend the scenario by adding some road AFTER the goal areas

                # Make sure we enlong the lanelets nearby goal areas or the planner will fail to reach them
                extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters=LANELET_EXTENSION)

                # Time Shift so at least one agent starts at time_step=0
                time_shifted_planning_problem_set = time_shift(scenario, planning_problem_set)

                # Make sure we execute the extended_scenario, not the original one
                # Both will be available afterwards, but the original shall be used for evolution
                simulation = scenario_executor.execute(extended_scenario, time_shifted_planning_problem_set)

                # Run evaluation and postprocessing outside the budget
                for agent in simulation.agents:
                    if agent.agent_state.status == AgentStatus.ERROR and agent.agent_state.last_timestep == 0:
                        self.logger.critical(f"The scenario is invalid. Agent {agent.id} did not even start")
                        scenario.is_valid = False
                        scenario.validation_message = "Agent {agent.id} could not compute a phyisically possible trajectory since the beginning."
                    
            except Exception as e:
                # We need to get rid of the execution folder!
                msg = f"A critical error prevented {scenario.generation_id} to complete. Mark it as Error and set the validity flag to False"
                self.logger.critical( "=" * ( len(msg) + 2 ) )
                self.logger.critical( f" {msg} ")
                self.logger.critical( "=" * ( len(msg) + 2) )
                scenario.is_valid = False
                scenario.validation_message = "Error during the simulation execution"

        # Evaluation
        if scenario.is_valid:
            try:
                self.logger.info(f"Evaluating scenario: {scenario.generation_id}")

                # we do not consider the time spent in collecting the metrics
                self.termination_criterion.update_elapsed_time()

                criticality_dataframe = scenario_evaluator.evaluate(simulation)
                self.termination_criterion.restart()
            
                # Compute the interaction map for the scenario
                interaction_map = compute_interaction_map(simulation, criticality_dataframe)
            
                return {
                    "individual": (scenario, planning_problem_set),
                    "agent-data": simulation.agents,
                    "evaluation-data": criticality_dataframe,
                    "interaction-map": interaction_map
                }
            except Exception as e:
                msg = f"A critical error prevented {scenario.generation_id} to be evaluated. Mark it as invalid"
                self.logger.critical( "=" * ( len(msg) + 2 ) )
                self.logger.critical( f" {msg} ")
                self.logger.critical( "=" * ( len(msg) + 2) )

                # If an error is triggered during the evaluation, get rid of the scenario
                # This should take care of those DSE error
                self.logger.critical(traceback.format_exc())
                # Try to move any folder away
                backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)
                return {
                    "individual": (scenario, planning_problem_set),
                    "agent-data": None,
                    "evaluation-data": None,
                    "interaction-map": None
                }
        elif scenario.validation_message == "Error during the simulation execution":
            # If the scenario is invalid because of an issue during the simulation, move the folder away
            backup_debug(self.output_folder, self.debug_folder, scenario, self.logger)
            return {
                    "individual": (scenario, planning_problem_set),
                    "agent-data": None,
                    "evaluation-data": None,
                    "interaction-map": None
            }
        else:
            # Invalid scenarios have no folder, no need to move it anywhere
            return {
                "individual": (scenario, planning_problem_set),
                "agent-data": None,
                "evaluation-data": None,
                "interaction-map": None
            }
        
class EVITADoppelTestNSGAIIproblem(NSGAIIproblem):
    def __init__(self, map_file_path, driving_agent_name, pop_size, termination_criterion, records_file_path, **kwargs):
        super().__init__(map_file_path, driving_agent_name, pop_size, termination_criterion, **kwargs)

        self.records_file_path = records_file_path
        self.scenario_validator = EVITAScenarioValidator(self.scenario_validator_configuration)

    def plot_pps(self, _scenario, planning_problems, output_folder: str):
        for pp in planning_problems:
            fig, ax = plt.subplots()
            rnd = MPRenderer(ax = ax)

            rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
            rnd.draw_params.lanelet_network.traffic_sign.scale_factor = 0.7
            rnd.draw_params.lanelet_network.traffic_light.scale_factor = 0.7
            rnd.draw_params.lanelet_network.lanelet.stop_line_color = "black"

            # Plot the road and the signals
            _scenario.lanelet_network.draw(rnd)

            rnd.draw_params.shape.facecolor = 'lightgreen'
            rnd.draw_params.shape.edgecolor = 'green'
            rnd.draw_params.shape.opacity = 0.6

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
                dpi = 800,            # very high resolution
                bbox_inches = "tight", # trims extra whitespace around the figure
                pad_inches = 0,        # no padding
                transparent = False,   # keep background white (or True if you want transparent)
            )
            plt.close()

    def _evaluate(self, x, out, *args, **kwargs):
        # Get scenario and planning problems to be evaluated     
        scenario = x[0]
        planning_problem_set = x[1]

        # ensure that the scenario has an unique number
        ensure_generation_id([(scenario, planning_problem_set)])

        current_generation = self.algorithm.n_gen
        current_individual = (self.valid_scenarios + self.invalid_scenarios) % self.pop_size        
        
        print(f"\n{'-' * 25} Evaluation of scenario {current_individual} {'-' * 25}\n")
        
        # This requires a list of tuples
        # Validate the generated scenarios. This will add the attribute: is_valid to the scenario object
        self.scenario_validator.validate_all([(scenario, planning_problem_set)])

        print(f"SCENARIO {current_individual} is valid {scenario.is_valid}")
              
        if scenario.is_valid:
            # Do not simulate invalid scenarios
            try:
                # Convert CommonRoad scenario to a Baidu Apollo compatible .json format
                json_scenario = build_json_scenario(scenario, deepcopy(planning_problem_set), self.map_file)

                # Save converted .json scenario to file
                output_folder_path = os.path.join(self.records_file_path.replace("/records/", "/evita-records/"), f'Generation_{current_generation}', f'Scenario_{current_individual}')
                os.makedirs(output_folder_path, exist_ok = True)

                evita_scenario_file_path = os.path.join(output_folder_path, f"evita_scenario.json")
                with open(evita_scenario_file_path, "w") as f:
                    json.dump(json_scenario, f, indent = 2)

                # Save CommonRoad scenario
                scenario_file_path = os.path.join(output_folder_path, f"scenario.xml")

                writer = CommonRoadFileWriter(
                    scenario,
                    planning_problem_set
                )

                writer.write_to_file(scenario_file_path)

                # Define python 3.8 environment and script to run scenario using Apollo
                python_38 = "../.venv38/bin/python"
                scenario_running_script = f"./apollo_scenario_runner.py"

                print(f"===== Started scenario execution with Apollo =====")
                print(f"===== Record files will be saved at: {os.path.join(self.records_file_path, f'Generation_{current_generation}', f'Scenario_{current_individual}')}")

                try:
                    # Launch script to run scenario with Apollo
                    # The script also converts the Apollo executed scenario to CommonRoad format and saves collisions, interactions, and perceptions data
                    _ = subprocess.run(
                        [python_38, scenario_running_script, evita_scenario_file_path, str(current_generation), str(current_individual), self.records_file_path],
                        check = True,
                    )
                except subprocess.CalledProcessError as e:
                    if e.returncode == 10:
                        raise RecordFileError("Record file issue") from e
                    
                    print("Subprocess failed!")
                    print("Return code:", e.returncode)
                    print("stdout:", e.stdout)
                    print("stderr:", e.stderr)
                    raise

                print(f"===== Completed scenario execution with Apollo =====")
                print(f"===== CommonRoad executed scenario saved at: {output_folder_path}\n")
                
                # Plot the scenario's planning problems
                planning_problems_plots_folder = os.path.join(output_folder_path, "planning_problems_plots")
                os.makedirs(planning_problems_plots_folder, exist_ok = True)
                self.plot_pps(scenario, planning_problem_set._planning_problem_dict.values(), planning_problems_plots_folder)
                
                # Open executed scenario
                executed_scenario_file_path = os.path.join(output_folder_path, "executed_scenario.xml")
                executed_scenario, _ = XMLFileReader(executed_scenario_file_path).open(lanelet_assignment = True)

                # Open scenario's collisions
                scenario_collisions_file_path = os.path.join(output_folder_path, "apollo_scenario_collisions.csv")
                scenario_collisions = pd.read_csv(scenario_collisions_file_path)
                
                # Open planned trajectories
                scenario_planned_trajectories_file_path = os.path.join(output_folder_path, "planned_trajectories.json")
                with open(scenario_planned_trajectories_file_path, "r") as f:
                    scenario_planned_trajectories = json.load(f)

                # Convert planned trajectories Dict to a DataFrame
                scenario_planned_trajectories_df = planned_trajectories_to_df(scenario_planned_trajectories)

                # Compute scenario's collisions
                collisions_df, issue_rows = augment_with_collisions(executed_scenario, scenario_collisions)

                # Save scenario's collisions
                scenario_collisions_file_path = os.path.join(output_folder_path, "collisions.csv")
                collisions_df.to_csv(scenario_collisions_file_path, index = False)

                # Save issue rows
                if len(issue_rows) > 0:
                    issue_rows_df = pd.DataFrame(issue_rows, columns = scenario_collisions.columns.tolist() + ["detected_obstacles"])
                    issue_rows_df.to_csv(os.path.join(output_folder_path, "invalid_collisions.csv"), index = False)

                # Open scenario's vehicles perceptions
                scenario_vehicles_perceptions_file_path = os.path.join(output_folder_path, "vehicles_perception.csv")
                scenario_vehicles_perceptions = pd.read_csv(scenario_vehicles_perceptions_file_path)

                # Open scenario's traffic lights perceptions
                scenario_traffic_lights_perceptions_file_path = os.path.join(output_folder_path, "traffic_lights_perceptions.csv")
                scenario_traffic_lights_perceptions = pd.read_csv(scenario_traffic_lights_perceptions_file_path)
                                                                              
                # Compute scenario's interactions and the interaction map
                interactions_df, interactions_map = augment_with_interactions(executed_scenario, 
                                                                              scenario_planned_trajectories_df, 
                                                                              scenario_vehicles_perceptions, 
                                                                              scenario_traffic_lights_perceptions,
                                                                              scenario_collisions
                                                                            )

                # Save scenario's interactions and interactions map to file
                interactions_df.to_csv(f"{output_folder_path}/interactions.csv", index = False)
                interactions_map_df = pd.DataFrame(interactions_map)
                interactions_map_df.to_csv(f"{output_folder_path}/interactions_map.csv", index = False, header = False)

                # Compute and return scenario's fitness value
                if interactions_df is None or interactions_map is None:
                    out["F"] = [sys.maxsize, sys.maxsize, sys.maxsize] # large values for a non-valid solution
                    self.invalid_scenarios = self.invalid_scenarios + 1
                else:
                    agents_data = executed_scenario.dynamic_obstacles
                    fitness_num_interactions = interactions_df["is_interacting"].sum()
                    fitness_num_cars = len(agents_data)
                    
                    min_dist = sys.maxsize
                    for i in range(1, (self.individual_number - (self.individual_number % self.pop_size)) + 1):
                        dist = np.linalg.norm(interactions_map - self.interaction_maps.__getitem__(i - 1))
                        min_dist = min(dist, min_dist)

                    self.individual_number += 1
                    self.interaction_maps.append(interactions_map)

                    # maximize number of interactions
                    # minimise the number of cars
                    # maximize minimum distance
                    out["F"] = [-fitness_num_interactions, fitness_num_cars, -min_dist]
                    self.valid_scenarios = self.valid_scenarios + 1
            except (ValueError, RecordFileError):
                out["F"] = [sys.maxsize, sys.maxsize, sys.maxsize] # large values for a non-valid solution
                self.invalid_scenarios = self.invalid_scenarios + 1      

                pass 

        # out["F"] = [2, 3]
        print(f'\nFitness Value: {out["F"]}')
        print(f"\n{'-' * 150}\n")