from ..interfaces import AbstractScenarioExecutor

import os
import sys
import logging

import traceback
from datetime import datetime

from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder

from commonroad.visualization.mp_renderer import MPRenderer

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_reader import CommonRoadFileReader

from ..utils import FRENETIX_ROOT, EVITA_MAIN_LOGGER_NAME

def _get_cmap(n, name = 'hsv'):
    """
    Get a function that maps each index in 0, 1, ..., n - 1 to a distinct RGB color
    the keyword argument name must be a standard mpl colormap name
    """
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(name, n)

class PatchedConfigurationBuilder(ConfigurationBuilder):
    """
    This is a patched version of the ConfigurationBuilder that does not consider the commandline configuration, since in the stitched execution we want to have a fixed configuration for all the tranches and not consider the commandline configuration that may be different for each tranche
    """
    @classmethod
    def build_configuration(cls, scenario_name: str, path_root: str = None, dir_config = "configurations", module = "simulation"):
        """
        Builds configuration from default, scenario-specific, but NOT considering the commandline config

        Args:
            scenario_name (str): considered scenario
            path_root (str): root path of the package
            dir_config (str): folder of configs
            module (str): module name of config
        """

        # Set the paths for the configuration builder
        if path_root is None:
            path_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))

        cls.set_paths(path_root = path_root, dir_config = dir_config, dir_config_default = module)

        config_default = cls.construct_configuration(scenario_name)

        return config_default
    
import copy
import time
from queue import Empty
from typing import List
# from cr_scenario_handler.simulation.simulation import Simulation
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from cr_scenario_handler.utils.agent_status import TIMEOUT, AgentStatus
import cr_scenario_handler.utils.prediction_helpers as ph

class EVITASimulation(Simulation):
    """
    Custom subclass of the EVITA Simulation to prevent the predictor from 
    crashing on empty histories when agents collide/finish.
    """

    def __init__(self, config_sim, config_planner):
        super().__init__(config_sim, config_planner)
        
        # Ensure the log directory exists for all batches
        log_dir = os.path.join(self.mod_path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok = True)

    def _step_parallel_simulation(self):
            """
            Does a single simulation step in the multi-process setting:
            - In the prestep, compute collisions based on the state computed at t - 1 about vehicles at time t (planned next state). t should be now self.global_timestep)
            - In the prestep, update the scenario with the state planned at time t
            - In the prestep, transform AV to NPC for those who have collided at time t based on their computed state at time t - 1
            - In the prestep, all the collided must have a stationary trajectory
            - In the step_simulation, send the updated scenario to all the agents and collect their planned states

            Enforce that the simulation is reported as finished also if the provided timeout triggers.
            Note. we recompute the self.replanning_counter because the original implementation is buggy.
            See: https://github.com/TUM-AVS/Frenetix-Motion-Planner/issues/8
            We also fix the issue with visualization of time_step = -1
            """
            # Ensure we update the replanning counter to avoid predicting at every step, which currently
            # expect to be synched with planner effective planning cycle
            if int(self.replanning_counter / self.config_planner.planning.replanning_frequency) == 1:
                self.replanning_counter = 0

            # Patched version of the original code
            # Update the scenario, calculate new predictions and check for collision of previous iteration
            # We also enforce StationaryTrajectories for agents and NPCs involved in collisions
            self.global_predictions, colliding_agents = self.prestep_simulation()

            # Send data to each batch process
            for batch in self.batch_list:
                batch.in_queue.put([self.scenario, self.global_timestep, self.global_predictions, colliding_agents])

            # We plot the status of the simulation at the previous time, when collisions are detected and scenario updated
            if self.global_timestep > 0:
                self.visualize_simulation(self.global_timestep - 1)
        
            syn_time = time.perf_counter()

            # Receive simulation step results
            for batch in reversed(self.batch_list):
                # If the event is set, it means that one of the child processes has failed and we need to terminate the simulation
                # Otherwise, we wait for the batch to send us the results of the simulation step
                if self.event.is_set():
                    self.msg_logger.error(f"Simulation received a termination event from child processes!")
                    raise ChildProcessError(f"Simulation received a termination event from child processes!")
                try:
                    queue_dict = batch.out_queue.get(block = True, timeout = TIMEOUT)

                    queue_agent_ids = list(queue_dict.keys())
                    actual_agent_ids = [agent.id for agent in self.agents]
                    agent_ids = [agent_id for agent_id in queue_agent_ids if agent_id in actual_agent_ids]

                    # Update the agents with the results from the batch
                    for agent_id in agent_ids:
                        agent = self.get_agent_by_id(agent_id)
                        for attr, value in queue_dict[agent_id].items():
                            if hasattr(agent, attr):
                                if type(getattr(agent, attr)) != list:
                                    setattr(agent, attr, value)
                                elif attr == "all_trajectories":
                                    setattr(agent, attr, value)
                                else:
                                    getattr(agent, attr).append(value)
                            else:
                                self.event.set()
                                raise AttributeError(f"{attr} is no valid attribute for the agents")
                    
                    # Check if batch is finished and log the processing times
                    [batch_finished, proc_times] = batch.out_queue.get(block = True, timeout = TIMEOUT)
                    self.msg_logger.critical(f"Simulation received batch infos from batch {batch.name}")
                    self.process_times[batch.name] = proc_times

                    # If simulation is finished, we need to close the batch and remove it from the list of active batches
                    if batch_finished:
                        self.msg_logger.critical(f"Start closing Batch {batch.name}")
                        
                        # Send the terminate event to the batch
                        batch.in_queue.put("END", block = True)
                        
                        # Wait for it to die
                        batch.join()
                        
                        # Remove the batch from the observed ones
                        self.batch_list.remove(batch)
                        
                        # Remove all the vehicle in this batch from the simulation
                        self.msg_logger.critical(f"Closing Batch {batch.name}")
                except Empty:
                    if self.event.is_set():
                        # An error occured in subprocesses
                        self.msg_logger.error(f"Simulation received a termination event from child processes!")
                        raise ChildProcessError(f"Simulation received a termination event from child processes!")
                    
                    self.msg_logger.error(f"Timeout while waiting for step results of batch {batch.name}!")
                    self.event.set()
                    raise Empty(f"Timeout while waiting for step results of batch {batch.name}!")
            
            # Update the processing time for the synchronization step
            self.process_times["time_sync"] = time.perf_counter() - syn_time

            # Ensure we update the replanning counter to avoid predicting at every step, which currently
            # expect to be synched with planner effective planning cycle
            self.replanning_counter += 1    

            '''
            # Ensure we transform collided agents into NPCs.
            for collided_agent_id in colliding_agents:
                agent = next(iter([a for a in self.agents if a.id == collided_agent_id]))
                self.colliding_agents.append(agent)
                self.agents.remove(agent)
            
            # We need to keep the simulation running if there are NPCs and the global timeout has not triggered
            any_active_npc = any([o.prediction.trajectory.state_list[-1].time_step >= self.expected_last_time_step for o in self.scenario.obstacles if o.obstacle_id not in self.agent_id_list])
            '''

            return len(self.batch_list) > 0

    def update_scenario(self, colliding_agents: List[int]):
        # 1. Let the original class do its normal update
        super().update_scenario()
        
        # 2. Apply the stationary padding for non-running agents
        if self.global_timestep == 0:
            return
            
        for agent_id in colliding_agents:
            obstacle = self.scenario.obstacle_by_id(agent_id)
            
            # The base class truncated this trajectory to the crash timestep.
            # We need to extract it and pad it up to the current global_timestep.
            state_list = obstacle.prediction.trajectory.state_list
            
            if state_list:
                last_state = state_list[-1]
                
                # If the trajectory doesn't reach the current simulation timestep, pad it
                if last_state.time_step < self.global_timestep:
                    for t in range(last_state.time_step + 1, self.global_timestep + 1):
                        # Create a stationary state (same position/orientation as the crash)
                        stationary_state = copy.deepcopy(last_state)
                        stationary_state.time_step = t
                        
                        # Zero out velocities and accelerations to simulate a stopped vehicle
                        if hasattr(stationary_state, 'velocity'):
                            stationary_state.velocity = 0.0
                        if hasattr(stationary_state, 'acceleration'):
                            stationary_state.acceleration = 0.0
                        if hasattr(stationary_state, 'yaw_rate'):
                            stationary_state.yaw_rate = 0.0
                            
                        state_list.append(stationary_state)
                        
                    # Re-assign the padded prediction to the obstacle
                    shape = obstacle.obstacle_shape
                    lanelet_assigned = obstacle.prediction.center_lanelet_assignment
                    obstacle.prediction = TrajectoryPrediction(
                        Trajectory(initial_time_step = state_list[0].time_step, state_list=state_list), 
                        shape, 
                        center_lanelet_assignment = lanelet_assigned
                    )

    def prestep_simulation(self):
        """
        Performs all steps necessary before a trajectory planning step:
        - Collision check
        - Prediction calculation
        - Global scenario update

        Exactly as super, but call the "local" update scenario, passing the colliding agents
        """

        # Retrieve the current timestep of the simulation and log it
        preproc_time = time.perf_counter()
        self.msg_logger.critical(f"Scenario {self.scenario.scenario_id} in timestep {self.global_timestep}")

        # Remove from the scenario agents that have completed the simulation
        ids_of_agents_to_remove = [agent.id for agent in self.agents if agent.status not in [AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.COLLISION]]
        obstacles_to_remove = [obstacle for obstacle in self.scenario.obstacles if obstacle.obstacle_id in ids_of_agents_to_remove]
        if len(obstacles_to_remove) > 0:
            for obstacle in obstacles_to_remove:
                # Copy the obstacle history until now to save it
                shape = obstacle.obstacle_shape
                lanelet_assigned = obstacle.prediction.center_lanelet_assignment
                keep_state_list = [state for state in obstacle.prediction.trajectory.state_list if state.time_step < self.global_timestep]
                
                obstacle.prediction = TrajectoryPrediction(Trajectory(initial_time_step = keep_state_list[0].time_step, state_list = keep_state_list), shape,
                                                            center_lanelet_assignment = lanelet_assigned)
        
        # At this point, we can check collisions with the remaining agents and NPCs        
        # Report the id of any obstacle involved in any collision
        colliding_agents = self.check_collision()
        
        # Update scenario forcing the colliding agents to become NPC
        self.update_scenario(colliding_agents)

        # Compute predictions for all the agents, if we are at the beginning of the simulation or if we are not replanning at every step
        if ((self.replanning_counter == 0 or self.config_planner.planning.replanning_frequency < 2) or self.config.behavior.use_behavior_planner):
            predictions = ph.get_predictions(self.config, self._predictor, self.scenario, self.global_timestep, self.prediction_horizon)
        else:
            predictions = None

        self.process_times["preprocessing"] = time.perf_counter() - preproc_time

        return predictions, colliding_agents
    
class FrenetixScenarioExecution(AbstractScenarioExecutor):
    """
    This class is responsible for executing a scenario using the Frenetix motion planner
    """
    def __init__(self, output_folder:str, **kwargs) -> None:           
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
        if "debug" in kwargs and kwargs["debug"]:
            self.debug = kwargs["debug"]
        else:
            self.debug = False

        self.output_folder = output_folder
        self.execution_counter = 0
        self.driving_agent_name = kwargs["driving_agent_name"]

        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
        
    def visualize_the_scenario(self, scenario, planning_problem_set, output_folder):
        """
        Visualize and plot the initial scenario
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        # Sort the planning problems by their ID to ensure consistent coloring
        planning_problems = sorted(planning_problem_set.planning_problem_dict.values(), key = lambda pp : pp.planning_problem_id)
        # Get a unique color for each planning problem
        unique_colors = _get_cmap(len(planning_problems), 'rainbow')

        # Initialize the renderer
        rnd = MPRenderer(figsize = (20, 10))

        # Set renderer draw params
        rnd.draw_params.time_begin = 0
        rnd.draw_params.dynamic_obstacle.draw_icon = True

        # Draw Road and Obstacles at time stamp
        scenario.draw(rnd)
        
        # Iterate over the planning problems and draw them with unique colors
        for i, planning_problem in enumerate(planning_problems):
            from commonroad.planning.planning_problem import PlanningProblem
            planning_problem: PlanningProblem
            # Set renderer draw params
            rnd.draw_params.planning_problem.initial_state.state.facecolor = colors.to_hex(unique_colors(i))
            rnd.draw_params.planning_problem.initial_state.state.draw_arrow = True
            rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
            
            # Add labels
            rnd.draw_params.planning_problem.initial_state.label= f"{planning_problem.planning_problem_id} - {planning_problem.initial_state.velocity}"

            rnd.draw_params.planning_problem.goal_region.shape.facecolor = colors.to_hex(unique_colors(i))
            rnd.draw_params.planning_problem.goal_region.shape.opacity = 0.5
                
            # Plot the planning problem
            planning_problem.draw(rnd)

        # Render the plot
        rnd.render()

        # Save the plot to a file
        plt.savefig(f"{output_folder}/scenario.png", format = 'png', dpi = 500, bbox_inches = 'tight')

    def execute(self, scenario, planning_problem_set) -> Simulation:
        """
        Return the original scenario, the simulated scenario, and additional information about the AVs
        """

        # Increment the execution counter for each execution
        self.execution_counter = self.execution_counter + 1 

        # Create the folders for the execution using the generation_id
        execution_folder_name = str(scenario.generation_id).rjust(5, '0')
        execution_folder = os.path.join(self.output_folder, execution_folder_name)
        os.makedirs(execution_folder, exist_ok=False)

        # Store here the input scenario, configurations, default configurations
        scenario_folder = os.path.join(execution_folder, "scenario")
        os.makedirs(scenario_folder, exist_ok=False)

        # Build the scenario file name and file path
        scenario_name = f"scenario_{execution_folder_name}"
        scenario_file_name = os.path.join(scenario_folder, f"{scenario_name}.xml")

        # Collect here the execution logs
        logs_folder = os.path.join(execution_folder, "logs")
        os.makedirs(logs_folder, exist_ok=False)
        
        # Load the default simulation configuration from the Frenetix Motion Planner configuration module
        config_sim = PatchedConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, root_path = FRENETIX_ROOT, module = "simulation")

        config_sim.simulation.use_multiagent = True
        # Make sure we write the logs in the right folder
        config_sim.simulation.log_path = logs_folder
        
        # Load the default planner configuration from the Frenetix Motion Planner configuration module
        config_planner = PatchedConfigurationBuilder.build_frenetplanner_configuration(scenario_name, module = "frenetix_motion_planner", root_path = FRENETIX_ROOT)
        config_planner.debug.use_cpp = True

        # Add debugging options if the debug flag is set
        if self.debug:
            config_sim.visualization.save_plots = True
            config_sim.visualization.save_gif = True

        # Iterate over scenario's planning problems
        for planning_problem in planning_problem_set.planning_problem_dict.values():
            original_initial_time = planning_problem.initial_state.time_step
            adjusted_initial_time = original_initial_time
            # Increment vehicle initial timestep until it matches the planner replanning frequency
            while adjusted_initial_time % config_planner.planning.replanning_frequency != 0:
                adjusted_initial_time = adjusted_initial_time + 1

            # If the initial timestep is modified, adjust it and log it
            if adjusted_initial_time != original_initial_time:
                self.logger.critical(f"Adjust the starting time of agent {planning_problem.planning_problem_id} from {original_initial_time} to {adjusted_initial_time}")
                planning_problem.initial_state.time_step = adjusted_initial_time

        # Write the scenario to a CommonRoad file
        # CommonRoad truncate floating point numbers, so we need to ensure that we are indeed executing what we generate
        # Or at least be consistent to avoid reproducibility issues
        crfw = CommonRoadFileWriter(
            scenario,
            planning_problem_set,
            author = "EVITA",
            decimal_precision = 10)
        crfw.write_to_file(scenario_file_name)

        # To enforce consistency we re-read what we just wrote to file
        crfr = CommonRoadFileReader(scenario_file_name)
        scenario, planning_problem_set = crfr.open()

        # Plot the scenario for reference
        self.visualize_the_scenario(scenario, planning_problem_set, scenario_folder)

        # All the configurations used for this execution will be stored in simulation.db, table meta
        simulation = None

        try:
            # If any agent fails to start, this will trigger an exception, and stop the simulation
            simulation = EVITASimulation(config_sim, config_planner)
            simulation.run_simulation()
        except Exception as e:
            error_traceback = traceback.format_exc()  # This gets the entire error traceback
            current_time = datetime.now().strftime('%H:%M:%S')
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            print("Error time: " + str(current_time) + " ; " +
                  "In Scenario Timestep: " + current_timestep + " ; " +
                  "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n")
            
            return None
        finally:
            if simulation is not None:
                try:
                    # close sim_logger
                    # simulation.sim_logger.con.close()
                    # close child processes
                    simulation.close_processes()
                except:
                    pass
        
        return simulation