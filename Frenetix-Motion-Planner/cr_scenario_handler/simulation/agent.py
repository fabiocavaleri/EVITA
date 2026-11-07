__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# standard imports
import inspect
import os
import time
from copy import deepcopy

# third party
import commonroad_dc.pycrcc as pycrcc
from commonroad.planning.planning_problem import PlanningProblem

# commonroad-io
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory

# scenario handler
import cr_scenario_handler.planner_interfaces as planner_interfaces
import cr_scenario_handler.utils.multiagent_helpers as hf
import cr_scenario_handler.utils.prediction_helpers as ph
import cr_scenario_handler.utils.visualization as visu
from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface
from cr_scenario_handler.evaluation.collision_report import coll_report
from cr_scenario_handler.evaluation.agent_evaluation import evaluate
from cr_scenario_handler.utils.agent_status import AgentStatus, AgentState
from cr_scenario_handler.utils.visualization import visualize_agent_at_timestep


class Agent:

    def __init__(self, agent_id: int, planning_problem: PlanningProblem,
                 scenario: Scenario, config_planner, config_sim, msg_logger, log_path: str, mod_path: str):
        """Represents one agent of a multiagent or single-agent simulation.

        EVITA: There's an issue with the predictions of agents if their initial_state.time_step != 0.
        So we force it to be 0, but keep the agent in "pause" until then, so the agent is indeed active, but disabled

        Manages the agent's local view on the scenario, the planning problem,
        planner interface, collision detection, and per-agent plotting and logging.
        Contains the step function of the agent.

        :param agent_id: The agent ID, equal to the obstacle_id of the
            DynamicObstacle it is represented by.
        :param planning_problem: The planning problem of this agent.
        :param scenario: The scenario to be solved. May not contain the ego obstacle.
        :param config_planner: The configuration object for the trajectory planner.
        :param config_sim: The configuration object for the simulation framework
        :param msg_logger: Logger for msg-printing
        :param log_path: Path for logging and visualization.
        :param mod_path: Working directory of the planner.
        """
        # Agent id, equals the id of the dummy obstacle
        self.dt = scenario.dt
        self.id = agent_id

        self.msg_logger = msg_logger

        self.config = deepcopy(config_sim)
        self.config_simulation = self.config.simulation
        self.config_visu = self.config.visualization
        self.config_planner = deepcopy(config_planner)
        self.vehicle = config_sim.vehicle

        self.mod_path = mod_path
        self.log_path = os.path.join(config_sim.simulation.log_path, str(agent_id)) if (
                                     self.config_simulation.use_multiagent) else log_path

        self.save_plot = (self.id in self.config_visu.save_specific_individual_plots
                          or self.config_visu.save_all_individual_plots)
        self.gif = (self.config_visu.save_all_individual_gifs
                    or self.id in self.config_visu.save_specific_individual_gifs)
        self.show_plot = (self.id in self.config_visu.show_specific_individual_plots
                          or self.config_visu.show_all_individual_plots)

        try:
            self.max_time_steps_scenario = int(
                self.config_simulation.max_steps * planning_problem.goal.state_list[0].time_step.end)
        except NameError:
            self.max_time_steps_scenario = 200
        self.msg_logger.debug(f"Agent {self.id}: Max time steps {self.max_time_steps_scenario}")

        self.planning_problem = planning_problem

        self.scenario = hf.scenario_without_obstacle_id(scenario=deepcopy(scenario), obs_ids=[self.id])

        # TODO CR-Reach/Spot/ Occlusion / Sensor

        self.planning_times = list()

        self.predictions = None
        self.visible_area = None

        # Initialize Planning Problem
        problem_init_state = deepcopy(planning_problem.initial_state)
        
        if not hasattr(problem_init_state, 'acceleration'):
            problem_init_state.acceleration = 0.
        
        x_0 = deepcopy(problem_init_state)

        self.collision_objects = list()
        self._create_collision_object(x_0, problem_init_state.time_step)

        self._all_trajectories = None
        # Initialize Planner
        used_planner = self.config_simulation.used_planner_interface

        try:
            planner_interface = [cls for _, module in inspect.getmembers(planner_interfaces, inspect.ismodule)
                                 for name, cls in inspect.getmembers(module, inspect.isclass) if
                                 issubclass(cls, PlannerInterface)
                                 and name == used_planner][0]
        except:
            raise ModuleNotFoundError(f"No such planner class found in planner_interfaces: {used_planner}")
        
        self.planner_interface = planner_interface(self.id, self.config_planner, self.config, self.scenario,
                                                   self.planning_problem, self.log_path, self.mod_path)

        self.agent_state = AgentState(planning_problem=self.planning_problem, reference_path=self.reference_path,
                                      coordinate_system=self.coordinate_system)
        
        self.already_collided = False

        # Here's the thing that makes everything in motion
        if planning_problem.initial_state.time_step == 0:
            self.agent_state.log_running(0)


    @property
    def computed_trajectories(self):
        return self.planner_interface.computed_trajectories

    @property
    def reference_path(self):
        return self.planner_interface.reference_path

    @property
    def record_input_list(self):
        return self.planner_interface.record_input_list

    @property
    def record_disengagement(self):
        """
        Sometimes the planner cannot find any physically possible trajectory - leading to a disengagement
        """
        return self.planner_interface.record_disengagement
    
    @property
    def record_emergency(self):
        """
        Sometimes the planner cannot find an optimal trajectory and switches to emergency mode
        """
        return self.planner_interface.record_emergency
    

    @property
    def record_state_list(self):
        return self.planner_interface.record_state_list

    @property
    def vehicle_history(self):
        return self.planner_interface.vehicle_history

    @property
    def coordinate_system(self):
        return self.planner_interface.coordinate_system

    @property
    def all_trajectories(self):
        return self._all_trajectories if self._all_trajectories is not None else self.planner_interface.all_trajectories

    @all_trajectories.setter
    def all_trajectories(self, traj):
        self._all_trajectories = traj

    @property
    def status(self):
        return self.agent_state.status

    @property
    def current_timestep(self):
        return self.agent_state.last_timestep

    def update_agent(self, scenario: Scenario, time_step: int, global_predictions: dict,
                     collision: bool = False, colliding_agents: list = []):
        """ 
        Update the scenario to synchronize the agents.


        :param scenario:
        :param time_step:
        :param global_predictions:
        :param collision:
        """
        self.agent_state.log_running(time_step)

        if not collision:
            if self.config_simulation.use_multiagent:
                self.scenario = hf.scenario_without_obstacle_id(scenario=deepcopy(scenario), obs_ids=[self.id])
            if global_predictions:
                self.predictions, self.visible_area = ph.filter_global_predictions(self.scenario, global_predictions,
                                                                                   self.vehicle_history[-1],
                                                                                   time_step,
                                                                                   self.config,
                                                                                   occlusion_module=self.planner_interface.occlusion_module,
                                                                                   ego_id=self.id,
                                                                                   msg_logger=self.msg_logger)
            else:
                self.predictions, self.visible_area = None, None
        else:
            # This avoids inconsistency in collision reporting (see tests/data/flaky-collisions.xml)
            self.scenario = hf.scenario_without_obstacle_id(scenario=deepcopy(scenario), obs_ids=[self.id])
            
            if self.agent_state.status == AgentStatus.COLLISION:
                self.msg_logger.debug(f"Agent {self.id} already collided at time_step {time_step}")    
                # This triggers one step after colliding
                self.already_collided = True
            else:
                self.msg_logger.debug(f"Agent {self.id} collided at time_step {time_step}. Colliding agents {colliding_agents}")
                # Set the status of the agent to COLLISION and the last time step to the current one
                # But what's the status of the agent at this point?
                self.agent_state.log_collision(time_step)
                # TODO Is this really necessary?
                self.colliding_agents = colliding_agents
                

    def step_agent(self, timestep):
        """
        Execute one planning step.
        """
        #
        # Check for collisions in previous timestep - done only at collision time
        #
        if self.agent_state.status == AgentStatus.COLLISION and not self.already_collided:
            self.planning_times.append(0)
            # Report the collision ONLY if its a new collision
            if self.config.evaluation.collision_report and not self.already_collided:
                # TODO It does not report the collision of the second vehicle?
                # It does report the wrong value for the collision speed of the obstacle
                
                #Report the collisions in the run folder, not inside the the agent path!
                collision_folder = os.path.dirname(self.log_path)

                coll_report(self.vehicle_history, self.planner_interface.planner, self.scenario, self.planning_problem,
                            self.agent_state.last_timestep, self.config, collision_folder, self.colliding_agents)
                
            self.postprocessing()

            self.already_collided = True
        
        # Note this one. When the agent collides, it's current state is not computed, leaving an hole in the dataset that breaks the evaluation

        if timestep > self.max_time_steps_scenario and self.agent_state.status != AgentStatus.COLLISION:
            # Do not log a different requirement violation if already collided
            self.planning_times.append(0)
            self.agent_state.log_timelimit(timestep)
            self.postprocessing()

        elif self.planner_interface.planner.x_cl[0][0] > self.agent_state.goal_checker.last_goal_position and self.agent_state.status != AgentStatus.COLLISION:
            # Do not log a different requirement violation if already collided
            self.agent_state.log_max_s_position(timestep)
            self.planning_times.append(0)
            self.postprocessing()
        
        else:
            # Otherwise, the planner is either running or init ?
            # check for completion of this agent - 
            collided = self.agent_state.status == AgentStatus.COLLISION
            # this changes the AgentStatus.RUNNING and overwrite the collision state
            self.agent_state.check_goal_reached(self.record_state_list, self.planner_interface.planner.x_cl)
            # Ensure collisions are preserved
            self.agent_state.status = AgentStatus.COLLISION if collided else self.agent_state.status

            if self.agent_state.status not in [AgentStatus.RUNNING, AgentStatus.COLLISION]:
                self.planning_times.append(0)
                self.agent_state.log_finished(timestep)
                self.postprocessing()
            
            else:
                self.msg_logger.debug(f"Agent {self.id} planning current time step {timestep}")

                comp_time_start = time.time()

                if self.agent_state.status == AgentStatus.COLLISION and self.already_collided:
                    self.msg_logger.debug(f"Agent {self.id}: Already crashed at current time step: {timestep}. Computing its collided trajectory")
                    
                    # Replan the agent trajectory such that it stays still 

                    # state_list : List[TraceState]
                    initial_time_step = timestep
                    # Take the latest state and replicate
                    
                    # TODO There might be an hole corresponding to the timestep of the collision...

                    # Check that is the same timestemp! record_state_list[-1] == timestep -1 !
                    # assert timestep - 1 == self.record_state_list[-1].time_step, f"timestep {timestep} but last recorded state {self.record_state_list[-1]}"
                    current_state = deepcopy(self.record_state_list[-1])
                    state_list = list()
                    for i in range(int(self.config_planner.planning.planning_horizon / self.config_planner.planning.dt) + 1):
                        current_state = deepcopy(self.record_state_list[-1])
                        # From the initial time count the necessasry steps to build the trajectory
                        # Will this also update the computed_trajectory and the rest?
                        current_state.time_step = initial_time_step + i
                        current_state.velocity = 0.0
                        current_state.acceleration = 0.0
                        current_state.yaw_rate = 0.0

                        state_list.append(current_state)

                    trajectory =  Trajectory(initial_time_step, state_list)
                    replanning_counter = 0

                    # Force the planner_interface state as it would do with step_interface
                    # Maybe this is enough? it
                    # record the new state for planner-internal logging
                    self.planner_interface.planner.record_state_and_input(trajectory.state_list[1])

                    # record the entire trajectory
                    self.planner_interface.planner.record_computed_trajectory(trajectory)

                    current_ego_vehicle = self.planner_interface.planner.convert_state_list_to_commonroad_object(trajectory.state_list[replanning_counter:],
                                                                                                                 self.config_simulation.ego_agent_id)

                    # This one sets the ego_vehicle history which is used (also) for plotting        
                    self.planner_interface.planner.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

                    # This requires the curvilinear trajectory for logging
                    # self.planner_interface.planner.plan_postprocessing(optimal_trajectory=trajectory, planning_time=0.0, replanning_counter=replanning_counter)
                    
                else:
                    # **************************
                    # Cycle Occlusion Module
                    # **************************
                    # if config.occlusion.use_occlusion_module:
                    #     occlusion_module.step(predictions=predictions, x_0=planner.x_0, x_cl=planner.x_cl)
                    #     predictions = occlusion_module.predictions

                    # **************************
                    # Set Planner Subscriptions
                    # **************************

                    # TODO Are the agents that reached goal area still in this scenario!?
                    # Set the agent.already_collided flag
                    # Do they set history?
                    self.planner_interface.update_planner(self.scenario, self.predictions)

                    # **************************
                    # Execute Planner
                    # This includes computing a standing still trajectory until initial delay is passed
                    # **************************
                    # comp_time_start = time.time()
                    trajectory, replanning_counter = self.planner_interface.step_interface(timestep)

                #### At this point we computed - trajectory, replanning_counter
                comp_time_end = time.time()

                # END TIMER
                self.planning_times.append(comp_time_end - comp_time_start)
                self.msg_logger.info(f"Agent {self.id}: Total Planning Time: \t\t{self.planning_times[-1]:.5f} s")

                if trajectory:
                    self._create_collision_object(self.vehicle_history[-1].initial_state,
                                                self.vehicle_history[-1].initial_state.time_step)
                    self.agent_state.log_running(timestep)

                    # plot own view on scenario
                    if (self.save_plot or self.show_plot or self.gif or ((self.config_visu.save_plots or
                                                                        self.config_visu.show_plots) and
                                                                        not self.config.simulation.use_multiagent)):
                        visualize_agent_at_timestep(self.scenario, self.planning_problem,
                                                    self.vehicle_history[-1], timestep,
                                                    self.config, self.log_path,
                                                    traj_set=self.all_trajectories,
                                                    optimal_traj=self.planner_interface.trajectory_pair[0],
                                                    ref_path=self.planner_interface.reference_path,
                                                    predictions=self.predictions,
                                                    visible_area=self.visible_area,
                                                    plot_window=self.config_visu.plot_window_dyn, save=self.save_plot,
                                                    show=self.show_plot, gif=self.gif,
                                                    replanning_counter=replanning_counter)

                else:
                    self.msg_logger.critical(f"Agent {self.id}: No Kinematic Feasible and Optimal Trajectory Available!")
                    self.agent_state.log_error(timestep)
                    self.postprocessing()

    def postprocessing(self):
        """ Execute post-simulation tasks.

        Create a gif from plotted images, and run the evaluation function.
        """

        self.msg_logger.info(f"Agent {self.id}: timestep {self.agent_state.last_timestep}: {self.agent_state.message}")
        self.msg_logger.debug(f"Agent {self.id} current state : {self.agent_state.status}")
        self.msg_logger.debug(f"Agent {self.id} current goal message: {self.agent_state.goal_message}")
        self.msg_logger.debug(f"Agent {self.id}: {self.agent_state.goal_checker_status}")

        if self.config.evaluation.evaluate_agents:
            evaluate(self.scenario, self.planning_problem, self.id, self.record_state_list, self.record_input_list,
                     self.config, self.log_path,
                     )

        # plot final trajectory
        show = (self.config_visu.show_all_individual_final_trajectories or
                self.id in self.config_visu.show_specific_final_trajectories)
        save = (self.config_visu.save_all_final_trajectory_plots or
                self.id in self.config_visu.save_specific_final_trajectory_plots)
        if show or save:
            visu.plot_final_trajectory(self.scenario, self.planning_problem, self.record_state_list,
                                       self.config, self.log_path, ref_path=self.reference_path, save=save, show=show)

    def make_gif(self):
        # make gif
        if self.gif:
            visu.make_gif(self.scenario,
                          range(self.planning_problem.initial_state.time_step,
                                self.agent_state.last_timestep),
                          self.log_path, duration=0.1)

    def _create_collision_object(self, state, timestep):
        ego = pycrcc.TimeVariantCollisionObject(timestep)
        ego.append_obstacle(pycrcc.RectOBB(0.5 * self.config.vehicle.length, 0.5 * self.config.vehicle.width,
                                           state.orientation,
                                           state.position[0],
                                           state.position[1]))
        self.collision_objects.append(ego)

