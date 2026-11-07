__author__ = "Rainer Trauth, Marc Kaufeld"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

from copy import deepcopy
import numpy as np
import logging

from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad_route_planner.route_planner import RoutePlanner

import cr_scenario_handler.utils.multiagent_logging as lh
from behavior_planner.behavior_module import BehaviorModule
from cr_scenario_handler.planner_interfaces.planner_interface import PlannerInterface
import cr_scenario_handler.utils.goalcheck as gc
from cr_scenario_handler.utils.utils_coordinate_system import smooth_ref_path, extend_ref_path_both_ends
from cr_scenario_handler.utils.velocity_planner import VelocityPlanner


from frenetix_motion_planner.reactive_planner import ReactivePlannerPython
# EVITA: Preconfigured planner that implement a AEB system
from frenetix_motion_planner.emergency_reactive_planner import EmergencyReactivePlannerPython

from frenetix_motion_planner.reactive_planner_cpp import ReactivePlannerCpp
from frenetix_motion_planner.state import ReactivePlannerState

# from frenetix_occlusion.interface import FOInterface

# msg_logger = logging.getLogger("Message_logger")


class FrenetPlannerInterface(PlannerInterface):

    def __init__(self, agent_id: int, config_planner, config_sim, scenario: Scenario,
                 planning_problem: PlanningProblem, log_path: str, mod_path: str):
        """ Class for using the frenetix_motion_planner Frenet planner with the cr_scenario_handler.

        Implements the PlannerInterface.

        :param config_planner: The configuration object for the trajectory planner.
        :param config_sim: The configuration object for the simulation framework
        :param scenario: The scenario to be solved. May not contain the ego obstacle.
        :param planning_problem: The planning problem of the ego vehicle.
        :param log_path: Path for writing planner-specific log files to.
        :param mod_path: Working directory of the planner.
        """
        self.config_plan = config_planner
        self.config_sim = config_sim
        self.scenario = scenario
        self.id = agent_id
        self.config_sim.simulation.ego_agent_id = agent_id
        self.DT = self.config_plan.planning.dt
        self.replanning_counter = 0
        self.replanning_traj = None
        self.behavior_module_state = None
        

        self.log_path = log_path
        self.mod_path = mod_path

        # *************************************
        # Message Logger of Run
        # *************************************
        # self.msg_logger = logging.getLogger("Message_logger_" +str(self.id ))
        self.msg_logger = lh.logger_initialization(self.config_plan, log_path, "Message_logger_" + str(self.id))
        self.msg_logger.critical("Start Planner Vehicle ID: " + str(self.id))

        # Make sure to account for the initial delay
        self.planning_problem = planning_problem
        
        assert hasattr(self.planning_problem, "initial_delay"), "Planning problem does not have any initial_delay"

        self.initial_delay = self.planning_problem.initial_delay

        self.msg_logger.info(f"Planner {str(self.id)} initial delay {self.initial_delay}")

        # Init and Goal State

        # Initialize planner
        self.planner = ReactivePlannerCpp(self.config_plan, self.config_sim, scenario, self.planning_problem, log_path, mod_path, self.msg_logger) \
            if self.config_plan.debug.use_cpp else \
              ReactivePlannerPython(self.config_plan, self.config_sim, scenario, self.planning_problem, log_path, mod_path, self.msg_logger)

        problem_init_state = self.planning_problem.initial_state

        if not hasattr(problem_init_state, 'acceleration'):
            problem_init_state.acceleration = 0.
        x_0 = deepcopy(problem_init_state)

        shape = Rectangle(self.planner.vehicle_params.length, self.planner.vehicle_params.width)
        ego_vehicle = DynamicObstacle(agent_id, ObstacleType.CAR, shape, x_0, None)
        self.planner.set_ego_vehicle_state(current_ego_vehicle=ego_vehicle)

        # Set initial state and curvilinear state
        self.x_0 = ReactivePlannerState.create_from_initial_state(
            deepcopy(self.planning_problem.initial_state),
            self.config_sim.vehicle.wheelbase,
            self.config_sim.vehicle.wb_rear_axle
        )
        self.planner.record_state_and_input(self.x_0)

        self.x_cl = None
        self.desired_velocity = None
        self.occlusion_module = None
        self.behavior_module = None
        self.route_planner = None

        # lanelet_network: LaneletNetwork,
        # planning_problem: PlanningProblem,
        # scenario: Union[Scenario, None] = None,
        # extended_search: bool = False,
        # prohibited_lanelet_ids: List[int] = None,
        # logging_level: int = logging.WARNING,
        # Set reference path
        if not self.config_sim.behavior.use_behavior_planner:
            self.route_planner = RoutePlanner(scenario=scenario, planning_problem=self.planning_problem,
                                              extended_search=False, logging_level=logging.CRITICAL)
            shortest_route = self.route_planner.plan_routes().retrieve_shortest_route(retrieve_shortest=True)

            # # Init route extendor
            # route_extendor: RouteExtendor = RouteExtendor(shortest_route, extrapolation_length=50)
            #
            # # Extend reference path at start and end
            # route_extendor.extend_reference_path_at_start_and_end()
            reference_path = extend_ref_path_both_ends(shortest_route.reference_path)

            self.reference_path = smooth_ref_path(reference_path)

        else:
            self.behavior_modul = BehaviorModule(scenario=scenario,
                                                 planning_problem=self.planning_problem,
                                                 init_ego_state=x_0,
                                                 dt=config_planner.planning.dt,
                                                 config=config_sim,
                                                 log_path=self.log_path)
            self.reference_path = smooth_ref_path(self.behavior_modul.reference_path)

        self.goal_area = gc.get_goal_area_shape_group(planning_problem=self.planning_problem, scenario=scenario)

        # *******************************************************************************
        # Initialize Occlusion Module GO to: https://github.com/TUM-AVS/Frenetix-Occlusion
        # ********************************************************************************
        # if self.config_sim.occlusion.use_occlusion_module:
        #     self.occlusion_module = FOInterface(scenario, self.reference_path, self.config_sim.vehicle, self.DT,
        #                             os.path.join(self.mod_path, "configurations", "simulation", "occlusion.yaml"))

        # **************************
        # Set External Planner Setups
        # **************************
        self.planner.update_externals(x_0=self.x_0, reference_path=self.reference_path, goal_area=self.goal_area,
                                      occlusion_module=self.occlusion_module)
        self.x_cl = self.planner.x_cl

        # ***************************
        # Initialize Velocity Planner
        # ***************************
        self.velocity_planner = VelocityPlanner(scenario=scenario, planning_problem=self.planning_problem,
                                                coordinate_system=self.coordinate_system)

    @property
    def computed_trajectories(self):
        """Return the recorded optimal trajecotires computed by the controller."""
        return self.planner.computed_trajectories

    @property
    def all_trajectories(self):
        """Return the sampled trajectory bundle for plotting purposes."""
        return self.planner.all_traj

    @property
    def record_input_list(self):
        return self.planner.record_input_list

    @property
    def record_state_list(self):
        return self.planner.record_state_list

    @property
    def record_disengagement(self):
        return self.planner.record_disengagement
    
    @property
    def record_emergency(self):
        return self.planner.record_emergency

    @property
    def vehicle_history(self):
        return self.planner.ego_vehicle_history

    @property
    def coordinate_system(self):
        return self.planner.coordinate_system

    @property
    def optimal_trajectory(self):
        return self.planner.optimal_trajectory

    @property
    def trajectory_pair(self):
        return self.planner.trajectory_pair

    def update_planner(self, scenario: Scenario, predictions: dict):
        """ Update the planner before the next time step.

        Updates the scenario and the internal states, and sets the new predictions.

        :param scenario: Updated scenario reflecting the new positions of other agents.
        :param predictions: Predictions for the other obstacles in the next time steps.
        """

        # TODO: Check if the obstacles corresponding to the planners that have already reached goal are still there!
        self.scenario = scenario

        if not self.config_sim.behavior.use_behavior_planner:
            # set desired velocity
            self.desired_velocity = self.velocity_planner.calculate_desired_velocity(self.x_0, self.x_cl[0][0])
        else:
            behavior = self.behavior_modul.execute(predictions=predictions, ego_state=self.x_0, time_step=self.DT)
            self.desired_velocity = behavior.desired_velocity
            if behavior.reference_path is not None:
                self.reference_path = behavior.reference_path
            self.behavior_module_state = behavior.behavior_planner_state

        self.planner.update_externals(scenario=scenario, x_0=self.x_0, x_cl=self.x_cl,
                                      desired_velocity=self.desired_velocity, predictions=predictions)

    def step_interface(self, current_timestep=None):
        """ Execute one planing step.

        update_planner has to be called before this function.
        Plans the trajectory for the next time step, updates the
        internal state of the FrenetInterface, and shifts the trajectory
        to the global representation.

        :return: error, trajectory
            where error is:
                0: If an optimal trajectory has been found.
                1: Otherwise.
            and trajectory is:
                A Trajectory object containing the planned trajectory,
                    using the vehicle center for the position: If error == 0
                None: Otherwise
        """

        if int(self.replanning_counter / self.config_plan.planning.replanning_frequency) == 1:
            self.replanning_counter = 0

        
        # Make sure that we do not screw up the trajectories switching back to 


        if self.replanning_counter == 0 or self.config_plan.planning.replanning_frequency < 2 or self.planner.emergency:

            if self.planner.emergency:
                # If the planner was in emergency in the past step, it might be that it recovered the situation
                #   so we update its current state and reference path and retry a regular plan
                self.planner.emergency = False
                self.replanning_counter = 0 # Make sure we reset
                self.planner.set_new_ref_path = True # Force recomputing x_cl
                # Doing so we should tell the planner that it has to recompute the current curvilinear state x_cl
                # otherwise it will teleport! But the resulting trajectories are kinda weird...
                self.planner.update_externals(x_0=self.x_0)


            if self.occlusion_module is not None:
                self.occlusion_module.evaluate_scenario(predictions=self.planner.predictions,
                                                        ego_pos=self.x_0.position,
                                                        ego_v=self.x_0.velocity,
                                                        ego_orientation=self.x_0.orientation,
                                                        ego_pos_cl=np.array([self.x_cl[0][0], self.x_cl[1][0]]),
                                                        timestep=current_timestep,
                                                        cosy_cl=self.coordinate_system.ccosy)

            if current_timestep < self.initial_delay:
                self.msg_logger.debug(f"Planner initial delay has {self.planning_problem.initial_state.time_step - current_timestep} steps to go")
                # Plan a standing sill trajectory
                optimal_trajectory_pair = self.planner.stand_still()
                # We did not really replanned here!
                self.replanning_counter = self.replanning_counter - 1
            else:
                # plan trajectory
                optimal_trajectory_pair = self.planner.plan()

            
            if not optimal_trajectory_pair:
                self.msg_logger.critical(f"Planner {self.id} - No Optimal Trajectory Available. Try Stand Still")
                optimal_trajectory_pair = self.planner.stand_still()

            if not optimal_trajectory_pair:
                self.msg_logger.critical(f"Planner {self.id} - No Optimal Trajectory Available. Try Emergency model!")
                optimal_trajectory_pair = self.planner.emergency_stop()
                # Make sure we do not skip the next replanning!
                # TODO Not sure why this should be something for the planner...
                self.planner.emergency = True

            if not optimal_trajectory_pair:
                self.msg_logger.critical(f"Planner {self.id} - IT SHOULD NOT BE THERE... CANNOT FIND ANY TRAJECTORY AT ALL?!")
                return None, self.replanning_counter

            # record the new state for planner-internal logging
            self.planner.record_state_and_input(optimal_trajectory_pair[0].state_list[1])

            # recird the entire trajectory
            # self.msg_logger.debug(f"Agent {self.id} - Appending trajectory {self.planner_interface.trajectory_pair[0]} for agent {self}")
            self.planner.record_computed_trajectory(optimal_trajectory_pair[0])


            # update init state and curvilinear state
            self.x_0 = deepcopy(self.planner.record_state_list[-1])
            self.x_cl = (optimal_trajectory_pair[2][1], optimal_trajectory_pair[3][1])

            self.msg_logger.info(f"current time step: {current_timestep}")
            self.msg_logger.info(f"current velocity: {self.x_0.velocity}")
            self.msg_logger.info(f"current target velocity: {self.desired_velocity}")

            self.replanning_traj = optimal_trajectory_pair
            selected_trajectory = optimal_trajectory_pair[0]

        else:
            # After emergency planning this does not really work... because replanning_counter does not consider the resets of emergency
            # record the new state for planner-internal logging
            self.planner.record_state_and_input(self.replanning_traj[0].state_list[1+self.replanning_counter])

            # update init state and curvilinear state
            self.x_0 = deepcopy(self.planner.record_state_list[-1])
            self.x_cl = (self.replanning_traj[2][1+self.replanning_counter], self.replanning_traj[3][1+self.replanning_counter])

            current_ego_vehicle = self.planner.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list[self.replanning_counter:],
                                                                                       self.config_sim.simulation.ego_agent_id)
            # THIS ONE SETS THE EGO VEHICLE HISTORY NOT ITS STATE!
            self.planner.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

            self.planner.plan_postprocessing(optimal_trajectory=self.planner.optimal_trajectory, planning_time=0.0,
                                             replanning_counter=self.replanning_counter)
            
            # Store also the "replanned" trajectories
            self.planner.record_computed_trajectory(self.replanning_traj[0])

            self.msg_logger.info(f"current time step: {current_timestep}")
            self.msg_logger.info(f"current velocity: {self.x_0.velocity}")
            self.msg_logger.info(f"current target velocity: {self.desired_velocity}")
            selected_trajectory = self.replanning_traj

        self.replanning_counter += 1

        return selected_trajectory, self.replanning_counter-1
