__author__ = "Rainer Trauth, Gerald Würsching, Christian Pek"
__credits__ = ["BMW Group CAR@TUM, interACT"]
__version__ = "1.0"
__maintainer__ = "Rainer Trauth"
__email__ = "rainer.trauth@tum.de"
__status__ = "Beta"

# python packages
import math
import time

import numpy as np
from typing import List
import multiprocessing
from multiprocessing.context import Process

# frenetix_motion_planner imports
from frenetix_motion_planner.polynomial_trajectory import QuinticTrajectory, QuarticTrajectory
from frenetix_motion_planner.trajectories import TrajectoryBundle, TrajectorySample, CartesianSample, CurviLinearSample
from frenetix_motion_planner.cost_functions.cost_function import CostFunction


from cr_scenario_handler.planner_interfaces.frenet_interface import ReactivePlannerPython
from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem, interpolate_angle

from omegaconf import OmegaConf

# precision value
_EPS = 1e-5

# get logger
# msg_logger = logging.getLogger("Message_logger")

# This must follow the logic of partial functions to evaluate each single cost factor...
def sum_them(trajectory: TrajectorySample, planner=None, scenario=None, desired_speed: float=0) -> float:
     return np.sum(trajectory.cartesian.v)

class StopAsSoonAsPossibleCostFunction(CostFunction):
    """
    Cost function that stops the vehicle as soon as possible (closer to the initial point)
    Based on AdaptableCostFunction
    """

    def __init__(self, rp, configuration):
        super(StopAsSoonAsPossibleCostFunction, self).__init__()

        self.scenario = None
        self.rp = rp
        self.reachset = None
        self.desired_speed = None
        self.predictions = None
        self.configuration = configuration

        self.vehicle_params = rp.vehicle_params
        
        # Load of all the possible cost elements from the given yml configuration
        self.cost_weights = OmegaConf.to_object(configuration.cost.cost_weights)
        
        # Remove anything which is NOT velocity

        # No idea what's this
        self.save_unweighted_costs = False
        self.cost_weights_names = None

        for category in list(self.cost_weights.keys()):
            if category != "velocity":
                del self.cost_weights[category]
            else:
                # Force the weight for velocity cost
                self.cost_weights[category] = 1.0
        
        assert len(self.cost_weights.keys()) == 1, f"Wrong Configuration for Emergency Cost Function {self.cost_weights.keys()}"

        names = list()
        names.append("velocity")
        self.cost_weights_names = names

        self.functions = dict()
        self.functions["velocity"] = sum_them

    def update_state(self, scenario, rp, predictions, reachset):
        self.scenario = scenario
        self.rp = rp
        self.reachset = reachset
        self.desired_speed = rp.desired_velocity
        self.predictions = predictions

     # calculate all costs for all trajcetories
    def evaluate(self, trajectories: List[TrajectorySample]):
        self.calc_cost(trajectories)

    # calculate all costs and weigh them using the 
    def calc_cost(self, trajectories: List[TrajectorySample]):

        for i, trajectory in enumerate(trajectories):
            costlist = np.zeros(len(self.cost_weights))
            costlist_weighted = np.zeros(len(self.cost_weights))

            for num, function in enumerate(self.cost_weights_names):
                costlist[num] = self.functions[function](trajectory=trajectory, planner=self.rp,
                                                         scenario=self.scenario, desired_speed=self.desired_speed)
                costlist_weighted[num] = self.cost_weights[function] * costlist[num]

            total_cost = np.sum(costlist_weighted)

            trajectory.set_costs(total_cost, costlist, costlist_weighted, self.cost_weights_names)


class EmergencyReactivePlannerPython(ReactivePlannerPython):
    """
    THIS IS A COPY OF REACTIVE PLANNER. PROBABLY IT WOULD BE ENOUGH TO INSTANTIATE THAT ONE AND SIMPLY REPLACE SOME CODE?
    Reactive planner class that plans trajectories in a sampling-based fashion and TAKES the vehicle to a stop as soon as possible
    """
    def __init__(self, config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger):
        """
        Constructor of the reactive planner
        : param config: Configuration object holding all planner-relevant configurations
        """
        assert len(scenario.obstacles) == 0, "Emergency Reactive Planner cannot have obstacles in scenario!"

        super().__init__(config_plan, config_sim, scenario, planning_problem, log_path, work_dir, msg_logger)

        # **************************
        # Cost Function Setting
        # **************************
        cost_function = StopAsSoonAsPossibleCostFunction(rp=self, configuration=self.config_plan)
        self.set_cost_function(cost_function=cost_function)
        # This must be set with update_externals
        # self.desired_velocity = 0.0

    def _get_optimal_trajectory(self, trajectory_bundle: TrajectoryBundle, samp_lvl):
        """
        Computes the optimal trajectory from a given trajectory bundle. This is the same a reactive planner, without set risk, since we do not account 
        for any prediction during emergency/panic

        :param trajectory_bundle: The trajectory bundle
        :return: The optimal trajectory if exists (otherwise None)
        """

        # reset statistics
        self._collision_counter = 0
        self._infeasible_count_kinematics = None
        infeasible_count_kinematics = [0] * 11

        # check kinematics of each trajectory
        if self._multiproc:
            # with multiprocessing
            # divide trajectory_bundle.trajectories into chunks
            chunk_size = math.ceil(len(trajectory_bundle.trajectories) / self._num_workers)
            chunks = [trajectory_bundle.trajectories[ii * chunk_size: min(len(trajectory_bundle.trajectories),
                                                                          (ii+1)*chunk_size)] for ii in range(0, self._num_workers)]

            # initialize list of Processes and Queues
            list_processes = []
            trajectories_all = []
            queue_1 = multiprocessing.Queue()
            queue_2 = multiprocessing.Queue()

            for chunk in chunks:
                p = Process(target=self.check_feasibility, args=(chunk, queue_1, queue_2))
                list_processes.append(p)
                p.start()

            # get return values from queue
            for p in list_processes:
                trajectories_all.extend(queue_1.get())
                if self._kinematic_debug:
                    temp = queue_2.get()
                    infeasible_count_kinematics = [x + y for x, y in zip(infeasible_count_kinematics, temp)]

            # wait for all processes to finish
            for p in list_processes:
                p.join()
        else:
            # without multiprocessing
            trajectories_all = self.check_feasibility(trajectory_bundle.trajectories)

        feasible_trajectories = [obj for obj in trajectories_all if obj.valid is True and obj.feasible is True]
        infeasible_trajectories = [obj for obj in trajectories_all if obj.valid is False or obj.feasible is False]

        # update number of infeasible trajectories
        self._infeasible_count_kinematics = infeasible_count_kinematics
        self._infeasible_count_kinematics[0] = len(infeasible_trajectories)
        self.infeasible_kinematics_percentage = float(len(feasible_trajectories) / len(trajectories_all)) * 100

        # print size of feasible trajectories and infeasible trajectories
        self.msg_logger.debug(
            'Found {} feasible trajectories and {} infeasible trajectories'.format(str(len(feasible_trajectories)),
                                                                                   str(len(infeasible_trajectories))))
        self.msg_logger.debug('Percentage of valid & feasible trajectories: %s %%' % str(self.infeasible_kinematics_percentage))

        # for visualization store all trajectories with validity level based on kinematic validity
        if self._draw_traj_set or self.save_all_traj:
            trajectory_bundle.trajectories = trajectories_all
            trajectory_bundle.sort(occlusion_module=self.occlusion_module)
            self.all_traj = trajectory_bundle.trajectories
            trajectory_bundle.trajectories = list(filter(lambda x: x.feasible is True, trajectory_bundle.trajectories))
        else:
            # set feasible trajectories in bundle
            trajectory_bundle.trajectories = feasible_trajectories
            # sort trajectories according to their costs
            trajectory_bundle.sort(occlusion_module=self.occlusion_module)

        # ******************************************
        # Check Feasible Trajectories for Collisions
        # ******************************************
        optimal_trajectory = self.trajectory_collision_check(feasible_trajectories=
                                                             trajectory_bundle.get_sorted_list(
                                                                 occlusion_module=self.occlusion_module))

        if samp_lvl >= self._sampling_max - 1 and optimal_trajectory is None and feasible_trajectories:
            
            self.msg_logger.critical("Select a trajectory from the pool of feasible ones without computing risks!")
            # for traje in feasible_trajectories:
            #     self.set_risk_costs(traje)
            # sort_risk = sorted(feasible_trajectories, key=lambda traj: traj._ego_risk + traj._obst_risk,
                            #    reverse=False)
            # self.msg_logger.warning("No optimal trajectory available. Select lowest risk trajectory!")
            # optimal_trajectory = sort_risk[0]
            if len(feasible_trajectories) > 0:
                optimal_trajectory = feasible_trajectories[0]
                return optimal_trajectory
            else:
                self.msg_logger.critical("There are no feasible trajectory. Return None!")
                return None
        else:
            return optimal_trajectory


    def plan_optimal_trajectory(self):
        """
        Same as plan but retun directly the optimal trajectory
        """
        assert self.desired_velocity == 0.0, "Emergency Planner cannot have aim for positive velocity!"
        
        # Convert current x_cl state to the new coordinate system
        self.x_cl = self._compute_initial_states(self.x_0)

        self.msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(self.x_cl[0], self.x_cl[1]))
        self.msg_logger.debug('Desired velocity is {} m/s'.format(self.desired_velocity))
        
        # Initialization of while loop
        optimal_trajectory = None

        # Initial index of sampling set to use
        samp_level = self._sampling_min

        # sample until trajectory has been found or sampling sets are empty
        while optimal_trajectory is None and samp_level < self._sampling_max:

            self.cost_function.update_state(scenario=self.scenario, rp=self, predictions=self.predictions, reachset=self.reach_set)

            bundle = self._create_trajectory_bundle(self.x_cl[0], self.x_cl[1], self.cost_function, samp_level=samp_level)

            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step

            optimal_trajectory = self._get_optimal_trajectory(bundle, samp_level)

            # increase sampling level (i.e., density) if no optimal trajectory could be found
            samp_level += 1


        self.msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(self._infeasible_count_kinematics))
        self.msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(self.infeasible_count_collision))

        if optimal_trajectory is None and self.x_0.velocity <= 0.1:
            self.msg_logger.warning('Planning standstill for the current scenario')
            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step
            optimal_trajectory = self._compute_standstill_trajectory()

        return optimal_trajectory

    def plan(self) -> tuple:
        """
        Plans an optimal trajectory: follow the orientation of the car, slow down as fast as possible.
        Do not care about obstacles and other planners
        """
        assert self.desired_velocity == 0.0, "Emergency Planner cannot have aim for positive velocity!"
        
        # Convert current x_cl state to the new coordinate system
        self.x_cl = self._compute_initial_states(self.x_0)

        self.msg_logger.debug('Initial state is: lon = {} / lat = {}'.format(self.x_cl[0], self.x_cl[1]))
        self.msg_logger.debug('Desired velocity is {} m/s'.format(self.desired_velocity))
        

        # Initialization of while loop
        optimal_trajectory = None
        t0 = time.time()

        # Initial index of sampling set to use
        samp_level = self._sampling_min

        # sample until trajectory has been found or sampling sets are empty
        while optimal_trajectory is None and samp_level < self._sampling_max:

            self.cost_function.update_state(scenario=self.scenario, rp=self, predictions=self.predictions, reachset=self.reach_set)

            bundle = self._create_trajectory_bundle(self.x_cl[0], self.x_cl[1], self.cost_function, samp_level=samp_level)

            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step

            optimal_trajectory = self._get_optimal_trajectory(bundle, samp_level)

            # increase sampling level (i.e., density) if no optimal trajectory could be found
            samp_level += 1


        planning_time = time.time() - t0

        self.msg_logger.debug('Rejected {} infeasible trajectories due to kinematics'.format(self._infeasible_count_kinematics))
        self.msg_logger.debug('Rejected {} infeasible trajectories due to collisions'.format(self.infeasible_count_collision))


        # ******************************************
        # Update Trajectory Pair & Commonroad Object
        # ******************************************
        self.trajectory_pair = self._compute_trajectory_pair(optimal_trajectory) if optimal_trajectory is not None else None
        if self.trajectory_pair is not None:
            current_ego_vehicle = self.convert_state_list_to_commonroad_object(self.trajectory_pair[0].state_list)
            self.set_ego_vehicle_state(current_ego_vehicle=current_ego_vehicle)

        if optimal_trajectory is None and self.x_0.velocity <= 0.1:
            self.msg_logger.warning('Planning standstill for the current scenario')
            if self.logger:
                self.logger.trajectory_number = self.x_0.time_step
            optimal_trajectory = self._compute_standstill_trajectory()

        # # Failed to find any feasible trajectory
        # if not optimal_trajectory == 0:
        #     # At this poi
        #     self.msg_logger.warning("No feasible trajectory exists - Disengage!")
        #     self.record_a_disengagement(self.x_0.time_step)

        # ************************************
        # Set Risk Costs to Optimal Trajectory
        # ************************************
        if optimal_trajectory is not None and self.log_risk:
            optimal_trajectory = self.set_risk_costs(optimal_trajectory)

        self.optimal_trajectory = optimal_trajectory

        self.plan_postprocessing(optimal_trajectory=optimal_trajectory, planning_time=planning_time)

        return self.trajectory_pair