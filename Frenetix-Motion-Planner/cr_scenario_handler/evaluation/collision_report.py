import traceback
import numpy as np
from typing import List

from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad_dc.collision.collision_detection.scenario import create_collision_checker_scenario
from commonroad_dc.boundary.boundary import create_road_boundary_obstacle

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleRole, DynamicObstacle

from commonroad.planning.planning_problem import PlanningProblem

from cr_scenario_handler.utils.configuration import Configuration
from cr_scenario_handler.utils.multiagent_helpers import create_tvobstacle
from risk_assessment.helpers.collision_helper_function import angle_range
from risk_assessment.harm_estimation import harm_model
from risk_assessment.visualization.collision_visualization import collision_vis
from risk_assessment.utils.logistic_regression_symmetrical import \
    get_protected_inj_prob_log_reg_ignore_angle

# If the vehicles are less than a centimeter we consider them collided
# Hopefully this is not too large.
MIN_DISTANCE_FOR_COLLISION = 0.5

def coll_report(ego_vehicle_history: List[DynamicObstacle], planner,
                scenario: Scenario, planning_problem: PlanningProblem,
                timestep: int, config: Configuration, collision_report_path: str,
                colliding_agents: List):
    """Collect and present detailed information about a collision.

    :param ego_vehicle_history: List of ego obstacles for at least the last two time steps.
    :param planner: The planner used by the ego vehicle.
    :param scenario: The simulated scenario.
    :param planning_problem: The planning problem of the ego vehicle.
    :param timestep: Time step at which the collision occurred.
    :param config: Configuration containing the vehicle parameters.
    :param collision_report_path: The path to write the report to.

    Colliding agents reported by the simulation. to double check what's going on here!

    """

    # print(f"Reported colliding agents for {planning_problem.planning_problem_id} are {colliding_agents}")

    # check if the current state is collision-free
    # TODO This does not seem to be even used...
    vel_list = []
    # get ego position and orientation
    ego_pos = None
    ego_pos_last = None
    ego_orientation = None
    try:
        ego_pos = ego_vehicle_history[-1].state_at_time(timestep).position
        ego_orientation = ego_vehicle_history[-1].state_at_time(timestep).orientation

    except AttributeError:
        print("None-type error")
        traceback.print_exc()

    (
        _,
        road_boundary,
    ) = boundary.create_road_boundary_obstacle(
        scenario=scenario,
        method="aligned_triangulation",
        axis=2,
    )

    if timestep == 0:
        ego_vel = ego_vehicle_history[-1].initial_state.velocity
        ego_yaw = ego_vehicle_history[-1].initial_state.orientation

        vel_list.append(ego_vel)
    else:
        ego_pos_last = ego_vehicle_history[-2].state_at_time(timestep).position

        delta_ego_pos = ego_pos - ego_pos_last

        ego_vel = np.linalg.norm(delta_ego_pos) / scenario.dt

        vel_list.append(ego_vel)
        # Why not ego_orientation?
        # Why not ego occupancy at time?
        # This seems to be broken !
        # ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])
        ego_yaw = ego_orientation
        
    # Collision obj of ego at this exact time
    current_state_collision_object = create_tvobstacle(
        traj_list=[
            [
                ego_pos[0],
                ego_pos[1],
                ego_yaw,
            ]
        ],
        # Why only half of it?
        box_length=config.vehicle.length / 2,
        box_width=config.vehicle.width / 2,
        start_time_step=timestep,
    )

    # Add road boundary to collision checker
    # TODO Why do we create a second road boundary object?
    cc = create_collision_checker_scenario(scenario)
    _, road_boundary_sg_obb = create_road_boundary_obstacle(scenario)
    cc.add_collision_object(road_boundary_sg_obb)
    cc.add_collision_object(road_boundary)

    # Get the colliding obstacle
    # The original code behaves werdly: for the same pair of vehicles it reports different distance, so different collision status.
    # This happens despite simulation.py reports the vehicles as colliding
    #
    obs_id = None
    for obs in [ o for o in scenario.obstacles if o.obstacle_id in colliding_agents]:

        # Note: if a vehicle reached the goal area before timestep this assertion triggers. However, those agents should not be reported as obstacles anymore in the scenario!
        # If the obstacles are still in the scenario, they must be there at the right time!

        # assert obs.state_at_time(timestep), f"Missing state for obstacles {obs.obstacle_id} at time {timestep}"
        if not obs.state_at_time(timestep):
            # print(f">>>>>> WARNING: Missing state for obstacles {obs.obstacle_id} at time {timestep}. Check if GOAL REACHED!")
            # If the obstacle is not there at the given timestep, skip it
            continue

        # DEBUG
        def debug_plot():
            import os
            import matplotlib.pyplot as plt
            from commonroad.visualization.mp_renderer import MPDrawParams, MPRenderer, DynamicObstacleParams, OccupancyParams

            ego_params = DynamicObstacleParams()
            ego_params.time_begin = timestep
            ego_params.draw_icon = False
            ego_params.vehicle_shape.occupancy.shape.facecolor = "#E37222"
            ego_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
            ego_params.vehicle_shape.occupancy.shape.zorder = 50
            ego_params.vehicle_shape.occupancy.shape.opacity = 1

            npc_params = DynamicObstacleParams()
            npc_params.time_begin = timestep
            npc_params.draw_icon = False
            npc_params.vehicle_shape.occupancy.shape.facecolor = "#00FF00"
            npc_params.vehicle_shape.occupancy.shape.edgecolor = "#9C4100"
            npc_params.vehicle_shape.occupancy.shape.zorder = 50
            npc_params.vehicle_shape.occupancy.shape.opacity = 1

            # set plot limits to show the road section around the ego vehicle
            position = ego_vehicle_history[-1].state_at_time(timestep).position
            plot_limits = [position[0] - 20,
                   position[0] + 20,
                   position[1] - 20,
                   position[1] + 20]

            scenario_params = MPDrawParams()
            scenario_params.time_begin = timestep
            scenario_params.dynamic_obstacle.show_label = True
            scenario_params.dynamic_obstacle.draw_icon = True
            scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#0065BD"
            scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"
            scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 50
            scenario_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 1

            rnd = MPRenderer(plot_limits=plot_limits)
            # plot the scenario at the current time step
            scenario.draw(rnd, draw_params=scenario_params)

            plt.gca().set_aspect('equal')

            ego_vehicle_history[-1].draw(rnd, draw_params=ego_params)
            obs.draw(rnd, draw_params=npc_params)

            rnd.render()

            # Replot the shapely objects
            x,y = ego_vehicle_history[-1].occupancy_at_time(timestep).shape.shapely_object.exterior.xy
            plt.plot(x,y, "b-", zorder=55)

            x,y = obs.occupancy_at_time(timestep).shape.shapely_object.exterior.xy
            plt.plot(x,y, "g-", zorder=55)

            debug_collision_report_path = os.path.join(collision_report_path, f"debug_{planning_problem.planning_problem_id}_{obs.obstacle_id}.png")
            plt.savefig(debug_collision_report_path)

        # Manual debugging
        # debug_plot()

        # Note: It seems that here the collision is reported one timestamp after than it actually occurred. So check both timestep and timestep - 1
        # For whatever effect, in cases which the distance at timestep-1 is 0, it then becomes > 0 at timestep.
        # now this might not be needed anymore
        distance = ego_vehicle_history[-1].occupancy_at_time(timestep).shape.shapely_object.distance(obs.occupancy_at_time(timestep).shape.shapely_object)

        # print(f"\n\n Min distance at collision time between {planning_problem.planning_problem_id} and {obs.obstacle_id} is {distance}")

        if distance <= MIN_DISTANCE_FOR_COLLISION:
            # Collision detected
            if obs.obstacle_id != ego_vehicle_history[-1].obstacle_id:
                # print(f"Agent {planning_problem.planning_problem_id}: collided with {obs.obstacle_id} at time {timestep} (distance {distance} <= {MIN_DISTANCE_FOR_COLLISION})")
                if obs_id is None:
                    obs_id = obs.obstacle_id
                else:
                    print(f"WARNING: Agent {planning_problem.planning_problem_id}: More than one collision detected. We cannot handle this case, skip report generation")
                    return

        #
        # TODO There are two issues: (1) the collision state is reported with one step delay; (2) at this time the rectangles might not really touch because they moved away
        # (following their original trajectories)
        #
        #
        # if ego_vehicle_history[-1].occupancy_at_time(timestep).shape.shapely_object.intersects(obs.occupancy_at_time(timestep).shape.shapely_object):
        #     print(f"\n\n Shapely reported intersections between ego {planning_problem.planning_problem_id} and {obs.obstacle_id}")
        # elif distance < MIN_DISTANCE_FOR_COLLISION:
        #     print(f"\n\n Shapely reported objects almost touching for ego {planning_problem.planning_problem_id} and {obs.obstacle_id}")
        # co = create_collision_object(obs)
        # if current_state_collision_object.collide(co):
        #     if obs.obstacle_id != ego_vehicle_history[-1].obstacle_id:
        #         print(f"Agent {planning_problem.planning_problem_id}: collided with {obs.obstacle_id} at time {timestep}")
        #         if obs_id is None:
        #             obs_id = obs.obstacle_id
        #         else:
        #             print(f"Agent {planning_problem.planning_problem_id}: More than one collision detected")
        #             return

    # If we get there, we did not find any collision between ego and the reported colliding_agents
    # so we need to check if the ego collided with the road instead?

    # Is the ego_vehicle inside any road?
    is_in_lanelet = False
    # for lanelet_ids in [ l for l in scenario.lanelet_network.find_lanelet_by_position([ego_pos[0], ego_pos[1]]) if len(l) > 0]:
    for lanelet_ids in [ l for l in scenario.lanelet_network.find_lanelet_by_position([ego_pos]) if len(l) > 0]:
        for lanelet in [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in lanelet_ids]:
            # print(f"Agent {planning_problem.planning_problem_id} is in lanelet {lanelet.lanelet_id}")
            is_in_lanelet = True

    if not is_in_lanelet:
        print(f"WARNING: Agent {planning_problem.planning_problem_id} is out of the road.")
        return

    # Not sure what's this. It reports a collision with road or static.
    # Maybe it means the vehicle is still ON the road (at least partially)
    # I notice this when the vehicle is across lanes, but might be just a coincidence
    # if not cc.collide(current_state_collision_object):
    #     # find lanelet by position
        
    # At this point, if the vehicle's center is (partially) inside a road but it was reported as colliding,
    # we can only conlude that it collided against some "exterior" road boundaries
    
    # TODO We should check whether this is really the case!!

    # Collision with boundary
    if obs_id is None:
        ego_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=ego_vel, coeff=planner.params_harm
        )
        total_harm = ego_harm

        print(f"WARNING: Agent {planning_problem.planning_problem_id}: " + "Collision with road boundary. (Harm: {:.2f})".format(ego_harm))

        return


    # get information of colliding obstacle
    obs_pos = (
        scenario.obstacle_by_id(obstacle_id=obs_id)
        .occupancy_at_time(time_step=timestep)
        .shape.center
    )
    obs_pos_last = (
        scenario.obstacle_by_id(obstacle_id=obs_id)
        .occupancy_at_time(time_step=timestep - 1)
        .shape.center
    )
    obs_size = (
            scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.length
            * scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.width
    )

    # filter initial collisions
    if timestep < 1:
        print(f"WARNING: Agent {planning_problem.planning_problem_id}: Collision at initial state")
        return
    if (
            scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
            == ObstacleRole.ENVIRONMENT
    ):
        obs_vel = 0
        obs_yaw = 0
    else:
        pos_delta = obs_pos - obs_pos_last

        # Replace with Velocity
        obs_vel = np.linalg.norm(pos_delta) / scenario.dt
        if (
                scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                == ObstacleRole.DYNAMIC
        ):
            # TODO Replace with Orientation
            obs_yaw = np.arctan2(pos_delta[1], pos_delta[0])
        else:
            obs_yaw = scenario.obstacle_by_id(
                obstacle_id=obs_id
            ).initial_state.orientation

    # calculate crash angle
    pdof = angle_range(obs_yaw - ego_yaw + np.pi)
    rel_angle = np.arctan2(
        obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
    )
    ego_angle = angle_range(rel_angle - ego_yaw)
    obs_angle = angle_range(np.pi + rel_angle - obs_yaw)

    # calculate harm
    ego_harm, obs_harm, ego_obj, obs_obj = harm_model(
        scenario=scenario,
        ego_vehicle_sc=ego_vehicle_history[-1],
        vehicle_params=config.vehicle,
        ego_velocity=ego_vel,
        ego_yaw=ego_yaw,
        obstacle_id=obs_id,
        obstacle_size=obs_size,
        obstacle_velocity=obs_vel,
        obstacle_yaw=obs_yaw,
        pdof=pdof,
        ego_angle=ego_angle,
        obs_angle=obs_angle,
        modes=planner.params_risk,
        coeffs=planner.params_harm,
    )

    # if collision report should be shown
    collision_vis(
        scenario=scenario,
        ego_vehicle=ego_vehicle_history[-1],
        destination=collision_report_path,
        ego_harm=ego_harm,
        ego_type=ego_obj.type,
        ego_v=ego_vel,
        ego_mass=ego_obj.mass,
        obs_harm=obs_harm,
        obs_type=obs_obj.type,
        obs_v=obs_vel,
        obs_mass=obs_obj.mass,
        pdof=pdof,
        ego_angle=ego_angle,
        obs_angle=obs_angle,
        time_step=timestep,
        modes=planner.params_risk,
        marked_vehicle=ego_vehicle_history[-1].obstacle_id,
        planning_problem=planning_problem,
        global_path=None,
        driven_traj=None,
    )