import ast
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from copy import deepcopy

from shapely.geometry import LineString
from shapely.affinity import scale

from evita.generators.utils import compute_RIV, compute_RIA, compute_impact_position
from evita.data_analysis import get_interaction_map_bins


from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import Trajectory, Occupancy
from commonroad_dc.geometry.util import resample_polyline
from commonroad.scenario.obstacle import ObstacleRole
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.scenario import Scenario, DynamicObstacle
from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem
from cr_scenario_handler.utils.configuration import VehicleParameterMapping, VehicleType

TRAFFIC_LIGHTS_SAFETY_DISTANCE = 100
STOP_SIGN_SAFETY_DISTANCE = 75
VEHICLES_SAFETY_DISTANCE = 10

def check_stop_line_crossing(scenario: Scenario, trajectory_at_time_next_t: Trajectory) -> List:
    """
    Check if the vehicle's planned trajectory intersects a stop line associated to a traffic light or a stop sign
    """
    intersecting_stop_line_obstacles = set()

    # Iterate over each map lanelet
    for lanelet in scenario.lanelet_network.lanelets:
        # Extract lanelet's stop line and check if it is a valid value
        stop_line = lanelet.stop_line
        if stop_line is not None:
            # Convert stop line and planned trajectory to LineString
            stop_line_path = LineString([stop_line.start, stop_line.end])
            coords = [
                (state.position[0], state.position[1])
                for state in trajectory_at_time_next_t.state_list
            ]

            if len(coords) < 2:
                return []

            traj_line = LineString(coords)

            # Check if they intersect
            if stop_line_path.intersects(traj_line):
                # Save the ids of the "interacting" traffic light or stop sign
                if stop_line.traffic_sign_ref:
                    intersecting_stop_line_obstacles.update(stop_line.traffic_sign_ref)

                if stop_line.traffic_light_ref:
                    intersecting_stop_line_obstacles.update(stop_line.traffic_light_ref)

    return list(intersecting_stop_line_obstacles)

def augment_with_interactions(executed_scenario: Scenario, planned_trajectories: pd.DataFrame, scenario_vehicles_perceptions: pd.DataFrame, scenario_traffic_lights_perceptions: pd.DataFrame, scenario_collisions: pd.DataFrame, trajectory_difference_threshold: float) -> pd.DataFrame:
    """
    Given the scenario vehicles planned trajectories and perceptions data, detect the occurrence of interactions between scenario's "obstacles"
    """
    interactions_count_df = None
    PLANNING_TIMESTEP_OFFSET = (2.0 / executed_scenario.dt)  # 1.5 seconds in timesteps

    # Extract scenario vehicles ids
    simulation_agents_ids = [obstacle.obstacle_id for obstacle in executed_scenario.obstacles]

    # Iterate over each scenario vehicle
    for agent_id in sorted(simulation_agents_ids):

        # Retrieve dynamic obstacle object
        agent = executed_scenario.obstacle_by_id(agent_id)
        
        # Extract vehicle's planning timesteps
        agent_planning_timesteps = planned_trajectories.loc[planned_trajectories['vehicle_id'] == agent_id, 'timestep']

        # Extract vehicle's collision timestep, if exists
        if scenario_collisions[scenario_collisions['vehicle_id'] == agent_id].empty:
            agent_collision_timestep = None
        else:
            agent_collision_timestep = scenario_collisions.loc[scenario_collisions['vehicle_id'] == agent_id, 'collision_timestep'].iloc[0]

        is_interacting = {}
        is_interacting_with = {}
        diff_long = {}
        diff_lat = {}

        # Iterate over the vehicle's planning timesteps
        for t_index in range(len(agent_planning_timesteps) - 1):
            # Extract two consecutive planning timesteps
            t = int(agent_planning_timesteps.iloc[t_index])
            next_t = int(agent_planning_timesteps.iloc[t_index + 1])

            # Skip planning timesteps after vehicles final history timestep
            # Skip planning timesteps that are too far apart
            # Skip planning timesteps after vehicle's collision
            if (t >= agent.prediction.final_time_step or next_t >= agent.prediction.final_time_step) or \
               (next_t - t) > PLANNING_TIMESTEP_OFFSET or \
               (agent_collision_timestep is not None and (t >= agent_collision_timestep or next_t >= agent_collision_timestep)):
                continue

            # Extract planned trajectories at time `t` and `next_t`
            trajectory_at_time_t = dict_to_trajectory(planned_trajectories.loc[(planned_trajectories['vehicle_id'] == agent_id) & (planned_trajectories['timestep'] == t), 'trajectory'].iloc[0]["planned_trajectory"])
            trajectory_at_time_next_t_data = planned_trajectories.loc[(planned_trajectories['vehicle_id'] == agent_id) & (planned_trajectories['timestep'] == next_t), 'trajectory'].iloc[0]
            trajectory_at_time_next_t = dict_to_trajectory(trajectory_at_time_next_t_data["planned_trajectory"])

            replanning_decision = trajectory_at_time_next_t_data["decision"]
            decision_object = trajectory_at_time_next_t_data["decision_object"]


            interacting_obstacles = []

            if replanning_decision in ["cruise", "STOP_REASON_OBSTACLE", "STOP_REASON_SIDE", "STOP_REASON_SIDEPASS_SAFETY"]:
                # Replanning due to another Vehicle
                if decision_object in [None, "Side_Pass_Stop"]:
                    other_agents = [a for a in executed_scenario.obstacles if a.obstacle_id != agent_id]
                    nearby_agents_ids = list_anyone_nearby(next_t, agent, other_agents)
                    interacting_obstacles.extend(nearby_agents_ids)
                else:
                    interacting_agent_id = decision_object

                    # Extract perceived vehicles at time `next_t`
                    perceived_vehicles_at_next_t = scenario_vehicles_perceptions.loc[
                        (scenario_vehicles_perceptions["vehicle_id"] == agent_id) &
                        (scenario_vehicles_perceptions["timestep"] == next_t), ["obstacle_id", "obstacle_position", "obstacle_orientation"]]

                    # Check if interacting obstacle is perceived by the agent
                    perceived_obstacles_ids = perceived_vehicles_at_next_t["obstacle_id"]
                    if interacting_agent_id in perceived_obstacles_ids:
                        # Retrieve standard vehicle size
                        vehicle_parameters = VehicleParameterMapping.from_vehicle_type(VehicleType(2))
                        length = vehicle_parameters.l
                        width = vehicle_parameters.w

                        interacting_obstacle = perceived_vehicles_at_next_t[perceived_vehicles_at_next_t["obstacle_id"] == interacting_agent_id]

                        obstacle_position = ast.literal_eval(interacting_obstacle["obstacle_position"])
                        obstacle_orientation = interacting_obstacle["obstacle_orientation"]
                        
                        # Reconstruct perceived obstacle shape
                        obstacle_shape = Rectangle(length, width, np.array([obstacle_position[0], obstacle_position[1]]), obstacle_orientation)

                        # Get `agent` occupancy at time `next_t`
                        agent_occupancy_at_time_next_t = agent.occupancy_at_time(next_t)

                        if agent_occupancy_at_time_next_t is not None:
                            # Compute distance to observed agent
                            distance = agent_occupancy_at_time_next_t.shape.shapely_object.distance(obstacle_shape.shapely_object)

                            nearby_agents_ids = [interacting_agent_id]

                            interacting_obstacles.extend(nearby_agents_ids)

                intersecting_stop_line_agents = check_stop_line_crossing(executed_scenario, trajectory_at_time_next_t)
                interacting_obstacles.extend(intersecting_stop_line_agents)
            else:
                # Check if there is any close vehicle anyway
                other_agents = [a for a in executed_scenario.obstacles if a.obstacle_id != agent_id]
                nearby_agents_ids = list_anyone_nearby(next_t, agent, other_agents)
                interacting_obstacles.extend(nearby_agents_ids)
                if replanning_decision in ["STOP_REASON_SIGNAL"]:
                    # Replanning due to Traffic Light

                    # Extract traffic_light_id
                    traffic_light_id = int(decision_object.split("_")[-1]) + 10000

                    # Extract perceived traffic lights at time `next_t`
                    perceived_traffic_lights_at_next_t = scenario_traffic_lights_perceptions.loc[
                        (scenario_traffic_lights_perceptions["vehicle_id"] == agent_id) &
                        (scenario_traffic_lights_perceptions["timestep"] == next_t), "tl_id"
                    ].tolist()

                    # Check that traffic_light_id is perceived
                    if traffic_light_id in perceived_traffic_lights_at_next_t:
                        # Retrieve traffic_light object
                        traffic_light = executed_scenario.lanelet_network.find_traffic_light_by_id(traffic_light_id)

                        # Rebuild traffic light shape
                        traffic_light_shape = Rectangle(
                            center = traffic_light.position,
                            width = 0.3,
                            length = 0.3,
                        )
                        
                        if agent_occupancy_at_time_next_t is not None:
                            # Compute distance of the two shapes
                            distance = agent_occupancy_at_time_next_t.shape.shapely_object.distance(traffic_light_shape.shapely_object)
                            
                            # If the traffic light is within the vehicle's safety distance, the second condition (C2) for interactions holds
                            if distance < TRAFFIC_LIGHTS_SAFETY_DISTANCE:
                                interacting_obstacles.append(traffic_light_id)

                            # print(agent_id, next_t, "Traffic Light", decision_object, distance, trajectory_length, similar, diff_0, diff_1)
                elif replanning_decision in ["STOP_REASON_STOP_SIGN", "STOP_REASON_CREEPER"]:
                    # Replanning due to Stop Sign

                    # Extract traffic_light_id
                    stop_sign_id = int(decision_object.split("_")[-1]) + 5000

                    # Retrieve stop_sign object
                    stop_sign = executed_scenario.lanelet_network.find_traffic_sign_by_id(stop_sign_id)

                    if stop_sign is not None:
                        # Rebuild stop_sign shape
                        stop_sign_shape = Rectangle(
                            center = stop_sign.position,
                            width = 0.3,
                            length = 0.3,
                        )
                        
                        if agent_occupancy_at_time_next_t is not None:
                            # Compute distance of the two shapes
                            distance = agent_occupancy_at_time_next_t.shape.shapely_object.distance(stop_sign_shape.shapely_object)
                            
                            # If the traffic light is within the vehicle's safety distance, the second condition (C2) for interactions holds
                            if distance < STOP_SIGN_SAFETY_DISTANCE:
                                interacting_obstacles.append(stop_sign_id)

                            # print(agent_id, next_t, "Stop Sign", decision_object, distance, trajectory_length, similar, diff_0, diff_1)
                elif replanning_decision not in ["mission_complete", "STOP_REASON_DESTINATION"]:
                    # print(agent_id, replanning_decision, decision_object)
                    continue

            try:
                cropped_traj_t, cropped_traj_next_t = crop_trajectories(trajectory_at_time_t, trajectory_at_time_next_t)
            except (ValueError, StopIteration):
                continue

            if (not is_trajectory_similar(cropped_traj_t, cropped_traj_next_t)) and len(interacting_obstacles) > 0:
                # Compute difference in consecutive planned trajectories
                try:
                    diff_0, diff_1 = compute_difference_in_cc_space(executed_scenario, trajectory_at_time_t, trajectory_at_time_next_t)
                except:
                    # If this fails, we cannot do much
                    diff_0, diff_1 = compute_difference_in_cc_space(executed_scenario, trajectory_at_time_t, trajectory_at_time_next_t, use_relative_path = True)

                # Skip interaction in case of errors in trajectory difference computation
                if diff_0 is None or diff_1 is None:
                    continue

                is_interacting[next_t] = 1.0
                is_interacting_with[next_t] = interacting_obstacles
                diff_long[next_t] = diff_0
                diff_lat[next_t] = diff_1

        agent_interactions_count_df = pd.DataFrame({
            "is_interacting": is_interacting,
            "is_interacting_with": is_interacting_with,
            "diff_long": diff_long,
            "diff_lat": diff_lat
        })

        agent_interactions_count_df.index.name = "timestep"
        agent_interactions_count_df = agent_interactions_count_df.set_index(agent_interactions_count_df.index)
        agent_interactions_count_df.index.names = ["timestep"]

        if interactions_count_df is None:
            # promote the agent df to global danger df
            interactions_count_df = agent_interactions_count_df.set_index([[agent_id] * len(agent_interactions_count_df), agent_interactions_count_df.index])
        else:
            agent_interactions_count_df = agent_interactions_count_df.set_index([[agent_id] * len(agent_interactions_count_df), agent_interactions_count_df.index])
            interactions_count_df = pd.concat([interactions_count_df, agent_interactions_count_df])

    # At this point interactions_count_df should be organized exactly as criticality_dataframe so they can be merged
    # so we reset the index
    interactions_count_df.reset_index(inplace = True)
    
    interactions_count_df = interactions_count_df.rename(columns = {"level_0": "agent_id", "level_1": "timestep"})

    return interactions_count_df

def planned_trajectories_to_df(planned_trajectories: dict) -> pd.DataFrame:
    """
    Convert planned trajectories representation from .json like dictionary representation to a pandas DataFrame
    
    :param planned_trajectories: Dictionary with the scenario planned trajectories
    :type planned_trajectories: dict
    :return: DataFrame with the scenario planned trajectories
    :rtype: DataFrame
    """

    # Initialize planned trajectories DataFrame
    trajectories_df = pd.DataFrame(columns = ["vehicle_id", "timestep", "trajectory"])

    # Iterate over scenario vehicles
    for vehicle_id in planned_trajectories.keys():
        # Extract vehicle's planning timesteps and planned trajectories
        vehicle_planned_trajectories = planned_trajectories[vehicle_id]
        for planning_timestep, planned_trajectory in vehicle_planned_trajectories.items():
            # Append a new row to the DataFrame
            trajectories_df.loc[len(trajectories_df)] = [vehicle_id, planning_timestep, planned_trajectory]

    trajectories_df["vehicle_id"] = trajectories_df["vehicle_id"].astype(int)
    trajectories_df["timestep"] = trajectories_df["timestep"].astype(int)

    return trajectories_df

def dict_to_trajectory(trajectory_dict: dict) -> Trajectory:
    """
    Method to convert a .json like representation of a planned trajectory to a Trajectory object
    """
    # Extract trajectory initial timestep and state list
    initial_timestep = trajectory_dict["initial_time_step"]
    state_list = trajectory_dict["state_list"]

    converted_state_list = []
    # Extract info from each state
    for state in state_list:
        timestep = state["time_step"]
        position = np.array([state["position"][0], state["position"][1]])
        orientation = state['orientation']
        velocity = state['velocity']
        acceleration = state['acceleration']
        yaw_rate = state['yaw_rate']

        # For the initial timestep create an InitialState object, otherwise create a CustomState object
        if timestep == initial_timestep:
            converted_state_list.append(InitialState(time_step = timestep, position = position, orientation = orientation, velocity = velocity, acceleration = acceleration, yaw_rate = yaw_rate, slip_angle = 0))
        else:
            converted_state_list.append(CustomState(time_step = timestep, position = position, orientation = orientation, velocity = velocity, acceleration = acceleration, yaw_rate = yaw_rate, slip_angle = 0))

    return Trajectory(initial_time_step = initial_timestep, state_list = converted_state_list)

def augment_with_collisions(executed_scenario: Scenario, scenario_collisions: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    For each agent and time when Apollo oracle detected a collision, extract RIA, RIV, and Impact Location.
    """

    extended_collisions_rows = []
    issue_rows = []

    # Iterate over each collision detected by Apollo oracle
    for _, row in deepcopy(scenario_collisions).iterrows():
        # Extract the two colliding agents and the collision timestep
        agent = executed_scenario.obstacle_by_id(row["vehicle_id"])

        collision_timestep = int(row["collision_timestep"])

        # Check if the other agent involved in the collision is known, otherwise we need to find it based on the collision position
        if row["has_collided_with"] in [obstacle.obstacle_id for obstacle in executed_scenario.obstacles]:
            collided_agent = executed_scenario.obstacle_by_id(row["has_collided_with"])
        else:
            collided_agent_position_x, collided_agent_position_y = ast.literal_eval(row["colliding_vehicle_position"])

            epsilon = 2  # tolerance (meters)
            collided_agent_position_intervals = [
                Interval(collided_agent_position_x - epsilon, collided_agent_position_x + epsilon),
                Interval(collided_agent_position_y - epsilon, collided_agent_position_y + epsilon)
            ]

            collided_agents = executed_scenario.obstacles_by_position_intervals(
                position_intervals = collided_agent_position_intervals,
                obstacle_role = (ObstacleRole.DYNAMIC,),
                time_step = collision_timestep
            )

            if len(collided_agents) <= 1:
                # Not enough obstacles are detected at the collision position, we cannot compute the collision features, so we skip this collision
                row["detected_obstacles"] = [o.obstacle_id for o in collided_agents]
                issue_rows.append(row)
                continue
            elif len(collided_agents) == 2:
                # Get the other agent involved in the collision
                collided_agent = collided_agents[0] if collided_agents[0].obstacle_id != agent.obstacle_id else collided_agents[1]
            else:
                # Get the closer agent to the collision position
                distances = []
                for obstacle in collided_agents:
                    if obstacle.obstacle_id == agent.obstacle_id:
                        continue

                    obstacle_position = obstacle.state_at_time(collision_timestep).position
                    distance = compute_euclidean_distance((collided_agent_position_x, collided_agent_position_y), obstacle_position)
                    distances.append((obstacle, distance))
                    
                collided_agent, _ = min(distances, key = lambda x: x[1])

        # Compute collision metrics and extend original DataFrame row
        try:
            # Compute collision criticality features
            row["ria"], row["riv"], row["impact_location"] = compute_collision_features(collision_timestep, agent, collided_agent)
        except Exception as e:
            row["ria"], row["riv"], row["impact_location"] = None, None, None

        # if pd.notna(row["impact_location"]):
        extended_collisions_rows.append(row)

    return pd.DataFrame(extended_collisions_rows, columns = scenario_collisions.columns.tolist() + ["ria", "riv", "impact_location"]), issue_rows

def compute_collision_features(collision_timestep: int, agent: DynamicObstacle, collided_agent: DynamicObstacle) -> List[int]:
    """
    Compute RIA, RIV, and impact location for the most critical colliding object (e.g., the one for which RIV is max).
    Note: vehicles might not exactly touch each others, so we need to include the case of "nearby" vehicles

    The framework already compute those features. Compare and include or replace
    """
    ria = -1
    riv = -1
    impact_position = None

    # Returns the occupancy and state of the agent at collision timestep
    agent_occupancy_at_time_t = agent.occupancy_at_time(collision_timestep)    
    agent_state_at_time_t = agent.state_at_time(collision_timestep) 
    agent_state_at_time_t_minus_one = agent.state_at_time(collision_timestep - 1)

    if agent_occupancy_at_time_t is None or agent_state_at_time_t is None or agent_state_at_time_t_minus_one is None:
        module_logger.debug(f"Agent {agent.id} has missing state information at time {time_step}")
        return ria, riv, impact_position

    # Returns the occupancy and state of the other_agent at collision timestep
    other_agent_occupancy_at_time_t = collided_agent.occupancy_at_time(collision_timestep)
    other_agent_state_at_time_t = collided_agent.state_at_time(collision_timestep)
    other_agent_state_at_time_t_minus_one = collided_agent.state_at_time(collision_timestep - 1)

    # Compute Relative Impact Velocity (RIV)
    current_riv = compute_RIV(agent_state_at_time_t, other_agent_state_at_time_t)

    if current_riv > riv:
        riv = current_riv

        # Compute Relative Impact Angle (RIA)
        ria = compute_RIA(agent_state_at_time_t, agent_state_at_time_t_minus_one, other_agent_state_at_time_t, other_agent_state_at_time_t_minus_one)
        
        try:
            # Compute Impact Position
            impact_position = compute_impact_position(agent_occupancy_at_time_t, agent_state_at_time_t, other_agent_occupancy_at_time_t, other_agent_state_at_time_t)
        except AssertionError:
            # Fallback: vehicles are "near" but not intersecting
            impact_position = None
        
    return ria, riv, impact_position

def augment_with_danger(executed_scenario: Scenario) -> pd.DataFrame:
    """
    Extract the danger metric for each agent in the executed scenario.
    """
    danger_df = None

    # Extract scenario vehicles ids
    simulation_agents_ids = [obstacle.obstacle_id for obstacle in executed_scenario.obstacles]

    # Iterate over each scenario vehicle
    for agent_id in sorted(simulation_agents_ids):
        # For each agent, create a df and concatenate with the previous one
        agent = executed_scenario.obstacle_by_id(agent_id)

        # Extract initial and final timestep for the agent
        t_start = agent.initial_state.time_step
        t_end = agent.prediction.final_time_step

        # Initialize the max danger and the agent causing it with default values (0 and -1 respectively)
        max_danger = pd.Series(None, index = list(range(t_start, t_end + 1)))
        max_danger_caused_by = pd.Series(None, index = list(range(t_start, t_end + 1)))

        # Make sure we fill in all the values for this agent
        for t in range(t_start, t_end + 1):
            max_danger[t] = 0.0
            max_danger_caused_by[t] = -1

        # Iterate over agent timesteps
        for t in range(max([s.time_step for s in agent.prediction.state_list])):
            max_danger_at_time = 0
            max_danger_at_time_caused_by_agent_id = -1

            # Extract agent's state at timestep t
            state_of_agent_at_time_t = agent.state_at_time(t)

            # Check that the agent is active at this time step, otherwise we skip it
            if state_of_agent_at_time_t is None:
                max_danger[t] = max_danger_at_time
                max_danger_caused_by[t] = max_danger_at_time_caused_by_agent_id
                continue

            # Iterate over the other scenario agents
            for the_other_agent_id in [other_agent_id for other_agent_id in simulation_agents_ids if other_agent_id != agent_id]:
                the_other_agent = scenario.obstacle_by_id(the_other_agent_id)

                # Check that the other agent is active at this time step, otherwise we skip it
                if max([state.time_step for state in the_other_agent.prediction.state_list]) >= t:
                    # Extract the other agent's state at timestep t
                    state_of_other_agent_at_time_t = the_other_agent.state_at_time(t)

                    if state_of_other_agent_at_time_t is None:
                        continue

                    # Compute danger for the two states
                    danger = compute_danger_for_states(state_of_agent_at_time_t, state_of_other_agent_at_time_t) 

                    # Check if the computed danger is higher than the maximum danger for this time step
                    if danger > max_danger_at_time:
                        max_danger_at_time = danger
                        max_danger_at_time_caused_by_agent_id = the_other_agent_id
            
            # Overwrite the default value
            max_danger[t] = max_danger_at_time
            max_danger_caused_by[t] = max_danger_at_time_caused_by_agent_id

        # Update the agent df with the computed max danger and the agent causing it for each time step
        agent_danger_df = pd.DataFrame(None, index = list(range(t_start, t_end + 1)), columns = ["max_danger", "max_danger_caused_by_agent"])
        agent_danger_df["max_danger"] = max_danger
        agent_danger_df["max_danger_caused_by_agent"] = max_danger_caused_by
        
        # Concatenate the agent df with the previous ones
        if danger_df is None:
            danger_df = agent_danger_df.set_index([[agent_id] * len(agent_danger_df), agent_danger_df.index])
        else:
            agent_danger_df = agent_danger_df.set_index([[agent_id] * len(agent_danger_df), agent_danger_df.index])
            danger_df = pd.concat([danger_df, agent_danger_df])

    # At this point danger_df should be organized exactly as criticality_dataframe so they can be merged so we reset the index
    danger_df.reset_index(inplace = True)
    
    return danger_df