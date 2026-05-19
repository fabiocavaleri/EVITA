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

def compute_trajectory_length(state_list: List) -> float:
    """
    From the planned trajectory state list compute geometrical length
    """
    total_length = 0.0

    # Iterate over state list
    for i in range(len(state_list) - 1):
        # Get planned position from two consecutive states
        p1 = state_list[i].position
        p2 = state_list[i + 1].position

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        total_length += math.hypot(dx, dy)

    return total_length

def crop_trajectories(trajectory_before: Trajectory, trajectory_after: Trajectory) -> Tuple[Trajectory, Trajectory]:
    """
    Crop two consecutive planned trajectories to the same interval of observations, so they can be compared in a meaningful way.
    """

    # Check if the two trajectories have an overlapping time interval, otherwise we cannot crop them to the same interval
    if trajectory_after.initial_time_step < trajectory_before.initial_time_step:
        raise ValueError("trajectory_after starts earlier than trajectory_before")

    # Get cropped trajectories initial and final timestep
    initial_timestep = max(trajectory_before.initial_time_step, trajectory_after.initial_time_step)
    final_timestep = min(trajectory_before.final_state.time_step, trajectory_after.final_state.time_step)

    # Crop the trajectories to the same interval of observations
    cut_trajectory_b, cut_trajectory_a = get_cut_trajectory_start(trajectory_before, trajectory_after, initial_timestep)
    cut_trajectory_b, cut_trajectory_a = get_cut_trajectory_end(cut_trajectory_b, cut_trajectory_a, final_timestep)

    return cut_trajectory_a, cut_trajectory_b

def get_cut_trajectory_end(trajectory_before: Trajectory, trajectory_after: Trajectory, cut_timestep: int) -> Tuple[Trajectory, Trajectory]:
    """
    Crop the two trajectories to the same final timestep
    """
    cropped_trajectories = []

    for trajectory in [trajectory_before, trajectory_after]:
        # Extract trajectory state list
        states = deepcopy(trajectory.state_list)

        # Compute index of the state corresponding to the cut timestep
        cut_index = next((i for i, s in enumerate(states) if s.time_step == cut_timestep))

        # Cut trajectory state list
        states = states[:cut_index + 1]

        # Rebuild trajectory object
        cropped_trajectories.append(Trajectory(states[0].time_step, states))
    
    return cropped_trajectories[0], cropped_trajectories[1]

def get_cut_trajectory_start(trajectory_before: Trajectory, trajectory_after: Trajectory, cut_timestep: int) -> Tuple[Trajectory, Trajectory]:
    """
    Crop the two trajectories to the same initial timestep
    """
    cropped_trajectories = []

    for trajectory in [trajectory_before, trajectory_after]:
        # Extract trajectory state list
        states = deepcopy(trajectory.state_list)

        # Compute index of the state corresponding to the cut timestep
        cut_index = next((i for i, s in enumerate(states) if s.time_step == cut_timestep))

        # Cut trajectory state list
        states = states[cut_index:]
        
        # Rebuild trajectory object
        cropped_trajectories.append(Trajectory(states[0].time_step, states))
    
    return cropped_trajectories[0], cropped_trajectories[1]

def compute_difference_in_cc_space(scenario: Scenario, trajectory_at_time_t: Trajectory, trajectory_at_time_next_t: Trajectory, use_relative_path = False) -> Tuple[float, float]:
    """
    Compute difference in consecutive planned trajectories.
    This metric is aimed to measures the impact of on interaction on the planned trajectory 
    """
    # Uniform trajectories timesteps so they cover the same interval of observations
    cropped_traj_a, cropped_traj_b = crop_trajectories(trajectory_at_time_t, trajectory_at_time_next_t)

    # Extract the latest overlapping state (where the AV wants to go)
    pos_a = cropped_traj_a.final_state.position    
    pos_b = cropped_traj_b.final_state.position

    # It might happen that the final states are out of the road, because those are planned trajectories and those in emergency mode might result in this
    # So we need to consider an alternative strategy to label this interaction. If we can assume that NOT all the planned states are outside the road
    # we look up in the past states until we find a lanelet

    lanelet_id = None
    # Go backwards starting from last planned state until we find a position on a valid lanelet
    for target_state_a in reversed(cropped_traj_a.state_list):
        for _lanelet_id in [l for l in scenario.lanelet_network.find_lanelet_by_position([target_state_a.position]) if len(l) > 0]:
            lanelet_id = _lanelet_id[0]
            break

    if lanelet_id is None:
        return None, None

    # Get lanelet object
    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

    if not use_relative_path:
        # Use lanelet centerline as reference path to build the curvilinear coordinate system
        path = lanelet.center_vertices
    else:
        # Use the line passing from the middle of the vehicle, rotated according to the vechile's orientation,
        # as reference path to build the curvilinear coordinate system
        orientation_a = cropped_traj_a.final_state.orientation
        xs = [pos_a[0] + distance * math.cos(orientation_a) for distance in np.linspace(-100, 100, 100)]
        ys = [pos_a[1] + distance * math.sin(orientation_a) for distance in np.linspace(-100, 100, 100)]
        path = np.array([[px, py] for px, py in zip(xs, ys)])

    # Make sure the path is long enough to avoid the out of domain projection error
    try:
        extended_path = np.array(scale(LineString(path), 3.0, 3.0).coords)
        extended_path = resample_polyline(extended_path, step = 1)
        cosy = CoordinateSystem(extended_path)

        # (s,d) - s is distance from the beginning of the refpath (long), d is the lateral displacement
        cosy_a = cosy.convert_to_curvilinear_coords(pos_a[0], pos_a[1])
        cosy_b = cosy.convert_to_curvilinear_coords(pos_b[0], pos_b[1])
        diff_long = cosy_a[0] - cosy_b[0]
        diff_lat = cosy_a[1] - cosy_b[1]

        return diff_long, diff_lat
    except Exception as exc:
        def debug_plot():
            import os
            import matplotlib.pyplot as plt
            from commonroad.visualization.mp_renderer import MPRenderer

            rnd = MPRenderer()
            rnd.draw_params.lanelet_network.traffic_light.draw_traffic_lights = False
            scenario.lanelet_network.draw(rnd)
            rnd.render()

            print("=====")
            print(path)
            print("=====")
            print(pos_a)
            print("=====")
            print(pos_b)
            print("=====")
            print(cropped_traj_a)
            print("=====")
            print(cropped_traj_b)
            print("=====")

            plt.plot(path[:, 0], path[:, 1], "-", zorder=99, color="black", lw=2)
            plt.plot(extended_path[:, 0], extended_path[:, 1], "-", zorder=98, color="green", lw=2)

            # Plot the positions pos a and pos b
            plt.plot(pos_a[0], pos_a[1], "ro", zorder=100, lw=2)
            plt.plot(pos_b[0], pos_b[1], "g+", zorder=100, lw=2)
            

            plt.plot([s.position[0] for s in cropped_traj_a.state_list], [s.position[1] for s in cropped_traj_a.state_list], "r", zorder=98, lw=2)
            plt.plot([s.position[0] for s in cropped_traj_b.state_list], [s.position[1] for s in cropped_traj_b.state_list], "g", zorder=98, lw=2)
            # Plot the reference path used to build the coordinate system
            
            os.makedirs("debug_plots", exist_ok = True)
            count = len(os.listdir("debug_plots"))
            plt.savefig(f"debug_plots/{count}.png", dpi = 800)
            plt.close()

        if use_relative_path:
            # print("Debug Plot")
            # debug_plot()
            return None, None
        
        raise exc
    except ValueError:
        # One or both points are outside the curvilinear projection domain
        return None, None
    
def list_anyone_nearby(timestep: int, agent_occupancy: Occupancy, other_agents: List[DynamicObstacle]) -> List[int]:
    """
    List `other_agents` driving within safety distance from `agent`
    """
    distance = float('inf')
    nearby_agent_ids = []
    nearby_agent_distances = []

    # Iterate over the other agents
    for other_agent in other_agents:
        other_agent_occupancy = other_agent.occupancy_at_time(timestep)
        if agent_occupancy is not None and other_agent_occupancy is not None:
            distance = agent_occupancy.shape.shapely_object.distance(other_agent_occupancy.shape.shapely_object)
            # If the two vehicles are closer than the safety distance, the second condition (C2) for interactions holds
            if distance < VEHICLES_SAFETY_DISTANCE:
                nearby_agent_ids.append(other_agent.obstacle_id)
                nearby_agent_distances.append(distance)

    return nearby_agent_ids, nearby_agent_distances
    
def is_trajectory_similar(trajectory_a: Trajectory, trajectory_b: Trajectory, tolerance_distance: int = 1.5):
    """
    Compare trajectories (with same length) with given tolerance deviation
    """

    # Check if the two trajectories have the same length
    if len(trajectory_a.state_list) != len(trajectory_b.state_list):
        print(f'Trajectory comparison: Not the same length {len(trajectory_a.state_list)} != {len(trajectory_b.state_list)}')
        return False

    # Iterate over the two planned trajectories states
    for state_a, state_b in zip(trajectory_a.state_list, trajectory_b.state_list):
        # Check if the two planned states are similar in terms of position
        is_pos_equal = is_position_similar(state_a.position, state_b.position, tolerance_distance)
        
        if not is_pos_equal:
            return False
    
    # Return true only if all the planned states are similar
    return True

def is_position_similar(pos_a: Tuple[float, float], pos_b: Tuple[float, float], tolerance_distance: int = 1) -> bool:
    """
    Check if two planned states are similar
    """
    # Compute the euclidean distance between the two planned positions
    euclidean_distance = compute_euclidean_distance(pos_a, pos_b)

    if euclidean_distance < tolerance_distance:
        return True
    else:
        return False
    
def compute_euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]):
    position_diff = (pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2
    
    return math.sqrt(position_diff)
    """
    Compute distance between the observed agent and the perceived vehicles
    """
    
    
    perceived_vehicles = []
    # Iterate over perceived vehicles entries
    for _, row in perceived_vehicles_at_next_t.iterrows():
        # Extract perceived vehicle id, position and orientation
        obstacle_id = row["obstacle_id"]


        # Save data
        perceived_vehicles.append([obstacle_id, distance])

    return perceived_vehicles

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

def augment_with_interactions(executed_scenario: Scenario, planned_trajectories: pd.DataFrame, scenario_vehicles_perceptions: pd.DataFrame, scenario_traffic_lights_perceptions: pd.DataFrame, scenario_collisions: pd.DataFrame) -> pd.DataFrame:
    """
    Given the scenario vehicles planned trajectories and perceptions data, detect the occurrence of interactions between scenario's "obstacles"
    """
    interactions_count_df = None

    long_bins, lat_bins = get_interaction_map_bins()

    # Include 0 - B[0]:B[9] - 10 and len(bins)
    rows = len(long_bins) + 2 # 0 <, bins, len(bins) +1 or +2?
    cols = len(lat_bins) + 2 # 0 <, bins, len(bins)

    interaction_map = np.zeros((rows, cols))

    # Extract scenario vehicles ids
    simulation_agents_ids = [obstacle.obstacle_id for obstacle in executed_scenario.obstacles]

    # Iterate over each scenario vehicle
    for agent_id in tqdm(sorted(simulation_agents_ids), desc = "Interaction Metrics"):
        tqdm.write(f"Agent: {agent_id}")

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
               (next_t - t) > 15 or \
               (agent_collision_timestep is not None and (t >= agent_collision_timestep or next_t >= agent_collision_timestep)):
                continue

            # Extract planned trajectories at time `t` and `next_t`
            trajectory_at_time_t = dict_to_trajectory(planned_trajectories.loc[(planned_trajectories['vehicle_id'] == agent_id) & (planned_trajectories['timestep'] == t), 'trajectory'].iloc[0]["planned_trajectory"])
            trajectory_at_time_next_t_data = planned_trajectories.loc[(planned_trajectories['vehicle_id'] == agent_id) & (planned_trajectories['timestep'] == next_t), 'trajectory'].iloc[0]
            trajectory_at_time_next_t = dict_to_trajectory(trajectory_at_time_next_t_data["planned_trajectory"])

            replanning_decision = trajectory_at_time_next_t_data["decision"]
            decision_object = trajectory_at_time_next_t_data["decision_object"]

            # trajectory_length = compute_trajectory_length(trajectory_at_time_next_t.state_list)

            # Get `agent` occupancy at time `next_t`
            agent_occupancy_at_time_next_t = agent.occupancy_at_time(next_t)

            interacting_obstacles = []

            if replanning_decision in ["cruise", "STOP_REASON_OBSTACLE", "STOP_REASON_SIDE", "STOP_REASON_SIDEPASS_SAFETY"]:
                # Replanning due to another Vehicle
                if decision_object in [None, "Side_Pass_Stop"]:
                    other_agents = [a for a in executed_scenario.obstacles if a.obstacle_id != agent_id]
                    nearby_agents_ids, nearby_agents_distances = list_anyone_nearby(next_t, agent_occupancy_at_time_next_t, other_agents)
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

                        if agent_occupancy_at_time_next_t is not None:
                            # Compute distance to observed agent
                            distance = agent_occupancy_at_time_next_t.shape.shapely_object.distance(obstacle_shape.shapely_object)

                            nearby_agents_ids = [interacting_agent_id]
                            nearby_agents_distances = [distance]

                            interacting_obstacles.extend(nearby_agents_ids)
                    # else:
                    #     nearby_agents_ids = []
                    #     nearby_agents_distances = []

                intersecting_stop_line_agents = check_stop_line_crossing(executed_scenario, trajectory_at_time_next_t)
                interacting_obstacles.extend(intersecting_stop_line_agents)

                # nearby_agents_ids.extend(intersecting_stop_line_agents)

                # print(agent_id, next_t, "Vehicle", decision_object, nearby_agents_ids, nearby_agents_distances, trajectory_length, similar, diff_0, diff_1)
            else:
                # Check if there is any close vehicle anyway
                other_agents = [a for a in executed_scenario.obstacles if a.obstacle_id != agent_id]
                nearby_agents_ids, nearby_agents_distances = list_anyone_nearby(next_t, agent_occupancy_at_time_next_t, other_agents)
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
            # if len(interacting_obstacles) > 0:
                # Compute difference in consecutive planned trajectories
                try:
                    diff_0, diff_1 = compute_difference_in_cc_space(executed_scenario, trajectory_at_time_t, trajectory_at_time_next_t)
                except:
                    # If this fails, we cannot do much
                    diff_0, diff_1 = compute_difference_in_cc_space(executed_scenario, trajectory_at_time_t, trajectory_at_time_next_t, use_relative_path = True)

                # Skip interaction in case of errors in trajectory difference computation
                if diff_0 is None or diff_1 is None:
                    continue

                is_interacting[next_t] = 1

                is_interacting_with[next_t] = interacting_obstacles

                diff_long[next_t] = diff_0
                diff_lat[next_t] = diff_1

                # Compute interaction map indexes
                digitized_long = np.digitize(diff_0, long_bins)
                digitized_lat = np.digitize(diff_1, lat_bins)

                interaction_map[digitized_long][digitized_lat] += 1.0

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

    return interactions_count_df, interaction_map

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
        row["ria"], row["riv"], row["impact_position"] = compute_collision_features(collision_timestep, agent, collided_agent)
        if pd.notna(row["impact_position"]):
            extended_collisions_rows.append(row)

    return pd.DataFrame(extended_collisions_rows,
                        columns = scenario_collisions.columns.tolist() + ["ria", "riv", "impact_position"]), issue_rows

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
    # May need one timestep earlier
    agent_state_at_time_t_minus_one = agent.state_at_time(collision_timestep - 1)

    # Returns the occupancy and state of the other_agent at collision timestep
    other_agent_occupancy_at_time_t = collided_agent.occupancy_at_time(collision_timestep)
    other_agent_state_at_time_t = collided_agent.state_at_time(collision_timestep)
    # May need one timestep earlier
    other_agent_state_at_time_t_minus_one = collided_agent.state_at_time(collision_timestep - 1)

    # Compute Relative Impact Velocity (RIV)
    current_riv = compute_RIV(agent_state_at_time_t, other_agent_state_at_time_t)

    if current_riv > riv:
        riv = current_riv

        # Compute Relative Impact Angle (RIA)
        ria = compute_RIA(agent_state_at_time_t, agent_state_at_time_t_minus_one, 
                            other_agent_state_at_time_t, other_agent_state_at_time_t_minus_one)
        
        try:
            # Compute Impact Position
            impact_position = compute_impact_position(agent_occupancy_at_time_t, agent_state_at_time_t,
                                                        other_agent_occupancy_at_time_t, other_agent_state_at_time_t)
        except AssertionError:
            # Fallback: vehicles are "near" but not intersecting
            impact_position = None
        
    return ria, riv, impact_position