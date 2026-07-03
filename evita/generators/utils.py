# Contains Utility Methods to Generate and Evolve the Scenarios
import math
import random
import numpy as np

from typing import List, Optional, Tuple

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion, Interval
from commonroad.common.file_reader import CommonRoadFileReader

from scipy.cluster.hierarchy import DisjointSet

from cr_scenario_handler.utils.agent_status import AgentStatus
from cr_scenario_handler.utils.utils_coordinate_system import CoordinateSystem

# from cr_scenario_handler.evaluation.collision_report import MIN_DISTANCE_FOR_COLLISION
MIN_DISTANCE_FOR_COLLISION = 0.1

from shapely.affinity import rotate, translate, scale
from shapely.geometry import LineString, Point, Polygon

from commonroad_dc.geometry.util import resample_polyline

import geopandas

from ..config import ScenarioGeneratorConfig

# Get the main logger
import logging
from ..utils import EVITA_MAIN_LOGGER_NAME

main_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

def orientation_by_position(lanelet: Lanelet, position: np.array):
    """
    Compute the orientation of the lanelet at the given position
    The orientation is computed by finding the closest two vertices of the lanelet centerline and computing the angle between them
    """
    # Extract lanelet center vertices and resample them to have a point every 1 meter
    center_vertices = lanelet.center_vertices
    center_vertices = resample_polyline(center_vertices, step=1.0)

    # Find the closest two vertices of the lanelet centerline to the given position
    position_diff_square = np.sum((center_vertices - position) ** 2, axis=1)
    closest_vertex_index = np.argmin(position_diff_square)

    if closest_vertex_index == len(center_vertices) - 1:
        vertex1 = center_vertices[closest_vertex_index - 1, :]
        vertex2 = center_vertices[closest_vertex_index, :]
    else:
        vertex1 = center_vertices[closest_vertex_index, :]
        vertex2 = center_vertices[closest_vertex_index + 1, :]

    direction_vector = vertex2 - vertex1

    return np.arctan2(direction_vector[1], direction_vector[0])

def get_random_init_state(config: ScenarioGeneratorConfig, lanelet: Lanelet) -> Optional[InitialState]:
    """
    Return a random Initial State of the vehicle
    """
    # Extract a random velocity between min and max
    velocity = random.uniform(config.min_init_velocity, config.max_init_velocity)

    min_idx = 0
    if not lanelet.predecessor:
        # This is an entry lanelet, there's nothing before it. So we cannot pick the first centerpoints
        # Otherwise the vehicle might not be entirely inside the road!
        distance = 0.0
        # 2 is the minimum length to create a LineString
        min_idx = min_idx + 2
        # Loop until we find a point that is at least 3 meters away from the lanelet start, to make sure the vehicle is entirely inside the road
        while distance < 3.0:
            try:
                distance = LineString(lanelet.center_vertices[0:min_idx]).length
                min_idx += 1
            except IndexError as ind_err:
                return None
            except Exception as e:
                module_logger.critical(f"Unexpected exception {e}")
                return None
            
    # This may be triggered by too short lanelets
    if min_idx >= len(lanelet.center_vertices):
        return None

    # Randomly pick a point along the lanelet centerline, but not too close to the start if it's an entry lanelet
    random_index = random.choice(range(min_idx, len(lanelet.center_vertices) - 1))
    position = lanelet.center_vertices[random_index]

    # Get the next point to compute the orientation
    next_point = lanelet.center_vertices[random_index + 1]
    orientation = get_orientation_by_coords((position[0], position[1]), (next_point[0], next_point[1]))

    # Those are kept at zero at the beginning
    acceleration = 0.0
    yaw_rate = 0.0
    slip_angle = 0.0

    # Randomply pick a time step between 0 and the max wait time, to make sure that not all the vehicles start at the same time step
    from EVITA_DoppelTest.config import INSTANCE_MAX_WAIT_TIME
    initial_time_step = random.randint(0, INSTANCE_MAX_WAIT_TIME)
    
    return InitialState(time_step = initial_time_step, 
                        position = position,
                        orientation = orientation, 
                        velocity = velocity, 
                        acceleration = acceleration,
                        yaw_rate = yaw_rate,
                        slip_angle = slip_angle)

def get_possible_goal_for_lanelet(config: ScenarioGeneratorConfig, goal_lanelet: Lanelet) -> Rectangle:
    """
    Return the Rectangle corresponding to the Goal Area for the given lanelet
    """
    # Reverse center vertices to get the distance from lanelet end with cut()
    reversed_center_vertices = LineString(goal_lanelet.center_vertices[::-1])
    goal_center_vertices: LineString = cut(reversed_center_vertices, config.dist_to_end)[0]
    goal_center = goal_center_vertices.centroid

    # last_point is at index 0 because vertices were reversed previously
    last_point = goal_center_vertices.coords[0]
    orientation = get_orientation_by_coords((goal_center.x, goal_center.y), (last_point[0], last_point[1]))

    return Rectangle(config.length, config.width, np.array([goal_center.x, goal_center.y]), orientation)

def generate_goal(config: ScenarioGeneratorConfig, goal_area: Rectangle) -> GoalRegion:
    """
    Return a well formed Goal State object from the rectangle.
    By default, AV must reach the goal area before the end of the scenario
    """
    goal_state_list = [CustomState(position = goal_area, time_step=Interval(0, config.max_time_step))]
    return GoalRegion(goal_state_list)

def get_base_scenario_from_file(commonroad_scenario_file: str, remove_obstacles: bool, remove_planning_problems: bool) -> Scenario:
    """
    Load the scenario from the given file and remove obstacles and/or planning problems if specified
    """
    # Load CommonRoad scenario from File
    crfr = CommonRoadFileReader(commonroad_scenario_file)
    scenario: Scenario

    # Open the scenario, but discard the planning problems
    scenario, planning_problem_set = crfr.open()
    
    if remove_obstacles:
        # Remove all the obstacles from the scenario
        [scenario.remove_obstacle(obstacle) for obstacle in scenario.obstacles]
    
    if remove_planning_problems:
        # Remove all the planning problems
        planning_problem_set.planning_problem_dict.clear()

    # Return the basic scenario
    return scenario, planning_problem_set

def parse_scenario_generator_configuration_from_kwargs(**kwargs) -> ScenarioGeneratorConfig:
    """
    Create a ScenarioGeneratorConfiguration object from the given kwargs, by updating only the matching attributes
    """
    # Create Default ScenarioGeneratorConfiguration
    scenario_generator_config = ScenarioGeneratorConfig()

    # Update only matching attributes
    for key, value in kwargs.items():
        if hasattr(scenario_generator_config, key):
            setattr(scenario_generator_config, key, value)

    return scenario_generator_config

def get_adjacent_lanelets(lanelet: Lanelet, ll_network: LaneletNetwork) -> List[Lanelet]:
    """
    Recursively find adjacent lanelets with same drving direction
    """
    adjancet_lanelets = []
    processed_ids = [lanelet.lanelet_id]
    backlog = [lanelet]

    # Loop until there are no more lanelets to process
    while len(backlog) != 0:
        the_lanelet = backlog.pop()
        # Check left and right adjacent lanelets, if they exist, have the same driving direction and were not processed yet, add them to the list and to the backlog
        if the_lanelet.adj_left and the_lanelet.adj_left_same_direction and the_lanelet.adj_left not in processed_ids:
            the_left_lanelet = ll_network.find_lanelet_by_id(the_lanelet.adj_left)
            adjancet_lanelets.append(the_left_lanelet)
            backlog.append(the_left_lanelet)
            processed_ids.append(the_lanelet.adj_left)

        # Check right adjacent lanelet, if it exists, has the same driving direction and was not processed yet, add it to the list and to the backlog
        if the_lanelet.adj_right and the_lanelet.adj_right_same_direction and the_lanelet.adj_right not in processed_ids:
            the_right_lanelet = ll_network.find_lanelet_by_id(the_lanelet.adj_right)
            adjancet_lanelets.append(the_right_lanelet)
            backlog.append(the_right_lanelet)
            processed_ids.append(the_lanelet.adj_right)
        
    return adjancet_lanelets

def get_successor_lanelets(lanelet: Lanelet, ll_network: LaneletNetwork) -> List[Lanelet]:
    """
    Recursively find successor lanelets and merge them with the given lanelet to create a single lanelet with the same drving direction
    """
    full_lanelets = [lanelet]
    processed_ids = [lanelet.lanelet_id]
    # Loop until there are no more lanelets to process, i.e., until all the successor lanelets have been merged into the full_lanelet
    while _have_lanelets_successors(full_lanelets, processed_ids):
        new_full_lls = []
        for ll in full_lanelets:
            if len(ll.successor) == 0:
                new_full_lls.append(ll)
            else:
                for succ_id in ll.successor:
                    succ_ll = ll_network.find_lanelet_by_id(succ_id)
                    full_lanelet = Lanelet.merge_lanelets(ll, succ_ll)
                    new_full_lls.append(full_lanelet)
                    processed_ids.append(succ_ll.lanelet_id)

        full_lanelets = new_full_lls

    return full_lanelets

def _have_lanelets_successors(lanelets: List[Lanelet], visited_ll_ids: List[int]):
    """
    Check if at least one of the lanelets has a successor that was not visited yet
    """
    have_successors = any(len(ll.successor) != 0 and any(
        ll_id not in visited_ll_ids for ll_id in ll.successor) for ll in lanelets)
    
    return have_successors

def cut(line: LineString, distance: float):
    """
    Cuts a line in two at a specified distance from its starting point.
    """
    # Skip if the distance is out of bounds
    if distance <= 0.0 or distance >= line.length:
        return [line]
    
    coords = list(line.coords)
    # Iterate over line coordinates to find the point where the line should be cut
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        
        if pd == distance:
            return [
                LineString(coords[:i + 1]),
                LineString(coords[i:])]
        
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def get_orientation_by_coords(first_point: Tuple[float, float], next_point: Tuple[float, float]):
    """
    Compute orientation: https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors
    """
    a_x, a_y = next_point
    b_x, b_y = first_point

    return math.atan2(a_y - b_y, a_x - b_x)

# ========== Trajectory Utilities ==========
from commonroad.prediction.prediction import Trajectory
from copy import deepcopy

def crop_trajectories(trajectory_before: Trajectory, trajectory_after: Trajectory) -> Tuple[Trajectory, Trajectory]:
    """
    Crop the second trajectory to the first one, so both have the same length and initial time step
    """
    # Skip if the trajectory_after starts earlier than trajectory_before
    if trajectory_after.initial_time_step < trajectory_before.initial_time_step:
        raise Exception('trajectory_b has a larger time_step and therefore can not be cropped to trajectory_a')
    # Compute time delta between the two trajectories and crop the trajectory_after to the trajectory_before
    time_step_diff = trajectory_after.initial_time_step - trajectory_before.initial_time_step
    cut_trajectory_a = get_cut_trajectory_end(trajectory_after, time_step_diff)
    cut_trajectory_b = get_cut_trajectory_start(trajectory_before, time_step_diff)

    return cut_trajectory_a, cut_trajectory_b

def get_cut_trajectory_end(trajectory: Trajectory, amount: int) -> Trajectory:
    """
    Crop the trajectory by cutting the last "amount" states
    """
    states = deepcopy(trajectory.state_list)
    cut_off = len(states) - amount
    states = states[:cut_off]

    return Trajectory(states[0].time_step, states)

def get_cut_trajectory_start(trajectory: Trajectory, amount: int) -> Trajectory:
    """
    Crop the trajectory by cutting the first "amount" states
    """
    states = deepcopy(trajectory.state_list)
    states = states[amount:]

    return Trajectory(states[0].time_step, states)

# ========== Evaluation methods ==========
from enum import Enum

def is_position_similar(pos_a: Tuple[float, float], pos_b: Tuple[float, float], tolerance_distance: int = 1) -> bool:
    """
    Compare two positions with a given tolerance distance
    If the euclidean distance between the two positions is smaller than the tolerance distance, they are considered similar
    """
    euclidean_distance = compute_euclidean_distance(pos_a, pos_b)

    if euclidean_distance < tolerance_distance:
        return True
    else:
        return False
    
def is_trajectory_similar(trajectory_a: Trajectory, trajectory_b: Trajectory, tolerance_distance: int = 1):
    """
    Compare two trajectories with a given tolerance distance
    If the euclidean distance between the two trajectories is smaller than the tolerance distance, they are considered similar
    """
    # Skip compaison between trajectories of different length, as they are not comparable
    if len(trajectory_a.state_list) != len(trajectory_b.state_list):
        module_logger.debug(f'Trajectory comparison: Not the same length {len(trajectory_a.state_list)} != {len(trajectory_b.state_list)}')
        return False

    # Iterate over the two trajectories states and compare the positions of the states with the given tolerance distance
    for state_a, state_b in zip(trajectory_a.state_list, trajectory_b.state_list):
        is_pos_equal = is_position_similar(state_a.position, state_b.position, tolerance_distance)
        if not is_pos_equal:
            return False
        
    return True

def list_anyone_nearby(time_step: int, agent, other_agents, min_distance = 10) -> List[int]:
    """
    List the IDs of all agents that are nearby the given agent at the given time step
    """

    distance = float('inf')
    nearby_agent_ids = []

    # Check if the agent is active at time `time_step`
    if len([s for s in agent.prediction.trajectory.state_list if s.time_step == time_step]) == 0:
        return nearby_agent_ids
    
    # Get the occupancy of the agent at time `time_step`
    agent_occupancy_at_time_t = agent.occupancy_at_time(time_step)

    # Extract other active agents at time `time_step`
    other_agents_at_time_t =[]
    for other_agent in other_agents:
        if len([s for s in other_agent.prediction.trajectory.state_list if s.time_step == time_step]) > 0:
            other_agents_at_time_t.append(other_agent)

    # For each other agent, compute the distance to the given agent and check if it is below the minimum distance
    for other_agent in other_agents_at_time_t:
        # Compute distance of their shapes if possible
        other_agent_occupancy_at_time_t = other_agent.occupancy_at_time(time_step)
        
        if agent_occupancy_at_time_t and other_agent_occupancy_at_time_t:
            distance = agent_occupancy_at_time_t.shape.shapely_object.distance(other_agent_occupancy_at_time_t.shape.shapely_object)
            
            # If the distance is below the minimum distance, add the other agent's ID to the list of nearby agents
            if distance < min_distance:
                nearby_agent_ids.append(other_agent.obstacle_id)
        
    return list(set(nearby_agent_ids))

def evaluate_trajectories(trajectory_map, pp_id, index):
    """
    Evaluate the similarity of two trajectories of the same planning problem at two different time steps
    """
    # subtract -1 and -2 because list starts at 0 but simulation step starts at 1
    trajectory_new = trajectory_map[pp_id][index - 1]
    trajectory_old = trajectory_map[pp_id][index - 2]
    cropped_traj_a, cropped_traj_b = crop_trajectories(trajectory_new, trajectory_old)
    
    return not is_trajectory_similar(cropped_traj_a, cropped_traj_b)

def compute_euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]):
    """
    Compute the euclidean distance between two positions
    """
    position_diff = (pos_b[0] - pos_a[0]) ** 2 + (pos_b[1] - pos_a[1]) ** 2
    
    return math.sqrt(position_diff)

import pandas as pd
from cr_scenario_handler.simulation.simulation import Simulation
from commonroad.scenario.state import CustomState, InitialState

def compute_2D_velocity_vectors(velocity: float, sin, cos):
    """
    Compute the 2D velocity vectors from the velocity and the orientation
    """
    vector_x = velocity * cos
    vector_y = velocity * sin
    
    return vector_x, vector_y

def compute_sin_cos_from_orientation(orientation: float):
    """
    Compute the sin and cos of the orientation
    """
    cos = math.cos(orientation)
    sin = math.sin(orientation)
    
    return sin, cos

def compute_danger_for_states(state_a: CustomState, state_b: CustomState):
    """
    Compute the danger metric for two states
    """
    # Compute sin and cos of the two states' orientations
    sin_a, cos_a = compute_sin_cos_from_orientation(state_a.orientation)
    sin_b, cos_b = compute_sin_cos_from_orientation(state_b.orientation)
    # Compute the 2D velocity vectors of the two states
    velocity_x_a, velocity_y_a = compute_2D_velocity_vectors(state_a.velocity, sin_a, cos_a)
    velocity_x_b, velocity_y_b = compute_2D_velocity_vectors(state_b.velocity, sin_b, cos_b)
    # Compute the relative velocity between the two states
    diff_x = velocity_x_a - velocity_x_b
    diff_y = velocity_y_a - velocity_y_b
    relative_velocity = math.sqrt(diff_x ** 2 + diff_y ** 2)

    # Compute euclidean distance between the two states
    euclidean_distance = compute_euclidean_distance(state_a.position, state_b.position)

    # Compute danger criticality metric
    result = relative_velocity / euclidean_distance

    return result

def augment_with_danger(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the danger metric for each agent in the simulation and augment the criticality df with it
    """
    danger_df = None

    # Iterate over scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        # For each agent, create a df and concatenate with the previous one
        agent_id = agent.id

        # Extract initial and final timestep for the agent
        t_start = simulation.scenario.obstacle_by_id(agent_id).initial_state.time_step
        t_end = simulation.scenario.obstacle_by_id(agent_id).prediction.final_time_step

        module_logger.debug(f"Evaluate agent {agent_id} - DANGER")
        
        # Initialize the max danger and the agent causing it with default values (0 and -1 respectively)
        max_danger = pd.Series(None, index = list(range(t_start, t_end+1)))
        max_danger_caused_by = pd.Series(None, index = list(range(t_start, t_end+1)))

        # Make sure we fill in all the values for this agent
        for t in range(t_start, t_end + 1):
            max_danger[t] = 0.0
            max_danger_caused_by[t] = -1

        # Iterate over agent timesteps
        for t in range(max([s.time_step for s in agent.record_state_list])):
            max_danger_at_time = 0
            max_danger_at_time_caused_by_agent_id = -1

            # Check that the agent is active at this time step, otherwise we skip it
            if len([s for s in agent.record_state_list if s.time_step == t]) < 1:
                max_danger[t] = max_danger_at_time
                max_danger_caused_by[t] = max_danger_at_time_caused_by_agent_id
                continue

            # Extract agent's state at timestep t
            state_of_agent_at_time_t = [s for s in agent.record_state_list if s.time_step == t][0]

            # Iterate over the other scenario agents
            for the_other_agent in [a for a in simulation.agents if a.id != agent_id]:
                the_other_agent_id = the_other_agent.id

                # Check that the other agent is active at this time step, otherwise we skip it
                if max([s.time_step for s in the_other_agent.record_state_list]) >= t:
                    if len([s for s in the_other_agent.record_state_list if s.time_step == t]) < 1:
                        continue

                    # Extract the other agent's state at timestep t
                    state_of_other_agent_at_time_t = [s for s in the_other_agent.record_state_list if s.time_step == t][0]

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
        agent_danger_df = pd.DataFrame(None, index = list(range(t_start, t_end+1)), columns = ["max_danger", "max_danger_caused_by_agent"])
        agent_danger_df["max_danger"] = max_danger
        agent_danger_df["max_danger_caused_by_agent"] = max_danger_caused_by
        
        if danger_df is None:
            danger_df = agent_danger_df.set_index([[agent_id] * len(agent_danger_df), agent_danger_df.index])
        else:
            agent_danger_df = agent_danger_df.set_index([[agent_id] * len(agent_danger_df), agent_danger_df.index])
            danger_df = pd.concat([danger_df, agent_danger_df])

    # At this point danger_df should be organized exactly as criticality_dataframe so they can be merged so we reset the index
    danger_df.reset_index(inplace=True)
    
    # We merge the criticality dataframe
    criticality_dataframe = pd.merge(criticality_dataframe, danger_df)
    
    return criticality_dataframe

def list_vehicles_involved_in_collision(time_step: int, agent, other_agents: list, scenario: Scenario) -> List[int]:
    """
    This function groups vehicles involved in the same collision at the given time_step
    """
    # List all the agents collided at or after time_step
    collided_agents = [a for a in [agent] + other_agents if a.agent_state.status == AgentStatus.COLLISION and time_step >= a.agent_state.last_timestep]
    
    # Group the collided agents by distance. Initially everybody is its own group
    collisions = DisjointSet([a.id for a in collided_agents])

    # As soon as two agents intersect/collided, we merge their sets, i.e., their collisions groups
    for the_agent in collided_agents:
        # Extract the occupancy of the agent at time step t
        the_agent_occupancy_at_time_t = scenario.obstacle_by_id(the_agent.id).occupancy_at_time(time_step)
        
        # Consider only those that are NOT in the same collision group
        for the_other_agent in [oa for oa in collided_agents if not collisions.connected(the_agent.id, oa.id)]:
            # Extract the occupancy of the other agent at time step t
            the_other_agent_occupancy_at_time_t = scenario.obstacle_by_id(the_other_agent.id).occupancy_at_time(time_step)

            if the_agent_occupancy_at_time_t is None or the_other_agent_occupancy_at_time_t is None:
                continue

            # This is somehow subjected to floating point errors and approximation.
            distance = the_agent_occupancy_at_time_t.shape.shapely_object.distance(the_other_agent_occupancy_at_time_t.shape.shapely_object)

            if distance <= MIN_DISTANCE_FOR_COLLISION:
                collisions.merge(the_agent.id, the_other_agent.id)

    # Return only the subset relevant for the "agent"
    return list(collisions.subset(agent.id))
    
def angle_range(angle):
    """
    TAKEN FROM Frenetix Motion Planner: Collilsion)_helper_function
    Return an angle in [rad] in the interval ]-pi; pi].

    Args:
        angle (float): Angle in rad.
    Returns:
        float: angle in rad in the interval ]-pi; pi]

    """
    while angle <= -np.pi or angle > np.pi:
        if angle <= -np.pi:
            angle += 2 * np.pi
        elif angle > np.pi:
            angle -= 2 * np.pi

    return angle

def compute_RIA(agent_state_at_time_t, agent_state_at_time_t_minus_one, other_agent_state_at_time_t, other_agent_state_at_time_t_minus_one):
    """
    Compute the RIA metric for the given states of the ego vehicle and the other agent at time t and t - 1
    """
    # Extract agent's position and orientation at time t and t - 1
    obs_pos = other_agent_state_at_time_t.position
    ego_pos_last = agent_state_at_time_t_minus_one.position
    ego_yaw = agent_state_at_time_t.orientation

    # Extract other agent's position at time t and orientation at time t - 1
    obs_pos_last = other_agent_state_at_time_t_minus_one.position
    obs_yaw = other_agent_state_at_time_t.orientation

    # Calculate crash angle
    pdof = angle_range(obs_yaw - ego_yaw + np.pi)
    # Movement direction
    rel_angle = np.arctan2(obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0])
    ego_angle = angle_range(rel_angle - ego_yaw)

    return pdof

def crash_intersection(car1: Polygon, car2: Polygon):
    """
    Compute the intersection points between the boundaries of two polygons, which represent the shapes of the two vehicles at the time of the crash
    """
    polys1 = geopandas.GeoSeries([car1, car2])
    df1 = geopandas.GeoDataFrame({'geometry': polys1})
    geom1 = df1.geometry[0]
    geom2 = df1.geometry[1]

    # They can return, 2 or more points (3 might be a corner ?)
    intesection = geom1.boundary.intersection(geom2.boundary)  # multipoint
    points = list()
    for g in intesection.geoms:
        if isinstance(g, LineString):
            # take the middle point.
            assert len(g.xy) == 2, "Linestring has more than 2 points?!"
            a_x, a_y = g.xy[0][0], g.xy[0][1]
            b_x, b_y = g.xy[1][0], g.xy[1][1]
            points.append(Point((a_x + b_x) * 0.5, (a_y + b_y) * 0.5))

        elif isinstance(g, Point):
            points.append(g)
    
    # If they overlap
    point1 = points[0]
    point2 = points[1]

    return [point1.x, point1.y, point2.x, point2.y]

def crash_point(crash_points:list) -> list:
    """
    Compute the crash point as the center of the crash points, which are the intersection points between the boundaries of two polygons, which represent the shapes of the two vehicles at the time of the crash
    """
    point1 = crash_points[0]
    point2 = crash_points[1]
    point3 = crash_points[2]
    point4 = crash_points[3]
    center = [(point1 + point3) / 2, (point2 + point4) / 2]

    return center

def compute_impact_position(agent_occupancy_at_time_t, agent_state_at_time_t, other_agent_occupancy_at_time_t, other_agent_state_at_time_t):
    """
    Compute the impact position between the two vehicles at time t, as the intersection point between their shapes at time t
    """
    # Extract vehicles shape
    agent_shape: Rectangle
    agent_shape = agent_occupancy_at_time_t.shape
    agent_vehicle_length: float = agent_shape.length
    agent_vehicle_width: float = agent_shape.width
    agent_vehicle_center = agent_shape.center

    # Build agents shape at time t
    agent_rectangle = agent_occupancy_at_time_t.shape.shapely_object
    npc_rectangle = other_agent_occupancy_at_time_t.shape.shapely_object

    # Sometimes the vehicles are very close but do not touch. In that case, we replace their rectangle with a slightly bigger one
    if not agent_rectangle.intersects(npc_rectangle):
        agent_rectangle = agent_occupancy_at_time_t.shape.shapely_object.buffer(MIN_DISTANCE_FOR_COLLISION)
        npc_rectangle = other_agent_occupancy_at_time_t.shape.shapely_object.buffer(MIN_DISTANCE_FOR_COLLISION)

        assert agent_rectangle.intersects(npc_rectangle), f"The colliding vehicles do not touch also after increasing their size by {MIN_DISTANCE_FOR_COLLISION}"

    crash_intersection_points = []
    try:
        crash_intersection_points = crash_intersection(agent_rectangle, npc_rectangle)
    except AttributeError: # 'LineString' object has no attribute 'geoms'. Did you mean: '_geom'?:
        # The vehicles are very closed but their geometries do not touch.
        # Approximate the impact by simulating the movement of the vehicles in less than 0.1 steps
        for t in np.linspace(0.0, 2.0, 20):
            # Extract agent velocity, orientation and compute the distance traveled at time t
            agent_velocity = agent_state_at_time_t.velocity
            agent_orientation = agent_state_at_time_t.orientation
            agent_distance = agent_velocity * t

            # Extract other agent velocity, orientation and compute the distance traveled at time t
            other_agent_velocity = other_agent_state_at_time_t.velocity
            other_orientation = other_agent_state_at_time_t.orientation
            other_distance = other_agent_velocity * t

            # Approximate the next position of the two vehicles at time t and check if they intersect
            next_position_agent = translate(agent_rectangle, xoff=agent_distance*math.cos(agent_orientation), yoff=agent_distance*math.sin(agent_orientation))
            next_position_other = translate(npc_rectangle, xoff=other_distance*math.cos(other_orientation), yoff=other_distance*math.sin(other_orientation))

            # Check if the two vehicles intersect at their next position
            if next_position_agent.intersects(next_position_other):
                crash_intersection_points = crash_intersection(next_position_agent, next_position_other)
                break

    # At this point if there's no intersection we need to guess based on the closed points
    if len(crash_intersection_points) == 0:
        from shapely.ops import nearest_points
        point1, point2 = nearest_points(agent_rectangle, npc_rectangle)
        crash_intersection_points = [point1.x, point1.y, point2.x, point2.y]

    # Coordinate must be expressed in relation to the Bounding Box of the vehicle
    impact_position = None

    # We compute the intersection between the line connecting the center of the ego vehicle and the impact point, and the boundary of the ego vehicle
    ego_vehicle_center_point = Point(agent_vehicle_center)
    impact_point = None # Absolute coordinates
    if len(crash_intersection_points) == 2:
        # One contact point
        impact_point = Point(crash_intersection_points)
        pass
    elif len(crash_intersection_points) == 4:
        # Average position. This is NOT on the boundary of the vehicle
        impact_point = Point( [(crash_intersection_points[0] + crash_intersection_points[2] ) * 0.5,
            (crash_intersection_points[1] + crash_intersection_points[3]) * 0.5])
    else:
        raise AssertionError(f"Too many intersection points at crash {(len(crash_intersection_points) /2)}")
    
    # We extend the line connecting the center of the ego vehicle and the impact point, to make sure it intersects with the boundary of the ego vehicle, even in case of very close collision
    intersecting_out = scale(LineString([ego_vehicle_center_point, impact_point]), xfact = -2.0, yfact = -2.0, origin = (impact_point))

    intersecting = LineString(list(intersecting_out.coords) + list(ego_vehicle_center_point.coords))

    # Map the impact location to a sector in the car: FR, FL, MR, ML, BR, BL
    # Reduce the impact location and the ego_rectangle in their standard position (center == origin, rotated toward increasing y)
    standard_intersect = rotate(
            translate(intersecting, -ego_vehicle_center_point.x, -ego_vehicle_center_point.y),
            angle = math.pi * 0.5 - agent_state_at_time_t.orientation,
            origin = (0.0, 0.0),
            use_radians = True)

    standard_ego_rectangle = rotate(
            translate(agent_rectangle,-ego_vehicle_center_point.x, -ego_vehicle_center_point.y),
            angle = math.pi * 0.5 - other_agent_state_at_time_t.orientation,
            origin = (0.0, 0.0),
            use_radians = True)

    # If the vehicles do not touch we need to "guess" the standard_impact_location
    standard_impact_location = standard_ego_rectangle.boundary.intersection(standard_intersect)

    # Map the position to the labels
    impact_position = None
    if standard_impact_location.y > agent_vehicle_length / 6.0:
        impact_position = "F"
    # elif -agent_vehicle_length / 6 < standard_impact_location.x <= agent_vehicle_length / 6:
    elif standard_impact_location.y < - agent_vehicle_length / 6.0:
        impact_position = "B"
    else:
        impact_position = "M"
    
    if standard_impact_location.x > agent_vehicle_width / 6.0:
        impact_position += "R"
    elif standard_impact_location.x < - agent_vehicle_width / 6.0:
        impact_position += "L"
    else:
        impact_position += "C"
    
    assert impact_position, "Impact position cannot be None"
    
    return impact_position

def compute_RIV(agent_state_at_time_t, other_agent_state_at_time_t):
    """
    Compute the RIV metric for the given states of the ego vehicle and the other agent at time t
    """
    # Compute the relative speed by comparing vel_lon and vel_lat (or vel_x and vel_y)
    a_vel_x = agent_state_at_time_t.velocity * math.cos(agent_state_at_time_t.orientation)
    a_vel_y = agent_state_at_time_t.velocity * math.sin(agent_state_at_time_t.orientation)

    b_vel_x = other_agent_state_at_time_t.velocity * math.cos(other_agent_state_at_time_t.orientation)
    b_vel_y = other_agent_state_at_time_t.velocity * math.sin(other_agent_state_at_time_t.orientation)

    return ( (a_vel_x-b_vel_x) ** 2 + (a_vel_y-b_vel_y) ** 2) ** 0.5

def compute_collision_features(time_step: int, agent, other_agents, scenario: Scenario) -> List[int]:
    """
    Compute RIA, RIV, and impact location for the most critical colliding object (e.g., the one for which RIV is max).
    Note: vehicles might not exactly touch each others, so we need to include the case of "nearby" vehicles

    The framework already compute those features. Compare and include or replace
    """
    ria = -1
    riv = -1
    impact_position = None
    most_critical_agent = None

    assert time_step != 0, "Cannot compute collision features at time_step 0"

    # Extract the occupancy and state of the agent
    agent_occupancy_at_time_t = scenario.obstacle_by_id(agent.id).occupancy_at_time(time_step)
    agent_state_at_time_t = scenario.obstacle_by_id(agent.id).state_at_time(time_step)
    agent_state_at_time_t_minus_one = scenario.obstacle_by_id(agent.id).state_at_time(time_step - 1)

    # Find the agent colliding with it. The agent must be COLLISION and must be overlapping to agent
    for other_agent in [oa for oa in other_agents if oa.id != agent.id and oa.agent_state.status == AgentStatus.COLLISION]:
        # Extract the occupancy and state of the other agent at time step t and t - 1
        other_agent_occupancy_at_time_t = scenario.obstacle_by_id(other_agent.id).occupancy_at_time(time_step)
        other_agent_state_at_time_t = scenario.obstacle_by_id(other_agent.id).state_at_time(time_step)
        other_agent_state_at_time_t_minus_one = scenario.obstacle_by_id(other_agent.id).state_at_time(time_step - 1)

        # Compute distance between the two agents at time step t
        # If they do not touch, we skip them, as they are not involved in the same collision
        distance = agent_occupancy_at_time_t.shape.shapely_object.distance(other_agent_occupancy_at_time_t.shape.shapely_object)

        if distance > MIN_DISTANCE_FOR_COLLISION:
            module_logger.debug(f"Other collided agent {other_agent.id} does collide with {agent.id} (distance {distance} > {MIN_DISTANCE_FOR_COLLISION} )")
            continue

        # Compute RIV and check if it is the most critical collision (i.e., the one with the highest RIV)
        current_riv = compute_RIV(agent_state_at_time_t, other_agent_state_at_time_t)

        if current_riv > riv:
            # Compute RIA for the two agents at time step t and t - 1
            ria = compute_RIA(agent_state_at_time_t, agent_state_at_time_t_minus_one, other_agent_state_at_time_t, other_agent_state_at_time_t_minus_one)

            # Compute the impact position between the two vehicles at time t, as the intersection point between their shapes at time t            
            impact_position = compute_impact_position(agent_occupancy_at_time_t, agent_state_at_time_t, other_agent_occupancy_at_time_t, other_agent_state_at_time_t)
            
            riv = current_riv
            most_critical_agent = other_agent.id
        
    if most_critical_agent is None:
        # The vehicle collided with the road"
        module_logger.debug(f"Agent {agent.id} collided with the road at time {time_step}")        
        # For the moment, we cannot compute the angle and the position easily, so we keep all together
        impact_position = "-+" # Default
        # Approximate the roation of the vehicle
        ria = agent_state_at_time_t.orientation
        # Road is not moving
        riv = agent_state_at_time_t.velocity
        # Let's keep this way
        most_critical_agent = None

    return ria, riv, impact_position, most_critical_agent

def augment_with_emergencies_and_disengagements(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent and time check if at that time the agent triggered an emergency behavior
    """
    emergency_behavior_df = None

    t_start = 0
    t_end = -1
    # Compute the scenario final timestep
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        t_end = max(t_end, simulation.scenario.obstacle_by_id(agent.id).prediction.final_time_step)

    # Iterate over all scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        # Initialize the has_emergency and has_disengagement series with default values (0.0)
        has_emergency = pd.Series(None, index = list(range(t_start, t_end + 1)))
        has_disengagement = pd.Series(None, index = list(range(t_start, t_end + 1)))

        # Fill in the has_emergency and has_disengagement series with 1.0 for the time steps where the agent triggered an emergency behavior or a disengagement
        for t in range(t_start, t_end + 1):
            has_emergency[t] = 1.0 if t in agent.record_emergency else 0.0
            has_disengagement[t] = 1.0 if t in agent.record_disengagement else 0.0

        # Save the dataframe for this agent and concatenate with the previous ones
        agent_missed_goal_df = pd.DataFrame(None, index = list(range(t_start, t_end + 1)), columns = ["has_emergency", "has_disengagement"])
        agent_missed_goal_df["has_emergency"] = has_emergency
        agent_missed_goal_df["has_disengagement"] = has_disengagement

        if emergency_behavior_df is None:
            # Promote the agent df to global danger df
            emergency_behavior_df = agent_missed_goal_df.set_index([[agent.id] * len(agent_missed_goal_df), agent_missed_goal_df.index])
        else:
            agent_missed_goal_df = agent_missed_goal_df.set_index([[agent.id] * len(agent_missed_goal_df), agent_missed_goal_df.index])
            emergency_behavior_df = pd.concat([emergency_behavior_df, agent_missed_goal_df])

    # At this point collision_df should be organized exactly as criticality_dataframe so they can be merged so we reset the index
    emergency_behavior_df.reset_index(inplace=True)

    # We merge the criticality dataframe
    criticality_dataframe = pd.merge(criticality_dataframe, emergency_behavior_df)

    return criticality_dataframe

def augment_with_missed_goals(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent and time check whether the agent missed its goal by looking at its state to be Timeout
    (goal not reached in time) and Goal Area Passed (goal area passed before the timeout)
    """
    missed_goal_df = None

    t_start = 0
    t_end = -1
    # Compute scenario final timestep
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        t_end = max(t_end, simulation.scenario.obstacle_by_id(agent.id).prediction.final_time_step)

    # Iterate over all scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        # Initialize the has_timeout and has_missed_goal series with default values (0.0)
        has_timeout = pd.Series(None, index=list(range(t_start, t_end+1)))
        has_missed_goal = pd.Series(None, index=list(range(t_start, t_end+1)), dtype=object)

        # Iterate over scenario's timesteps and populate the has_timeout and has_missed_goal series
        for t in range(t_start, t_end + 1):
            has_timeout[t] = 1.0 if agent.agent_state.status == AgentStatus.TIMELIMIT else 0.0
            has_missed_goal[t] = 1.0 if agent.agent_state.status == AgentStatus.MAX_S_POSITION else 0.0

        # Save the dataframe for this agent and concatenate with the previous ones
        agent_missed_goal_df = pd.DataFrame(None, index=list(range(t_start, t_end + 1)), columns=["has_timeout", "has_missed_goal"])
        agent_missed_goal_df["has_timeout"] = has_timeout
        agent_missed_goal_df["has_missed_goal"] = has_missed_goal

        if missed_goal_df is None:
            missed_goal_df = agent_missed_goal_df.set_index([[agent.id] * len(agent_missed_goal_df), agent_missed_goal_df.index])
        else:
            agent_missed_goal_df = agent_missed_goal_df.set_index([[agent.id] * len(agent_missed_goal_df), agent_missed_goal_df.index])
            missed_goal_df = pd.concat([missed_goal_df, agent_missed_goal_df])

    # At this point collision_df should be organized exactly as criticality_dataframe so they can be merged so we reset the index
    missed_goal_df.reset_index(inplace=True)

    # We merge the criticality dataframe
    criticality_dataframe = pd.merge(criticality_dataframe, missed_goal_df)

    return criticality_dataframe

def augment_with_collisions(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent and time check whether the agent collided with someone or the road
    Then extract RIA, RIV, and IMPACT LOCATION. Use a simplified model if the collision is with the road
    """
    collision_df = None
    
    t_start = 0
    t_end = -1
    # Compute scenario final timestep
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        t_end = max(t_end, simulation.scenario.obstacle_by_id(agent.id).prediction.final_time_step)

    # Iterate over all scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        # Initialize the has_collided, has_collided_with, ria, riv, impact_position, and most_critical_agent series with default values (None)
        has_collided = pd.Series(None, index = list(range(t_start, t_end + 1)))
        has_collided_with = pd.Series(None, index = list(range(t_start, t_end + 1)), dtype = object)
        ria = pd.Series(None, index = list(range(t_start, t_end + 1)))
        riv = pd.Series(None, index = list(range(t_start, t_end + 1)))
        impact_position = pd.Series(None, index = list(range(t_start, t_end + 1)), dtype = str)
        most_critical_agent = pd.Series(None, index = list(range(t_start, t_end + 1)))

        # Iterate over scenario's timesteps
        for t in range(t_start, t_end+1):
            # Check if the agent collided at this time step and if it is the first time step after the collision            
            if agent.agent_state.status == AgentStatus.COLLISION and t >= agent.agent_state.last_timestep and t < len(agent.vehicle_history):
                # Compute collision features at the time it happened
                has_collided[t] = 1.0

                # Detect vehicles involved in the collision
                other_agents = [oa for oa in simulation.agents if oa.id != agent.id]
                vehicles_involved_in_collision = list_vehicles_involved_in_collision(t, agent, [a for a in simulation.agents if a.id != agent.id], simulation.scenario)

                # Compute the collision features only considering the agents involved in this collision
                # This generates some spurious entry for those already collided...
                collided_agents = [ca for ca in other_agents if ca.id in vehicles_involved_in_collision]

                # Compute collision criticality features
                ria[t], riv[t], impact_position[t], most_critical_agent[t] = compute_collision_features(t, agent, collided_agents, simulation.scenario)
                
                has_collided_with[t] = vehicles_involved_in_collision
            else:
                has_collided[t] = 0.0
                has_collided_with[t] = None
                ria[t] = None
                riv[t] = None
                impact_position[t] = None
                most_critical_agent[t] = None
                
        # Save the dataframe for this agent and concatenate with the previous ones
        agent_has_collided_df = pd.DataFrame(None, index = list(range(t_start, t_end + 1)), columns = ["has_collided", "has_collided_with", "ria", "riv", "impact_position", "most_critical_agent"])
        agent_has_collided_df["has_collided"] = has_collided
        agent_has_collided_df["has_collided_with"] = has_collided_with
        agent_has_collided_df["ria"] = ria
        agent_has_collided_df["riv"] = riv
        agent_has_collided_df["impact_position"] = impact_position
        agent_has_collided_df["most_critical_agent"] = most_critical_agent
        

        if collision_df is None:
            # promote the agent df to global danger df
            collision_df = agent_has_collided_df.set_index([[agent.id] * len(agent_has_collided_df), agent_has_collided_df.index])
        else:
            agent_has_collided_df = agent_has_collided_df.set_index([[agent.id] * len(agent_has_collided_df), agent_has_collided_df.index])
            collision_df = pd.concat([collision_df, agent_has_collided_df])
    
    # At this point collision_df should be organized exactly as criticality_dataframe so they can be merged so we reset the index
    collision_df.reset_index(inplace=True)
    
    # We merge the criticality dataframe
    criticality_dataframe = pd.merge(criticality_dataframe, collision_df)
    
    return criticality_dataframe

def augment_with_interactions(simulation: Simulation, trajectories: pd.DataFrame, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent and time check whether the agent interacted with someone else (i.e., changed its trajectory because of someone else)
    """
    interactions_count_df = None

    # Iterate over all scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        agent_id = agent.id

        # Extract the start and end time steps for the agent's trajectory
        t_start = simulation.scenario.obstacle_by_id(agent_id).initial_state.time_step
        t_end = simulation.scenario.obstacle_by_id(agent_id).prediction.final_time_step

        module_logger.debug(f"Evaluate Agent {agent_id} - Interactions")

        # Initialize the is_interacting and is_interacting_with series with default values (None)
        is_interacting = pd.Series(None, index = list(range(t_start, t_end + 1)))
        is_interacting_with = pd.Series(None, index = list(range(t_start, t_end + 1)), dtype = object)

        # Make sure we fill in all the values for this agent
        for t in range(t_start, t_end + 1):
            is_interacting[t] = 0.0

        # Iterate over the time steps of the agent's trajectory to check for interactions
        for t in range(t_start, t_end + 1):
            # Since we compare now with the next one, also the next one cannot be a collision otherwise it's trajectory is different
            if agent.agent_state.status == AgentStatus.COLLISION and (t >= agent.agent_state.last_timestep or t + 1 >= agent.agent_state.last_timestep):
                continue

            trajectory_at_time_t = trajectories.loc[(trajectories['vehicle_id'] == agent_id) & (trajectories['timestep'] == t), 'trajectory']
            trajectory_at_time_t_1 = trajectories.loc[(trajectories['vehicle_id'] == agent_id) & (trajectories['timestep'] == (t + 1)), 'trajectory']

            if len(trajectory_at_time_t) > 0 and len(trajectory_at_time_t_1) > 0:
                trajectory_at_time_t = trajectory_at_time_t.values[0]
                trajectory_at_time_t_1 = trajectory_at_time_t_1.values[0]
            else:
                module_logger.debug(f"Missing trajectory at time {t} for agent {agent_id}")
                continue

            # Crop planned trajectories to the same time interval
            cropped_traj_a, cropped_traj_b = crop_trajectories(trajectory_at_time_t, trajectory_at_time_t_1)

            # Detect nearby agents at time t that are within the safety distance
            obstacle = simulation.scenario.obstacle_by_id(agent_id)
            other_obstacles = [simulation.scenario.obstacle_by_id(a.id) for a in simulation.agents if a.id != agent_id]
            nearby_agent_ids = list_anyone_nearby(t, obstacle, other_obstacles)

            # Check if the consecutive planned trajectories are different and if there are nearby agents
            # If both conditions are true, we consider that the agent interacted with the nearby agents at time t
            if not is_trajectory_similar(cropped_traj_a, cropped_traj_b) and len(nearby_agent_ids) > 0:
                is_interacting[t] = 1.0
                is_interacting_with[t] = nearby_agent_ids
        
        # Save the dataframe for this agent and concatenate with the previous ones
        agent_interactions_count_df = pd.DataFrame(None, index = list(range(t_start, t_end + 1)), columns = ["is_interacting", "is_interacting_with"])
        agent_interactions_count_df["is_interacting"] = is_interacting
        agent_interactions_count_df["is_interacting_with"] = is_interacting_with
        
        if interactions_count_df is None:
            interactions_count_df = agent_interactions_count_df.set_index([[agent_id] * len(agent_interactions_count_df), agent_interactions_count_df.index])
        else:
            agent_interactions_count_df = agent_interactions_count_df.set_index([[agent_id] * len(agent_interactions_count_df), agent_interactions_count_df.index])
            interactions_count_df = pd.concat([interactions_count_df, agent_interactions_count_df])


    # At this point interactions_count_df should be organized exactly as criticality_dataframe so they can be merged so we reset the index
    interactions_count_df.reset_index(inplace=True)
    
    # We merge the criticality dataframe
    criticality_dataframe = pd.merge(criticality_dataframe, interactions_count_df)
    
    return criticality_dataframe

def compute_difference_in_cc_space(scenario: Scenario, trajectory_at_time_t: Trajectory, trajectory_at_time_t_1: Trajectory, use_relative_path = False) -> Tuple[float, float]:
    """
    Compute the difference between two trajectories in curvilinear coordinates (s, d) space
    This is useful to compute the difference between two planned trajectories, to check if the agent changed its trajectory because of someone else (i.e., interacted with someone else)
    """
    # Take the trajectories that lead to the interaction
    # Uniform them so they cover the same interval of observations
    cropped_traj_a, cropped_traj_b = crop_trajectories(trajectory_at_time_t, trajectory_at_time_t_1)

    # Extract the latest overlapping state (where the AV wants to go)
    pos_a = cropped_traj_a.final_state.position    
    pos_b = cropped_traj_b.final_state.position

    # Take one of the lanelets where the first state (a) is and use its center as curvilinear reference system
    # and project there a and b

    # It might happen that the final states are out of the road, because those are planned trajectories and those in emergency mode might result in this
    # So we need to consider an alternative strategy to label this interaction. If we can assume that NOT all the planned states are outside the road
    # we look up in the past states until we find a lanelet
    lanelet_id = None
    # Go backwards
    for target_state_a in reversed(cropped_traj_a.state_list):
        if lanelet_id:
            break

        for _lanelet_id in [l for l in scenario.lanelet_network.find_lanelet_by_position([target_state_a.position]) if len(l) > 0]:
            lanelet_id = _lanelet_id[0]
            break

    assert lanelet_id, f"Cannot find a lanelet over the entire cropped trajectory for agent"

    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

    if not use_relative_path:
        path = lanelet.center_vertices
    else:
        orientation_a = cropped_traj_a.final_state.orientation
        # Create a line passing from the middle of the vehicle rotated according to the vechile's orientation
        xs = [pos_a[0] + distance * math.cos(orientation_a) for distance in np.linspace(-100, 100, 100)]
        ys = [pos_a[1] + distance * math.sin(orientation_a) for distance in np.linspace(-100, 100, 100)]
        path = np.array([[px, py] for px, py in zip(xs, ys)])

    # Make sure the path is long enough to avoid the out of domain projection error. 50 is not enough...
    try:
        # This one is broken. It extends it ONLY in one side and crop the other side
        # path = extend_ref_path_both_ends(path, extension_length=100)
        
        extended_path = np.array(scale(LineString(path), 3.0, 3.0).coords)
        extended_path = resample_polyline(extended_path, step=1)
        cosy = CoordinateSystem(extended_path)

        # (s,d) - s is distance from the beginning of the refpath (long), d is the lateral displacement
        cosy_a = cosy.convert_to_curvilinear_coords(pos_a[0], pos_a[1])
        cosy_b = cosy.convert_to_curvilinear_coords(pos_b[0], pos_b[1])
        diff_long = cosy_a[0] - cosy_b[0]
        diff_lat = cosy_a[1] - cosy_b[1]

        return diff_long, diff_lat
    except Exception as exc:
        def debug_plot():
            import matplotlib.pyplot as plt
            from commonroad.visualization.mp_renderer import MPRenderer
            rnd = MPRenderer()
            scenario.lanelet_network.draw(rnd)
            rnd.render()

            plt.plot(path[:, 0], path[:, 1], "-", zorder=99, color="black", lw=2)
            plt.plot(extended_path[:, 0], extended_path[:, 1], "-", zorder=98, color="green", lw=2)

            # Plot the positions pos a and pos b
            plt.plot(pos_a[0], pos_a[1], "ro", zorder=100, lw=2)
            plt.plot(pos_b[0], pos_b[1], "g+", zorder=100, lw=2)
            

            plt.plot([s.position[0] for s in cropped_traj_a.state_list], [s.position[1] for s in cropped_traj_a.state_list], "r", zorder=98, lw=2)
            plt.plot([s.position[0] for s in cropped_traj_b.state_list], [s.position[1] for s in cropped_traj_b.state_list], "g", zorder=98, lw=2)
            # Plot the reference path used to build the coordinate system
            
            
            plt.show()

        # debug_plot()
        
        raise exc

def compute_interaction_map(simulation: Simulation, trajectories: pd.DataFrame, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the interaction map, which is a 3x3 matrix that counts the number of interactions in each of the 9 possible combinations of longitudinal and lateral differences in the curvilinear space (s, d) between two consecutive planned trajectories that lead to an interaction
    The longitudinal and lateral differences are categorized into three classes: negative, zero, and positive, based on a threshold
    The resulting interaction map can be used to analyze the distribution of interactions in terms of their longitudinal and lateral characteristics
    """
    interaction_map = np.zeros((3, 3))
    
    threshold_0 = 0.1
    threshold_1 = 0.1

    # List all the places where any interaction occurred
    index_of_interactions = criticality_dataframe[criticality_dataframe["is_interacting"] == 1.0].index

    # Iterate over all scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        # Find all the rows that belong to this agent
        for t in sorted([i[1] for i in index_of_interactions if i[0] == agent.id]):
            trajectory_at_time_t = trajectories.loc[(trajectories['vehicle_id'] == agent.id) & (trajectories['timestep'] == t), 'trajectory']
            trajectory_at_time_t_1 = trajectories.loc[(trajectories['vehicle_id'] == agent.id) & (trajectories['timestep'] == (t + 1)), 'trajectory']

            if len(trajectory_at_time_t) > 0 and len(trajectory_at_time_t_1) > 0:
                trajectory_at_time_t = trajectory_at_time_t.values[0]
                trajectory_at_time_t_1 = trajectory_at_time_t_1.values[0]
            else:
                module_logger.debug(f"Missing trajectory at time {t} for agent {agent.id}")
                continue

            # Compute interaction effect on the planned trajectory
            try:
                diff_0, diff_1 = compute_difference_in_cc_space(simulation.scenario, trajectory_at_time_t, trajectory_at_time_t_1)
            except:
                # If this fails, we cannot do much
                diff_0, diff_1 = compute_difference_in_cc_space(simulation.scenario, trajectory_at_time_t, trajectory_at_time_t_1, use_relative_path = True)

            # Categorize the differences into negative, zero, and positive based on the thresholds and update the interaction map
            if diff_0 < -threshold_0:
                index_0 = 0
            elif diff_0 > threshold_0:
                index_0 = 2
            else:
                index_0 = 1
            if diff_1 < -threshold_1:
                index_1 = 0
            elif diff_1 > threshold_1:
                index_1 = 2
            else:
                index_1 = 1

            interaction_map[index_0][index_1] += 1

    return interaction_map