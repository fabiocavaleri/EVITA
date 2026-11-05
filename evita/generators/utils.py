"""
Utilities to generate scenarios (start states, goal areas, routes) and to evaluate executed scenarios
(danger, interactions, collisions, missed goals, emergencies) for the evaluator's table.

The evaluation functions build, for each agent, a table with one row per time step, indexed by the columns
`level_0` (agent id) and `level_1` (time step), and merge it into the evaluator's table.
"""
import math
import random
import logging
import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import DisjointSet
from shapely.ops import nearest_points
from shapely.affinity import rotate, scale, translate
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import Trajectory
from commonroad.planning.goal import GoalRegion, Interval
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad_dc.geometry.util import resample_polyline

from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.agent_status import AgentStatus

from ..config import ScenarioGeneratorConfig, ValidationConfig
from ..utils import EVITA_MAIN_LOGGER_NAME

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

# Two vehicle shapes closer than this (metres) are touching: they collided
MIN_DISTANCE_FOR_COLLISION = 0.1
# Distance (metres) below which another vehicle counts as nearby (condition C2 of the interaction definition)
NEARBY_DISTANCE = 10.0
# Distance (metres) used instead of zero when two vehicle centres coincide, to keep the danger metric finite
MIN_DANGER_DISTANCE = 0.1
# Columns identifying the rows of the evaluator's table
AGENT_COLUMN, TIME_COLUMN = "level_0", "level_1"

# ========== Scenario generation ==========

def orientation_by_position(lanelet: Lanelet, position: np.ndarray) -> float:
    """
    Direction (radians) of the lanelet's centre line at the vertex closest to `position` (centre line resampled every meter).
    """
    # Extract lanelet center vertices and resample them to have a point every 1 meter
    center_vertices = resample_polyline(lanelet.center_vertices, step = 1.0)

    # Find the closest two vertices of the lanelet centerline to the given position
    closest = int(np.argmin(np.sum((center_vertices - position) ** 2, axis = 1)))

    # Determine the two closest vertices to compute the orientation
    # If the closest vertex is the last one, we take the previous vertex and the closest vertex
    # Otherwise, we take the closest vertex and the next vertex
    if closest == len(center_vertices) - 1:
        vertex1, vertex2 = center_vertices[closest - 1], center_vertices[closest]
    else:
        vertex1, vertex2 = center_vertices[closest], center_vertices[closest + 1]

    # Compute the direction vector between the two vertices and calculate the orientation using arctan2
    direction_vector = vertex2 - vertex1

    return float(np.arctan2(direction_vector[1], direction_vector[0]))

def get_random_init_state(config: ScenarioGeneratorConfig, lanelet: Lanelet) -> Optional[InitialState]:
    """
    Return a random Initial State of the vehicle
    """
    # Skip if the lanelet is too short to generate a valid initial state
    center_vertices = lanelet.center_vertices
    if len(center_vertices) < 2:
        return None

    # If the lanelet has no predecessor, it means that it is an entry lanelet.
    # In this case, we need to make sure that the vehicle is not too close to the start of the lanelet
    min_idx = 0
    if not lanelet.predecessor:
        # Compute the cumulative distance along the lanelet centerline and find the index of the first vertex that is far enough from the start
        cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(center_vertices, axis = 0), axis = 1))])
        far_enough = np.nonzero(cumulative >= ValidationConfig.min_dist_init)[0]

        # Skip if there are no vertices that are far enough from the start of the lanelet
        if len(far_enough) == 0:
            return None
        
        min_idx = int(far_enough[0])
            
    # The next vertex gives the orientation, so the last vertex cannot be used
    if min_idx >= len(lanelet.center_vertices) - 1:
        return None

    # Randomly pick a point along the lanelet centerline, but not too close to the start if it's an entry lanelet
    random_index = random.randrange(min_idx, len(center_vertices) - 1)
    position = center_vertices[random_index]
    next_point = center_vertices[random_index + 1]

    return InitialState(time_step = 0,
                        position = position,
                        orientation = get_orientation_by_coords((position[0], position[1]), (next_point[0], next_point[1])),
                        velocity = random.uniform(config.min_init_velocity, config.max_init_velocity),
                        acceleration = 0.0,
                        yaw_rate = 0.0,
                        slip_angle = 0.0)

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
    goal_state_list = [CustomState(position = goal_area, time_step = Interval(0, config.max_time_step))]
    
    return GoalRegion(goal_state_list)

def get_base_scenario_from_file(commonroad_scenario_file: str, remove_obstacles: bool, remove_planning_problems: bool) -> Scenario:
    """
    Load the scenario from the given file and remove obstacles and/or planning problems if specified
    """
    #n Load the scenario and planning problem set from the CommonRoad file
    scenario, planning_problem_set = CommonRoadFileReader(commonroad_scenario_file).open()
    
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
    Recursively find adjacent lanelets with same driving direction
    """
    adjacent_lanelets = []
    processed_ids = [lanelet.lanelet_id]
    backlog = [lanelet]

    # Loop until there are no more lanelets to process
    while backlog:
        the_lanelet = backlog.pop()
        
        # Check left and right adjacent lanelets, if they exist, have the same driving direction and were not processed yet, add them to the list and to the backlog
        if the_lanelet.adj_left and the_lanelet.adj_left_same_direction and the_lanelet.adj_left not in processed_ids:
            the_left_lanelet = ll_network.find_lanelet_by_id(the_lanelet.adj_left)
            adjacent_lanelets.append(the_left_lanelet)
            backlog.append(the_left_lanelet)
            processed_ids.append(the_lanelet.adj_left)

        # Check right adjacent lanelet, if it exists, has the same driving direction and was not processed yet, add it to the list and to the backlog
        if the_lanelet.adj_right and the_lanelet.adj_right_same_direction and the_lanelet.adj_right not in processed_ids:
            the_right_lanelet = ll_network.find_lanelet_by_id(the_lanelet.adj_right)
            adjacent_lanelets.append(the_right_lanelet)
            backlog.append(the_right_lanelet)
            processed_ids.append(the_lanelet.adj_right)
        
    return adjacent_lanelets

def routes_to_exits(start_lanelet: Lanelet, ll_network: LaneletNetwork) -> List[Tuple[Lanelet, int]]:
    """
    Every route that can be driven forward (through successors) from `start_lanelet` to an outgoing lanelet.
    A lanelet is never visited twice on the same route, so cycles in the road network end the route.
    """
    finished = []
    pending = [(start_lanelet, [start_lanelet.lanelet_id])]

    while pending:
        merged, ids = pending.pop()
        last = ll_network.find_lanelet_by_id(ids[-1])
        successors = [s for s in last.successor if s not in ids]

        if not successors:
            finished.append((merged, ids[-1]))
            continue

        for successor_id in successors:
            successor = ll_network.find_lanelet_by_id(successor_id)
            pending.append((Lanelet.merge_lanelets(merged, successor), ids + [successor_id]))

    return finished

def cut(line: LineString, distance: float) -> List[LineString]:
    """
    Cuts a line in two at a specified distance from its starting point.
    """
    # Skip if the distance is out of bounds
    if distance <= 0.0 or distance >= line.length:
        return [line]
    
    coords = list(line.coords)
    # Iterate over line coordinates to find the point where the line should be cut
    for i, p in enumerate(coords):
        projected = line.project(Point(p))
        
        if projected == distance:
            return [LineString(coords[:i + 1]), LineString(coords[i:])]
        
        if projected > distance:
            cut_point = line.interpolate(distance)
            return [LineString(coords[:i] + [(cut_point.x, cut_point.y)]), LineString([(cut_point.x, cut_point.y)] + coords[i:])]

def get_orientation_by_coords(first_point: Tuple[float, float], next_point: Tuple[float, float]) -> float:
    """
    Direction (radians) from `first_point` to `next_point`.
    """
    a_x, a_y = next_point
    b_x, b_y = first_point

    return math.atan2(a_y - b_y, a_x - b_x)

# ========== Evaluation: helpers for the evaluator's table ==========

def _agent_table(agent_id: int, time_steps: List[int], columns: Dict[str, list]) -> pd.DataFrame:
    """
    Table of one agent: one row per time step, with the identifying columns and the given value columns.
    """
    table = pd.DataFrame(columns, index = range(len(time_steps)))
    table.insert(0, TIME_COLUMN, list(time_steps))
    table.insert(0, AGENT_COLUMN, agent_id)

    return table

def _merge_into(criticality_dataframe: pd.DataFrame, tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge the agents' tables into the evaluator's table, matching agent and time step.
    Every row of the evaluator's table is kept; time steps an agent table does not cover get missing values.
    """
    if not tables:
        return criticality_dataframe
    
    return pd.merge(criticality_dataframe, pd.concat(tables, ignore_index = True), on = [AGENT_COLUMN, TIME_COLUMN], how = "left")

def _scenario_final_time_step(simulation: Simulation) -> int:
    """
    Last time step of any agent in the simulation.
    """
    return max(simulation.scenario.obstacle_by_id(agent.id).prediction.final_time_step for agent in simulation.agents)

def _agent_time_range(simulation: Simulation, agent_id: int) -> List[int]:
    """
    Time steps of the agent in the simulated scenario, from its initial to its final state.
    """
    obstacle = simulation.scenario.obstacle_by_id(agent_id)

    return list(range(obstacle.initial_state.time_step, obstacle.prediction.final_time_step + 1))

# ========== Evaluation: interactions ==========

def crop_trajectories(trajectory_before: Trajectory, trajectory_after: Trajectory) -> Tuple[Optional[Trajectory], Optional[Trajectory]]:
    """
    States of the two trajectories at the time steps both cover, in time order.
    (None, None) if they share no time step.
    """
    before = {state.time_step: state for state in trajectory_before.state_list}
    after = {state.time_step: state for state in trajectory_after.state_list}
    common = sorted(set(before) & set(after))
    
    if not common:
        return None, None
    return (Trajectory(common[0], [deepcopy(after[t]) for t in common]), Trajectory(common[0], [deepcopy(before[t]) for t in common]))

# ========== Evaluation methods ==========

def is_trajectory_similar(trajectory_a: Optional[Trajectory], trajectory_b: Optional[Trajectory], trajectory_difference_threshold: float) -> bool:
    """
    True if no planned position moved by trajectory_difference_threshold metres or more between the two plans.
    Plans without a common time window give no evidence of replanning, so they count as similar.
    """
    if trajectory_a is None or trajectory_b is None or not trajectory_a.state_list:
        return True
    
    for state_a, state_b in zip(trajectory_a.state_list, trajectory_b.state_list):
        if np.linalg.norm(np.asarray(state_a.position) - np.asarray(state_b.position)) >= trajectory_difference_threshold:
            return False
    
    return True

def list_anyone_nearby(time_step: int, agent, other_agents, min_distance: float = NEARBY_DISTANCE) -> List[int]:
    """
    Condition C2: ids of the other agents (obstacles of the simulated scenario) whose shape is closer than `min_distance` meters to the agent's shape at `time_step`.
    Empty if the agent is not active then.
    """

    # Get the occupancy of the agent at time `time_step`
    agent_occupancy = agent.occupancy_at_time(time_step)
    
    # Check if the agent is active at time `time_step`
    if agent_occupancy is None:
        return []
    
    agent_shape = agent_occupancy.shape.shapely_object

    # Extract other active agents at time `time_step`
    nearby_agent_ids = set()
    for other_agent in other_agents:
        other_occupancy = other_agent.occupancy_at_time(time_step)
        if other_occupancy is not None and agent_shape.distance(other_occupancy.shape.shapely_object) < min_distance:
            nearby_agent_ids.add(other_agent.obstacle_id)

    return list(nearby_agent_ids)

def compute_difference_in_cc_space(trajectory_at_time_t: Trajectory, trajectory_at_time_t_1: Trajectory) -> Tuple[Optional[float], Optional[float]]:
    """
    How the new plan (computed at t + 1) moved the vehicle compared with the old plan (computed at t), at the last time
    step both plans cover, in the old plan's frame.
    - diff_long > 0: further ahead;
    - diff_lat > 0: further left.
    - (None, None) if the plans share no time step.
    """
    before = {state.time_step: state for state in trajectory_at_time_t.state_list}
    after = {state.time_step: state for state in trajectory_at_time_t_1.state_list}
    common = sorted(set(before) & set(after))
    if not common:
        return None, None

    # Shared time step with the largest deviation (the last one among equal deviations)
    deviations = [np.linalg.norm(np.asarray(after[t].position, dtype = float) - np.asarray(before[t].position, dtype = float)) for t in common]
    t_peak = common[len(deviations) - 1 - int(np.argmax(deviations[::-1]))]

    p_old = np.asarray(before[t_peak].position, dtype = float)
    p_new = np.asarray(after[t_peak].position, dtype = float)

    # Direction of the old plan at t_peak: its last segment there, or its heading if the vehicle stands still
    tangent = None
    if t_peak - 1 in before:
        segment = p_old - np.asarray(before[t_peak - 1].position, dtype = float)
        if np.linalg.norm(segment) > 1e-3:
            tangent = segment / np.linalg.norm(segment)
    if tangent is None:
        theta = float(getattr(before[t_peak], "orientation", 0.0) or 0.0)
        tangent = np.array([np.cos(theta), np.sin(theta)])

    # Left of the direction of travel (tangent rotated by +90 degrees)
    normal = np.array([-tangent[1], tangent[0]])

    # Forward and leftward parts of the displacement between the two plans (dot products with unit vectors)
    displacement = p_new - p_old

    return float(displacement @ tangent), float(displacement @ normal)

def augment_with_interactions(simulation: Simulation, trajectories: pd.DataFrame, criticality_dataframe: pd.DataFrame, trajectory_difference_threshold: float) -> pd.DataFrame:
    """
    For each agent and time check whether the agent interacted with someone else (i.e., changed its trajectory because of someone else)
    """
    planned_trajectories = {(int(vehicle_id), int(timestep)): trajectory for vehicle_id, timestep, trajectory in zip(trajectories["vehicle_id"], trajectories["timestep"], trajectories["trajectory"])}

    tables = []

    # Iterate over all scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        agent_id = agent.id
        module_logger.debug(f"Evaluate Agent {agent_id} - Interactions")

        # Extract agent's timesteps
        time_steps = _agent_time_range(simulation, agent_id)

        # Initialize the is_interacting and is_interacting_with series with default values (None)
        is_interacting = {t: 0.0 for t in time_steps}
        is_interacting_with, diff_long, diff_lat = {}, {}, {}

        obstacle = simulation.scenario.obstacle_by_id(agent_id)
        other_obstacles = [simulation.scenario.obstacle_by_id(other.id) for other in simulation.agents if other.id != agent_id]
        collided = agent.agent_state.status == AgentStatus.COLLISION

        # Iterate over the time steps of the agent's trajectory to check for interactions
        for t in time_steps:
            # After a collision (at t or t + 1), the planned trajectories no longer reflect interactions
            if collided and t + 1 >= agent.agent_state.last_timestep:
                continue

            # Extract the planned trajectories at time t and t + 1 for the agent
            trajectory_at_time_t = planned_trajectories.get((agent_id, t))
            trajectory_at_time_t_1 = planned_trajectories.get((agent_id, t + 1))

            if trajectory_at_time_t is None or trajectory_at_time_t_1 is None:
                module_logger.debug(f"Missing trajectory at time {t} for agent {agent_id}")
                continue

            cropped_traj_a, cropped_traj_b = crop_trajectories(trajectory_at_time_t, trajectory_at_time_t_1)

            # Condition C1: the plan changed between t and t+1
            if is_trajectory_similar(cropped_traj_a, cropped_traj_b, trajectory_difference_threshold):
                continue

            # Condition C2: another vehicle is nearby
            nearby_agent_ids = list_anyone_nearby(t, obstacle, other_obstacles)

            if not nearby_agent_ids:
                continue
                
            diff_0, diff_1 = compute_difference_in_cc_space(trajectory_at_time_t, trajectory_at_time_t_1)

            # Skip interaction in case of errors in trajectory difference computation
            if diff_0 is None or diff_1 is None:
                continue

            is_interacting[t] = 1.0
            is_interacting_with[t] = nearby_agent_ids
            diff_long[t] = diff_0
            diff_lat[t] = diff_1
        
        tables.append(_agent_table(agent_id, time_steps, {
            "is_interacting": [is_interacting[t] for t in time_steps],
            "is_interacting_with": [is_interacting_with.get(t) for t in time_steps],
            "diff_long": [diff_long.get(t) for t in time_steps],
            "diff_lat": [diff_lat.get(t) for t in time_steps],
        }))

    return _merge_into(criticality_dataframe, tables)
    
def compute_interaction_map(criticality_dataframe: pd.DataFrame, n_bins = None) -> pd.DataFrame:
    """
    Compute the interaction map, which is a 2D histogram of the longitudinal and lateral differences between consecutive planned trajectories of agents that interacted with others
    The longitudinal and lateral differences are categorized into three classes: negative, zero, and positive, based on a threshold
    The resulting interaction map can be used to analyze the distribution of interactions in terms of their longitudinal and lateral characteristics
    """
    # Get the bins of the interaction map
    config = ScenarioGeneratorConfig()

    if n_bins is None:
        n_bins = config.n_bins

    # US Interstate standard lane width: 12 ft (3.7 m)
    lane_width = 3.7
    long_bins = np.linspace(-config.length, +config.length, n_bins)
    lat_bins = np.linspace(-lane_width, lane_width, n_bins)

    # Include 0 - B[0]:B[9] - 10 and len(bins)
    rows = len(long_bins) + 1
    cols = len(lat_bins) + 1

    # Initialize the interaction map and the counters for the requirement violations
    interaction_map = np.zeros((rows, cols))

    # List all the places where any interaction occurred
    interactions = criticality_dataframe[criticality_dataframe["is_interacting"] == 1.0]

    for diff_long, diff_lat in zip(interactions["diff_long"], interactions["diff_lat"]):
        if pd.isna(diff_long) or pd.isna(diff_lat):
            continue

        # Determine the bin indices for the longitudinal and lateral differences
        long_bin_index = np.digitize(diff_long, long_bins)
        lat_bin_index = np.digitize(diff_lat, lat_bins)

        # Increment the corresponding cell in the interaction map
        interaction_map[long_bin_index][lat_bin_index] += 1

    return interaction_map

# ========== Evaluation: danger (TIAV's criticality metric) ==========

def compute_danger_for_states(state_a: CustomState, state_b: CustomState) -> float:
    """
    Danger between two vehicles: magnitude of their relative velocity divided by the distance between their centres.
    """
    velocity_a = state_a.velocity * np.array([math.cos(state_a.orientation), math.sin(state_a.orientation)])
    velocity_b = state_b.velocity * np.array([math.cos(state_b.orientation), math.sin(state_b.orientation)])
    relative_velocity = float(np.linalg.norm(velocity_a - velocity_b))

    distance = float(np.linalg.norm(np.asarray(state_a.position) - np.asarray(state_b.position)))

    return relative_velocity / max(distance, MIN_DANGER_DISTANCE)

def augment_with_danger(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the danger metric for each agent in the simulation and augment the criticality df with it
    """
    # Recorded state of each agent at each time step
    states = {agent.id: {state.time_step: state for state in agent.record_state_list} for agent in simulation.agents}

    tables = []
    # Iterate over scenario agents
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        # For each agent, create a df and concatenate with the previous one
        agent_id = agent.id

        module_logger.debug(f"Evaluate agent {agent_id} - DANGER")

        # Extract agent's timesteps
        time_steps = _agent_time_range(simulation, agent_id)

        # Initialize the max danger and the agent causing it
        max_danger, caused_by = [], []

        # Iterate over agent timesteps
        for t in time_steps:
            max_danger_at_time, max_danger_at_time_caused_by_agent_id = 0.0, -1
            agent_state = states[agent_id].get(t)
            if agent_state is not None:
                for other_agent_id, other_states in states.items():
                    if other_agent_id != agent_id:
                        other_state = other_states.get(t)
                    else:
                        other_state = None
                    
                    if other_state is None:
                        continue
                    
                    danger = compute_danger_for_states(agent_state, other_state)
                    
                    if danger > max_danger_at_time:
                        max_danger_at_time, max_danger_at_time_caused_by_agent_id = danger, other_agent_id
            
            max_danger.append(max_danger_at_time)
            caused_by.append(max_danger_at_time_caused_by_agent_id)

        tables.append(_agent_table(agent_id, time_steps, {"max_danger": max_danger, "max_danger_caused_by_agent": caused_by}))

    return _merge_into(criticality_dataframe, tables)

# ========== Evaluation: collisions ==========

def list_vehicles_involved_in_collision(time_step: int, agent, other_agents: list, scenario: Scenario) -> List[int]:
    """
    This function groups vehicles involved in the same collision at the given time_step
    """
    # List all the agents collided at or after time_step
    collided_agents = [a for a in [agent] + other_agents if a.agent_state.status == AgentStatus.COLLISION and time_step >= a.agent_state.last_timestep]
    
    # Group the collided agents by distance. Initially everybody is its own group
    collisions = DisjointSet([agent.id for agent in collided_agents])

    shapes = {}
    # Save the shapes of the collided agents at time_step
    for a in collided_agents:
        occupancy = scenario.obstacle_by_id(a.id).occupancy_at_time(time_step)
        if occupancy is not None:
            shapes[a.id] = occupancy.shape.shapely_object
        else:
            shapes[a.id] = None

    for i, the_agent in enumerate(collided_agents):
        # Check the distance between the agent and all the other agents that have not been checked yet
        for the_other_agent in collided_agents[i + 1:]:
            if collisions.connected(the_agent.id, the_other_agent.id):
                continue

            shape, other_shape = shapes[the_agent.id], shapes[the_other_agent.id]
            
            if shape is None or other_shape is None:
                continue
            
            # Subject to floating point approximation, hence the tolerance
            if shape.distance(other_shape) <= MIN_DISTANCE_FOR_COLLISION:
                collisions.merge(the_agent.id, the_other_agent.id)

    return list(collisions.subset(agent.id))
    
def angle_range(angle: float) -> float:
    """
    Angle (radians) in the interval [-pi, pi].
    From FrenetiX's collision helper functions.
    """
    angle = math.fmod(angle, 2 * math.pi)
    
    if angle <= -math.pi:
        angle += 2 * math.pi
    elif angle > math.pi:
        angle -= 2 * math.pi
    
    return angle

def compute_RIA(agent_state_at_time_t, other_agent_state_at_time_t) -> float:
    """
    Relative impact angle (radians, in [-pi, pi]): the principal direction of force
    """
    return angle_range(other_agent_state_at_time_t.orientation - agent_state_at_time_t.orientation + np.pi)

def compute_RIV(agent_state_at_time_t, other_agent_state_at_time_t) -> float:
    """
    Relative impact velocity (m/s): magnitude of the difference between the two velocity vectors.
    """
    velocity_a = agent_state_at_time_t.velocity * np.array([math.cos(agent_state_at_time_t.orientation), math.sin(agent_state_at_time_t.orientation)])
    velocity_b = other_agent_state_at_time_t.velocity * np.array([math.cos(other_agent_state_at_time_t.orientation), math.sin(other_agent_state_at_time_t.orientation)])

    return float(np.linalg.norm(velocity_a - velocity_b))

def crash_intersection(car1: Polygon, car2: Polygon) -> List[float]:
    """
    Two points where the boundaries of the two vehicle shapes intersect, as [x1, y1, x2, y2].
    A boundary overlap counts by its midpoint
    [] if the boundaries meet in fewer than two points.
    """

    # Compute the intersection of the boundaries of the two vehicle shapes
    intersection = car1.boundary.intersection(car2.boundary)
    if hasattr(intersection, "geoms"):
        parts = list(intersection.geoms)
    else:
        parts = [intersection]

    points = []
    # Iterate over the intersection parts and extract the points of intersection
    for part in parts:
        if part.is_empty:
            continue
        # If the intersection part is a LineString, compute its midpoint and add it to the list of points
        if isinstance(part, LineString):
            (a_x, a_y), (b_x, b_y) = part.coords[0], part.coords[-1]
            points.append(Point((a_x + b_x) * 0.5, (a_y + b_y) * 0.5))
        # If the intersection part is a Point, add it to the list of points
        elif isinstance(part, Point):
            points.append(part)
        # If the intersection part is a MultiPoint, extend the list of points with its geometries
        elif isinstance(part, MultiPoint):
            points.extend(part.geoms)

    if len(points) < 2:
        return []
    
    return [points[0].x, points[0].y, points[1].x, points[1].y]

def compute_impact_position(agent_occupancy_at_time_t, agent_state_at_time_t, other_agent_occupancy_at_time_t, other_agent_state_at_time_t) -> str:
    """
    Point of impact on the agent's body:
    - F(ront),
    - M(iddle)
    - B(ack)
    = L(eft),
    = C(entre)
    = R(ight).
    The impact point is where the two shapes' boundaries cross, found on the agent's boundary along the line from the agent's centre through that point, in the agent's frame.
    """
    # Extract vehicles shape
    agent_shape: Rectangle = agent_occupancy_at_time_t.shape
    agent_rectangle = agent_shape.shapely_object
    other_agent_rectangle = other_agent_occupancy_at_time_t.shape.shapely_object

    # Vehicles very close but not touching: enlarge both shapes by the collision tolerance
    if not agent_rectangle.intersects(other_agent_rectangle):
        agent_rectangle = agent_rectangle.buffer(MIN_DISTANCE_FOR_COLLISION)
        other_agent_rectangle = other_agent_rectangle.buffer(MIN_DISTANCE_FOR_COLLISION)

    
    crash_points = crash_intersection(agent_rectangle, other_agent_rectangle)

    # Boundaries do not cross in two points (one shape inside the other, or only touching):
    # move both vehicles forward along their headings for up to 2 s, until their boundaries cross
    if not crash_points:
        for t in np.linspace(0.0, 2.0, 20):
            moved_agent = translate(agent_rectangle, xoff = agent_state_at_time_t.velocity * t * math.cos(agent_state_at_time_t.orientation),
                                    yoff = agent_state_at_time_t.velocity * t * math.sin(agent_state_at_time_t.orientation))
            moved_other = translate(other_agent_rectangle, xoff = other_agent_state_at_time_t.velocity * t * math.cos(other_agent_state_at_time_t.orientation),
                                    yoff = other_agent_state_at_time_t.velocity * t * math.sin(other_agent_state_at_time_t.orientation))
            if moved_agent.intersects(moved_other):
                crash_points = crash_intersection(moved_agent, moved_other)
                if crash_points:
                    break

    # Still nothing: use the closest points of the two shapes
    if not crash_points:
        point1, point2 = nearest_points(agent_rectangle, other_agent_rectangle)
        crash_points = [point1.x, point1.y, point2.x, point2.y]

    # Impact point: the midpoint of the two crossing points
    impact_point = Point((crash_points[0] + crash_points[2]) * 0.5, (crash_points[1] + crash_points[3]) * 0.5)
    ego_center = Point(agent_shape.center)

    # Line from the agent's centre through the impact point and beyond, so it crosses the agent's boundary
    extended = scale(LineString([ego_center, impact_point]), xfact = -2.0, yfact = -2.0, origin = impact_point)
    impact_line = LineString(list(extended.coords) + list(ego_center.coords))

    # Agent's frame: centre at the origin, heading towards +y (front: y > 0; right: x > 0)
    angle = math.pi * 0.5 - agent_state_at_time_t.orientation
    standard_line = rotate(translate(impact_line, -ego_center.x, -ego_center.y), angle = angle, origin = (0.0, 0.0), use_radians = True)
    standard_rectangle = rotate(translate(agent_rectangle, -ego_center.x, -ego_center.y), angle = angle, origin = (0.0, 0.0), use_radians = True)

    # Find the intersection of the agent's boundary and the line from its centre through the impact point, in the agent's frame
    impact_location = standard_rectangle.boundary.intersection(standard_line)
    if impact_location.is_empty:
        impact_location = nearest_points(standard_rectangle.boundary, standard_line)[0]
    elif isinstance(impact_location, LineString):
        impact_location = impact_location.interpolate(0.5, normalized = True)
    elif not isinstance(impact_location, Point):
        impact_location = impact_location.representative_point()

    # Front, middle or back third of the length; left, centre or right third of the width
    if impact_location.y > agent_shape.length / 6.0:
        impact_position = "F"
    elif impact_location.y < -agent_shape.length / 6.0:
        impact_position = "B"
    else:
        impact_position = "M"

    if impact_location.x > agent_shape.width / 6.0:
        impact_position += "R"
    elif impact_location.x < -agent_shape.width / 6.0:
        impact_position += "L"
    else:
        impact_position += "C"

    return impact_position

def compute_collision_features(time_step: int, agent, other_agents, scenario: Scenario) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[int]]:
    """
    (RIA, RIV, point of impact, id of the other vehicle) of the agent's collision at `time_step`, for the colliding vehicle with the highest relative impact velocity.
    For a collision with the road (no colliding vehicle): point of impact "-+", the agent's orientation as RIA and its velocity as RIV, and no other vehicle.
    (-1, -1, None, None) if the agent's states are missing.
    """
    if time_step == 0:
        raise ValueError("Cannot compute collision features at timestep 0")
    
    obstacle = scenario.obstacle_by_id(agent.id)
    agent_occupancy = obstacle.occupancy_at_time(time_step)
    agent_state = obstacle.state_at_time(time_step)
    agent_state_before = obstacle.state_at_time(time_step - 1)

    if agent_occupancy is None or agent_state is None or agent_state_before is None:
        module_logger.debug(f"Agent {agent.id} has missing state information at time {time_step}")
        return -1, -1, None, None

    ria = -1
    riv = -1
    impact_position = None
    most_critical_agent = None

    # Find the agent colliding with it. The agent must be COLLISION and must be overlapping to agent
    for other_agent in other_agents:
        # Skip the agent itself and any other agent that is not in COLLISION state
        if other_agent.id == agent.id or other_agent.agent_state.status != AgentStatus.COLLISION:
            continue

        # Extract the occupancy and state of the other agent at time step t and t - 1
        other_obstacle = scenario.obstacle_by_id(other_agent.id)
        other_occupancy = other_obstacle.occupancy_at_time(time_step)
        other_state = other_obstacle.state_at_time(time_step)

        if other_occupancy is None or other_state is None:
            continue

        # Only vehicles touching the agent take part in this collision
        if agent_occupancy.shape.shapely_object.distance(other_occupancy.shape.shapely_object) > MIN_DISTANCE_FOR_COLLISION:
            continue

        # Compute RIV and check if it is the most critical collision (i.e., the one with the highest RIV)
        current_riv = compute_RIV(agent_state, other_state)
        if current_riv > riv:
            ria = compute_RIA(agent_state, other_state)
            impact_position = compute_impact_position(agent_occupancy, agent_state, other_occupancy, other_state)
            riv = current_riv
            most_critical_agent = other_agent.id

    if most_critical_agent is None:
        # Collision with the road: approximate the features
        module_logger.debug(f"Agent {agent.id} collided with the road at time {time_step}")
        return agent_state.orientation, agent_state.velocity, "-+", None

    return ria, riv, impact_position, most_critical_agent

def augment_with_collisions(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    For each agent and time check whether the agent collided with someone or the road
    Then extract RIA, RIV, and Impact Location.
    Use a simplified model if the collision is with the road
    """
    final_timestep = _scenario_final_time_step(simulation)
    timesteps = list(range(0, final_timestep + 1))

    tables = []
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        rows = {name: [None] * len(timesteps) for name in ["has_collided_with", "ria", "riv", "impact_location", "most_critical_agent"]}
        has_collided = [0.0] * len(timesteps)

        if agent.agent_state.status == AgentStatus.COLLISION:
            other_agents = [other for other in simulation.agents if other.id != agent.id]

            for i, t in enumerate(timesteps):
                # Skip time steps before the collision or after the agent's last recorded state
                if t < agent.agent_state.last_timestep or t >= len(agent.vehicle_history):
                    continue

                involved = list_vehicles_involved_in_collision(t, agent, other_agents, simulation.scenario)
                collided_agents = [other for other in other_agents if other.id in involved]

                try:
                    ria, riv, impact_location, most_critical_agent = compute_collision_features(t, agent, collided_agents, simulation.scenario)
                except Exception as e:
                    module_logger.error(f"Error computing collision features for agent {agent.id} at time {t}: {e}")
                    continue

                # Collisions with the road have no other vehicle: count only collisions with complete features
                if ria is None or riv is None or impact_location is None or most_critical_agent is None:
                    continue

                has_collided[i] = 1.0
                rows["has_collided_with"][i] = involved
                rows["ria"][i] = ria
                rows["riv"][i] = riv
                rows["impact_location"][i] = impact_location
                rows["most_critical_agent"][i] = most_critical_agent

        tables.append(_agent_table(agent.id, timesteps, {"has_collided": has_collided, **rows}))

    return _merge_into(criticality_dataframe, tables)

# ========== Evaluation: missed goals, emergencies ==========

def augment_with_missed_goals(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add, for each agent, whether its final status was a timeout (has_timeout: goal not reached in time)
    or a missed goal (has_missed_goal: goal area passed), repeated on every time step.
    """
    time_steps = list(range(0, _scenario_final_time_step(simulation) + 1))

    tables = []
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        status = agent.agent_state.status
        if status == AgentStatus.TIMELIMIT:
            has_timeout = 1.0
        else:
            has_timeout = 0.0

        if status == AgentStatus.MAX_S_POSITION:
            has_missed_goal = 1.0
        else:
            has_missed_goal = 0.0

        tables.append(_agent_table(agent.id, time_steps, {"has_timeout": [has_timeout] * len(time_steps), "has_missed_goal": [has_missed_goal] * len(time_steps)}))

    return _merge_into(criticality_dataframe, tables)

def augment_with_emergencies_and_disengagements(simulation: Simulation, criticality_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add, for each agent and time step, whether the agent was in emergency mode (has_emergency) or disengaged
    (has_disengagement), from the time steps recorded by the simulation.
    """
    time_steps = list(range(0, _scenario_final_time_step(simulation) + 1))

    tables = []
    for agent in sorted(simulation.agents, key = lambda a: a.id):
        emergency, disengagement = set(agent.record_emergency), set(agent.record_disengagement)
        tables.append(_agent_table(agent.id, time_steps, {
            "has_emergency": [1.0 if t in emergency else 0.0 for t in time_steps],
            "has_disengagement": [1.0 if t in disengagement else 0.0 for t in time_steps],
        }))

    return _merge_into(criticality_dataframe, tables)