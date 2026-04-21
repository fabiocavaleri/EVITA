#
# Implementation of basic mutation operators. Those are building blocks that can be used to build more complex search operators
import random
import numpy as np

from copy import deepcopy
from shapely.geometry import LineString

from . import utils as generation_utils
from ..config import ScenarioGeneratorConfig

import os, json
from typing import List
from config import INSTANCE_MAX_WAIT_TIME

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from commonroad.scenario.traffic_light import TrafficLightState
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

from conversion_scripts.utils.conversion_utils import get_traffic_light_configuration, get_traffic_light_cycle, compute_map_elements_mapping

class EVITAPlanningProblem(PlanningProblem):
    def __init__(self, planning_problem_id: int, initial_state: InitialState, goal_region: GoalRegion, path: List[int]):
        super().__init__(planning_problem_id, initial_state, goal_region)
        self.path = path

# Ideally, those must be initialized using HO Functions
def mutate_initial_position(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario) -> PlanningProblem:
    """
    Mutate the initial position of the given planning problem. Return a new planning problem
    """
    _planning_problem = deepcopy(planning_problem)

    # Get planning problem initial lanelet from staring position
    current_position = _planning_problem.initial_state.position
    ll_network = scenario.lanelet_network
    lanelets = ll_network.find_lanelet_by_position([current_position])
    
    # planning_problem is initially offroad
    if not (len(lanelets) != 0 and len(lanelets[0]) != 0):
        if type(_planning_problem) == EVITAPlanningProblem:
            lanelet = ll_network.find_lanelet_by_id(_planning_problem.path[0])
        else:
            raise ValueError
    else:
        # Get starting lanelet center line
        
        lanelet: Lanelet = ll_network.find_lanelet_by_id(lanelets[0][0])
    center_line = LineString(lanelet.center_vertices)
    # Randomly sample a new position on the lanelet center line
    position_modifier = random.uniform(-config.max_position_modifier, config.max_position_modifier)
    new_position = center_line.interpolate(position_modifier)
    _planning_problem.initial_state.position = np.array([new_position.x, new_position.y])

    return _planning_problem

def mutate_initial_velocity(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario) -> PlanningProblem:
    """
    Mutate the initial velocity of the given planning problem. Return a new planning problem
    """
    _planning_problem = deepcopy(planning_problem)

    # Randomly sample a new velocity. 
    new_velocity = random.uniform(config.min_velocity, config.max_init_velocity)
    _planning_problem.initial_state.velocity = new_velocity

    return _planning_problem

def mutate_goal_area_position(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario) -> PlanningProblem:
    """
    Mutate the position of the Goal Area velocity of the given planning problem. Return a new planning problem ensuring the goal area
    is reacheable
    """
    _planning_problem = deepcopy(planning_problem)

    # Lanelet of the current goal area. # Flattened as a set
    # Note goal_area.position is a rectangle!
    # list_of_list_of_lanelet_ids_to_avoid = scenario.lanelet_network.find_lanelet_by_position([s.position for s in planning_problem.goal.state_list])
    # TODO Why not simply 
    
    # Get the lanelet ids of the goal area
    lanelet_ids_to_avoid = []
    if planning_problem.goal.lanelets_of_goal_position:
        lanelet_ids_to_avoid = planning_problem.goal.lanelets_of_goal_position
    else:
        list_of_list_of_lanelet_ids_to_avoid = [scenario.lanelet_network.find_lanelet_by_shape(s.position) for s in planning_problem.goal.state_list]
        lanelet_ids_to_avoid = set([item for sublist in list_of_list_of_lanelet_ids_to_avoid for item in sublist])

    # Get the lanelet ids of the initial state    
    list_of_list_of_lanelet_ids_of_initial_state = scenario.lanelet_network.find_lanelet_by_position([planning_problem.initial_state.position])
    lanelet_ids_of_initial_state = set([item for sublist in list_of_list_of_lanelet_ids_of_initial_state for item in sublist])

    # Compute the set of reacheable goal areas from the initial lanelets
    full_lanelets = []
    for lanelet in [scenario.lanelet_network.find_lanelet_by_id(lanelet_id) for lanelet_id in lanelet_ids_of_initial_state]:
        full_lanelets = full_lanelets + generation_utils.get_successor_lanelets(lanelet, scenario.lanelet_network)

    # Make this a set
    full_lanelets = set(full_lanelets)

    # Remove the ones to avoid
    exit_points = full_lanelets.difference(lanelet_ids_to_avoid)
    
    if len(exit_points) == 0:
        print("Cannot find a new position of the goal area!")
        # TODO What to do? nothing for the moment
    else:
        # Take one at random
        goal_lanelet = random.choice(list(exit_points))

        # Who provides this?
        goal_area = generation_utils.get_possible_goal_for_lanelet(config, goal_lanelet)
        goal = generation_utils.generate_goal(config, goal_area)

        _planning_problem.goal = goal

    return _planning_problem

def add_random_planning_problem(config: ScenarioGeneratorConfig, scenario: Scenario, problem_set: PlanningProblemSet) -> PlanningProblemSet:
    """
    Return a new PlanningProblemSet containing the same problems plus a random one
    """

    _scenario = deepcopy(scenario)
    _problem_set = deepcopy(problem_set)

    # Pick a random lanelet to place the AV 
    lanelet = random.choice(_scenario.lanelet_network.lanelets)

    # Get any adjacent random lanelet with same direction
    adjacent_lanelets = generation_utils.get_adjacent_lanelets(lanelet, _scenario.lanelet_network)

    # Get a random "exit" lanelet reacheable from the selected one or its adjacent lanelets
    full_lanelets = generation_utils.get_successor_lanelets(lanelet, _scenario.lanelet_network)

    for adjacent in adjacent_lanelets:
        full_lanelets = full_lanelets + generation_utils.get_successor_lanelets(adjacent, _scenario.lanelet_network)

    # Pick one
    random_exit_lanelet = random.choice(full_lanelets)

    goal_area = generation_utils.get_possible_goal_for_lanelet(config, random_exit_lanelet)
    goal = generation_utils.generate_goal(config, goal_area)

    # TODO Ensure that they do not have the same ID is a tricky businness...
    if _problem_set.planning_problem_dict:
        last_key, pp = max(list(_problem_set.planning_problem_dict.items()))
    else:
        # Note start with the predefined ID to ensure uniqueness of IDs
        last_key = config.av_base_id
    new_id = last_key + 1

    init_state = generation_utils.get_random_init_state(config, random.choice(full_lanelets))
    new_pp = PlanningProblem(new_id, init_state, goal)
    _problem_set.add_planning_problem(new_pp)

    return _problem_set

def point_at_distance_with_next(lanelet, s):
    """
    Returns:
        point: interpolated (x, y) at distance s
        next_point: next center vertex after s
    """
    distances = lanelet.distance
    center = lanelet.center_vertices

    # Clamp s
    s = np.clip(s, 0.0, distances[-1])

    idx = np.searchsorted(distances, s)

    # If at or before start
    if idx == 0:
        return center[0], center[1]

    # If beyond last segment
    if idx >= len(distances):
        return center[-1], center[-1]

    # Interpolate
    s0, s1 = distances[idx - 1], distances[idx]
    p0, p1 = center[idx - 1], center[idx]

    ratio = (s - s0) / (s1 - s0)
    point = p0 + ratio * (p1 - p0)

    next_point = center[idx]

    return point, next_point

def add_evita_random_planning_problem(config: ScenarioGeneratorConfig, scenario: Scenario, problem_set: PlanningProblemSet, scenario_map_file_path: str = None) -> PlanningProblemSet:
    """
    Return a new PlanningProblemSet containing the same problems plus a random one
    """

    _scenario = deepcopy(scenario)
    _problem_set = deepcopy(problem_set)

    scenario_map_path = scenario_map_file_path.removesuffix(".xml")

    # Open .json file containing map lanes that are not in an intersection
    lanelets_not_in_junction_file_path = os.path.join(f"{scenario_map_path}_lanes_not_in_junction.json")
    with open(lanelets_not_in_junction_file_path, "r") as f:
        lanelets_not_in_junction = json.load(f)

    while True:
        # Pick a random lanelet to place the AV 
        initial_lanelet = scenario.lanelet_network.find_lanelet_by_id(random.choice(lanelets_not_in_junction))

        # Check that the initial lanelet has at least a successor (not only adjacent lanes)
        if len(initial_lanelet.successor) > 0:
            break

    # Open .json file with all the feasible paths for the map lanelets
    map_feasible_paths_file_path = os.path.join(f"{scenario_map_path}_feasible_paths.json")
    with open(map_feasible_paths_file_path, "r") as f:
        feasible_paths = json.load(f)

    # Extract input lanelet feasible paths
    lanelet_feasible_paths = feasible_paths[str(initial_lanelet.lanelet_id)]

    while True:
        # Pick a random goal lanelet
        random_path = random.choice(lanelet_feasible_paths)
        random_exit_lanelet = _scenario.lanelet_network.find_lanelet_by_id(random_path[-1])
        # Check that the goal lanelet is not in a junction (to avoid short routes) and that the path is not within a single lanelet
        if random_path[-1] in lanelets_not_in_junction and len(random_path) > 1:
            break

    # Get initial and final lanelet lengths
    start_lane_length = np.sum(np.linalg.norm(np.diff(initial_lanelet.center_vertices, axis = 0), axis = 0))
    final_lane_length = np.sum(np.linalg.norm(np.diff(random_exit_lanelet.center_vertices, axis = 0), axis = 0))

    # Select start position
    if start_lane_length > 5:
        start_position_distance = round(random.random() * (start_lane_length - 5), 1)
    else:
        start_position_distance = 0.0

    # Extract start position and next point to compute orientation
    start_point = initial_lanelet.interpolate_position(start_position_distance)[0]
    start_next_point = initial_lanelet.interpolate_position(start_position_distance + 1.0)[0]

    # Compute initial velocity, orientation, acceleration, yaw_rate and slip angle
    velocity = random.uniform(config.min_init_velocity, config.max_init_velocity)
    orientation = generation_utils.get_orientation_by_coords((start_point[0], start_point[1]), (start_next_point[0], start_next_point[1]))
    acceleration = 0.0
    yaw_rate = 0.0
    slip_angle = 0.0

    # Randomply select the initial timestep
    initial_time_step = random.randint(0, INSTANCE_MAX_WAIT_TIME * 10)
    
    init_state = InitialState(time_step = initial_time_step, 
                            position = start_point,
                            orientation = orientation, 
                            velocity = velocity, 
                            acceleration = acceleration,
                            yaw_rate = yaw_rate,
                            slip_angle = slip_angle)


    # Select goal position
    goal_position_distance = round(final_lane_length / 2, 1)

    # Extract goal position and next point to compute orientation
    goal_center_point = random_exit_lanelet.interpolate_position(goal_position_distance)[0]
    goal_next_point = random_exit_lanelet.interpolate_position(goal_position_distance + 1.0)[0]

    # Compute goal position orientation
    orientation = generation_utils.get_orientation_by_coords((goal_center_point[0], goal_center_point[1]), (goal_next_point[0], goal_next_point[1]))

    # Create goal area object
    goal_area = Rectangle(config.length, config.width, np.array([goal_center_point[0], goal_center_point[1]]), orientation)
    goal = generation_utils.generate_goal(config, goal_area)

    if _problem_set.planning_problem_dict:
        last_key, _ = max(list(_problem_set.planning_problem_dict.items()))
    else:
        # Note start with the predefined ID to ensure uniqueness of IDs
        last_key = config.av_base_id

    new_id = last_key + 1

    new_pp = EVITAPlanningProblem(new_id, init_state, goal, random_path)
    _problem_set.add_planning_problem(new_pp)
    
    return _problem_set

def mutate_evita_goal_area_position(config: ScenarioGeneratorConfig, planning_problem: EVITAPlanningProblem, scenario: Scenario, scenario_map_file_path: str) -> EVITAPlanningProblem:
    """
    Mutate the position of the Goal Area velocity of the given planning problem. Return a new planning problem ensuring the goal area
    is reacheable
    """
    _planning_problem = deepcopy(planning_problem)

    # Get the lanelet ids of the initial state    
    lanelet_id_of_initial_state = _planning_problem.path[0]

    scenario_map_path = scenario_map_file_path.removesuffix(".xml")

    # Open .json file containing map lanes that are not in an intersection
    lanelets_not_in_junction_file_path = os.path.join(f"{scenario_map_path}_lanes_not_in_junction.json")
    with open(lanelets_not_in_junction_file_path, "r") as f:
        lanelets_not_in_junction = json.load(f)

    # Open .json file with all the feasible paths for the map lanelets
    map_feasible_paths_file_path = os.path.join(f"{scenario_map_path}_feasible_paths.json")
    with open(map_feasible_paths_file_path, "r") as f:
        feasible_paths = json.load(f)

    # Extract input lanelet feasible paths
    lanelet_feasible_paths = feasible_paths[str(lanelet_id_of_initial_state)]

    correct = False
    attempted_paths = []
    while len(attempted_paths) < len(lanelet_feasible_paths):
        # Pick a random goal lanelet
        random_path = random.choice(lanelet_feasible_paths)

        # Skip already attempted paths
        if random_path in attempted_paths:
            continue
        
        attempted_paths.append(random_path)
        random_exit_lanelet = scenario.lanelet_network.find_lanelet_by_id(random_path[-1])

        # Check that the goal lanelet is not in a junction (to avoid short routes) and that the path is not within a single lanelet
        if random_path[-1] in lanelets_not_in_junction and len(random_path) > 1 and random_path[-1] != _planning_problem.path[-1]:
            correct = True
            break
    
    # If there is not any other path available, keep the same goal lane and compute a new goal position
    if not correct:
        random_path = _planning_problem.path
        random_exit_lanelet = scenario.lanelet_network.find_lanelet_by_id(random_path[-1])

    # Get final lanelet length
    final_lane_length = np.sum(np.linalg.norm(np.diff(random_exit_lanelet.center_vertices, axis = 0), axis = 0))

    # Select goal position
    goal_position_distance = round(final_lane_length / 2, 1)

    # Extract goal position and next point to compute orientation
    goal_center_point = random_exit_lanelet.interpolate_position(goal_position_distance)[0]
    goal_next_point = random_exit_lanelet.interpolate_position(goal_position_distance + 1.0)[0]

    # Compute goal position orientation
    orientation = generation_utils.get_orientation_by_coords((goal_center_point[0], goal_center_point[1]), (goal_next_point[0], goal_next_point[1]))

    # Create goal area object
    goal_area = Rectangle(config.length, config.width, np.array([goal_center_point[0], goal_center_point[1]]), orientation)
    goal = generation_utils.generate_goal(config, goal_area)

    _planning_problem.goal = goal

    return _planning_problem

def remove_random_planning_problem(config: ScenarioGeneratorConfig, scenario: Scenario, problem_set: PlanningProblemSet) -> PlanningProblemSet:
    """
    Return a new PlanningProblemSet containing one planning problem less than the input one
    """
    _problem_set = deepcopy(problem_set)
    
    # Select a random planning problem to remove
    pp = random.choice(list(_problem_set.planning_problem_dict.values()))
    pp_id_to_remove = _problem_set.find_planning_problem_by_id(pp.planning_problem_id)
    
    _problem_set.planning_problem_dict.pop(pp_id_to_remove.planning_problem_id)

    return _problem_set

def mutate_tl_initial_configuration(_: ScenarioGeneratorConfig, scenario: Scenario, map_file_path: str) -> Scenario:
    """
    Mutate the initial configuration of the scenario's traffic lights. Return the updated scenario.
    """
    # Easy mapping between CommonRoad traffic light colors and string representation
    color_map = {
        TrafficLightState.RED : "RED",
        TrafficLightState.YELLOW : "YELLOW",
        TrafficLightState.GREEN : "GREEN"
    }

    # Extract CommonRoad scenario's traffic lights
    scenario_traffic_lights = scenario.lanelet_network.traffic_lights

    # Get the traffic light cycle elements of one signal in the scenario
    traffic_light_cycle_elements = scenario_traffic_lights[0].traffic_light_cycle.cycle_elements
    
    # Extract traffic lights color durations
    traffic_light_durations = {
        color_map[element.state]: element.duration
        for element in traffic_light_cycle_elements
    }

    durations = {
        "red_wait_duration": traffic_light_durations['RED'],
        "yellow_duration": traffic_light_durations['YELLOW'],
        "green_duration": traffic_light_durations['GREEN']
    }

    # Generate a new random traffic light configuration
    _, signals_id_map, reverse_signals_id_map, signals_relations_map = compute_map_elements_mapping(map_file_path)
    traffic_light_configurations = get_traffic_light_configuration(scenario, signals_id_map, reverse_signals_id_map, signals_relations_map)

    # Assign traffic light configuration to each signal in the map
    for signal_id, color in traffic_light_configurations.items():
        signal = scenario.lanelet_network.find_traffic_light_by_id(signal_id)

        signal.traffic_light_cycle = get_traffic_light_cycle(color, durations)
        signal.initial_state = (TrafficLightState.RED if color == "RED" else TrafficLightState.GREEN)
        signal.time_offset = 0
    
    return scenario

def mutate_green_light_duration(config: ScenarioGeneratorConfig, scenario: Scenario, _: str) -> Scenario:
    """
    Mutate the initial configuration of the scenario's traffic lights. Return the updated scenario.
    """
    # Generate the traffic lights durations
    traffic_light_durations = {
        "red_wait_duration": 2 * 10,
        "yellow_duration": 3 * 10,
        "green_duration": random.randint(5, int(config.max_time_step / 2))
    }

    durations = {
        TrafficLightState.RED : (traffic_light_durations["red_wait_duration"] + traffic_light_durations["yellow_duration"] + traffic_light_durations["green_duration"]),
        TrafficLightState.YELLOW : traffic_light_durations["yellow_duration"],
        TrafficLightState.GREEN : traffic_light_durations["green_duration"]
    }

    # Extract CommonRoad scenario's traffic lights
    scenario_traffic_lights = scenario.lanelet_network.traffic_lights

    # Assign traffic light configuration to each signal in the map
    for signal in scenario_traffic_lights:
        for cycle_element in signal.traffic_light_cycle.cycle_elements:
            cycle_element.duration = durations[cycle_element.state]
            
    return scenario