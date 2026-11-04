#
# Implementation of basic mutation operators. Those are building blocks that can be used to build more complex search operators
import random
import numpy as np
import logging
from typing import Tuple

from copy import deepcopy
from shapely.geometry import LineString

from . import utils as generation_utils
from ..config import ScenarioGeneratorConfig

import os, json
from typing import List
from EVITA_DoppelTest.config import INSTANCE_MAX_WAIT_TIME

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from commonroad.scenario.traffic_light import TrafficLightState
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

from EVITA_DoppelTest.conversion_scripts.utils.conversion_utils import get_traffic_light_configuration, get_traffic_light_cycle, compute_map_elements_mapping

from ..utils import EVITA_MAIN_LOGGER_NAME

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

class EVITAPlanningProblem(PlanningProblem):
    """
    Custom class for EVITA planning problems, which includes the path of lanelet ids to reach the goal
    """
    def __init__(self, planning_problem_id: int, initial_state: InitialState, goal_region: GoalRegion, path: List[int]):
        super().__init__(planning_problem_id, initial_state, goal_region)
        self.path = path

def mutate_initial_position(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario, scenario_map_file_path: str = None) -> PlanningProblem:
    """
    Mutate the initial position of the given planning problem.
    Return a new planning problem
    """
    _planning_problem = deepcopy(planning_problem)

    # Get planning problem initial lanelet from starting position
    current_position = _planning_problem.initial_state.position
    lanelet_network = scenario.lanelet_network
    lanelets = lanelet_network.find_lanelet_by_position([current_position])
    
    # Get initial lanelet
    if lanelets and lanelets[0]:
        lanelet = lanelet_network.find_lanelet_by_id(lanelets[0][0])
    elif isinstance(_planning_problem, EVITAPlanningProblem):
        lanelet = lanelet_network.find_lanelet_by_id(_planning_problem.path[0])
    else:
        raise ValueError(f"Could not find lanelet for initial position {current_position} and planning problem type {type(_planning_problem).__name__}")
    
    center_line = LineString(lanelet.center_vertices)

    # Randomly sample a new position on the lanelet center line
    position_modifier = random.uniform(-config.max_position_modifier, config.max_position_modifier)
    new_position = center_line.interpolate(position_modifier)
    _planning_problem.initial_state.position = np.array([new_position.x, new_position.y])

    return _planning_problem

def mutate_initial_velocity(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario, scenario_map_file_path: str = None) -> PlanningProblem:
    """
    Mutate the initial velocity of the given planning problem. Return a new planning problem
    """
    _planning_problem = deepcopy(planning_problem)

    # Randomly sample a new velocity. 
    new_velocity = random.uniform(config.min_velocity, config.max_init_velocity)
    _planning_problem.initial_state.velocity = new_velocity

    return _planning_problem

def _routes_with_exit(start_lanelet: Lanelet, ll_network) -> List[Tuple[Lanelet, int]]:
    """
    All routes that can be driven forward from start_lanelet, each returned as (merged lanelet of the whole route, id of the exit lanelet where the route ends)
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

def mutate_goal_area_position(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario, scenario_map_file_path: str = None) -> PlanningProblem:
    """
    Move the goal area of the planning problem to the end of a different exit lane
    reachable from the initial lanelet (in the driving direction)
    """
    _planning_problem = deepcopy(planning_problem)
    ll_network = scenario.lanelet_network
    initial_state = planning_problem.initial_state

    # Step 1: Lanelets the current goal area lies on
    goal_lanelet_ids = set()
    for state in planning_problem.goal.state_list:
        if hasattr(state, "position"):
            goal_lanelet_ids.update(ll_network.find_lanelet_by_shape(state.position))

    if planning_problem.goal.lanelets_of_goal_position:
        for ids in planning_problem.goal.lanelets_of_goal_position.values():
            goal_lanelet_ids.update(ids)

    # Step 2: Lanelets under the vehicle that point in its driving direction
    ids_at_start = ll_network.find_lanelet_by_position([initial_state.position])[0]
    start_lanelets = [ll for ll in (ll_network.find_lanelet_by_id(i) for i in ids_at_start)]

    # Step 3: Reachable routes whose exit lane is NOT the current one
    candidates = [route for start in start_lanelets for route, exit_id in _routes_with_exit(start, ll_network) if exit_id not in goal_lanelet_ids]

    # Step 4: Pick one at random, or keep the current goal if there is no alternative exit
    if not candidates:
        module_logger.debug(f"No alternative exit lane for planning problem {planning_problem.planning_problem_id}")

        return _planning_problem

    goal_area = generation_utils.get_possible_goal_for_lanelet(config, random.choice(candidates))
    _planning_problem.goal = generation_utils.generate_goal(config, goal_area)

    return _planning_problem

def add_random_planning_problem(config: ScenarioGeneratorConfig, scenario: Scenario, problem_set: PlanningProblemSet, map_file_path: str = None) -> PlanningProblemSet|EVITAPlanningProblem:
    """
    Return a new PlanningProblemSet containing the same problems plus a random one
    """
    # Copy the scenario and problem set
    _scenario = deepcopy(scenario)
    _problem_set = deepcopy(problem_set)

    # Pick a random lanelet to place the AV 
    lanelet = random.choice(_scenario.lanelet_network.lanelets)

    # Get all the successor lanelets of the initial lanelet
    full_lanelets = generation_utils.get_successor_lanelets(lanelet, _scenario.lanelet_network)

    # Get any adjacent random lanelet with same direction
    adjacent_lanelets = generation_utils.get_adjacent_lanelets(lanelet, _scenario.lanelet_network)

    # Add the adjacent lanelets successors to the list of possible exit lanelets
    for adjacent in adjacent_lanelets:
        full_lanelets = full_lanelets + generation_utils.get_successor_lanelets(adjacent, _scenario.lanelet_network)

    # Pick one of the exit lanelets at random
    random_exit_lanelet = random.choice(full_lanelets)

    # Generate a new random goal area and goal position
    goal_area = generation_utils.get_possible_goal_for_lanelet(config, random_exit_lanelet)
    goal = generation_utils.generate_goal(config, goal_area)

    # Check for planning problem ID uniqueness and generate a new ID
    if _problem_set.planning_problem_dict:
        last_key, pp = max(list(_problem_set.planning_problem_dict.items()))
    else:
        # Note start with the predefined ID to ensure uniqueness of IDs
        last_key = config.av_base_id

    new_id = last_key + 1

    # Generate a new initial state for the planning problem
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

def add_evita_random_planning_problem(config: ScenarioGeneratorConfig, scenario: Scenario, problem_set: PlanningProblemSet, scenario_map_file_path: str) -> PlanningProblemSet:
    """
    Return a new PlanningProblemSet containing the same problems plus a random one
    This is a custom version of the add_random_planning_problem function that ensures the generated planning problem is compatible with the EVITA + DoppelTest framework
    """

    # Copy the scenario and problem set
    _scenario = deepcopy(scenario)
    _problem_set = deepcopy(problem_set)
    lanelet_network = _scenario.lanelet_network

    # Retrieve scenario's map name
    scenario_map_path = scenario_map_file_path.removesuffix(".xml")

    # Open .json file containing map lanes that are not in an intersection
    lanelets_not_in_junction_file_path = os.path.join(f"{scenario_map_path}_lanes_not_in_junction.json")
    with open(lanelets_not_in_junction_file_path, "r") as f:
        lanelets_not_in_junction = json.load(f)

    # Open .json file with all the feasible paths for the map lanelets
    map_feasible_paths_file_path = os.path.join(f"{scenario_map_path}_feasible_paths.json")
    with open(map_feasible_paths_file_path, "r") as f:
        feasible_paths = json.load(f)

    # Iterate until a valid planning problem is found
    while True:
        # Pick a random lanelet to place the AV 
        initial_lanelet = scenario.lanelet_network.find_lanelet_by_id(random.choice(lanelets_not_in_junction))

        # Check that the initial lanelet has at least a successor (not only adjacent lanes)
        if len(initial_lanelet.successor) == 0:
            continue

        # Extract input lanelet feasible paths
        lanelet_feasible_paths = feasible_paths[str(initial_lanelet.lanelet_id)]
        # Filter only the paths with a valid goal lanelet
        filtered_feasible_paths = [path for path in lanelet_feasible_paths if path[-1] in lanelets_not_in_junction and not lanelet_network.find_lanelet_by_id(path[-1]).successor]

        # Check there is at least one valid path
        if len(filtered_feasible_paths) == 0:
            continue
        
        random_path = random.choice(filtered_feasible_paths)

        # Generate an initial state for the planning problem
        init_state = generation_utils.get_random_init_state(config, initial_lanelet)
        if init_state is None:
            continue

        # Randomly select the initial timestep
        init_state.time_step = random.randint(0, INSTANCE_MAX_WAIT_TIME * 10)

        # Generate a goal area
        goal_area = generation_utils.get_possible_goal_for_lanelet(config, _merge_path(lanelet_network, random_path))
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

def _merge_path(lanelet_network, path: List[int]) -> Lanelet:
    """
    Merge the lanelets of a path into a single lanelet, as get_successor_lanelets does for the Frenetix routes
    """
    merged = None

    for lanelet_id in path:
        lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
        if merged is None:
            merged = deepcopy(lanelet)
        else:
            merged = Lanelet.merge_lanelets(merged, lanelet)
    
    return merged

def mutate_evita_goal_area_position(config: ScenarioGeneratorConfig, planning_problem: EVITAPlanningProblem, scenario: Scenario, scenario_map_file_path: str) -> EVITAPlanningProblem:
    """
    Mutate the position of the Goal Area velocity of the given planning problem
    Return a new planning problem ensuring the goal area is reacheable
    """
    _planning_problem = deepcopy(planning_problem)
    lanelet_network = scenario.lanelet_network

    # Get the lanelet ids of the initial state    
    lanelet_id_of_initial_state = _planning_problem.path[0]

    # Extract scenario's map name
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
    # Filter only the paths with a valid goal lanelet
    filtered_feasible_paths = [path for path in lanelet_feasible_paths if path[-1] in lanelets_not_in_junction and not lanelet_network.find_lanelet_by_id(path[-1]).successor and path[-1] != _planning_problem.path[-1]]

    if len(filtered_feasible_paths) == 0:
        module_logger.debug("No other path available")
        # If there is not any other path available, keep the same goal lane and compute a new goal position
        random_path = _planning_problem.path
    else:
        # Pick a random goal lanelet
        random_path = random.choice(filtered_feasible_paths)
    
    # Generate a goal area
    goal_area = generation_utils.get_possible_goal_for_lanelet(config, _merge_path(lanelet_network, random_path))
    goal = generation_utils.generate_goal(config, goal_area)

    _planning_problem.goal = goal
    _planning_problem.path = random_path

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