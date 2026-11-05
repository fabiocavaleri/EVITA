"""
Mutation operators on scenarios: building blocks of the search operators.

Mutations of a vehicle's planning problem: initial position, initial velocity, goal area.
Mutations of the set of vehicles: add a random vehicle, remove a random vehicle.
Mutations of the traffic lights (Apollo): initial configuration, green light duration.

Every operator works on a copy and returns it; its inputs are never modified.
The functions named *_evita_* and the traffic-light mutations belong to the Apollo pipeline.
"""
import os
import json
import logging
import random
import numpy as np

from copy import deepcopy
from typing import List, Optional
from shapely.geometry import LineString, Point

from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from commonroad.scenario.traffic_light import TrafficLightState

from EVITA_DoppelTest.config import INSTANCE_MAX_WAIT_TIME
from EVITA_DoppelTest.conversion_scripts.utils.conversion_utils import compute_map_elements_mapping, get_traffic_light_configuration, get_traffic_light_cycle

from . import utils as generation_utils
from ..config import ScenarioGeneratorConfig
from ..utils import EVITA_MAIN_LOGGER_NAME

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

# Attempts to place a new vehicle before giving up and returning the planning problems unchanged
MAX_PLACEMENT_ATTEMPTS = 100

class EVITAPlanningProblem(PlanningProblem):
    """
    Custom class for EVITA planning problems, which includes the path of lanelet ids to reach the goal
    """
    def __init__(self, planning_problem_id: int, initial_state: InitialState, goal_region: GoalRegion, path: List[int]):
        super().__init__(planning_problem_id, initial_state, goal_region)
        self.path = path

# ========== Mutations of a vehicle ==========

def _initial_lanelet(planning_problem: PlanningProblem, scenario: Scenario) -> Lanelet:
    """
    Lanelet the vehicle starts on.
    """
    if isinstance(planning_problem, EVITAPlanningProblem):
        return lanelet_network.find_lanelet_by_id(planning_problem.path[0])
    
    lanelet_network = scenario.lanelet_network
    lanelet_ids = lanelet_network.find_lanelet_by_position([planning_problem.initial_state.position])[0]

    if lanelet_ids:
        return lanelet_network.find_lanelet_by_id(lanelet_ids[0])
    
    raise ValueError(f"Could not find lanelet for initial position {planning_problem.initial_state.position} and planning problem type {type(planning_problem).__name__}")

def _place_on_line(initial_state: InitialState, center_line: LineString, s: float) -> None:
    """
    Move the initial state to distance `s` along the centre line (clamped to the line), heading along it.
    """
    s = min(max(s, 0.0), center_line.length)
    point = center_line.interpolate(s)
    ahead = center_line.interpolate(min(s + 0.5, center_line.length))
    behind = center_line.interpolate(max(s - 0.5, 0.0))

    initial_state.position = np.array([point.x, point.y])
    initial_state.orientation = generation_utils.get_orientation_by_coords((behind.x, behind.y), (ahead.x, ahead.y))

def mutate_initial_position(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario, scenario_map_file_path: str = None) -> PlanningProblem:
    """
    Mutate the vehicle's initial position, with equal probability:
      - shift it along its lanelet by a random value in [-max_position_modifier, +max_position_modifier] metres, or
      - move it to the same distance along an adjacent lanelet with the same driving direction, if there is one.
    The heading follows the centre line at the new position.
    """
    _planning_problem = deepcopy(planning_problem)

    # Get planning problem initial lanelet and its center line
    lanelet = _initial_lanelet(_planning_problem, scenario)    
    center_line = LineString(lanelet.center_vertices)
    s_current = center_line.project(Point(_planning_problem.initial_state.position))

    # Randomly decide whether to move along the same lanelet or switch to an adjacent lanelet
    adjacent_lanelets = generation_utils.get_adjacent_lanelets(lanelet, scenario.lanelet_network)
    if adjacent_lanelets and random.random() < 0.5:
        # Same distance along a different lane
        new_lanelet = random.choice(adjacent_lanelets)
        new_line = LineString(new_lanelet.center_vertices)
        _place_on_line(_planning_problem.initial_state, new_line, s_current / center_line.length * new_line.length)
    else:
        position_modifier = random.uniform(-config.max_position_modifier, config.max_position_modifier)
        _place_on_line(_planning_problem.initial_state, center_line, s_current + position_modifier)

    return _planning_problem

def mutate_initial_velocity(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario, scenario_map_file_path: Optional[str] = None) -> PlanningProblem:
    """
    Mutate the vehicle's initial velocity
    Change it by a random amount in [min_init_velocity, max_init_velocity] m/s, up or down with equal probability, keeping it within
    [min_velocity, max_velocity].
    """
    _planning_problem = deepcopy(planning_problem)

    change = random.uniform(config.min_init_velocity, config.max_init_velocity) * random.choice((-1.0, 1.0))
    new_velocity = _planning_problem.initial_state.velocity + change
    _planning_problem.initial_state.velocity = min(max(new_velocity, config.min_velocity), config.max_velocity)

    return _planning_problem

def mutate_goal_area_position(config: ScenarioGeneratorConfig, planning_problem: PlanningProblem, scenario: Scenario, scenario_map_file_path: str = None) -> PlanningProblem:
    """
    Move the goal area to the end of a different exit lane reachable (forward) from the vehicle's initial lanelet
    The goal is kept if there is no other reachable exit lanelets.
    """
    _planning_problem = deepcopy(planning_problem)
    ll_network = scenario.lanelet_network

    # Step 1: Lanelets the current goal area lies on
    goal_lanelet_ids = set()
    for state in planning_problem.goal.state_list:
        if hasattr(state, "position"):
            goal_lanelet_ids.update(ll_network.find_lanelet_by_shape(state.position))

    if planning_problem.goal.lanelets_of_goal_position:
        for ids in planning_problem.goal.lanelets_of_goal_position.values():
            goal_lanelet_ids.update(ids)

    # Step 2: Routes from the lanelets under the vehicle to an exit other than the current one
    ids_at_start = ll_network.find_lanelet_by_position([planning_problem.initial_state.position])[0]
    candidates = [route for start_id in ids_at_start
                  for route, exit_id in generation_utils.routes_to_exits(ll_network.find_lanelet_by_id(start_id), ll_network)
                  if exit_id not in goal_lanelet_ids]

    # Step 4: Pick one at random, or keep the current goal if there is no alternative exit
    if not candidates:
        module_logger.debug(f"No alternative exit lane for planning problem {planning_problem.planning_problem_id}")

        return _planning_problem

    goal_area = generation_utils.get_possible_goal_for_lanelet(config, random.choice(candidates))
    _planning_problem.goal = generation_utils.generate_goal(config, goal_area)

    return _planning_problem

# ========== Mutations of the set of vehicles ==========

def _next_planning_problem_id(config: ScenarioGeneratorConfig, problem_set: PlanningProblemSet) -> int:
    """
    Id for a new planning problem: one more than the largest id in use, or than av_base_id if there is none.
    """
    return max(problem_set.planning_problem_dict.keys(), default = config.av_base_id) + 1

def add_random_planning_problem(config: ScenarioGeneratorConfig, scenario: Scenario, problem_set: PlanningProblemSet, map_file_path: Optional[str] = None) -> PlanningProblemSet|EVITAPlanningProblem:
    """
    Copy of the planning problems plus a new vehicle
    On a random lanelet, take the routes to the exits from it and from its adjacent lanes with the same direction;
    The planning problems are returned unchanged if no start can be placed after MAX_PLACEMENT_ATTEMPTS attempts.
    """
    # Copy the scenario and problem set
    _problem_set = deepcopy(problem_set)
    lanelet_network = scenario.lanelet_network

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        lanelet = random.choice(lanelet_network.lanelets)

        # Routes from the lanelet and from its adjacent lanes with the same direction
        routes = [route for route, _ in generation_utils.routes_to_exits(lanelet, lanelet_network)]
        for adjacent in generation_utils.get_adjacent_lanelets(lanelet, lanelet_network):
            routes += [route for route, _ in generation_utils.routes_to_exits(adjacent, lanelet_network)]

        random_route = random.choice(routes)

        init_state = generation_utils.get_random_init_state(config, random_route[0])
        if init_state is None:
            continue

        goal = generation_utils.generate_goal(config, generation_utils.get_possible_goal_for_lanelet(config, random_route[-1]))
        _problem_set.add_planning_problem(PlanningProblem(_next_planning_problem_id(config, _problem_set), init_state, goal))

        return _problem_set

    module_logger.debug(f"Could not place a new vehicle after {MAX_PLACEMENT_ATTEMPTS} attempts")
    return _problem_set

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

# ========== Apollo pipeline ==========

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