import os
import json
import numpy as np
from random import randint, shuffle
from typing import  List, Dict, Literal, Tuple, Any

from commonroad.scenario.lanelet import Lanelet
from commonroad.planning.goal import GoalRegion
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.traffic_light import TrafficLightCycleElement, TrafficLightCycle, TrafficLightState

def build_ad_section(scenario: Scenario, planning_problem_set: PlanningProblemSet, reverse_lane_id_map: Dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    Method to build the .json representation of the CommonRoad scenario autonomous vehicles
    
    :param scenario: CommonRoad scenario
    :type scenario: Scenario
    :param planning_problem_set: CommonRoad scenario planning problems
    :type planning_problem_set: PlanningProblemSet
    """

    adcs = []

    # Iterate over scenario's planning problems
    for planning_problem in planning_problem_set.planning_problem_dict.values():
        # Extract vehicle goal lanelet
        goal_position = planning_problem.goal.state_list[0].position.center
        lanelets_goal =  scenario.lanelet_network.find_lanelet_by_position([goal_position])[0]
        lanelets_of_goal_position = {0: lanelets_goal}

        # Build goal region object
        new_goal = GoalRegion(
            state_list = planning_problem.goal.state_list,
            lanelets_of_goal_position = lanelets_of_goal_position
        )
        planning_problem.goal = new_goal

        # Extract vehicle routing: retrieve sequence of lanelets from start lane to goal lane
        routing, start_lanelet, goal_lanelet = extract_routing(scenario, planning_problem, reverse_lane_id_map)
        
        # Extract start time and position
        st = extract_st(planning_problem.initial_state, start_lanelet)
        
        # Extract goal position
        dest_s = extract_dest_s(goal_position, goal_lanelet)

        # Build the adcs section of the .json file
        adcs.append({
            "routing": routing,
            "start_s": st["start_s"],
            "dest_s": dest_s,
            "start_t": st["start_t"]
        })

    return {"adcs": adcs}

def build_tc_section(scenario: Scenario, signals_id_map: Dict[str, int], reverse_signals_id_map: Dict[str, str], signals_relations_map: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Dict[str, Dict[str, str] | int]]:
    """
    Method to build the .json representation of the CommonRoad scenario traffic lights
    
    :param scenario: CommonRoad scenario
    :type scenario: Scenario
    :param signals_id_map: Mapping between CommonRoad and Apollo traffic light ids
    :type signals_id_map: Dict[str, int]
    :param reverse_signals_id_map: Mapping between Apollo and CommonRoad traffic light ids
    :type reverse_signals_id_map: Dict[str, str]
    :param signals_relations_map: Mapping between traffic light ids
    :type signals_relations_map: Dict[str, List[Tuple[str, str]]]
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
    # Convert them from timesteps (with timestep duration 0.1) to seconds representation by dividing by 10
    durations = {
        color_map[element.state]: (element.duration / 10)
        for element in traffic_light_cycle_elements
    }

    # Apollo scenarios representation stores only the additional waiting time for red light
    # Since we build CommonRoad red light duration as red_wait_time + yellow_duration + green_duration
    # we subtract yellow_duration and green_duration
    durations["RED"] -= (durations["YELLOW"] + durations["GREEN"])

    # Extract the initial color for each traffic light in the scenario
    initial_light_colors = {}
    for traffic_light in scenario_traffic_lights:
        first_cycle_element = traffic_light.traffic_light_cycle.cycle_elements[0]
        initial_light_colors[traffic_light.traffic_light_id] = color_map[first_cycle_element.state]

    # Generate the final traffic light color following the Apollo logic
    preferences = [signal_id for signal_id in initial_light_colors if initial_light_colors[signal_id] == 'RED']
    final_light_colors = get_traffic_light_configuration(scenario, signals_id_map, reverse_signals_id_map, signals_relations_map, preferences)

    # Replace CommonRoad map traffic_light ids with Apollo original ones
    for traffic_light_id in initial_light_colors.copy().keys():
        apollo_signal_id = reverse_signals_id_map[traffic_light_id]

        initial_light_color = initial_light_colors.pop(traffic_light_id)
        final_light_color = final_light_colors.pop(traffic_light_id)

        initial_light_colors[apollo_signal_id] = initial_light_color
        final_light_colors[apollo_signal_id] = final_light_color      

    tc_section = {
        "initial": initial_light_colors,
        "final": final_light_colors,
        "duration_g": int(durations["GREEN"]),
        "duration_y": int(durations["YELLOW"]),
        "duration_b": int(durations["RED"]),
    }

    return tc_section

def build_json_scenario(scenario: Scenario, planning_problems: PlanningProblemSet, map_path: str) -> Dict:
    '''
    Method to convert a CommonRoad scenario to a Baidu Apollo compatible .json format
    
    :param scenario: CommonRoad scenario
    :type scenario: Scenario
    :param planning_problems: CommonRoad scenario's planning problems
    :type planning_problems: PlanningProblemSet
    :param scenario_map_path: Path to the CommonRoad scenario's map
    :type scenario_map_path: str
    '''
    
    reverse_lane_id_map, signals_id_map, reverse_signals_id_map, signals_relations_map = compute_map_elements_mapping(map_path)

    # Build the scenario .json representation 
    json_scenario = {
        # Build the ego-vehicles section
        "ad_section": build_ad_section(
            scenario,
            planning_problems,
            reverse_lane_id_map
        ),
        # We don't generate scenarios with pedestrians, so pedestrians section is always empty
        "pd_section": {"pds": []},
        # Build traffic lights section
        "tc_section": build_tc_section(scenario, signals_id_map, reverse_signals_id_map, signals_relations_map)
    }

    return json_scenario

def extract_routing(scenario: Scenario, planning_problem: PlanningProblemSet, reverse_lane_id_map: Dict):
    """
    Returns list of route lanelet IDs as strings
    
    :param scenario: CommonRoad scenario
    :type scenario: Scenario
    :param planning_problem: CommonRoad scenario's planning problems
    :type planning_problem: PlanningProblemSet
    """

    # Create route planner
    route_lanelet_ids = planning_problem.path

    # From the generated route extract CommonRoad map start and goal lanes
    start_lanelet = scenario.lanelet_network.find_lanelet_by_id(int(route_lanelet_ids[0]))
    goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(int(route_lanelet_ids[-1]))

    # Convert CommonRoad lanelet ids to Apollo map ones
    output_route = []

    for lanelet_id in route_lanelet_ids:
        apollo_lane_id = reverse_lane_id_map[int(lanelet_id)]
        output_route.append(apollo_lane_id)

    return output_route, start_lanelet, goal_lanelet

def extract_st(initial_state: InitialState, start_lanelet: Lanelet) -> Dict[str, float]:
    """
    Method to extract vehicle's initial state time and position
    
    :param initial_state: Vehicle's initial state
    :type initial_state: InitialState
    :param start_lanelet: Vehicle's initial lanelet
    :type start_lanelet: Lanelet
    """

    # Retrieve vehicle's initial position and timestep
    x, y = initial_state.position
    initial_timestep = initial_state.time_step / 10 # Convert from CommonRoad timestep representation (with timestep duration 0.1) to seconds representation by dividing by 10

    # Find closest centerline point to vehicle's initial position
    centerline = start_lanelet.center_vertices
    distances = np.linalg.norm(centerline - np.array([x, y]), axis = 1)
    min_idx = np.argmin(distances)

    # Compute point distance along lanelet centerline
    start_s = start_lanelet.distance[min_idx]

    return {
        "start_s": round(float(start_s), 2),
        "start_t": float(initial_timestep)
    }

def extract_dest_s(goal_position: np.ndarray, goal_lanelet: Lanelet) -> float:
    """
    Method to extract vehicle's goal area position
    
    :param goal_lanelet: Vehicle's goal lanelet
    :type goal_lanelet: Lanelet
    """

    # Retrieve vehicle's goal position and timestep
    x, y = goal_position

    # Find closest centerline point to vehicle's goal position
    centerline = goal_lanelet.center_vertices
    distances = np.linalg.norm(centerline - np.array([x, y]), axis = 1)
    min_idx = np.argmin(distances)

    # Compute point distance along lanelet centerline
    dest_s = goal_lanelet.distance[min_idx]

    return round(float(dest_s), 2)

def generate_traffic_lights(scenario: Scenario, scenario_max_duration: int, scenario_map_path: str) -> List[Scenario]:
    """
    Method to generate traffic lights in the CommonRoad scenario
    
    :param scenarios: CommomRoad scenario
    :type scenarios: Scenario
    :param scenario_max_duration: Scenario max duration defined in DoppelTest configuration file
    :type scenario_max_duration: int
    """

    _, signals_id_map, reverse_signals_id_map, signals_relations_map = compute_map_elements_mapping(scenario_map_path)

    # Generate a random traffic light configuration
    traffic_light_configurations = get_traffic_light_configuration(scenario, signals_id_map, reverse_signals_id_map, signals_relations_map)

    # Generate the traffic lights durations
    traffic_light_durations = {
        "red_wait_duration": 2 * 10,
        "yellow_duration": 3 * 10,
        "green_duration": randint(5, int(scenario_max_duration / 20)) * 10
    }

    # Assign traffic light configuration to each signal in the map
    for signal_id, color in traffic_light_configurations.items():
        signal = scenario.lanelet_network.find_traffic_light_by_id(signal_id)

        signal.traffic_light_cycle = get_traffic_light_cycle(color, traffic_light_durations)
        signal.initial_state = (TrafficLightState.RED if color == "RED" else TrafficLightState.GREEN)
        signal.time_offset = 0

    return scenario

def get_traffic_light_cycle(initial_color: Literal["RED", "GREEN"], traffic_light_durations: Dict[str, int]) -> TrafficLightCycle:
    """
    Method to build the CommonRoad traffic light cycle
    
    :param initial_color: Traffic light initial color (can be either red or green)
    :type initial_color: Literal["RED", "GREEN"]
    :param traffic_light_durations: Traffic light color durations
    :type traffic_light_durations: Dict[str, int]
    """

    red_cycle_element = TrafficLightCycleElement(
            state = TrafficLightState.RED,
            duration = traffic_light_durations["red_wait_duration"] + traffic_light_durations["yellow_duration"] + traffic_light_durations["green_duration"]
        )
    
    yellow_cycle_element = TrafficLightCycleElement(
            state = TrafficLightState.YELLOW,
            duration = traffic_light_durations["yellow_duration"]
        )
    
    green_cycle_element = TrafficLightCycleElement(
            state = TrafficLightState.GREEN,
            duration = traffic_light_durations["green_duration"]
        )
    
    # Build the color sequence according to the initial traffic light color
    if initial_color == "RED":
        new_cycle = [red_cycle_element, green_cycle_element, yellow_cycle_element]
    else:
        new_cycle = [green_cycle_element, yellow_cycle_element, red_cycle_element]

    return TrafficLightCycle(cycle_elements = new_cycle)

def get_traffic_light_configuration(scenario: Scenario, signals_id_map: Dict[str, int], reverse_signals_id_map: Dict[str, str], signals_relations_map: Dict[str, List[Tuple[str, str]]], preferences: List[str] = []) -> Dict[str, str]:
    """
    Method to generate a suitable traffic light colors configuration for the input scenario
    This logic is extracted from DoppelTest
    
    :param scenario: CommonRoad scenario
    :type scenario: Scenario
    :param signals_id_map: Mapping between CommonRoad and Apollo traffic light ids
    :type signals_id_map: Dict[str, int]
    :param reverse_signals_id_map: Mapping between Apollo and CommonRoad traffic light ids
    :type reverse_signals_id_map: Dict[str, str]
    :param signals_relations_map: Mapping between traffic lights and their relational conditions
    :type signals_relations_map: Dict[str, List[Tuple[str, str]]]
    :param preferences: List of traffic lights to be prioritized
    :type preferences: List[str]
    """

    traffic_light_colors = {}  

    # Extract scenario's traffic lights ids and shuffle them
    scenario_traffic_lights_ids = [signal.traffic_light_id for signal in scenario.lanelet_network.traffic_lights]
    shuffle(scenario_traffic_lights_ids)

    # Loop until each traffic light in the scenario has a color
    while len(scenario_traffic_lights_ids) > 0:
        # Assign first the prioritized traffic lights
        if len(preferences) > 0:
            signal_id = preferences.pop()
            scenario_traffic_lights_ids.remove(signal_id)
        else:
            signal_id = scenario_traffic_lights_ids.pop()
    
        # Assign green light color to the traffic light
        traffic_light_colors[signal_id] = "GREEN"
        
        # Extract traffic lights related to the current one
        apollo_signal_id = reverse_signals_id_map[signal_id]
        related_signals = signals_relations_map[apollo_signal_id] # This map contains Apollo traffic lights ids

        # Iterate over each related traffic light with the associated equality condition
        for apollo_signal_id, condition in related_signals:
            # Retrieve CommonRoad traffic light id
            signal_id = signals_id_map[apollo_signal_id]

            if signal_id in preferences:
                preferences.remove(signal_id)

            if signal_id in scenario_traffic_lights_ids:
                scenario_traffic_lights_ids.remove(signal_id)

                # Assign light color according to the relational condition
                if condition == 'EQ':
                    traffic_light_colors[signal_id] = "GREEN"
                else:
                    # NE
                    traffic_light_colors[signal_id] = "RED"

    return traffic_light_colors

def compute_map_elements_mapping(scenario_map_path: str) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str], Dict[str, List[Tuple[str, str]]]]:
    """
    Retrive Apollo map elements id mapping files
    """
    # Extract map name
    scenario_map = scenario_map_path.removesuffix(".xml")

    # Open .json file containing Apollo-CommonRoad traffic lights ids mapping
    map_signal_ids_map_file_path = os.path.join(f"{scenario_map}_signals_ids_map.json")
    with open(map_signal_ids_map_file_path, "r") as f:
        signals_id_map = json.load(f)

    # Swap map keys and values
    reverse_signals_id_map = {v: k for k, v in signals_id_map.items()}

    # Open .json file containing Apollo traffic lights relations
    map_signal_relations_map_file_path = os.path.join(f"{scenario_map}_signals_relations_map.json")
    with open(map_signal_relations_map_file_path, "r") as f:
        signals_relations_map = json.load(f)

    # Open .json file containing Apollo-CommonRoad lanes ids mapping
    map_signal_ids_map_file_path = f"{scenario_map}_lane_ids_map.json"
    with open(map_signal_ids_map_file_path, "r") as f:
        map_lane_id_map = json.load(f)

    # Swap map keys and values
    reverse_lane_id_map = {v: k for k, v in map_lane_id_map.items()}

    return reverse_lane_id_map, signals_id_map, reverse_signals_id_map, signals_relations_map