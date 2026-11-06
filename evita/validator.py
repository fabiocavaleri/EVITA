"""
Scenario validation (paper, Section 4.5): rejects scenarios that cannot be solved or are irrelevant to testing,
before they are executed.
"""
import logging
import traceback
import numpy as np
from copy import deepcopy

from itertools import combinations
from typing import Dict, List, Optional, Tuple
from shapely.geometry import LineString, Point

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet

from .config import ScenarioGeneratorConfig, ValidationConfig
from .timing import TASK_TIMER
from .utils import EVITA_MAIN_LOGGER_NAME

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

# Minimum distance (metres) between the centres of two vehicles in the same lane, whatever their speed
MIN_CENTRE_DISTANCE = 5.0

class ScenarioValidator:
    """
    Validates generated scenarios.
    A scenario is valid if:
      - It has at least min_AVs vehicles;
      - Every vehicle starts on a lanelet, with a non-negative velocity not above max_velocity, at least
        min_dist_init metres into an entry lanelet, and is outside its goal area;
      - Every goal area is reachable along a route and at least min_dist_goal metres ahead along it (FrenetiX only);
      - Vehicles in the same lane keep a safety distance;
      - Every goal area lies on an outgoing lanelet (one without successors).
    The result is stored in the scenario as attributes `is_valid` and `validation_message`.
    """   
    def __init__(self, configuration: ValidationConfig, generation_config: Optional[ScenarioGeneratorConfig] = None, apollo: bool = False) -> None:
        self.config = configuration

        if generation_config:
            self.generation_config = generation_config
        else:
            self.generation_config = ScenarioGeneratorConfig()

        # Apollo plans its own routes: the route check is skipped for Apollo scenarios
        self.apollo = apollo

        # Merged successor paths of each lanelet, per road network: (scenario id, lanelet id) -> lanelets
        self._successor_paths_cache: Dict[Tuple[str, int], List[Lanelet]] = {}

    def validate(self, scenario: Scenario, planning_problem_set: PlanningProblemSet) -> None:
        """
        Validate the scenario and store the result in its `is_valid` and `validation_message` attributes.
        A scenario whose validation fails with an error is invalid.
        """
        try:
            validity_flag, validity_message = self.is_scenario_valid(scenario, planning_problem_set)
        except Exception as e:
            module_logger.critical(f"Cannot complete the validation of scenario {getattr(scenario, 'generation_id', '?')}. Force it to be invalid")
            module_logger.critical(traceback.format_exc())
            validity_flag, validity_message = False, f"Fail to validate {e}"

        setattr(scenario, "is_valid", validity_flag)
        setattr(scenario, "validation_message", validity_message)

    def validate_all(self, scenario_tuples: List[Tuple[Scenario, PlanningProblemSet]]) -> None:
        """
        Validate all the scenarios in the list of scenario tuples (scenario, planning_problem_set)
        """
        for scenario, planning_problem_set in scenario_tuples:
            # Check if scenario has already been validated, if so skip it
            if hasattr(scenario, "is_valid"):
                module_logger.debug(f"Scenario {scenario.generation_id} already validated. Is valid? {scenario.is_valid}")
                continue

            # Validate scenario
            self.validate(scenario, planning_problem_set)

    def is_scenario_valid(self, scenario: Scenario, problem_set: PlanningProblemSet) -> Tuple[bool, str]:
        """
        (validity, reason) of the scenario; the reason is empty for valid scenarios. Timed as task "validation".
        """
        with TASK_TIMER.measure("validation"):
            return self._is_scenario_valid(scenario, problem_set)

    def _is_scenario_valid(self, scenario: Scenario, problem_set: PlanningProblemSet) -> Tuple[bool, str]:
        """
        (validity, reason) of the scenario; the reason is empty for valid scenarios.
        """
        # Ignore validation if requested
        if self.config.ignore_validation:
            module_logger.critical("Ignoring validation!")
            return True, "Disabled"
        
        # Validate lanelet network
        is_lanelet_network_valid, lanelets_message = self.validate_lanelets(scenario)
        if not is_lanelet_network_valid:
            return False, lanelets_message
        
        # Validate planning problems
        is_planning_problem_valid, planning_problem_message = self.validate_planning_problems(scenario, problem_set)

        return is_planning_problem_valid, planning_problem_message

    # ========== Road network ==========

    def validate_lanelets(self, scenario: Scenario) -> Tuple[bool, str]:
        """
        Check the type and direction of every lanelet, if enabled in the configuration.
        """
        if not self.config.validate_lanelet_type_and_direction:
            return True, ""

        for lanelet in scenario.lanelet_network.lanelets:
            validity_flag, validity_message = self._validate_lanelet_basics(lanelet)
            if not validity_flag:
                return validity_flag, validity_message

        return True, ""

    def _validate_lanelet_basics(self, lanelet: Lanelet) -> Tuple[bool, str]:
        """
        The lanelet has one of the allowed types and, if required, the same direction as its adjacent lanelets.
        """
        # Check if lanelet has correct type
        if not lanelet.lanelet_type.intersection(self.config.lanelet_types):
            module_logger.debug("Scenario not valid: Wrong Lanelet Types")
            return False, "Wrong Lanelet Types"
        
        # Check if lanelet has correct direction with respect to adjacent lanelets
        if self.config.lanelet_same_direction_only and (lanelet.adj_right_same_direction is False or lanelet.adj_left_same_direction is False):
            module_logger.debug("Scenario not valid: Lanelets not all in same direction")
            return False, "Lanelets not all in same direction"

        return True, ""
    
    # ========== Planning problems ==========
    
    def validate_planning_problems(self, scenario: Scenario, problem_set: PlanningProblemSet) -> Tuple[bool, str]:
        """
        Check the number of vehicles, each vehicle's start and goal, the distances between vehicles and the placement
        of the goal areas; stop at the first failed check.
        """

        planning_problems = list(problem_set.planning_problem_dict.values())
        
        # If there are not enough planning problems, the scenario is not valid
        if len(planning_problems) < self.generation_config.min_amount_pp:
            return False, f"The scenario must have at least {self.generation_config.min_amount_pp} AVs"

        # Validate initial state of each planning problem and check that it is not in the goal region
        for planning_problem in planning_problems:
            validity_flag, validity_message = self._validate_initial_state(scenario, planning_problem)
            # Exit if any initial state is not valid
            if not validity_flag:
                return validity_flag, validity_message

            # Check if the goal area is reachable from the initial state of each planning problem, only if the scenario is not an Apollo scenario            
            if not self.apollo:
                # Validate if goal area is reachable
                validity_flag, validity_message = self._validate_reacheable_goal(scenario, planning_problem)

                # Exit if any initial state is not valid
                if not validity_flag:
                    return validity_flag, validity_message
        
        # Validate initial distances between agents and obstacles on the same lanelet
        validity_flag, validity_message = self._validate_init_distances(scenario, planning_problems)
        # Exit if any initial state is not valid
        if not validity_flag:
            return validity_flag, validity_message
        
        # Validate goal areas positionings
        validity_flag, validity_message = self._validate_goal_areas_placement(scenario, planning_problems)

        return validity_flag, validity_message

    def _validate_initial_state(self, scenario: Scenario, planning_problem: PlanningProblem) -> Tuple[bool, str]:
        """
        The vehicle starts with a velocity in [min_velocity, max_velocity], on a lanelet, at least min_dist_init
        metres into any entry lanelet it starts on, and outside its goal area.
        """
        pp_id = planning_problem.planning_problem_id
        initial_state = planning_problem.initial_state

        if planning_problem.initial_state.velocity < 0.0:
            module_logger.debug("Scenario not valid: Initial velocity cannot be negative!")
            return False, "Initial velocity cannot be negative!"

        if initial_state.velocity > self.generation_config.max_velocity:
            module_logger.debug(f"Scenario not valid: Initial velocity of {pp_id} above {self.generation_config.max_velocity} m/s")
            return False, f"Initial velocity of {pp_id} above {self.generation_config.max_velocity} m/s"

        initial_position = np.asarray(initial_state.position)
        initial_lanelet_ids = scenario.lanelet_network.find_lanelet_by_position([initial_position])[0]
        if not initial_lanelet_ids:
            module_logger.debug(f"Scenario not valid: Initial State of {pp_id} outside lanelets")
            return False, f"Initial State of {pp_id} outside lanelets"
    
        # On an entry lanelet (no predecessor), the start must be at least min_dist_init metres from its beginning.
        # Measured along the centre line, so that the whole vehicle is on the road
        for lanelet_id in initial_lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

            if lanelet.predecessor:
                # Skip checking if the lanelet has a predecessor -> Is not an ENTRY lanelets
                continue

            # Distance of the start from the beginning of the lanelet, measured along its centre line
            center_line = LineString(lanelet.center_vertices)
            distance_from_lanelet_start = center_line.project(Point(initial_position))

            if distance_from_lanelet_start < self.config.min_dist_init:
                return False, f"Initial State of {planning_problem.planning_problem_id} too close to lanelet {lanelet.lanelet_id} initial edge"

        # Check if initial state is not in goal region
        for goal_state in planning_problem.goal.state_list:
            # If the goal region is an area, check that the initial state is not inside of it
            if hasattr(goal_state, "position") and goal_state.position.contains_point(initial_position):
                module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} inside of Goal Region")
                return False, f"Initial State of {planning_problem.planning_problem_id} inside of Goal Region"

        # If nothing fails, we assume valid. In case this turns out to be wrong, we catch the issue after the execution
        return True, ""
    
    def _validate_reacheable_goal(self, scenario: Scenario, planning_problem: PlanningProblem) -> Tuple[bool, str]:
        """
        A route exists from the start to the goal (planned with FrenetiX's route planner), and every goal area is at
        least min_dist_goal metres ahead of the start along that route.
        """
        agent_id = planning_problem.planning_problem_id

        # 1. Route, as planned by Frenetix
        try:
            route_planner = RoutePlanner(scenario = scenario, planning_problem = planning_problem, extended_search = False, logging_level = logging.CRITICAL)
            route = route_planner.plan_routes().retrieve_shortest_route(retrieve_shortest = True)
        except Exception as e:
            module_logger.debug(f"No route for agent {agent_id}: {e}")
            return False, f"No route from the initial state to the goal of agent {agent_id}"

        reference_path = LineString(route.reference_path)
        s_start = reference_path.project(Point(planning_problem.initial_state.position))

        # 2. Each goal area ahead, along the route, by at least min_dist_goal
        for goal_state in planning_problem.goal.state_list:
            if not hasattr(goal_state, "position"):
                continue

            driving_distance = reference_path.project(Point(goal_state.position.center)) - s_start
            
            if driving_distance < self.config.min_dist_goal:
                return False, f"Goal area of agent {agent_id} is {driving_distance:.1f} m ahead along its route (min {self.config.min_dist_goal})"

        return True, ""
       
    def _validate_init_distances(self, scenario: Scenario, planning_problems: List[PlanningProblem]) -> Tuple[bool, str]:
        """
        Vehicles (and obstacles) starting in the same lane keep the safety distance.
        Lanes are followed along their successors, so vehicles on consecutive lanelets of the same lane are compared too.
        """
        # Extract scenario's lanelets, obstacles and planning problems
        lanelet_network: LaneletNetwork = scenario.lanelet_network
        obstacle_states: Dict[int, State] = scenario.obstacle_states_at_time_step(0)

        for lanelet in lanelet_network.lanelets:
            # This one MERGEs the lanelets into a big one
            full_lanelets = self._successor_paths(scenario, lanelet)

            for full_ll in full_lanelets:        
                polygon = full_ll.polygon

                # Extract planning problems and obstacles that are on this lanelet (or its successors)
                objects_on_lanelet = {pp.planning_problem_id: pp.initial_state for pp in planning_problems if polygon.contains_point(pp.initial_state.position)}
                objects_on_lanelet.update({obstacle_id: state for obstacle_id, state in obstacle_states.items() if polygon.contains_point(state.position)})

                # Check that the distance between all the agents and obstacles on this lanelet is larger than a minimum distance defined in the configuration file
                validity_flag, validity_message = self._is_distance_in_lane_kept(full_ll, objects_on_lanelet)
                if not validity_flag:
                    return False, validity_message
                
        return True, ""

    def _validate_goal_areas_placement(self, scenario: Scenario, planning_problems: List[PlanningProblem]) -> Tuple[bool, str]:
        """
        Every goal area lies on an outgoing lanelet (one without successors), not in the middle of the map.
        """
        lanelet_network = scenario.lanelet_network

        for pp in planning_problems:
            goal_state = pp.goal.state_list[0]
            # Goals may be defined by time only, without an area
            if not hasattr(goal_state, "position"):
                continue

            goal_lanelet_ids = lanelet_network.find_lanelet_by_position([goal_state.position.center])[0]
            if not any(len(lanelet_network.find_lanelet_by_id(lanelet_id).successor) == 0 for lanelet_id in goal_lanelet_ids):
                return False, f"Goal area of {pp.planning_problem_id} is not on an outgoing lanelet"

        return True, ""

    # ========== Distances in a lane ==========

    @staticmethod
    def _lane_coordinates(line: LineString, position) -> Tuple[float, float]:
        """
        Position of a point relative to a lane line:
        s = distance along the line to the closest point on it
        d = signed distance from that point (positive to the left of the line's direction)
        """
        point = Point(position[0], position[1])
        s = line.project(point)
        foot = line.interpolate(s)

        # Direction of the line around the closest point, to decide the side
        ahead = line.interpolate(min(s + 0.5, line.length))
        behind = line.interpolate(max(s - 0.5, 0.0))
        cross = (ahead.x - behind.x) * (point.y - foot.y) - (ahead.y - behind.y) * (point.x - foot.x)

        d = point.distance(foot)

        if cross >= 0:
            return s, d
        else:
            return s, -d
    
    def _is_distance_in_lane_kept(self, lanelet: Lanelet, objects: Dict[int, State]) -> Tuple[bool, str]:
        """
        Every pair of objects in the lane is far enough apart: the distance between their centres, measured in the lane's coordinates,
        is at least the safety distance of the faster of the two plus one vehicle length, and never below MIN_CENTRE_DISTANCE.
        """
        if len(objects) < 2:
            return True, ""
        
        # Lane line
        center_line = LineString(lanelet.center_vertices)

        # Position of every object along the lane and to its side, computed once
        lane_positions = {key: self._lane_coordinates(center_line, state.position) for key, state in objects.items()}

        # Check every pair once
        for (key, state), (other_key, other_state) in combinations(objects.items(), 2):
            # Extract the s and d coordinates of the two objects along the lane
            (s_1, d_1), (s_2, d_2) = lane_positions[key], lane_positions[other_key]
            # Compute the actual distance between the two objects in the lane's coordinate system
            actual_distance = ((s_1 - s_2) ** 2 + (d_1 - d_2) ** 2) ** 0.5

            # Compute the minimum safety distance between the two objects, based on their velocities and the configured safety time
            car_distance = max(self.config.car_distance_formula(state.velocity), self.config.car_distance_formula(other_state.velocity))
            min_safety_distance = max(car_distance + self.generation_config.length, MIN_CENTRE_DISTANCE)

            if actual_distance < min_safety_distance:
                module_logger.debug(f"Scenario not valid: Distance between {key} and {other_key} is too short {actual_distance:.1f} (min {min_safety_distance:.1f})")
                return False, f"Distance between {key} to next object in lane {other_key} too short"

        return True, ""

    # ========== Lanes as successor paths ==========

    def _successor_paths(self, scenario: Scenario, lanelet: Lanelet) -> List[Lanelet]:
        """
        The lanelet itself, and the lanelets obtained by merging it with each chain of successors up to an outgoing lanelet.
        The road network does not change within a run, so the result is computed once per lanelet.
        """
        key = (str(scenario.scenario_id), lanelet.lanelet_id)

        if key not in self._successor_paths_cache:
            self._successor_paths_cache[key] = self.get_successor_lanelets(lanelet, scenario.lanelet_network)
        
        return self._successor_paths_cache[key]

    @staticmethod
    def get_successor_lanelets(lanelet: Lanelet, ll_network: LaneletNetwork) -> List[Lanelet]:
        """
        The lanelet itself, plus one merged lanelet for each outgoing lanelet reachable from it through successors:
        the lanelets on the (breadth-first) path from `lanelet` to that outgoing lanelet, merged into one.
        """
        path_to = {lanelet.lanelet_id: [lanelet]}
        queue = [lanelet]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current.lanelet_id in visited:
                continue
            visited.add(current.lanelet_id)

            for successor_id in current.successor:
                successor = ll_network.find_lanelet_by_id(successor_id)
                path_to[successor_id] = path_to[current.lanelet_id] + [successor]
                queue.append(successor)

        full_lanelets = [lanelet]
        for lanelet_id, path in path_to.items():
            if len(ll_network.find_lanelet_by_id(lanelet_id).successor) > 0:
                continue

            full_lanelet = deepcopy(path[0])
            
            for next_lanelet in path[1:]:
                full_lanelet = Lanelet.merge_lanelets(full_lanelet, next_lanelet)
            
            full_lanelets.append(full_lanelet)

        return full_lanelets