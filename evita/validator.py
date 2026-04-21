import numpy as np
import traceback

from numpy.core.numeric import full

from typing import Dict, List, Tuple
from itertools import combinations

from commonroad.geometry.shape import Polygon, Rectangle, ShapeGroup
from commonroad.scenario.trajectory import State
from commonroad.scenario.state import InitialState
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
from commonroad.scenario.scenario import Scenario
from commonroad_route_planner.route_planner import RoutePlanner

from cr_scenario_handler.utils.utils_coordinate_system import smooth_ref_path, CoordinateSystem, extend_ref_path_both_ends, extend_path_linearly

from shapely.ops import nearest_points, split, snap, unary_union
from shapely.geometry import Point, LineString, Polygon

from shapely.affinity import scale
from commonroad_dc.geometry.util import resample_polyline

import matplotlib.pyplot as plt


from .config import ValidationConfig, ScenarioGeneratorConfig


NODE_DISTANCE_TOLERANCE = 0.01  # this is in meters
ADJACENT_WAY_DISTANCE_TOLERANCE = 0.05
DISTANCE = 50

# Get the main logger
import logging
from .utils import EVITA_MAIN_LOGGER_NAME

main_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

# Note: Validator cannot check whether the min/max number of AV is correct, as that is under control of hte scenario generator.
# TODO COnsider mering configurations together?
class ScenarioValidator:
    
    def __init__(self, configuration: ValidationConfig, generation_config: ScenarioGeneratorConfig = None):
        self.config = configuration
        self.generation_config = generation_config if generation_config else ScenarioGeneratorConfig()


    def validate(self, scenario: Scenario, planning_problem_set: PlanningProblemSet) -> None:
        # Store the attribute inside the scenario object
        # TODO Refactor the "Scenario" inside a proper object, not tuple!
        try:
            validity_flag, validity_message = self.is_scenario_valid(scenario, planning_problem_set)
            setattr(scenario, "is_valid", validity_flag)
            setattr(scenario, "validation_message", validity_message)
            # module_logger.info(f"Validating scenario {scenario.generation_id} - Valid? {validity_flag} " + "" if validity_flag else validity_message)
        except Exception as e:
            module_logger.critical(f"Cannot complete the validation of scenario {scenario.generation_id}. Force it to be invalid")
            module_logger.critical(traceback.format_exc())
            setattr(scenario, "is_valid", False)
            setattr(scenario, "validation_message", f"Fail to validate {e}")


    def validate_all(self, scenario_tuples: List[Tuple[Scenario, PlanningProblemSet]]) -> None:
        for scenario, planning_problem_set in scenario_tuples:
            if hasattr(scenario, "is_valid"):
                # This one has already validated!
                module_logger.debug(f"Scenario {scenario.generation_id} already validated. Is valid? {scenario.is_valid}")
                continue
            else:
                # This needs validation
                self.validate(scenario, planning_problem_set)

    def is_scenario_valid(self, scenario: Scenario, problem_set: PlanningProblemSet) -> Tuple[bool, str]:
        # TODO validation is not very robust as later during development many validation criteria were
        #  implemented inside the mutation operators
        if self.config.ignore_validation:
            module_logger.critical("Ignoring validation!!")
            return True, "Disabled"
        
        messages = []
        # Ideally here we should use a Monoid
        is_lanelet_network_valid, lanelets_message = self.validate_lanelets(scenario)
        messages.append(lanelets_message)
        
        # is_planning_problem_valid = True
        is_planning_problem_valid, planning_problem_message = self.validate_planning_problems(scenario, problem_set)
        messages.append(planning_problem_message)

        return is_lanelet_network_valid and is_planning_problem_valid, ",".join(messages)

    def validate_lanelets(self, scenario: Scenario):
        lanelet_network: LaneletNetwork = scenario.lanelet_network
        lanelets: Dict[int, Lanelet] = lanelet_network.lanelets
        for lanelet in lanelets:
            validity_flag, validity_message = self._validate_lanlet_basics(lanelet)
            
            if not validity_flag:
                return validity_flag, validity_message
             
            # those next variables only contain the lanelet IDs
            # predecessors: list = lanelet.predecessor
            # successors: list = lanelet.successor
            left_lanelet = lanelet.adj_left
            right_lanelet = lanelet.adj_right

            validity_flag, validity_message = self._validate_lanelets_orientation(lanelet, left_lanelet, right_lanelet)
            if not validity_flag:
                # module_logger.debug("Scenario not valid: Bad Lanelet Orientation")
                return False, validity_message #"Bad Lanelet Orientation"
        return True, ""

    def validate_planning_problems(self, scenario: Scenario, problem_set: PlanningProblemSet, evita: bool = False) -> Tuple[bool, str]:
        
        lanelet_network: LaneletNetwork = scenario.lanelet_network
        planning_problems = list(problem_set.planning_problem_dict.values())
        
        if len(planning_problems) == 0:
            validity_flag = False
            validity_message = "The scenario must have at least one AV"
            return validity_flag, validity_message

        # TODO Move to a method
        for key, problem in problem_set.planning_problem_dict.items():
            validity_flag, validity_message = self._validate_initial_state(scenario, problem)
            if not validity_flag:
                return validity_flag, validity_message
            
        if not evita:
            # This seems overkilling
            validity_flag, validity_message = self._validate_reacheable_goal(scenario, problem)
            if not validity_flag:
                return validity_flag, validity_message
        
        validity_flag, validity_message = self._validate_init_distances(scenario, problem_set)
        if not validity_flag:
            return validity_flag, validity_message
        
        validity_flag, validity_message = self._validate_goal_areas_placement(scenario, problem_set)
        if not validity_flag:
            return validity_flag, validity_message

        # This measures the distance to the goal area with a straight line
        validity_flag, validity_message = self._validate_distance_to_goal(scenario, problem_set)
        if not validity_flag:
            return validity_flag, validity_message
        
        return True, ""

    from typing import Set

    def DFS(self, scenario: Scenario, lanelet: Lanelet):
        setattr(lanelet, "visited", True)
        # Lanelets reacheable from u at one step
        reach = lanelet.successor[:]
        if lanelet.adj_left_same_direction: reach.append(lanelet.adj_left)
        if lanelet.adj_right_same_direction: reach.append(lanelet.adj_right)
        reacheable_at_one_step = [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in reach]

        for v in reacheable_at_one_step:
            if not v.visited:
                self.DFS(scenario,v)

    def _compute_reacheable_lanelets(self, scenario: Scenario, lanelet: Lanelet) -> Set[Lanelet]:
        # Use a depth first algorithm
        all_lanelets = scenario.lanelet_network.lanelets
        for l in all_lanelets:
            setattr(l, "visited", False)
        # Mark all the one reacheable
        self.DFS(scenario, lanelet)
        # Filter those that were reached
        reacheable_from_lanelet = [l for l in all_lanelets if l.visited]
        # Reset the attributes
        [delattr(l, "visited") for l in all_lanelets]
        return reacheable_from_lanelet


    def _validate_reacheable_goal(self, scenario: Scenario, planning_problem: PlanningProblem):

        initial_lanelets = [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in self._flatten(scenario.lanelet_network.find_lanelet_by_position([planning_problem.initial_state.position]))]
        # Compute reacheability using lanelets first and
        # check if initial state is not in goal region
        for goal_state in planning_problem.goal.state_list:
            # If the lateral distance is larger than twice the longitudinal distance the vechile is too much on the side of the goal area!
            center_of_goal = goal_state.position.center
            goal_lanelets = [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in self._flatten(scenario.lanelet_network.find_lanelet_by_position([center_of_goal]))]
            # Compute lanelets reacheable from initial lanelets

            for initial_lanelet in initial_lanelets:
                if any([goal_lanelet in self._compute_reacheable_lanelets(scenario, initial_lanelet) for goal_lanelet in goal_lanelets]):
                    return True, ""
        
        return False, f"There's no drivable path for agent {planning_problem.planning_problem_id}"

    def _validate_reacheable_goal_full(self, scenario: Scenario, planning_problem: PlanningProblem):
        """
        Check whether the agent can find a route to reach its goal area. This ensures also that the planner is NOT "after" the goal area
        (see tests/data/out-of-domain-projection.xml)
        
        This is expensive and generally overkilling. A better approach would be to parse the map and compute all the possibilities once and for all at the beginning since the map will not change (at least for the moment)
        """
        route_planner = RoutePlanner(scenario=scenario, planning_problem=planning_problem, extended_search=True, logging_level=logging.CRITICAL)

        # Computing reacheability takes a long time when agents need to cross multiple nearby lanes.
        # Here what we need to ensure is that the initial position is NOT after the goal area
        initial_position = planning_problem.initial_state.position

        # Compute reacheability using lanelets first and
        # check if initial state is not in goal region
        for goal_state in planning_problem.goal.state_list:
            
            # If the lateral distance is larger than twice the longitudinal distance the vechile is too much on the side of the goal area!
            center_of_goal = goal_state.position.center

            # If the agent is close enough to the goal area we need to check for lateral displacement
            distance_to_goal = Point(initial_position[0], initial_position[1]).distance(Point(center_of_goal[0], center_of_goal[1]))
            
            if distance_to_goal > DISTANCE:
                # Compute reacheability using lanelets
                route = route_planner.plan_routes().retrieve_first_route(retrieve_shortest=False)
                if route:
                    # module_logger.critical(f"Far away. F?ZZZ {route}. Scenario has valid reacheable goal")
                    return True, ""
                else:
                    module_logger.critical(f"Scenario Validator: Cannot find route for {planning_problem.planning_problem_id}")
                    return False, f"Scenario Validator: Cannot find route for {planning_problem.planning_problem_id}"
                    
            else:
                # Here the initial position and goal area are very close, computing the ref path is expensive (interpolation, weird angles)
                # So we check only that the initial state is "before" the goal area using their longitudinal distances 
                goal_lanelets = [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in self._flatten(scenario.lanelet_network.find_lanelet_by_position([center_of_goal]))]
                
                # This is an error
                if len(goal_lanelets) == 0:
                    module_logger.critical(f"Scenario Validator: Cannot find route for {planning_problem.planning_problem_id}")
                    return False, f"Goal area not reacheable by agent {planning_problem.planning_problem_id}"
            
                for goal_lanelet in goal_lanelets:
                    center_lane = goal_lanelet.center_vertices
                    # This seems to be broken. After extension, the new path is disjoint from the original one...
                    # extended_path = extend_path_linearly(at_start=True, extension_length=100)
                    # extended_path = extend_ref_path_both_ends(center_lane, extension_length=200)
                    extended_path = np.array(scale(LineString(center_lane), 3.0, 3.0).coords)
                
                    # THIS IS THE KEY TO AVOID OUT OF PROJECTION DOMAIN... But it works only f
                    # path = smooth_ref_path(extended_path)
                    path = resample_polyline(extended_path, 1)

                    try:
                        cosy = CoordinateSystem(path)
                        initial_position_cc = cosy.convert_to_curvilinear_coords(initial_position[0], initial_position[1])
                        goal_area_cc = cosy.convert_to_curvilinear_coords(center_of_goal[0], center_of_goal[1])
                        longitudinal_distance = goal_area_cc[0] - initial_position_cc[0]

                        if longitudinal_distance < 0.0:
                            # This means that goal is before, not good
                            module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} is beyond Goal Region")
                            return False, f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} is beyond Goal Region"
                    except Exception as e:
                        def debug_plot():
                            from commonroad.visualization.mp_renderer import MPRenderer
                            rnd = MPRenderer()
                            scenario.lanelet_network.draw(rnd)
                            rnd.render()
                            
                            plt.plot([p[0] for p in center_lane], [p[1] for p in center_lane], "-", color="blue", lw=6, zorder=100)
                            plt.plot([p[0] for p in extended_path], [p[1] for p in extended_path], "-", color="red", lw=4, zorder=100)
                            plt.plot([p[0] for p in path], [p[1] for p in path], "-", color="green", lw=2, zorder=100)
                            plt.plot(initial_position[0], initial_position[1], "o", lw=2, zorder=100)
                            plt.plot(center_of_goal[0], center_of_goal[1], "+", lw=2, zorder=100)
                            plt.show()
                        # We cannot project... assume FALSE
                        module_logger.critical(traceback.format_exc())
                        
                        # Use this only during DEBUGGING
                        # debug_plot()
                        
                        module_logger.debug(f"Scenario not valid: Cannot establish reacheability for {planning_problem.planning_problem_id}")
                        return False, f"Scenario not valid: Cannot establish reacheability for {planning_problem.planning_problem_id}"

                # At this point, we checked all the cases and none was bad
                return True, ""
            
        return False, "The scenario has no goal states?"

    def _flatten(self, xs): 
        # Initialize list for this layer
        # See: https://stackoverflow.com/questions/74696830/how-do-i-flatten-a-list-that-contains-nested-lists-of-varying-nestedness-level
        #  
        flat_list = []
        for x in xs: 
            # If it's a list, recurse down and return the interior list
            if isinstance(x, list): 
                flat_list += self._flatten(x)
            # Otherwise, add to this layer's list
            else: 
                flat_list.append(x) 
        return flat_list
    
    def _validate_initial_state(self, scenario: Scenario, planning_problem: PlanningProblem):
        if planning_problem.initial_state.velocity < 0.0:
            module_logger.debug("Scenario not valid: Initial velocity cannot be negative!")
            return False, "Initial velocity cannot be negative!"
        
        # The car must be entirely inside the lanelet
        initial_lanelets = scenario.lanelet_network.find_lanelet_by_position([np.array(planning_problem.initial_state.position)])
        
        if all([len(l) == 0  for l in initial_lanelets]):
            module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} outside lanelets")

            def debug_plot():
                from commonroad.visualization.mp_renderer import MPRenderer
                rnd = MPRenderer()
                scenario.lanelet_network.draw(rnd)
                planning_problem.draw(rnd)
                rnd.render()
                plt.show()

            # debug_plot()
            
            return False, f"Initial State of {planning_problem.planning_problem_id} outside lanelets"
        # Check the agent is at least 5 meters inside any of the initial lanelets, only if those are ENTRY lanelets
        for lanelet_id in  [item for sublist in initial_lanelets for item in sublist]:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

            if lanelet.predecessor:
                # Skip checking if the lanelet has a predecessor, i.e., is not an ENTRY lanelets
                continue

            reference_path = resample_polyline(lanelet.center_vertices, step=1.0)
            cosy = CoordinateSystem(reference_path)

            initial_position_cc = cosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])
            if initial_position_cc[0] < self.config.min_dist_init:
                return False, f"Initial State of {planning_problem.planning_problem_id} too close to lanelet {lanelet.lanelet_id} initial edge"

        goal_region = planning_problem.goal
        # check if initial state is not in goal region
        for goal_state in goal_region.state_list:
            # when no position provided the goal most likely is to not crash until a certain time step
            if hasattr(goal_state, "position") and goal_state.position.contains_point(planning_problem.initial_state.position):
                module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} inside of Goal Region")
                return False, f"Initial State of {planning_problem.planning_problem_id} inside of Goal Region"

            # If the lateral distance is larger than twice the longitudinal distance the vehicle is too much on the side of the goal area!
            center_of_goal = goal_state.position.center

            # If the agent is close enough to the goal area we need to check for lateral displacement
            distance_to_goal = Point(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1]).distance(Point(center_of_goal[0], center_of_goal[1]))
            if distance_to_goal > DISTANCE:
                continue
            
            goal_lanelets = [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in self._flatten(scenario.lanelet_network.find_lanelet_by_position([center_of_goal]))]
            for goal_lanelet in goal_lanelets:
                center_lane = goal_lanelet.center_vertices
                extended_path = np.array(scale(LineString(center_lane), 3.0, 3.0).coords)
                path = resample_polyline(extended_path, 1)

                try:
                    cosy = CoordinateSystem(path)
                                
                    initial_position_cc = cosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])
                    goal_area_cc = cosy.convert_to_curvilinear_coords(center_of_goal[0], center_of_goal[1])
                    
                    longitudinal_distance = goal_area_cc[0] - initial_position_cc[0]
                    abs_longitudinal_distance = abs(longitudinal_distance)
                    abs_lateral_distance = abs(initial_position_cc[1] - goal_area_cc[1])

                    if longitudinal_distance < 0.0:
                        # This means that goal is before, not good
                        module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} is beyond Goal Region")
                        return False, f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} is beyond Goal Region"


                    if abs_longitudinal_distance <= 2 * abs_lateral_distance:
                        module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} too close and too on the side wrt Goal Region")
                        return False, f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} too close and too on the side wrt Goal Region"
                except (ValueError, Exception) as e:
                    def debug_plot():
                        from commonroad.visualization.mp_renderer import MPRenderer
                        rnd = MPRenderer()
                        scenario.lanelet_network.draw(rnd)
                        rnd.render()
                        # center_lane
                        plt.plot([p[0] for p in center_lane], [p[1] for p in center_lane], "-", lw=6, color="blue", zorder=100)
                        plt.plot([p[0] for p in path], [p[1] for p in path], "-", lw=4, color="red", zorder=100)
                        plt.plot([p[0] for p in path], [p[1] for p in path], "-", lw=2, color="green", zorder=100)
                        plt.plot(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1], "o", lw=2, zorder=100)
                        plt.plot(center_of_goal[0], center_of_goal[1], "+", lw=2, zorder=100)
                        plt.show()

                    module_logger.critical(f"Cannot establish validity of initial state for {planning_problem.planning_problem_id}. Assume valid")
                    module_logger.critical(traceback.format_exc())
                    # Use this only during debugging
                    # debug_plot()
            
        # If nothing fails, we assume valid. In case this turns out to be wrong, we catch the issue after the execution
        return True, ""
    
    def _validate_init_distances(self, scenario: Scenario, problem_set: PlanningProblemSet):
        lanelet_network: LaneletNetwork = scenario.lanelet_network
        lanelets: Dict[int, Lanelet] = lanelet_network.lanelets
        obstacle_states: Dict[int, State] = scenario.obstacle_states_at_time_step(0)
        planning_problems = list(problem_set.planning_problem_dict.values())
        for lanelet in lanelets:
            # This one MERGEs the lanelets into a big one, but cannot deal correctly with adjacent ones...
            full_lanelets = get_successor_lanelets(lanelet, lanelet_network)
            for full_ll in full_lanelets:
                
                # collect planning problems on this lanelet
                pp_on_lanelet = dict()
                for pp in planning_problems:
                    if full_ll.polygon.contains_point(pp.initial_state.position):
                        pp_on_lanelet[pp.planning_problem_id] = pp.initial_state
                
                # collect obstacles on this lanelet
                obstacles_on_lanelet = dict()
                for obstacle_key, state in obstacle_states.items():
                    if full_ll.polygon.contains_point(state.position):
                        obstacles_on_lanelet[obstacle_key] = state

                validity_flag, validity_message = self._is_distance_in_lane_kept(scenario, full_ll, obstacles_on_lanelet, pp_on_lanelet)
                if not validity_flag:
                    return False, validity_message
        return True, ""

    def _validate_goal_areas_placement(self, scenario: Scenario, problem_set: PlanningProblemSet):
        """
        Check that the goal area is placed ONLY at exiting lanelet, i.e., there cannot be goal areas in the middle of the map
        """
        planning_problems = list(problem_set.planning_problem_dict.values())
        for pp in planning_problems:
            goal = pp.goal
            # skip planning problems that don't have an area goal
            # it is possible that goals only have a required time_step
            if not hasattr(goal.state_list[0], 'position'):
                continue
            goal_state = goal.state_list[0]
            goal_area = goal_state.position
            goal_area_placement = goal_area.center
            
            # Check if at least one of the lanelets corresponding to the goal is a TERMINAL/OUTGOING one, i.e., without any successor lanelet
            if not any([len(scenario.lanelet_network.find_lanelet_by_id(l_id).successor) == 0 for l_id in self._flatten(scenario.lanelet_network.find_lanelet_by_position([goal_area_placement]))]):
                return False, f"Goal area of {pp.planning_problem_id} is not on an outgoing lanelet"

        return True, ""

    def _validate_distance_to_goal(
            self, scenario: Scenario, problem_set: PlanningProblemSet):
        """
        Check the agent is x meters BEFORE the goal area
        """
        planning_problems = list(problem_set.planning_problem_dict.values())
        for pp in planning_problems:
            init_position = pp.initial_state.position
            goal = pp.goal
            # distance to goal is always valid when the goal area is a full lanelet
            if goal.lanelets_of_goal_position is not None and len(goal.lanelets_of_goal_position.values()) > 0:
                return True, ""
            # skip planning problems that don't have an area goal
            # it is possible that goals only have a required time_step
            if not hasattr(goal.state_list[0], 'position'):
                continue
            goal_state = goal.state_list[0]
            goal_area = goal_state.position
            line = LineString([goal_area.center, init_position])
            distance = line.length
            if distance < self.config.min_dist_goal:
                module_logger.debug(f"Planning Problem {pp.planning_problem_id} not valid: Minimum distance to Goal Region not fulfilled. Actual distance {distance}, minimum distance {self.config.min_dist_goal}")
                return False, f"Minimum distance to Goal Region not fulfilled. Actual distance {distance}, minimum distance {self.config.min_dist_goal}"
        return True, ""

    # TODO should this only be checked for planning problems or also dynamic obstacles?
    def _is_distance_in_lane_kept(self, scenario: Scenario, lanelet: Lanelet, obstacles: Dict[int, State],
                                  planning_problems: Dict[int, State]):
        # check if lanelet is empty
        if not obstacles and not planning_problems:
            return True, ""

        # all_obstacles = {**obstacles, **planning_problems}
        all_obstacles = {**planning_problems}
        if len(all_obstacles.values()) < 2:
            return True, ""
        
        # Find the position in the lanelet coordinate system of the agent
        # This looks weird!
        path = lanelet.center_vertices
        extended_path = resample_polyline(path, step=1.0)

        for key, obstacle in all_obstacles.items():

            
            try:
                # This one fails if extended path is not smooth
                cosy = CoordinateSystem(extended_path)

                obstacle_center_cc = cosy.convert_to_curvilinear_coords(obstacle.position[0], obstacle.position[1])

                # distance_to_obstacle = path.project(obstacle_center)
                
                # get the closest point from obstacle_center to the road center line
                # obstacle_point_path = path.interpolate(distance_to_obstacle)
                # split_paths = cut(path, distance_to_obstacle)
                # when path could not be split a car is not centered or way off lanelet and the scenario is invalid
                # if len(split_paths) != 2:
                    # return False, "INVALID - UNKNOW REASON!"
                # TODO is obstacle.orientation relevant for selecting the correct split lanelet
                #  in theory lanelet direction should automatically solve this
                # path = split_paths[1]
                
                # Keep at least a minimum distance (from the center of the vehicle) 
                min_safety_distance = max(self.config.car_distance_formula(obstacle.velocity) + self.generation_config.length * 0.5, 5)

                # path = cut(path, distance)[0]
                # required_driving_area = path.buffer(1)
                # plt.clf()
                # plt.plot(obstacle.position[0], obstacle.position[1], "go")
                for other_key, other_obstacle in all_obstacles.items():
                    if key == other_key:
                        continue
                    # plt.plot(*split_paths[1].buffer(1).exterior.xy, color="blue")
                    # plt.plot(*split_paths[0].buffer(1).exterior.xy, color="purple")
                    # plt.plot(other_obstacle.position[0],
                    #          other_obstacle.position[1], "ro")
                    # plt.show()
                    other_obstacle_center_cc = cosy.convert_to_curvilinear_coords(other_obstacle.position[0], other_obstacle.position[1])

                    # Compute the distance on the reference path and check this is larger than the minimum
                    actual_distance = ( (obstacle_center_cc[0] - other_obstacle_center_cc[0])**2 + (obstacle_center_cc[0] - other_obstacle_center_cc[0])**2 ) ** 0.5
                    # if required_driving_area.intersects(other_obstacle_center):
                    if actual_distance < min_safety_distance:
                        module_logger.debug(f"Scenario not valid: Distance between {key} to next object in lane {other_key} is too short {actual_distance} (min {min_safety_distance})")
                        return False, f"Distance between {key} to next object in lane {other_key} too short"
                # plt.clf()
            except Exception as exc:
                def debug_plot():
                    from commonroad.visualization.mp_renderer import MPRenderer

                    rnd = MPRenderer()
                    scenario.lanelet_network.draw(rnd)
                    rnd.render()
                    
                    plt.plot([p[0] for p in path], [p[1] for p in path], "-", color="blue", lw=6, zorder=100)
                    plt.plot([p[0] for p in extended_path], [p[1] for p in extended_path], "-", color="red", lw=4, zorder=100)
                    plt.plot(obstacle.position[0], obstacle.position[1], "o", lw=2, zorder=100)
                    plt.plot(other_obstacle.position[0], other_obstacle.position[1], "+", lw=2, zorder=100)
                    plt.show()

                module_logger.critical(f"Exception while validatin scenario {exc}")
                debug_plot()

        return True, ""

    # adjacent lanelets share the middle line with each other as a boundary for the right/left lanelet
    # so we need to check if the left/right boundary are the same for each other
    def _validate_lanelets_orientation(self, lanelet, left_lanelet, right_lanelet):
        # not really needed since we are currently not mutating the lanelet network
        return True, "Ignored"

    # validate same direction of neighbour lanelets
    def _validate_lanlet_basics(self, lanelet: Lanelet) -> Tuple[bool, str]:
        if not self.config.validate_lanelet_type_and_direction:
            return True, ""
        
        # check if lanelet has correct type
        if lanelet.lanelet_type.intersection(self.config.lanelet_types) == set():
            module_logger.debug("Scenario not valid: Wrong Lanelet Types")
            return False, "Wrong Lanelet Types"
        
        if self.config.lanelet_same_direction_only and \
                ((lanelet.adj_right_same_direction is not None and not lanelet.adj_right_same_direction)
                 or (lanelet.adj_left_same_direction is not None and not lanelet.adj_left_same_direction)):
            module_logger.debug("Scenario not valid: Lanelets not all in same direction")
            return False, "Lanelets not all in same direction"

        return True, ""

    # code taken from commonroad scenario designer
    def _find_adjacencies_of_coinciding_ways(
            self,
            lanelet: Lanelet,
            first_left_node: str,
            first_right_node: str,
            last_left_node: str,
            last_right_node: str,
    ):
        """Find adjacencies of a lanelet by checking if its vertices coincide with vertices of other lanelets.

        Set new adjacent left or right if it finds neighbors.

        Args:
          lanelet: Lanelet to check potential adjacencies for.
          first_left_node: Id of first left node of the lanelet relation in OSM.
          first_right_node: Id of first right node of the lanelet relation in OSM.
          last_left_node: Id of last left node of the lanelet relation in OSM.
          last_right_node: Id of last right node of the lanelet relation in OSM.

        """
        # first case: left adjacent, same direction
        if lanelet.adj_left is None:
            potential_left_front = self._find_lanelet_ids_of_suitable_nodes(
                self.first_right_pts, first_left_node
            )
            potential_left_back = self._find_lanelet_ids_of_suitable_nodes(
                self.last_right_pts, last_left_node
            )
            potential_left_same_direction = list(
                set(potential_left_front) & set(potential_left_back)
            )
            for lanelet_id in potential_left_same_direction:
                nb_lanelet = self.lanelet_network.find_lanelet_by_id(
                    lanelet_id)
                if nb_lanelet is not None and _two_vertices_coincide(
                        lanelet.left_vertices, nb_lanelet.right_vertices
                ):
                    self.lanelet_network.set_adjacent_left(
                        lanelet, nb_lanelet.lanelet_id, True
                    )
                    break

        # second case: right adjacent, same direction
        if lanelet.adj_right is None:
            potential_right_front = self._find_lanelet_ids_of_suitable_nodes(
                self.first_left_pts, first_right_node
            )
            potential_right_back = self._find_lanelet_ids_of_suitable_nodes(
                self.last_left_pts, last_right_node
            )
            potential_right_same_direction = list(
                set(potential_right_front) & set(potential_right_back)
            )
            for lanelet_id in potential_right_same_direction:
                nb_lanelet = self.lanelet_network.find_lanelet_by_id(
                    lanelet_id)
                if nb_lanelet is not None and _two_vertices_coincide(
                        lanelet.right_vertices, nb_lanelet.left_vertices
                ):
                    self.lanelet_network.set_adjacent_right(
                        lanelet, nb_lanelet.lanelet_id, True
                    )
                    break

        # third case: left adjacent, opposite direction
        if lanelet.adj_left is None:
            potential_left_front = self._find_lanelet_ids_of_suitable_nodes(
                self.last_left_pts, first_left_node
            )
            potential_left_back = self._find_lanelet_ids_of_suitable_nodes(
                self.first_left_pts, last_left_node
            )
            potential_left_other_direction = list(
                set(potential_left_front) & set(potential_left_back)
            )
            for lanelet_id in potential_left_other_direction:
                nb_lanelet = self.lanelet_network.find_lanelet_by_id(
                    lanelet_id)
                # compare right vertice of nb_lanelet with left vertice of lanelet
                if nb_lanelet is not None and _two_vertices_coincide(
                        lanelet.left_vertices, nb_lanelet.left_vertices[::-1]
                ):
                    self.lanelet_network.set_adjacent_left(
                        lanelet, nb_lanelet.lanelet_id, False
                    )
                    break

        # fourth case: right adjacent, opposite direction
        if lanelet.adj_right is None:
            potential_right_front = self._find_lanelet_ids_of_suitable_nodes(
                self.last_right_pts, first_right_node
            )
            potential_right_back = self._find_lanelet_ids_of_suitable_nodes(
                self.first_right_pts, last_right_node
            )
            potential_right_other_direction = list(
                set(potential_right_front) & set(potential_right_back)
            )
            for lanelet_id in potential_right_other_direction:
                nb_lanelet = self.lanelet_network.find_lanelet_by_id(
                    lanelet_id)
                if nb_lanelet is not None and self._two_vertices_coincide(
                        lanelet.right_vertices, nb_lanelet.right_vertices[::-1]
                ):
                    self.lanelet_network.set_adjacent_right(
                        lanelet, nb_lanelet.lanelet_id, True
                    )
                    break

    def _find_lanelet_ids_of_suitable_nodes(
            self, nodes_dict: dict, node_id: str
    ) -> List:
        """Find values of a dict where the keys are node ids.

        Return the entries if there is a value in the node_dict
        for the node_id, but also the values for nodes which are in
        the proximity of the node with the node_id.

        Args:
          nodes_dict: Dict which saves lanelet ids with node ids as keys.
          node_id: Id of node for which the other entries are searched for.
        Returns:
          List of lanelet ids which match the above-mentioned rules.
        """
        suitable_lanelet_ids = []
        suitable_lanelet_ids.extend(nodes_dict.get(node_id, []))
        for nd, lanelet_ids in nodes_dict.items():
            if self.node_distance(nd, node_id) < NODE_DISTANCE_TOLERANCE:
                suitable_lanelet_ids.extend(lanelet_ids)
        return suitable_lanelet_ids

    def node_distance(self, node_id1: str, node_id2: str) -> float:
        """Calculate distance of one node to other node in the projection.

        Args:
          node_id1: Id of first node.
          node_id2: id of second node.
        Returns:
          Distance in
        """
        node1 = self.osm.find_node_by_id(node_id1)
        node2 = self.osm.find_node_by_id(node_id2)
        vec1 = np.array(self.proj(float(node1.lon), float(node1.lat)))
        vec2 = np.array(self.proj(float(node2.lon), float(node2.lat)))
        return np.linalg.norm(vec1 - vec2)

class EVITAScenarioValidator(ScenarioValidator):
    
    def __init__(self, configuration: ValidationConfig, generation_config: ScenarioGeneratorConfig = None):
        super().__init__(configuration, generation_config)

    def extend_path_linearly(self, coords, extension_distance = 20.0):
        """Extends the start and end of a polyline along its tangents."""
        # Start tangent
        p0, p1 = coords[0], coords[1]
        tangent_start = (p0 - p1) / np.linalg.norm(p0 - p1)
        extended_start = p0 + tangent_start * extension_distance
        
        # End tangent
        pn_1, pn = coords[-2], coords[-1]
        tangent_end = (pn - pn_1) / np.linalg.norm(pn - pn_1)
        extended_end = pn + tangent_end * extension_distance
        
        return np.vstack([extended_start, coords, extended_end])
    
    def _validate_initial_state(self, scenario: Scenario, planning_problem: PlanningProblem):
        if planning_problem.initial_state.velocity < 0.0:
            module_logger.debug("Scenario not valid: Initial velocity cannot be negative!")
            return False, "Initial velocity cannot be negative!"
        
        # The car must be entirely inside the lanelet
        initial_lanelets = scenario.lanelet_network.find_lanelet_by_position([np.array(planning_problem.initial_state.position)])
        
        if all([len(l) == 0  for l in initial_lanelets]):
            module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} outside lanelets")

            def debug_plot():
                from commonroad.visualization.mp_renderer import MPRenderer
                rnd = MPRenderer()
                scenario.lanelet_network.draw(rnd)
                planning_problem.draw(rnd)
                rnd.render()
                plt.show()

            # debug_plot()
            
            return False, f"Initial State of {planning_problem.planning_problem_id} outside lanelets"
        # Check the agent is at least 5 meters inside any of the initial lanelets, only if those are ENTRY lanelets
        for lanelet_id in  [item for sublist in initial_lanelets for item in sublist]:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

            if lanelet.predecessor:
                # Skip checking if the lanelet has a predecessor, i.e., is not an ENTRY lanelets
                continue

            reference_path = resample_polyline(lanelet.center_vertices, step=1.0)
            cosy = CoordinateSystem(reference_path)

            initial_position_cc = cosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])
            if initial_position_cc[0] < self.config.min_dist_init:
                return False, f"Initial State of {planning_problem.planning_problem_id} too close to lanelet {lanelet.lanelet_id} initial edge"

        goal_region = planning_problem.goal
        # check if initial state is not in goal region
        for goal_state in goal_region.state_list:
            # when no position provided the goal most likely is to not crash until a certain time step
            if hasattr(goal_state, "position") and goal_state.position.contains_point(planning_problem.initial_state.position):
                module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} inside of Goal Region")
                return False, f"Initial State of {planning_problem.planning_problem_id} inside of Goal Region"

            # If the lateral distance is larger than twice the longitudinal distance the vehicle is too much on the side of the goal area!
            center_of_goal = goal_state.position.center

            # If the agent is close enough to the goal area we need to check for lateral displacement
            distance_to_goal = Point(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1]).distance(Point(center_of_goal[0], center_of_goal[1]))
            if distance_to_goal > DISTANCE:
                continue
            '''
            goal_lanelets = [scenario.lanelet_network.find_lanelet_by_id(l_id) for l_id in self._flatten(scenario.lanelet_network.find_lanelet_by_position([center_of_goal]))]
            for goal_lanelet in goal_lanelets:
                center_lane = goal_lanelet.center_vertices
                # extended_path = np.array(scale(LineString(center_lane), 3.0, 3.0).coords)
                # path = resample_polyline(extended_path, 1)
                extended_path = resample_polyline(center_lane, 1)
                path = self.extend_path_linearly(extended_path, extension_distance=30.0)

                try:
                    cosy = CoordinateSystem(path)
                                
                    initial_position_cc = cosy.convert_to_curvilinear_coords(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1])
                    goal_area_cc = cosy.convert_to_curvilinear_coords(center_of_goal[0], center_of_goal[1])
                    
                    longitudinal_distance = goal_area_cc[0] - initial_position_cc[0]
                    abs_longitudinal_distance = abs(longitudinal_distance)
                    abs_lateral_distance = abs(initial_position_cc[1] - goal_area_cc[1])

                    if longitudinal_distance < 0.0:
                        # This means that goal is before, not good
                        module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} is beyond Goal Region")
                        return False, f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} is beyond Goal Region"


                    if abs_longitudinal_distance <= 2 * abs_lateral_distance:
                        module_logger.debug(f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} too close and too on the side wrt Goal Region")
                        return False, f"Scenario not valid: Initial State of {planning_problem.planning_problem_id} too close and too on the side wrt Goal Region"
                except (ValueError, Exception) as e:
                    def debug_plot():
                        from commonroad.visualization.mp_renderer import MPRenderer
                        rnd = MPRenderer()
                        scenario.lanelet_network.draw(rnd)
                        rnd.render()
                        # center_lane
                        # plt.plot([p[0] for p in center_lane], [p[1] for p in center_lane], "-", lw=6, color="blue", zorder=100)
                        plt.plot([p[0] for p in path], [p[1] for p in path], "-", lw=2, color="red", zorder=100)
                        plt.plot([p[0] for p in path], [p[1] for p in path], "-", lw=1, color="green", zorder=100)
                        plt.plot(planning_problem.initial_state.position[0], planning_problem.initial_state.position[1], "o", markersize=4, markeredgewidth=1, zorder=100)
                        plt.plot(center_of_goal[0], center_of_goal[1], "+", markersize=6, markeredgewidth=1, zorder=100)
                        # plt.show()
                        plt.axis('off')
                        os.makedirs("test", exist_ok = True)
                        plt.savefig("test/check_plot.png",
                                dpi = 800,            # very high resolution
                                bbox_inches = "tight", # trims extra whitespace around the figure
                                pad_inches = 0,        # no padding
                                transparent = False)


                    module_logger.critical(f"Cannot establish validity of initial state for {planning_problem.planning_problem_id}. Assume valid")
                    module_logger.critical(traceback.format_exc())
                    # Use this only during debugging
                    debug_plot()
                    input("Check plot...")
            '''
        # If nothing fails, we assume valid. In case this turns out to be wrong, we catch the issue after the execution
        return True, ""

    def _validate_init_distances(self, scenario: Scenario, problem_set: PlanningProblemSet):
        obstacle_states = {}

        planning_problems = list(problem_set.planning_problem_dict.values())
        for pp in planning_problems:
            obstacle_states[pp.planning_problem_id] = pp.initial_state

        for (vehicle_id_1, state_1), (vehicle_id_2, state_2) in combinations(obstacle_states.items(), 2):
            '''
            lanelet_1 = scenario.lanelet_network.find_lanelet_by_position([state_1.position])
            lanelet_2 = scenario.lanelet_network.find_lanelet_by_position([state_2.position])

            ll_1 = scenario.lanelet_network.find_lanelet_by_id(lanelet_1[0][0])
            ll_2 = scenario.lanelet_network.find_lanelet_by_id(lanelet_2[0][0])

            # Check if vehicles are in the same or compatible direction
            is_same_direction = (
                ll_2.lanelet_id in ll_1.successor or 
                (hasattr(ll_1, "adjacent_right") and ll_2.lanelet_id == ll_1.adjacent_right and ll_1.adjacent_right_same_direction) or
                (hasattr(ll_1, "adjacent_left") and ll_2.lanelet_id == ll_1.adjacent_left and ll_1.adjacent_left_same_direction)
            )

            if not is_same_direction:
                continue
            '''
            result, distance = self.is_too_close_commonroad(state_1, state_2)

            if result:
                # module_logger.debug(f"Scenario not valid: Distance between {vehicle_id_1} to next object in lane {vehicle_id_2} too short: {distance} (min 7.0)")
                return False, f"Distance between {vehicle_id_1} to next object in lane {vehicle_id_2} too short"

        return True, ""

    def is_too_close_commonroad(self, state1: InitialState, state2: InitialState) -> Tuple[bool, str]:
        polygon_1 = self.state_to_polygon(state1, self.generation_config.length, self.generation_config.width)
        polygon_2 = self.state_to_polygon(state2, self.generation_config.length, self.generation_config.width)

        distance = polygon_1.distance(polygon_2)

        if distance < 7.0:
            return True, distance

        return False, distance

    def state_to_polygon(self, state: InitialState, length: float, width: float) -> Polygon:
        """
        Convert a CommonRoad state to a vehicle polygon.
        """
        x, y = state.position
        theta = state.orientation

        # Rectangle corners centered at (0,0)
        half_l = length / 2
        half_w = width / 2
        corners = np.array([
            [-half_l, -half_w],
            [-half_l,  half_w],
            [ half_l,  half_w],
            [ half_l, -half_w]
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # Rotate + translate
        transformed = (R @ corners.T).T + np.array([x, y])

        return Polygon(transformed)

    def validate_planning_problems(self, scenario: Scenario, problem_set: PlanningProblemSet) -> Tuple[bool, str]:
        super().validate_planning_problems(scenario, problem_set, True)

        validity_flag, validity_message = self._validate_init_distances(scenario, problem_set)
        if not validity_flag:
            return validity_flag, validity_message

        # validity_flag, validity_message = self.has_conflict(scenario, problem_set)
        # if not validity_flag:
        #     return validity_flag, validity_message
        
        return True, ""

    def has_conflict(self, scenario: Scenario, problem_set: PlanningProblemSet) -> Tuple[bool, str]:
        for pp_1, pp_2 in combinations(problem_set.planning_problem_dict.values(), 2):
            for ll_1 in pp_1.path:
                for ll_2 in pp_2.path:
                    if ll_1 == ll_2:
                        continue

                    lane1_vertices = scenario.lanelet_network.find_lanelet_by_id(ll_1).center_vertices
                    lane2_vertices = scenario.lanelet_network.find_lanelet_by_id(ll_2).center_vertices

                    lane1 = LineString(lane1_vertices)
                    lane2 = LineString(lane2_vertices)

                    if lane1.intersects(lane2):
                        return True, ""

        return False, "Scenario does not include any conflicting planning problem"

from copy import deepcopy

def get_successor_lanelets(lanelet: Lanelet, ll_network: LaneletNetwork) -> List[Lanelet]:
    # Given the lanelet, create longer lanelets by merging them together, but only following the successor relation.
    # Creats a new path for each successor (is a tree that becomes multiple paths)

    # TODO only need to check for remaining distance of the drivable area, do this with additional dist param
    full_lanelets = [lanelet]
    processed_ids = [lanelet.lanelet_id]
    
    queue = []
    visited = set()

    path_from_initial_lanelet = [lanelet]
    setattr(lanelet, "path", path_from_initial_lanelet)

    queue.append(lanelet)
    
    while queue:
        current_lanelet = queue.pop(0)
        
        if current_lanelet in visited:
            continue

        visited.add(current_lanelet)

        for successor_lanelet in [ll_network.find_lanelet_by_id(l) for l in current_lanelet.successor]:
            # For every successor we duplicate the path so far, so each reacheable lane contains the path from the initial lane
            path_from_initial_lanelet = current_lanelet.path[:]
            path_from_initial_lanelet.append(successor_lanelet)

            setattr(successor_lanelet, "path", path_from_initial_lanelet)

            queue.append(successor_lanelet)


    # At this point we can recreate the paths from the initial lane to all the ending lanes
    for exiting_lanelet in [l for l in ll_network.lanelets if len(l.successor) == 0 and hasattr(l, "path")]:
        # This is an exiting lane reacheable from the initial lanelet, let's generate the merged lane
        full_lanelet = None
        for ll in exiting_lanelet.path:
            full_lanelet = Lanelet.merge_lanelets(full_lanelet, ll) if full_lanelet else deepcopy(ll)

        full_lanelets.append(full_lanelet)
    
    # Clean up the mess
    [delattr(l, "path") for l in ll_network.lanelets if hasattr(l, "path")]

    return full_lanelets

    # while _have_lanelets_successors(full_lanelets, processed_ids):
    #     new_full_lls = []
    #     for ll in full_lanelets:
    #         if len(ll.successor) == 0:
    #             new_full_lls.append(ll)
    #         else:
    #             for succ_id in ll.successor:
    #                 succ_ll = ll_network.find_lanelet_by_id(succ_id)
    #                 full_lanelet = Lanelet.merge_lanelets(ll, succ_ll)
    #                 new_full_lls.append(full_lanelet)
    #                 processed_ids.append(succ_ll.lanelet_id)

    #     full_lanelets = new_full_lls
    # return full_lanelets


# def _have_lanelets_successors(lanelets: List[Lanelet], visited_ll_ids: List[int]):
#     have_successors = any(len(ll.successor) != 0 and any(
#         ll_id not in visited_ll_ids for ll_id in ll.successor) for ll in lanelets)
#     return have_successors


def cut(line: LineString, distance: float):
    # cuts a line in two at distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [line]
    coords = list(line.coords)
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


def _two_vertices_coincide(
        vertices1: List[np.ndarray], vertices2: List[np.ndarray]
) -> bool:
    """Check if two vertices coincide and describe the same trajectory.

    For each vertice of vertices2 the minimal distance to the trajectory
    described by vertices1 is calculated. If this distance crosses a certain
    threshold, the vertices are deemed to be different.

    Args:
    vertices1: List of vertices which describe first trajectory.
    vertices2: List of vertices which describe second trajectory.

    Returns:
    True if the vertices coincide, else False.
    """
    segments = np.diff(vertices1, axis=0)

    for vert in vertices2:
        distances = np.empty([len(vertices1) + 1])
        distances[0] = np.linalg.norm(vert - vertices1[0])
        distances[-1] = np.linalg.norm(vert - vertices1[-1])
        for i, diff in enumerate(segments):
            distances[i + 1] = np.abs(
                np.cross(diff, vertices1[i] - vert)
            ) / np.linalg.norm(diff)
        if np.min(distances) > ADJACENT_WAY_DISTANCE_TOLERANCE:
            return False

    return True
