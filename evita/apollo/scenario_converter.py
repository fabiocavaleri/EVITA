"""
CommonRoad scenario -> DoppelTest's scenario JSON specification.

    {
      "ad_section": {"adcs": [{"routing": [...], "start_s": ..., "dest_s": ..., "start_t": ...}, ...]},
      "pd_section": {"pds": []},
      "tc_section": {...},                 # traffic_control.TrafficControl.to_doppeltest
      "evita_vehicle_ids": [...]           # EVITA's id of each vehicle, in the order of "adcs" (ignored by DoppelTest)
    }

Each vehicle (planning problem) becomes one Apollo instance:
- routing: Apollo lanes from the start to the goal. It is the planning problem's route when it has one,
    otherwise the shortest chain of successors from the start lanelet to the goal.
- start_s, dest_s: distances along the first and last lane, by projection on the lanelet's centerline.
- start_t: the planning problem's initial time, in seconds.

Positions: a CommonRoad vehicle's position is the center of its rectangle; Apollo's is a reference point at
back_edge_to_center from the rear edge. start_s and dest_s are those of the reference point, so that the vehicle's
center is where the planning problem puts it (start) and at the goal area's center (destination).

What Apollo does not take: the initial velocity, orientation and acceleration (DoppelTest starts every vehicle at rest,
heading along its lane), the goal's time interval and orientation.
The scenario's traffic lights must have been written by traffic_control.
"""

import json
import math
import heapq
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shapely.geometry import LineString, Point

from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.scenario import Scenario

from cr_scenario_handler.utils.vehicle_models import APOLLO_LINCOLN_MKZ, back_edge_to_center, vehicle_parameters

from .map_data import ApolloMapData, MapDataError
from .traffic_control import TrafficControl

module_logger = logging.getLogger(__name__)

# Largest distance (m) between a position and its lanelet's centerline, beyond the lanelet's half width
LATERAL_TOLERANCE = 0.5

class ScenarioSpecError(Exception):
    """
    A scenario cannot be described in DoppelTest's format (e.g. a vehicle off its route, a route that is not a chain of
    successors, a destination behind the start)
    """

@dataclass(frozen = True)
class ApolloVehicle:
    """
    Apollo's vehicle: length, width and back_edge_to_center (distance from the rear edge to Apollo's reference point)
    """
    length: float
    width: float
    back_edge_to_center: float

    @property
    def center_offset(self) -> float:
        """
        Distance (m) from Apollo's reference point forward to the center of the vehicle
        """
        return self.length / 2 - self.back_edge_to_center

    @classmethod
    def from_vehicle_model(cls, cr_vehicle_id: int) -> "ApolloVehicle":
        """
        Dimensions of a vehicle model of frenetix_patches/vehicle_models.py (the reference point is its rear axle)
        """
        parameters = vehicle_parameters(cr_vehicle_id)

        return cls(length = parameters.l, width = parameters.w, back_edge_to_center = back_edge_to_center(cr_vehicle_id))

# Lincoln MKZ, the vehicle of DoppelTest's Apollo setup (the custom vehicle model 101); it must match Apollo's
# vehicle_param.pb.txt and DoppelTest's config.py, whose values the runner records in every trace
APOLLO_VEHICLE = ApolloVehicle.from_vehicle_model(APOLLO_LINCOLN_MKZ)

# ---------- Positions along a lanelet ----------

def centerline(lanelet: Lanelet) -> LineString:
    return LineString(lanelet.center_vertices)

def position_along(lanelet: Lanelet, position) -> Tuple[float, float]:
    """
    Distance along the lanelet's centerline of the point closest to the position, and the position's distance from it
    """
    line = centerline(lanelet)
    point = Point(float(position[0]), float(position[1]))
    s = line.project(point)
    return s, point.distance(line.interpolate(s))

def apollo_heading(line: LineString, s: float) -> Tuple[float, float]:
    """
    Unit direction Apollo gives a vehicle placed at distance s along a lane: the direction of the centerline segment
    nearest to the point.
    The converted map keeps Apollo's centerline points, so the result is Apollo's.
    """
    point = line.interpolate(s)
    coordinates = list(line.coords)
    best, best_distance = 0, float("inf")

    # Get closest segment of the centerline to the point
    for i in range(len(coordinates) - 1):
        distance = point.distance(LineString(coordinates[i:i + 2]))
        if distance < best_distance:
            best, best_distance = i, distance
    
    # Get the unit vector of that segment
    (x1, y1), (x2, y2) = coordinates[best], coordinates[best + 1]
    norm = math.hypot(x2 - x1, y2 - y1) or 1.0
    
    # Return the unit vector along the segment
    return (x2 - x1) / norm, (y2 - y1) / norm

def reference_point_s(lanelet: Lanelet, center_s: float, offset: float, resolution: float = 0.01) -> float:
    """
    Distance along the lanelet at which Apollo must put its reference point so that the vehicle's center is the lanelet's point at center_s.
    On straight lanes this is center_s - offset; on curves the vehicle's body is a chord of the curve and Apollo's
    heading is the segment's, so the solution is searched within 1 m of that value.
    """
    # Get the lanelet's centerline and the target point at center_s
    line = centerline(lanelet)
    target = line.interpolate(center_s)
    guess = center_s - offset

    best_s, best_error = guess, float("inf")

    # Search for the distance along the lanelet that minimizes the distance between the vehicle's center and the target point
    steps = int(round(2.0 / resolution))
    for k in range(steps + 1):
        # Get the distance along the lanelet to test
        s = guess - 1.0 + k * resolution

        # Skip it if it is outside the lanelet
        if s < 0 or s > line.length:
            continue
        
        # Get the point at that distance and the heading of the lanelet there
        point = line.interpolate(s)
        hx, hy = apollo_heading(line, s)

        # Compute the distance between the vehicle's center (point + offset * heading) and the target point
        error = math.hypot(point.x + offset * hx - target.x, point.y + offset * hy - target.y)
        if error < best_error:
            best_s, best_error = s, error

    return best_s

def check_on_lanelet(lanelet: Lanelet, position, what: str) -> float:
    """
    Distance of the position along the lanelet.
    Return ScenarioSpecError if the position is not on the lanelet
    """
    s, lateral = position_along(lanelet, position)

    # Check that the position is not too far from the lanelet's centerline (the lanelet's half width + tolerance)
    half_width = max(Point(lanelet.left_vertices[i]).distance(Point(lanelet.right_vertices[i]))
                     for i in range(len(lanelet.center_vertices))) / 2

    if lateral > half_width + LATERAL_TOLERANCE:
        raise ScenarioSpecError(f"{what} is {lateral:.2f} m from the centerline of lanelet {lanelet.lanelet_id}")
    
    return s

def goal_position(planning_problem: PlanningProblem):
    """
    Center of the goal area (the first goal state's position)
    """
    states = planning_problem.goal.state_list
    
    if not states or not hasattr(states[0], "position") or states[0].position is None:
        raise ScenarioSpecError(f"Vehicle {planning_problem.planning_problem_id} has no goal position")
    
    centroid = states[0].position.shapely_object.centroid
    return (centroid.x, centroid.y)

# ---------- Routes ----------

def lanelets_at(lanelet_network: LaneletNetwork, position, tolerance: float = 0.01) -> List[int]:
    """
    Lanelets containing the position whose centerline is closest to it (within `tolerance` of the closest one).
    Where lanelets overlap (e.g. just after a split), the position belongs to the lanelet it is centered on (EVITA puts
    starts and goals on their lanelet's centerline); lanelets whose centerlines coincide there are all kept.
    """
    candidates = list(lanelet_network.find_lanelet_by_position([list(position)])[0])

    if len(candidates) <= 1:
        return candidates

    distances = {lanelet_id: position_along(lanelet_network.find_lanelet_by_id(lanelet_id), position)[1]
                 for lanelet_id in candidates}
    closest = min(distances.values())

    return [lanelet_id for lanelet_id in candidates if distances[lanelet_id] <= closest + tolerance]

def shortest_successor_route(lanelet_network: LaneletNetwork, starts: List[int], goals: List[int]) -> Optional[List[int]]:
    """
    Shortest chain of successors (by centerline length) from one of the start lanelets to one of the goal lanelets
    """
    goals = set(goals)
    # Initialize queue with (length, lanelet_id, route) for each start lanelet
    queue = [(0.0, start, [start]) for start in starts]
    heapq.heapify(queue)
    visited = set()

    while queue:
        # Pop the tuple (length, lanelet_id, route) with the smallest length
        length, lanelet_id, route = heapq.heappop(queue)
        
        # Check if the current lanelet is one of the goal lanelets
        if lanelet_id in goals:
            return route

        if lanelet_id in visited:
            continue
        
        visited.add(lanelet_id)

        # Add successors of the current lanelet to the queue with updated length and route
        for successor in lanelet_network.find_lanelet_by_id(lanelet_id).successor:
            if successor not in visited:
                successor_length = centerline(lanelet_network.find_lanelet_by_id(successor)).length
                heapq.heappush(queue, (length + successor_length, successor, route + [successor]))

    return None

def vehicle_route(lanelet_network: LaneletNetwork, planning_problem: PlanningProblem) -> List[int]:
    """
    Lanelets from the vehicle's start to its goal, each the successor of the previous one
    """
    route = list(getattr(planning_problem, "path", None) or [])

    # If the planning problem has no route, compute the shortest chain of successors from the start lanelet to the goal lanelet
    if not route:
        starts = lanelets_at(lanelet_network, planning_problem.initial_state.position)
        goals = lanelets_at(lanelet_network, goal_position(planning_problem))
        route = shortest_successor_route(lanelet_network, starts, goals) or []

        if not route:
            raise ScenarioSpecError(f"Vehicle {planning_problem.planning_problem_id}: no chain of successors from lanelets {starts} to the goal lanelets {goals}")

    # Check that the route is a chain of successors
    for previous, following in zip(route, route[1:]):
        if following not in lanelet_network.find_lanelet_by_id(previous).successor:
            raise ScenarioSpecError(f"Vehicle {planning_problem.planning_problem_id}: in its route {route}, lanelet {following} is not a successor of {previous}")
    
    return route

# ---------- One vehicle ----------

def build_adc(lanelet_network: LaneletNetwork, planning_problem: PlanningProblem, map_data: ApolloMapData, dt: float, vehicle: ApolloVehicle = APOLLO_VEHICLE) -> Dict:
    """
    One entry of ad_section.adcs
    """
    vehicle_id = planning_problem.planning_problem_id
    route = vehicle_route(lanelet_network, planning_problem)
    first = lanelet_network.find_lanelet_by_id(route[0])
    last = lanelet_network.find_lanelet_by_id(route[-1])

    # Reference point: center_offset behind the center, along the lane
    start_center = check_on_lanelet(first, planning_problem.initial_state.position, f"Vehicle {vehicle_id}'s start")
    start_s = start_center - vehicle.center_offset

    # If the reference point is before the start of the lanelet, move it to the start of the lanelet and warn
    if start_s >= 0:
        start_s = reference_point_s(first, start_center, vehicle.center_offset)
    else:
        module_logger.warning(f"Vehicle {vehicle_id}: start moved {-start_s:.2f} m forward, to the beginning of lanelet {route[0]} (its rear would be on the previous lane)")
        start_s = 0.0

    dest_center = check_on_lanelet(last, goal_position(planning_problem), f"Vehicle {vehicle_id}'s goal")
    if dest_center >= vehicle.center_offset:
        dest_s = reference_point_s(last, dest_center, vehicle.center_offset)
    else:
        dest_s = 0.0

    # Check if the goal is ahead of the start on the route (if the route has only one lanelet)
    if len(route) == 1 and dest_s <= start_s:
        raise ScenarioSpecError(f"Vehicle {vehicle_id}: the goal ({dest_s:.2f} m) is not ahead of the start ({start_s:.2f} m) on lanelet {route[0]}")

    # Convert the route's lanelet ids to Apollo's lane ids
    try:
        routing = [map_data.lane_id(lanelet_id) for lanelet_id in route]
    except MapDataError as exc:
        raise ScenarioSpecError(f"Vehicle {vehicle_id}: {exc}") from exc

    return {
        "routing": routing,
        "start_s": start_s,
        "dest_s": dest_s,
        "start_t": round(planning_problem.initial_state.time_step * dt, 6)
    }

# ---------- Whole scenario ----------

def build_scenario_json(scenario: Scenario, planning_problem_set: PlanningProblemSet, map_data: ApolloMapData, vehicle: ApolloVehicle = APOLLO_VEHICLE) -> Dict:
    """
    DoppelTest's JSON of a CommonRoad scenario: one Apollo instance per planning problem, no pedestrians, the traffic control written in the scenario's traffic lights
    """
    lanelet_network = scenario.lanelet_network
    for warning in map_data.check(lanelet_network):
        module_logger.warning(warning)

    planning_problems = list(planning_problem_set.planning_problem_dict.values())
    if not planning_problems:
        raise ScenarioSpecError("The scenario has no vehicle")

    return {
        "ad_section": {"adcs": [build_adc(lanelet_network, planning_problem, map_data, scenario.dt, vehicle)
                                for planning_problem in planning_problems]},
        "pd_section": {"pds": []},
        "tc_section": TrafficControl.from_commonroad(lanelet_network, scenario.dt).to_doppeltest(map_data),
        "evita_vehicle_ids": [planning_problem.planning_problem_id for planning_problem in planning_problems]
    }

def write_scenario_json(scenario_json: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(scenario_json, f, indent = 2)
