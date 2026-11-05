"""
Extension of a road network: a straight piece of road is added after each outgoing lanelet (one without
successors), so that planners do not stop in front of the end of the road. Based on STRETCH.
"""
import numpy as np

from copy import deepcopy
from typing import Tuple

from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

# Number of points of the added straight piece, including the lanelet's end
EXTENSION_POINTS = 10

def _extended_vertices(lanelet: Lanelet, extension_distance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Left, center and right vertices of the lanelet followed by a straight piece of `extension_distance` meters, perpendicular to the lanelet's end edge.
    """
    left_end = np.asarray(lanelet.left_vertices[-1], dtype = float)
    right_end = np.asarray(lanelet.right_vertices[-1], dtype = float)

    # Forward direction: the end edge (from left to right) rotated by +90 degrees
    edge = right_end - left_end
    edge_length = np.linalg.norm(edge)
    if edge_length == 0.0:
        raise ValueError(f"Lanelet {lanelet.lanelet_id} has a degenerate end edge")
    
    forward = np.array([-edge[1], edge[0]]) / edge_length

    # Distances of the new points beyond the end (the end itself is already a vertex)
    distances = np.linspace(0.0, extension_distance, EXTENSION_POINTS)[1:, None]
    new_left = left_end + distances * forward
    new_right = right_end + distances * forward

    return (np.vstack([lanelet.left_vertices, new_left]),
            np.vstack([lanelet.center_vertices, (new_left + new_right) / 2.0]),
            np.vstack([lanelet.right_vertices, new_right]))

def extend_the_outgoing_lanelets_with_a_straight(lanelet_network: LaneletNetwork, extension_distance: float) -> LaneletNetwork:
    """
    Copy of the road network in which each outgoing lanelet continues straight for `extension_distance` meters.
    Everything else (other lanelets, traffic signs and lights, intersections, and the attributes of the outgoing lanelets) is kept.
    Returns the network itself if the distance is 0.
    """
    # Skip extension if the extension distance is 0.0
    if extension_distance < 0.0:
        raise ValueError("extension_distance must be greater than or equal to 0.0")
    if extension_distance == 0.0:
        return lanelet_network

    new_lanelet_network = deepcopy(lanelet_network)

    # Select the outgoing lanelets as those that do not have any successor
    outgoing_lanelets = [lanelet for lanelet in lanelet_network.lanelets if len(lanelet.successor) == 0]

    # Iterate over lanelets and extend the outgoing lanelets by a straight line of the given length
    for lanelet in outgoing_lanelets:
        left_vertices, center_vertices, right_vertices = _extended_vertices(lanelet, extension_distance)
    
        extended = Lanelet(left_vertices = left_vertices,
                           center_vertices = center_vertices,
                           right_vertices = right_vertices,
                           lanelet_id = lanelet.lanelet_id,
                           predecessor = lanelet.predecessor,
                           successor = lanelet.successor,
                           adjacent_left = lanelet.adj_left,
                           adjacent_left_same_direction = lanelet.adj_left_same_direction,
                           adjacent_right = lanelet.adj_right,
                           adjacent_right_same_direction = lanelet.adj_right_same_direction,
                           line_marking_left_vertices = lanelet.line_marking_left_vertices,
                           line_marking_right_vertices = lanelet.line_marking_right_vertices,
                           stop_line = lanelet.stop_line,
                           lanelet_type = lanelet.lanelet_type,
                           user_one_way = lanelet.user_one_way,
                           user_bidirectional = lanelet.user_bidirectional,
                           traffic_signs = lanelet.traffic_signs,
                           traffic_lights = lanelet.traffic_lights)
        
        new_lanelet_network.remove_lanelet(lanelet.lanelet_id)
        new_lanelet_network.add_lanelet(extended)

    return new_lanelet_network