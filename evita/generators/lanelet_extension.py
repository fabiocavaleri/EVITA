import numpy as np
from copy import deepcopy

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

from .extension_primitives import create_straight, create_curve, fit_primitive


def compute_orientation(initial_point: [], terminal_point: []):
    # orientation from initial_point to terminal_point
    x = (terminal_point[0] - initial_point[0])
    y = (terminal_point[1] - initial_point[1])
    # special cases
    if x < 0 and y == 0:
        return np.pi
    if x > 0 and y == 0:
        return 0
    if x == 0 and y < 0:
        return np.pi * 3 / 2
    if x == 0 and y > 0:
        return np.pi / 2
    if x == 0 and y == 0:
        raise ValueError('invalid orientation computation')
    # compute orientation
    orientation = abs(np.arctan(y / x))
    # adjust orientation according to quadrant
    if x > 0 and y > 0:
        pass
    elif x < 0 and y > 0:
        orientation = np.pi - orientation
    elif x < 0 and y < 0:
        orientation = np.pi + orientation
    elif x > 0 and y < 0:
        orientation = 2 * np.pi - orientation
    return orientation


def rotate_lanelet(lanelet: Lanelet, scenario_rotation: float, cx: float, cy: float):
    left_vertices = lanelet.left_vertices
    center_vertices = lanelet.center_vertices
    right_vertices = lanelet.right_vertices
    for i, c in enumerate(left_vertices):
        x = c[0] - cx
        y = c[1] - cy
        left_vertices[i][0] = x * np.cos(scenario_rotation) - y * np.sin(scenario_rotation)
        left_vertices[i][1] = x * np.sin(scenario_rotation) + y * np.cos(scenario_rotation)
    for i, c in enumerate(center_vertices):
        x = c[0] - cx
        y = c[1] - cy
        center_vertices[i][0] = x * np.cos(scenario_rotation) - y * np.sin(scenario_rotation)
        center_vertices[i][1] = x * np.sin(scenario_rotation) + y * np.cos(scenario_rotation)
    for i, c in enumerate(right_vertices):
        x = c[0] - cx
        y = c[1] - cy
        right_vertices[i][0] = x * np.cos(scenario_rotation) - y * np.sin(scenario_rotation)
        right_vertices[i][1] = x * np.sin(scenario_rotation) + y * np.cos(scenario_rotation)
    new_vertices = {'left': left_vertices, 'center': center_vertices, 'right': right_vertices}
    return new_vertices


def rotate_lanelets(lanelet_network: LaneletNetwork, rotation: float, cx: float, cy: float):
    if rotation == 0.0:
        return lanelet_network
    else:
        new_lanelet_network = LaneletNetwork()
        for lanelet in lanelet_network.lanelets:
            new_vertices = rotate_lanelet(lanelet, rotation, cx, cy)
            new_lanelet = Lanelet(left_vertices=new_vertices['left'],
                                  center_vertices=new_vertices['center'],
                                  right_vertices=new_vertices['right'],
                                  lanelet_id=lanelet.lanelet_id,
                                  adjacent_left=lanelet.adj_left,
                                  adjacent_left_same_direction=lanelet.adj_left_same_direction,
                                  adjacent_right=lanelet.adj_right,
                                  adjacent_right_same_direction=lanelet.adj_right_same_direction,
                                  predecessor=lanelet.predecessor,
                                  successor=lanelet.successor,
                                  lanelet_type=lanelet.lanelet_type)
            new_lanelet_network.add_lanelet(new_lanelet)
        for intersection in lanelet_network.intersections:
            new_lanelet_network.add_intersection(intersection)
        return new_lanelet_network


def extend_straight(lanelet: Lanelet, extension_distance: float, extend_rear=False):
    # extract connection_line and width
    if extend_rear is True:
        connection_line = {'left': [lanelet.right_vertices[-1][0], lanelet.right_vertices[-1][1]],
                           'right': [lanelet.left_vertices[-1][0], lanelet.left_vertices[-1][1]]}
    else:
        connection_line = {'left': [lanelet.left_vertices[0][0], lanelet.left_vertices[0][1]],
                           'right': [lanelet.right_vertices[0][0], lanelet.right_vertices[0][1]]}
    width = np.sqrt(np.power(connection_line['left'][0] - connection_line['right'][0], 2) +
                    np.power(connection_line['left'][1] - connection_line['right'][1], 2))
    # create straight extension
    extension = create_straight(extension_distance, width)
    # translate and rotate extension
    transformed_extension = fit_primitive(extension, connection_line)
    # compute center vertices
    left = transformed_extension['left']
    right = transformed_extension['right']
    center = {'x': [], 'y': []}
    for i in range(len(left['x'])):
        center['x'].append((left['x'][i] + right['x'][i]) / 2)
        center['y'].append((left['y'][i] + right['y'][i]) / 2)
    # merge vertices
    left_vertices = deepcopy(lanelet.left_vertices)
    center_vertices = deepcopy(lanelet.center_vertices)
    right_vertices = deepcopy(lanelet.right_vertices)
    # skip first coordinate
    for i in range(1, len(left['x'])):
        if extend_rear is True:
            left_vertices = np.vstack((left_vertices, [right['x'][i], right['y'][i]]))
            center_vertices = np.vstack((center_vertices, [center['x'][i], center['y'][i]]))
            right_vertices = np.vstack((right_vertices, [left['x'][i], left['y'][i]]))
        else:
            left_vertices = np.vstack(([left['x'][i], left['y'][i]], left_vertices))
            center_vertices = np.vstack(([center['x'][i], center['y'][i]], center_vertices))
            right_vertices = np.vstack(([right['x'][i], right['y'][i]], right_vertices))
    new_vertices = {'left': left_vertices, 'center': center_vertices, 'right': right_vertices}
    return new_vertices


def extend_curve(lanelet: Lanelet, extension_angle: float, extension_radius: float, left_adj: int, right_adj: int):
    # extract connection_line and width
    connection_line = {'left': [lanelet.left_vertices[0][0], lanelet.left_vertices[0][1]],
                       'right': [lanelet.right_vertices[0][0], lanelet.right_vertices[0][1]]}
    width = np.sqrt(np.power(connection_line['left'][0] - connection_line['right'][0], 2) +
                    np.power(connection_line['left'][1] - connection_line['right'][1], 2))
    # adjust width if there are adjacent lanelets with same direction
    if left_adj > 0:
        if extension_angle < 0:
            extension_radius -= (left_adj * width / 2)
        else:
            extension_radius += (left_adj * width / 2)
    if right_adj > 0:
        if extension_angle < 0:
            extension_radius += (right_adj * width / 2)
        else:
            extension_radius -= (right_adj * width / 2)
    # create curve extension
    extension = create_curve(extension_angle, extension_radius, width)
    # mirror curve if extension_angle is negative
    if extension_angle < 0:
        tmp = extension['left']
        extension['left'] = extension['right']
        extension['right'] = tmp
    # translate and rotate extension
    transformed_extension = fit_primitive(extension, connection_line)
    # compute center vertices
    left = transformed_extension['left']
    right = transformed_extension['right']
    center = {'x': [], 'y': []}
    for i in range(len(left['x'])):
        center['x'].append((left['x'][i] + right['x'][i]) / 2)
        center['y'].append((left['y'][i] + right['y'][i]) / 2)
    # merge vertices
    left_vertices = deepcopy(lanelet.left_vertices)
    center_vertices = deepcopy(lanelet.center_vertices)
    right_vertices = deepcopy(lanelet.right_vertices)
    # skip first coordinate
    for i in range(1, len(left['x'])):
        left_vertices = np.vstack(([left['x'][i], left['y'][i]], left_vertices))
        center_vertices = np.vstack(([center['x'][i], center['y'][i]], center_vertices))
        right_vertices = np.vstack(([right['x'][i], right['y'][i]], right_vertices))
    new_vertices = {'left': left_vertices, 'center': center_vertices, 'right': right_vertices}
    return new_vertices


def extend_the_outgoing_lanelets_with_a_straight(lanelet_network: LaneletNetwork, extension_distance: float) -> LaneletNetwork:
    if extension_distance == 0.0:
        return lanelet_network
    elif extension_distance < 0.0:
        raise ValueError('extension_distance must be greater than or equal to 0.0')
    else:
        new_lanelet_network = LaneletNetwork()
        # Select the outgoing lanelets as those that do not have any successor
        outgoing_lanelets = [ lanelet for lanelet in lanelet_network.lanelets if len(lanelet.successor) == 0]

        for lanelet in lanelet_network.lanelets:
            if lanelet in outgoing_lanelets:
                new_vertices = extend_straight(lanelet, extension_distance, extend_rear=True)
            
                new_lanelet = Lanelet(left_vertices=new_vertices['left'],
                                      center_vertices=new_vertices['center'],
                                      right_vertices=new_vertices['right'],
                                      lanelet_id=lanelet.lanelet_id,
                                      adjacent_left=lanelet.adj_left,
                                      adjacent_left_same_direction=lanelet.adj_left_same_direction,
                                      adjacent_right=lanelet.adj_right,
                                      adjacent_right_same_direction=lanelet.adj_right_same_direction,
                                      predecessor=lanelet.predecessor,
                                      successor=lanelet.successor,
                                      lanelet_type=lanelet.lanelet_type)
                
                new_lanelet_network.add_lanelet(new_lanelet)
            else:

                new_lanelet_network.add_lanelet(lanelet)

        # no idea, likely to keep data about intersections around        
        for intersection in lanelet_network.intersections:
            new_lanelet_network.add_intersection(intersection)
        return new_lanelet_network


def extend_lanelets(lanelet_network: LaneletNetwork, lanelet_ids: [], extension_distance: float, extension_radius: float) -> LaneletNetwork:
    if extension_distance == 0.0:
        return lanelet_network
    elif extension_distance < 0.0:
        raise ValueError('extension_distance must be greater than or equal to 0.0')
    else:
        new_lanelet_network = LaneletNetwork()
        for lanelet in lanelet_network.lanelets:
            if lanelet.lanelet_id in lanelet_ids:
                if extension_radius == 0.0:
                    new_vertices = extend_straight(lanelet, extension_distance)
                else:
                    extension_angle = extension_distance / extension_radius
                    if abs(extension_angle) > np.pi:
                        raise ValueError('combination of extension_distance and extension_radius yields unfeasible curve')
                    # adjust width if there are adjacent lanelets with same direction
                    left_adj = 0
                    curr_lanelet = lanelet
                    while curr_lanelet.adj_left_same_direction:
                        left_adj += 1
                        curr_lanelet = lanelet_network.find_lanelet_by_id(curr_lanelet.adj_left)
                    right_adj = 0
                    curr_lanelet = lanelet
                    while curr_lanelet.adj_right_same_direction:
                        right_adj += 1
                        curr_lanelet = lanelet_network.find_lanelet_by_id(curr_lanelet.adj_right)
                    new_vertices = extend_curve(lanelet, extension_angle, abs(extension_radius), left_adj, right_adj)
                new_lanelet = Lanelet(left_vertices=new_vertices['left'],
                                      center_vertices=new_vertices['center'],
                                      right_vertices=new_vertices['right'],
                                      lanelet_id=lanelet.lanelet_id,
                                      adjacent_left=lanelet.adj_left,
                                      adjacent_left_same_direction=lanelet.adj_left_same_direction,
                                      adjacent_right=lanelet.adj_right,
                                      adjacent_right_same_direction=lanelet.adj_right_same_direction,
                                      predecessor=lanelet.predecessor,
                                      successor=lanelet.successor,
                                      lanelet_type=lanelet.lanelet_type)
                new_lanelet_network.add_lanelet(new_lanelet)
            else:
                new_lanelet_network.add_lanelet(lanelet)
        for intersection in lanelet_network.intersections:
            new_lanelet_network.add_intersection(intersection)
        return new_lanelet_network


def extend_all_lanelets(lanelet_network: LaneletNetwork, extension_distance: float):
    if extension_distance == 0.0:
        return lanelet_network
    elif extension_distance < 0.0:
        raise ValueError('extension_distance must be greater than or equal to 0.0')
    else:
        new_lanelet_network = LaneletNetwork()
        for lanelet in lanelet_network.lanelets:
            # extend rear
            new_vertices = extend_straight(lanelet, extension_distance, extend_rear=True)
            new_lanelet = Lanelet(left_vertices=new_vertices['left'],
                                  center_vertices=new_vertices['center'],
                                  right_vertices=new_vertices['right'],
                                  lanelet_id=lanelet.lanelet_id,
                                  adjacent_left=lanelet.adj_left,
                                  adjacent_left_same_direction=lanelet.adj_left_same_direction,
                                  adjacent_right=lanelet.adj_right,
                                  adjacent_right_same_direction=lanelet.adj_right_same_direction,
                                  predecessor=lanelet.predecessor,
                                  successor=lanelet.successor,
                                  lanelet_type=lanelet.lanelet_type)
            # extend front
            new_vertices = extend_straight(new_lanelet, extension_distance)
            new_lanelet = Lanelet(left_vertices=new_vertices['left'],
                                  center_vertices=new_vertices['center'],
                                  right_vertices=new_vertices['right'],
                                  lanelet_id=lanelet.lanelet_id,
                                  adjacent_left=lanelet.adj_left,
                                  adjacent_left_same_direction=lanelet.adj_left_same_direction,
                                  adjacent_right=lanelet.adj_right,
                                  adjacent_right_same_direction=lanelet.adj_right_same_direction,
                                  predecessor=lanelet.predecessor,
                                  successor=lanelet.successor,
                                  lanelet_type=lanelet.lanelet_type)
            new_lanelet_network.add_lanelet(new_lanelet)
        for intersection in lanelet_network.intersections:
            new_lanelet_network.add_intersection(intersection)
        return new_lanelet_network