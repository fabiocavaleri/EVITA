import numpy as np
import matplotlib.pyplot as plt


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


def fit_primitive(coordinates: {}, connection_line: {}, plot: bool = False):
    left_point = connection_line['left']
    right_point = connection_line['right']
    primitive_left = [coordinates['left']['x'][0], coordinates['left']['y'][0]]
    primitive_right = [coordinates['right']['x'][0], coordinates['right']['y'][0]]
    # compute orientation and rotate coordinates
    connection_line_orientation = compute_orientation(left_point, right_point)
    primitive_orientation = compute_orientation(primitive_left, primitive_right)
    orientation = primitive_orientation - connection_line_orientation
    rotated_primitive = rotate_primitive(coordinates, orientation)
    # translate coordinates
    translated_primitive = translate_primitive(rotated_primitive, left_point, right_point)
    left_border = translated_primitive['left']
    right_border = translated_primitive['right']
    if plot is True:
        x_values = [left_point[0], right_point[0]]
        y_values = [left_point[1], right_point[1]]
        plt.plot(x_values, y_values, c='g')
        plt.plot(left_border['x'], left_border['y'], c='b')
        plt.plot(right_border['x'], right_border['y'], c='r')
        plt.axis('scaled')
        plt.show()
    return {'left': left_border, 'right': right_border}


def translate_primitive(coordinates: {}, left_point: [], right_point: []):
    # compute translation orientation and translation distance
    figure_origin = (coordinates['left']['x'][0], coordinates['left']['y'][0])
    orientation = compute_orientation(figure_origin, left_point)
    distance = np.sqrt(np.power(figure_origin[0] - left_point[0], 2) + np.power(figure_origin[1] - left_point[1], 2))
    # translate left border
    left_border = {'x': [], 'y': []}
    for i in range(len(coordinates['left']['x'])):
        x = coordinates['left']['x'][i]
        y = coordinates['left']['y'][i]
        new_x = x + np.cos(orientation) * distance
        new_y = y + np.sin(orientation) * distance
        left_border['x'].append(new_x)
        left_border['y'].append(new_y)
    # translate right border
    right_border = {'x': [], 'y': []}
    for i in range(len(coordinates['right']['x'])):
        x = coordinates['right']['x'][i]
        y = coordinates['right']['y'][i]
        new_x = x + np.cos(orientation) * distance
        new_y = y + np.sin(orientation) * distance
        right_border['x'].append(new_x)
        right_border['y'].append(new_y)
    return {'left': left_border, 'right': right_border}


def rotate_primitive(coordinates: {}, rotation: float):
    # compute center point
    cx = (sum(coordinates['left']['x']) + sum(coordinates['right']['x'])) / (len(coordinates['left']['x']) * 2)
    cy = (sum(coordinates['left']['y']) + sum(coordinates['right']['y'])) / (len(coordinates['left']['y']) * 2)
    # rotate left border around (cx, cy)
    left_border = {'x': [], 'y': []}
    for i in range(len(coordinates['left']['x'])):
        x = coordinates['left']['x'][i]
        y = coordinates['left']['y'][i]
        new_x = ((x - cx) * np.cos(rotation) + (y - cy) * np.sin(rotation)) + cx
        new_y = ((- x + cx) * np.sin(rotation) + (y - cy) * np.cos(rotation)) + cy
        left_border['x'].append(new_x)
        left_border['y'].append(new_y)
    # rotate right border around (cx, cy)
    right_border = {'x': [], 'y': []}
    for i in range(len(coordinates['right']['x'])):
        x = coordinates['right']['x'][i]
        y = coordinates['right']['y'][i]
        new_x = ((x - cx) * np.cos(rotation) + (y - cy) * np.sin(rotation)) + cx
        new_y = ((- x + cx) * np.sin(rotation) + (y - cy) * np.cos(rotation)) + cy
        right_border['x'].append(new_x)
        right_border['y'].append(new_y)
    return {'left': left_border, 'right': right_border}


def create_straight(extension_distance: float, width: float, plot: bool = False):
    precision = 10
    x_coordinates = np.linspace(0, extension_distance, precision)
    left_border = {'x': x_coordinates, 'y': [0] * precision}
    right_border = {'x': x_coordinates, 'y': [width] * precision}
    if plot is True:
        plt.plot(left_border['x'], left_border['y'], c='b')
        plt.plot(right_border['x'], right_border['y'], c='r')
        plt.axis('scaled')
        plt.show()
    return {'left': left_border, 'right': right_border}


def create_curve(curve_angle: float, curve_radius: float, width: float, plot: bool = False):
    precision = 20
    left_border = {'x': [], 'y': []}
    right_border = {'x': [], 'y': []}
    for i in range(precision, -1, -1):
        left_border['x'].append(np.cos(i * curve_angle / precision) * (curve_radius - width / 2))
        left_border['y'].append(np.sin(i * curve_angle / precision) * (curve_radius - width / 2))
        right_border['x'].append(np.cos(i * curve_angle / precision) * (curve_radius + width / 2))
        right_border['y'].append(np.sin(i * curve_angle / precision) * (curve_radius + width / 2))
    if plot is True:
        plt.plot(left_border['x'], left_border['y'], c='b')
        plt.plot(right_border['x'], right_border['y'], c='r')
        plt.axis('scaled')
        plt.show()
    return {'left': left_border, 'right': right_border}


def create_three_way_intersection():
    return None


def create_four_way_intersection():
    return None
