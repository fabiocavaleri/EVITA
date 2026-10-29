import numpy as np
import matplotlib.pyplot as plt

def compute_orientation(initial_point: [], terminal_point: []):
    """
    Compute the orientation of a line segment defined by two points
    The orientation is defined as the angle between the positive x-axis and the line segment, measured counterclockwise in radians
    """
    # Extract x and y differences
    x = (terminal_point[0] - initial_point[0])
    y = (terminal_point[1] - initial_point[1])

    # Manage special cases
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
    
    # Compute orientation
    orientation = abs(np.arctan(y / x))
    
    # Adjust orientation according to quadrant
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
    """
    Fit a primitive defined by its coordinates to a connection line defined by two points
    The primitive is first rotated to match the orientation of the connection line and then translated to match the position of the connection line
    """

    # Extract left and right points of the connection line and primitive
    left_point = connection_line['left']
    right_point = connection_line['right']
    primitive_left = [coordinates['left']['x'][0], coordinates['left']['y'][0]]
    primitive_right = [coordinates['right']['x'][0], coordinates['right']['y'][0]]

    # Compute orientation and rotate coordinates
    connection_line_orientation = compute_orientation(left_point, right_point)
    primitive_orientation = compute_orientation(primitive_left, primitive_right)
    orientation = primitive_orientation - connection_line_orientation
    rotated_primitive = rotate_primitive(coordinates, orientation)
    
    # Translate coordinates
    translated_primitive = translate_primitive(rotated_primitive, left_point, right_point)
    left_border = translated_primitive['left']
    right_border = translated_primitive['right']
    
    # Plot the fitted primitive and connection line if requested
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
    """
    Translate a primitive defined by its coordinates to match the position of a connection line defined by two points
    The translation is defined by the distance between the left point of the connection line and the left point of the primitive and the orientation of the connection line
    """
    # Extract left point of the primitive and connection line
    figure_origin = (coordinates['left']['x'][0], coordinates['left']['y'][0])
    # Compute orientation and distance between left point of the primitive and left point of the connection line
    orientation = compute_orientation(figure_origin, left_point)
    distance = np.sqrt(np.power(figure_origin[0] - left_point[0], 2) + np.power(figure_origin[1] - left_point[1], 2))

    # Translate left border
    left_border = {'x': [], 'y': []}
    for i in range(len(coordinates['left']['x'])):
        x = coordinates['left']['x'][i]
        y = coordinates['left']['y'][i]
        new_x = x + np.cos(orientation) * distance
        new_y = y + np.sin(orientation) * distance
        left_border['x'].append(new_x)
        left_border['y'].append(new_y)
    
    # Translate right border
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
    """
    Rotate a primitive defined by its coordinates by a given angle
    The rotation is defined by the angle between the positive x-axis and the line segment connecting the left and right points of the primitive, measured counterclockwise in radians
    """
    # Compute center point
    cx = (sum(coordinates['left']['x']) + sum(coordinates['right']['x'])) / (len(coordinates['left']['x']) * 2)
    cy = (sum(coordinates['left']['y']) + sum(coordinates['right']['y'])) / (len(coordinates['left']['y']) * 2)
    
    # Rotate left border around (cx, cy)
    left_border = {'x': [], 'y': []}
    for i in range(len(coordinates['left']['x'])):
        x = coordinates['left']['x'][i]
        y = coordinates['left']['y'][i]
        new_x = ((x - cx) * np.cos(rotation) + (y - cy) * np.sin(rotation)) + cx
        new_y = ((- x + cx) * np.sin(rotation) + (y - cy) * np.cos(rotation)) + cy
        left_border['x'].append(new_x)
        left_border['y'].append(new_y)
    
    # Rotate right border around (cx, cy)
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
    """
    Create a straight primitive defined by its left and right borders
    The straight primitive is defined by two parallel lines at a given distance and width
    """
    precision = 10

    # Create x coordinates for the left and right borders
    x_coordinates = np.linspace(0, extension_distance, precision)
    left_border = {'x': x_coordinates, 'y': [0] * precision}
    right_border = {'x': x_coordinates, 'y': [width] * precision}

    # Plot the straight primitive if requested
    if plot is True:
        plt.plot(left_border['x'], left_border['y'], c='b')
        plt.plot(right_border['x'], right_border['y'], c='r')
        plt.axis('scaled')
        plt.show()

    return {'left': left_border, 'right': right_border}

def create_curve(curve_angle: float, curve_radius: float, width: float, plot: bool = False):
    """
    Create a curve primitive defined by its left and right borders
    The curve primitive is defined by two concentric arcs at a given radius and width, with the angle of the curve defined by the curve_angle parameter
    """
    precision = 20
    # Initialize left and right borders
    left_border = {'x': [], 'y': []}
    right_border = {'x': [], 'y': []}
    # Iterate over the angle of the curve and compute the coordinates of the left and right borders
    for i in range(precision, -1, -1):
        left_border['x'].append(np.cos(i * curve_angle / precision) * (curve_radius - width / 2))
        left_border['y'].append(np.sin(i * curve_angle / precision) * (curve_radius - width / 2))
        right_border['x'].append(np.cos(i * curve_angle / precision) * (curve_radius + width / 2))
        right_border['y'].append(np.sin(i * curve_angle / precision) * (curve_radius + width / 2))

    # Plot the curve primitive if requested
    if plot is True:
        plt.plot(left_border['x'], left_border['y'], c='b')
        plt.plot(right_border['x'], right_border['y'], c='r')
        plt.axis('scaled')
        plt.show()

    return {'left': left_border, 'right': right_border}