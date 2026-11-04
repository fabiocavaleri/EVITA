import os
import json
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

import sys
sys.path.append("../")

from hdmap.MapParser import MapParser
from modules.map.proto.map_pb2 import Map as ApolloMap

from commonroad.geometry.shape import Rectangle
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.scenario.scenario import Scenario, ScenarioID, Tag
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
from commonroad.common.common_lanelet import StopLine, LineMarking
from commonroad.scenario.traffic_light import TrafficLight, TrafficLightDirection, TrafficLightCycleElement, TrafficLightCycle, TrafficLightState
from commonroad.scenario.traffic_sign import TrafficSign, TrafficSignElement, TrafficSignIDGermany

#
# ------------------------------------------------------------
#

def get_lanelet_from_stopline(lanelet_network: LaneletNetwork, stopline_points: List) -> List[int]:
    """
    Associate a stop-line to the nearest lanelet centerline.

    :param lanelet_network: Scenarios's laneletNetwork
    :param stopline_points: Numpy array with the stopline segment points
    :param max_dist: Max allowed distance (meters) between lanelet center point and stopline center pointt
    :return: lanelet_id or None
    """

    if stopline_points is None or len(stopline_points) == 0:
        print("NONE")
        return None
    
    stopline = LineString(stopline_points)

    lanelets = []

    # Iterate over scenario lanelets
    for lanelet in lanelet_network.lanelets:
        # Get lanelet center points and compute distance from stop line
        centerline = LineString(lanelet.center_vertices)

        if stopline.intersects(centerline):
            lanelets.append(lanelet.lanelet_id)

    return lanelets

def curve_to_points(curve: List) -> List:
    points = []
    for segment in curve.segment:
        if segment.HasField("line_segment"):
            for p in segment.line_segment.point:
                # Avoid duplicate points
                if not points or (p.x, p.y) != points[-1]:
                    points.append((p.x, p.y))

    # If exactly three points, insert interpolated midpoints
    if len(points) == 3:
        p0, p1, p2 = points

        # Midpoint between p0 and p1
        mid01 = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)

        # Midpoint between p1 and p2
        mid12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        # Rebuild list with inserted points
        points = [p0, mid01, p1, mid12, p2]

    return points

def get_lane_id(id_list: list) -> list:
    """
    Convert a list of Apollo lanelet ids to CommonRoad map lanelet ids
    
    :param id_list: List of Apollo lanelet ids
    :type id_list: list
    :return: List of converted CommonRoad lanelet ids
    :rtype: list
    """

    if len(id_list) > 0:
        global lane_id_map
        return [lane_id_map[id_item.id] for id_item in id_list]
    
    return None   

def link_lanelets(lanelet: Lanelet, predecessors_id: List[int], successors_id: List[int]) -> Lanelet:
    """
    Link a CommonRoad lanelet object to its predecessor(s) and successor(s)
    
    :param lanelet: CommonRoad lanelet object
    :type lanelet: Lanelet
    :param predecessors_id: Predecessors lanelets identifiers
    :type predecessors_id: List[int]
    :param successors_id: Successors lanelets identifiers
    :type successors_id: List[int]
    :return: Updated lanelet object
    :rtype: Lanelet
    """

    if successors_id is not None:
        for id in successors_id:
            lanelet.successor.append(id)

    if predecessors_id is not None:
        for id in predecessors_id:
            lanelet.predecessor.append(id)

    return lanelet

def adjacent_lanes(adjacent_forward: List[int], adjacent_reverse: List[int]) -> Tuple[int, bool]:
    """
    Retrieve the adjacent lane identifier and its direction 
    
    :param adjacent_forward: Same direction adjacent lane
    :type adjacent_forward: List[int]
    :param adjacent_reverse: Opposite direction adjacent lane
    :type adjacent_reverse: List[int]
    :return: Commonroad identifier for adjacent lane and flag indicating the lane direction
    :rtype: Tuple[int, bool]
    """

    same_direction = False
    adjacent = None
    
    if adjacent_forward is not None:
        same_direction = True
        adjacent = adjacent_forward[0]

    elif adjacent_reverse is not None:
        adjacent = adjacent_reverse[0]

    return adjacent, same_direction

def lane_vertices(center: List, left: List, right: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CommonRoad needs left and right boundaries with the same number of points (and lane lines of at least 3 points).
    Keep Apollo's points when they already satisfy this; otherwise resample the three lines to a common number of points.
    """
    if len(center) == len(left) == len(right) >= 3:
        return np.array(left), np.array(right), np.array(center)

    n = max(len(center), len(left), len(right), 3)
    
    return resample(left, n), resample(right, n), resample(center, n)

def resample(points: List, n: int) -> np.ndarray:
    """
    n points evenly spaced along a polyline (by length)
    """
    line = LineString(points)

    return np.array([line.interpolate(f, normalized = True).coords[0] for f in np.linspace(0.0, 1.0, n)])

def controlled_lanelets(apollo_object) -> List[int]:
    """
    Lanelets controlled by a stop sign or signal, with CommonRoad's convention (the stop line is at the END of the incoming lanelet).
    Apollo lists the lanes crossing the stop line; for a lane that starts at the stop line (a junction lane),
    the incoming lanelets are its predecessors; a lane the stop line crosses elsewhere is kept.
    """
    lanelets = set()

    for overlap_id in apollo_object.overlap_id:
        overlap = apollo_overlaps.get(overlap_id.id)
        
        if overlap is None:
            continue
        
        for obj in overlap.object:
            if obj.WhichOneof("overlap_info") != "lane_overlap_info" or obj.id.id not in lane_id_map:
                continue
            
            lanelet = lanelet_network.find_lanelet_by_id(lane_id_map[obj.id.id])
            
            if obj.lane_overlap_info.start_s <= 1.0 and lanelet.predecessor:
                lanelets.update(lanelet.predecessor)
            else:
                lanelets.add(lanelet.lanelet_id)
    
    return sorted(lanelets)
 
def attach_stop_line(lanelet: Lanelet, stopline_points: np.ndarray, traffic_sign_id: int = None, traffic_light_id: int = None):
    """
    Stop line across the lanelet (the Apollo stop line cropped to its boundaries).
    Objects sharing the lanelet's stop line are all referenced
    """
    stopline = LineString(stopline_points)
    midpoint = stopline.interpolate(0.5, normalized = True)

    left_bound = LineString(lanelet.left_vertices)
    right_bound = LineString(lanelet.right_vertices)
    
    p_left = np.array(left_bound.interpolate(left_bound.project(midpoint)).coords[0])
    p_right = np.array(right_bound.interpolate(right_bound.project(midpoint)).coords[0])

    sign_refs = set(lanelet.stop_line.traffic_sign_ref or set()) if lanelet.stop_line else set()
    light_refs = set(lanelet.stop_line.traffic_light_ref or set()) if lanelet.stop_line else set()

    if traffic_sign_id is not None:
        sign_refs.add(traffic_sign_id)
    if traffic_light_id is not None:
        light_refs.add(traffic_light_id)

    lanelet.stop_line = StopLine(p_left, p_right, LineMarking.SOLID_SOLID, traffic_sign_ref = sign_refs or None, traffic_light_ref = light_refs or None)

# Apollo lane types from modules/map/proto/map_lane_pb2.py
APOLLO_TO_CR_TYPE = {
    "CITY_DRIVING": LaneletType.URBAN,
    "BIKING": LaneletType.BICYCLE_LANE,
    "SIDEWALK": LaneletType.SIDEWALK,
    "PARKING": LaneletType.PARKING,
    "SHOULDER": LaneletType.SHOULDER,
    "NONE": None
}

#
# ------------------------------------------------------------
#

# Set destination folder for CommonRoad converted maps
output_folder = "./converted_maps"
os.makedirs(output_folder, exist_ok = True)

# Set source folder for Baidu Apollo .bin maps
DT_maps = sorted(os.listdir("../data/maps/"))

# Iterate over available Baidu Apollo maps
for i, map in enumerate(DT_maps):
    # Build output file paths for CommonRoad converted map
    output_map_file_path = os.path.join(output_folder, f"{map}.xml")
    output_map_image_file_path = os.path.join(output_folder, f"{map}.png")
    output_map_image_with_labels_file_path = os.path.join(output_folder, f"{map}_with_labels.png")
    output_lane_ids_map_file_path = os.path.join(output_folder, f"{map}_lane_ids_map.json")
    output_signal_ids_map_file_path = os.path.join(output_folder, f"{map}_signals_ids_map.json")
    output_signal_relations_map_file_path = os.path.join(output_folder, f"{map}_signals_relations_map.json")
    output_lanes_not_in_junction_file_path = os.path.join(output_folder, f"{map}_lanes_not_in_junction.json")

    # Build source file path for Baidu Apollo .bin map
    map_file = os.path.join("../data/maps/", map, "base_map.bin")

    # Skip conversion if output file already exists or if source map file is missing
    if not os.path.exists(map_file) or os.path.exists(output_map_file_path):
        print(f"Skipping map {map} conversion (source file missing or output file already exists)")
        continue

    print(f"\n========== Start conversion of map {map} ==========")

    # Open .bin map file
    mp = MapParser(map_file)

    #
    # ----- Initialize scenario object -----
    #

    # Scenario Identifier
    scenario_id = ScenarioID(
        country_id = "DEU",
        map_name = map,
        map_id = (i + 1)
    )

    # Scenario object
    scenario = Scenario(
        dt = 0.1,
        scenario_id = scenario_id,
        author = "Fabio Cavaleri",
        tags = [Tag.URBAN],
        affiliation = "USI-AIT-NII",
        source = "BaiduApollo",
        location = None
    )

    # Initialize CommonRoad Lanelet Network object
    lanelet_network = LaneletNetwork()
    
    #
    # ----- Lane objects conversion -----
    #

    # Initialize lanelet ids map and counter
    # Use map to assign incremental identifiers to lanelets
    global lane_id_map
    lane_id_map = {}
    lane_id_counter = 1

    # Assign an id to each lanelet
    for lane_id in mp.get_lanes():
        lane_id_map[lane_id] = lane_id_counter
        lane_id_counter += 1

    # Save lane ids map to JSON
    with open(output_lane_ids_map_file_path, "w") as f:
        json.dump(lane_id_map, f, indent = 2)

    print("Lane ids map saved")

    # Iterate over Baidu Apollo map lanes
    for lane_id in mp.get_lanes():
        # Get lane object from map
        lane = mp.get_lane_by_id(lane_id)

        # Extract lane boundary points
        center = curve_to_points(lane.central_curve)
        left = curve_to_points(lane.left_boundary.curve)
        right = curve_to_points(lane.right_boundary.curve)
        
        # Extract predecessor and successor lane ids
        predecessors_id = get_lane_id(lane.predecessor_id)
        successors_id = get_lane_id(lane.successor_id)

        # Extract left and right adjacent lane ids
        right_adjacent_forward = get_lane_id(lane.right_neighbor_forward_lane_id)
        right_adjacent_reverse = get_lane_id(lane.right_neighbor_reverse_lane_id)
        left_adjacent_forward = get_lane_id(lane.left_neighbor_forward_lane_id)
        left_adjacent_reverse = get_lane_id(lane.left_neighbor_reverse_lane_id)

        # Get the non-empty lane id
        right_adjacent, right_adjacent_same_direction = adjacent_lanes(right_adjacent_forward, right_adjacent_reverse)
        left_adjacent, left_adjacent_same_direction = adjacent_lanes(left_adjacent_forward, left_adjacent_reverse)
        
        # Extract lanelet type and convert to CommonRoad lanelet type
        apollo_lane_type = lane.LaneType.Name(lane.type)
        cr_lane_type = APOLLO_TO_CR_TYPE.get(apollo_lane_type)
        
        # Retrieve assigned lanelet id
        lanelet_id = lane_id_map[lane_id]

        # Build CommonRoad lanelet Object
        left_vertices, right_vertices, center_vertices = lane_vertices(center, left, right)
        lanelet = Lanelet(
            left_vertices = left_vertices,
            right_vertices = right_vertices,
            center_vertices = center_vertices,
            lanelet_id = lanelet_id,
            adjacent_left = left_adjacent,
            adjacent_left_same_direction = left_adjacent_same_direction,
            adjacent_right = right_adjacent,
            adjacent_right_same_direction = right_adjacent_same_direction
        )

        # Set lanelet type
        if cr_lane_type is not None:
            lanelet.lanelet_type.add(cr_lane_type)

        # Link lanelet to eventual predecessor and successor lanelets
        lanelet = link_lanelets(lanelet, predecessors_id, successors_id)

        # Add lanelet object to scenario's lanelet network
        lanelet_network.add_lanelet(lanelet)

    # Add lanelet network to scenario
    scenario.add_objects(lanelet_network)

    #
    # ----- Traffic Signals Conversion -----
    #

    # Apollo's overlaps state exactly which lanes each stop sign and signal controls
    apollo_map = ApolloMap()
    with open(map_file, "rb") as f:
        apollo_map.ParseFromString(f.read())
    apollo_overlaps = {o.id.id: o for o in apollo_map.overlap}
    
    # Initialize traffic signs id map and relations map
    signals_id_map = {}
    signals_relations_map = {}

    # Iterate over all stop signs in the map
    for signal_id in mp.get_stop_signs():
        # Extract stop sign object from map
        signal = mp.get_stop_sign_by_id(signal_id)

        # Convert signal identifier to int
        # Add 5000 to avoid lanelet ids duplicates
        if "_" not in signal_id:
            traffic_sign_id = int(signal_id) + 5000
        else:
            traffic_sign_id = int(signal_id.split("_")[-1]) + 5000

        # Extract stop line points associated to the stop sign
        traffic_sign_stop_line = []
        for point in signal.stop_line[0].segment[0].line_segment.point:
            traffic_sign_stop_line.append([point.x, point.y])

        stopline_points = np.array(traffic_sign_stop_line)

        # Extract traffic sign position
        traffic_sign_position = signal.stop_line[0].segment[0].line_segment.point[0]

        # Retrieve the lanelet(s) associated to the stop line
        traffic_sign_lanelets = controlled_lanelets(signal) or get_lanelet_from_stopline(lanelet_network, stopline_points)

        # Stop line on each controlled lanelet
        for lanelet_id in traffic_sign_lanelets:
            attach_stop_line(scenario.lanelet_network.find_lanelet_by_id(lanelet_id), stopline_points, traffic_sign_id = traffic_sign_id)
 
        # Create CommonRoad traffic sign object and add it to the scenario
        ts = TrafficSign(
            traffic_sign_id,
            [TrafficSignElement(TrafficSignIDGermany.STOP)],
            traffic_sign_lanelets,
            np.array([traffic_sign_position.x, traffic_sign_position.y]),
        )

        scenario.add_objects(ts, traffic_sign_lanelets)
    #
    # ----- Traffic Lights Conversion -----
    #

    # Default rectangle size for traffic light (in meters)
    DEFAULT_WIDTH = 0.3
    DEFAULT_LENGTH = 0.3
    DEFAULT_HEADING = 0.0

    # Create basic traffic light cycle object
    # The real cycle will be defined when the Apollo executed scenario is converted
    cycle = TrafficLightCycle(
        cycle_elements=[
            TrafficLightCycleElement(
                state = TrafficLightState.RED,
                duration = 1000
            )
        ]
    )

    # Iterate over all traffic lights in the map
    for signal_id in mp.get_signals():
        # Extract traffic light object from map
        signal = mp.get_signal_by_id(signal_id)

        # Extract position of the signal
        x, y = signal.subsignal[0].location.x, signal.subsignal[0].location.y

        # Extract direction of the traffic light (ARROW_FORWARD, ARROW_LEFT, etc.)
        arrow_type = signal.subsignal[0].Type.Name(signal.subsignal[0].type)

        # Map Apollo arrow type to CommonRoad TrafficLightDirection
        # Default to ALL if unsure
        direction_mapping = {
            "ARROW_FORWARD": TrafficLightDirection.STRAIGHT,
            "ARROW_LEFT": TrafficLightDirection.LEFT,
            "ARROW_RIGHT": TrafficLightDirection.RIGHT,
            "ARROW_LEFT_AND_FORWARD": TrafficLightDirection.LEFT_STRAIGHT,
            "ARROW_RIGHT_AND_FORWARD": TrafficLightDirection.STRAIGHT_RIGHT,
        }
        direction = direction_mapping.get(arrow_type, TrafficLightDirection.ALL)

        # Build traffic light shape as a rectangle centered on the signal position
        traffic_light_shape = Rectangle(
            center = np.array([x, y]),
            width = DEFAULT_WIDTH,
            length = DEFAULT_LENGTH,
        )
        
        # Convert traffic light identifier to int
        # Add 10000 to avoid lanelet ids duplicates
        if '_' in signal_id:
            tl_id = int(signal_id.split("_")[1]) + 10000
        else:
            tl_id = int(signal_id) + 10000

        # Save traffic lights id mapping
        signals_id_map[signal_id] = tl_id

        # Create TrafficLight object with empty cycle and color
        tl = TrafficLight(
            traffic_light_id = tl_id,
            position = np.array([x, y]),
            direction = direction,
            traffic_light_cycle = cycle
        )

        # Get stopline associated to the traffic light signal
        stopline_points = []
        for sl in signal.stop_line:
            for point in sl.segment[0].line_segment.point:
                stopline_points.append([point.x, point.y])

        stopline_points = np.array(stopline_points)

        # Get lanelet(s) associated to the traffic light stop line
        traffic_light_lanelets = controlled_lanelets(signal) or get_lanelet_from_stopline(lanelet_network, stopline_points)

        # Add traffic light to scenario
        if not traffic_light_lanelets:
            print(f"Unable to find lanelet for traffic light {signal_id}")
        else:
            scenario.add_objects(tl, traffic_light_lanelets)

            # Stop line on each controlled lanelet (the stop line of the signal crossing that lanelet, if there are several)
            stop_lines = [np.array([[p.x, p.y] for p in sl.segment[0].line_segment.point]) for sl in signal.stop_line]
            for lanelet_id in traffic_light_lanelets:
                lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                end_point = Point(lanelet.center_vertices[-1])
                closest = min(stop_lines, key = lambda pts: LineString(pts).distance(end_point))
                attach_stop_line(lanelet, closest, traffic_light_id = tl_id)
        
        # Retrieve the traffic lights related to the current signal
        related_signals = mp.get_signals_wrt(signal_id)
        signals_relations_map[signal_id] = related_signals

    # Save traffic light ids mapping to JSON
    with open(output_signal_ids_map_file_path, "w") as f:
        json.dump(signals_id_map, f, indent = 2)

    # Save traffic light relations mapping to JSON
    with open(output_signal_relations_map_file_path, "w") as f:
        json.dump(signals_relations_map, f, indent = 2)
    
    print("Signal ids and relations maps saved")
        
    # Export CommonRoad converted scenario to .xml
    writer = CommonRoadFileWriter(
        scenario = scenario,
        planning_problem_set = PlanningProblemSet()
    )

    writer.write_to_file(output_map_file_path, OverwriteExistingFile.ALWAYS)

    #
    # ----- Extract lanes not in junction -----
    #

    # Extract lanelets not in junction and convert their ids to CommonRoad lanelet ids
    lanes_not_in_junction = list(mp.get_lanes_not_in_junction())
    converted_lanes_ids = [lane_id_map[lane_id] for lane_id in lanes_not_in_junction]

    # Save to JSON
    with open(output_lanes_not_in_junction_file_path, "w") as f:
        json.dump(converted_lanes_ids, f, indent = 2)

    print("Map lanes not in junction saved")

    #
    # ----- Compute map lanes reachable set -----
    #

    lanelets_reachable = {}
    feasible_paths = {}

    # Iterate over all lanes in the map
    for lane_id in mp.get_lanes():
        # Extract reachable paths from the lanelet
        reachable_paths = mp._MapParser__get_reachable_from(lane_id, 10)
        
        reachable_set = set()
        for path in reachable_paths:
            reachable_set.update(path)

        # Convert reachable lanelet ids to CommonRoad lanelet ids
        converted_reachable_lanes = [lane_id_map[l_id] for l_id in reachable_set]

        # Save converted raechable lanelets
        lanelets_reachable[lane_id_map[lane_id]] = converted_reachable_lanes
        
        converted_feasible_paths = [[lane_id_map[lane_id]]]
        for path in reachable_paths:
            converted_path = []
            for lane in path:
                converted_path.append(lane_id_map[lane])

            converted_feasible_paths.append(converted_path)

        feasible_paths[lane_id_map[lane_id]] = converted_feasible_paths


    # Save to JSON
    with open(os.path.join(output_folder, f"{map}_rechable_lanelets.json"), "w") as f:
        json.dump(lanelets_reachable, f, indent = 2)

    # Save to JSON
    with open(os.path.join(output_folder, f"{map}_feasible_paths.json"), "w") as f:
        json.dump(feasible_paths, f, indent = 2)

    print("Reachable lanes and feasible paths saved")

    #
    # ----- Plot CommonRoad converted map -----
    #

    fig, ax = plt.subplots()
    rnd = MPRenderer(ax = ax)

    rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
    rnd.draw_params.lanelet_network.traffic_sign.scale_factor = 0.7
    rnd.draw_params.lanelet_network.traffic_light.scale_factor = 0.7
    rnd.draw_params.lanelet_network.lanelet.stop_line_color = "black"

    # Plot the road and the signals
    scenario.lanelet_network.draw(rnd)
    
    rnd.render()

    plt.axis('off')
    plt.savefig(
        output_map_image_file_path,
        dpi = 1200,            # very high resolution
        bbox_inches = "tight", # trims extra whitespace around the figure
        pad_inches = 0,        # no padding
        transparent = False,   # keep background white (or True if you want transparent)
    )

    # ----- Plot lanelet IDs -----
    for lanelet in scenario.lanelet_network.lanelets:
        center_vertices = lanelet.center_vertices

        if center_vertices is None or len(center_vertices) == 0:
            continue

        # Use midpoint of centerline
        mid_idx = len(center_vertices) // 2
        x, y = center_vertices[mid_idx]

        ax.text(
            x, y,
            str(lanelet.lanelet_id),
            fontsize = 2,
            color = "black",
            ha = "center",
            va = "center",
            zorder = 100,
            bbox = dict(
                boxstyle = "round,pad=0.2",
                fc = "white",
                ec = "none",
                alpha = 0.5
            )
        )

    plt.savefig(
        output_map_image_with_labels_file_path,
        dpi = 1200,            # very high resolution
        bbox_inches = "tight", # trims extra whitespace around the figure
        pad_inches = 0,        # no padding
        transparent = False,   # keep background white (or True if you want transparent)
    )
    print(f"Map saved to {output_map_file_path}")

    """
    #
    # ----- Plot reachable lanes for each lanelet in the map -----
    #

    for lanelet_id, reachable_set in lanelets_reachable.items():
        fig, ax = plt.subplots()
        rnd = MPRenderer(ax=ax)

        rnd.draw_params.lanelet_network.traffic_light.draw_traffic_lights = False
        rnd.draw_params.lanelet_network.lanelet.draw_stop_line = False
        rnd.draw_params.lanelet_network.lanelet.draw_center_bound = False

        scenario.lanelet_network.draw(rnd)

        lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
    
        # Define the style for this specific lanelet
        # The key must match the renderer's expectation: 'lanelet'
        spec_params = ShapeParams()
        spec_params.facecolor = "red"
        spec_params.edgecolor = "black"
        spec_params.opacity = 0.5
        spec_params.zorder = 20
        spec_params.linewidth = 0.5
        spec_params.antialiased = True
        vertices = np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0)))
        # Call draw on the individual Lanelet instance
        rnd.draw_polygon(vertices, draw_params=spec_params)

        for id in reachable_set:
            if id == lanelet_id:
                continue
            lanelet = scenario.lanelet_network.find_lanelet_by_id(id)
    
            # Define the style for this specific lanelet
            # The key must match the renderer's expectation: 'lanelet'
            spec_params = ShapeParams()
            spec_params.facecolor = "blue"
            spec_params.edgecolor = "black"
            spec_params.opacity = 0.5
            spec_params.zorder = 20
            spec_params.linewidth = 0.5
            spec_params.antialiased = True
            vertices = np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0)))
            # Call draw on the individual Lanelet instance
            rnd.draw_polygon(vertices, draw_params=spec_params)
        
        rnd.render()
        plt.axis('off')

        os.makedirs(os.path.join(output_folder, f"{map}_reachable_lanes_plots"), exist_ok=True)
        output_map_lane_reachable_file_path = os.path.join(output_folder, f"{map}_reachable_lanes_plots", f"{lanelet_id}_rechable_lanelets.png")
        plt.savefig(
            output_map_lane_reachable_file_path,
            dpi = 600,            # very high resolution
            bbox_inches = "tight", # trims extra whitespace around the figure
            pad_inches = 0,        # no padding
            transparent = False,   # keep background white (or True if you want transparent)
        )
        plt.close(fig)
    """