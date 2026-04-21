import os
import json
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from shapely.geometry import LineString

import sys
sys.path.append("../")

from hdmap.MapParser import MapParser

from commonroad.geometry.shape import Rectangle
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_writer import CommonRoadFileWriter
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
    # Build output file path for CommonRoad converted map
    output_map_file_path = os.path.join(output_folder, f"{map}.xml")
    output_map_image_file_path = os.path.join(output_folder, f"{map}.png")
    output_map_image_with_labels_file_path = os.path.join(output_folder, f"{map}_with_labels.png")
    output_lane_ids_map_file_path = os.path.join(output_folder, f"{map}_lane_ids_map.json")
    output_signal_ids_map_file_path = os.path.join(output_folder, f"{map}_signals_ids_map.json")
    output_signal_relations_map_file_path = os.path.join(output_folder, f"{map}_signals_relations_map.json")
    output_lanes_not_in_junction_file_path = os.path.join(output_folder, f"{map}_lanes_not_in_junction.json")

    # Build source file path for Baidu Apollo .bin map
    # map_file = os.path.join(MAPS_ROOT_FOLDER, map, "base_map.bin")
    map_file = os.path.join("../data/maps/", map, "base_map.bin")

    if not os.path.exists(map_file) or os.path.exists(output_map_file_path):
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
        source = "DoppelTest",
        location = None
    )

    # Lanelet Network
    lanelet_network = LaneletNetwork()
    
    #
    # ----- Lane objects conversion -----
    #

    # Initialize lanelet ids map and counter
    # Use map to assign incremental identifiers to lanelets
    global lane_id_map
    lane_id_map = {}
    lane_id_counter = 0

    # Assign id to each lanelet
    for lane_id in mp.get_lanes():
        lane_id_map[lane_id] = lane_id_counter
        lane_id_counter += 1

    # Save to JSON
    with open(output_lane_ids_map_file_path, "w") as f:
        json.dump(lane_id_map, f, indent = 2)

    print("Lane ids map saved")

    # Iterate over Baidu Apollo map lanes
    for lane_id in mp.get_lanes():
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
        
        # Extract lanelet type
        apollo_lane_type = lane.LaneType.Name(lane.type)
        cr_lane_type = APOLLO_TO_CR_TYPE.get(apollo_lane_type)
        
        # apollo_lane_direction = lane.LaneDirection.Name(lane.direction)
        # apollo_lane_turn = lane.LaneTurn.Name(lane.turn)

        # Retrive assigned lanelet id
        lanelet_id = lane_id_map[lane_id]

        # Build CommonRoad lanelet Object
        lanelet = Lanelet(
            left_vertices = np.array(left),
            right_vertices = np.array(right),
            center_vertices = np.array(center),
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
    
    # Default rectangle size for traffic light (in meters)
    DEFAULT_WIDTH = 0.3
    DEFAULT_LENGTH = 0.3
    DEFAULT_HEADING = 0.0

    # Create basic traffic light cycle object
    # The real cycle will be defined when the DoppelTest
    # scenario is converted
    cycle = TrafficLightCycle(
        cycle_elements=[
            TrafficLightCycleElement(
                state = TrafficLightState.RED,
                duration = 1000
            )
        ]
    )
    
    signals_id_map = {}
    signals_relations_map = {}

    # Iterate over all stop_signs in the map
    for signal_id in mp.get_stop_signs():
        # print(signal_id)
        
        signal = mp.get_stop_sign_by_id(signal_id)

        if "_" not in signal_id:
            traffic_sign_id = int(signal_id) + 5000
        else:
            traffic_sign_id = int(signal_id.split("_")[-1]) + 5000

        traffic_sign_stop_line = []
        for point in signal.stop_line[0].segment[0].line_segment.point:
            traffic_sign_stop_line.append([point.x, point.y])

        stopline_points = np.array(traffic_sign_stop_line)


        traffic_sign_position = signal.stop_line[0].segment[0].line_segment.point[0]

        traffic_sign_lanelets = get_lanelet_from_stopline(lanelet_network, stopline_points)

        for lanelet_id in traffic_sign_lanelets:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

            stopline_geom = LineString(stopline_points)
            
            midpoint = stopline_geom.interpolate(0.5, normalized=True)

            left_bound = LineString(lanelet.left_vertices)
            right_bound = LineString(lanelet.right_vertices)

            # project midpoint to lane boundaries
            p_left = left_bound.interpolate(left_bound.project(midpoint))
            p_right = right_bound.interpolate(right_bound.project(midpoint))

            cropped_stopline = np.array([
                np.array(p_left.coords[0]),
                np.array(p_right.coords[0])
            ])

            stop_line = StopLine(
                cropped_stopline[0],
                cropped_stopline[-1],
                LineMarking.SOLID_SOLID,
                traffic_sign_ref = set([traffic_sign_id])
            )

            lanelet.stop_line = stop_line
            lanelet.traffic_signs = set([traffic_sign_id])

        ts = TrafficSign(
            traffic_sign_id,
            [TrafficSignElement(TrafficSignIDGermany.STOP)],
            traffic_sign_lanelets,
            np.array([traffic_sign_position.x, traffic_sign_position.y]),
        )

        scenario.add_objects(ts, traffic_sign_lanelets)

    # Iterate over all traffic lights in the map
    for signal_id in mp.get_signals():
        # print(signal_id)
        signal = mp.get_signal_by_id(signal_id)

        # Position of the signal
        x, y = signal.subsignal[0].location.x, signal.subsignal[0].location.y

        # Direction of the signal (ARROW_FORWARD, ARROW_LEFT, etc.)
        arrow_type = signal.subsignal[0].type

        # Map Apollo arrow type to CommonRoad TrafficLightDirection
        # Default to ALL if unsure
        direction_mapping = {
            "ARROW_STRAIGHT": TrafficLightDirection.STRAIGHT,
            "ARROW_LEFT": TrafficLightDirection.LEFT,
            "ARROW_LEFT_STRAIGHT": TrafficLightDirection.LEFT_STRAIGHT,
            "ARROW_RIGHT": TrafficLightDirection.RIGHT,
            "ARROW_STRAIGHT_RIGHT": TrafficLightDirection.STRAIGHT_RIGHT,
            "ARROW_LEFT_RIGHT": TrafficLightDirection.LEFT_RIGHT,
        }
        direction = direction_mapping.get(arrow_type, TrafficLightDirection.ALL)

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

        # Get stopline associated to the lanelet
        # This allows to retrieve the lanelet where the signal is located
        # The lanelet is required to add the signal to the scenario lanelet network
        stopline_points = []

        for sl in signal.stop_line:
            for point in sl.segment[0].line_segment.point:
                stopline_points.append([point.x, point.y])

        stopline_points = np.array(stopline_points)

        traffic_light_lanelets = get_lanelet_from_stopline(lanelet_network, stopline_points)

        # Add to scenario
        if traffic_light_lanelets is None:
            print(f"Unable to find lanelet for traffic light {signal_id}")
        else:
            scenario.add_objects(tl, traffic_light_lanelets)

            for lanelet_id in traffic_light_lanelets:
                lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)

                stopline_geom = LineString(stopline_points)

                midpoint = stopline_geom.interpolate(0.5, normalized=True)

                left_bound = LineString(lanelet.left_vertices)
                right_bound = LineString(lanelet.right_vertices)

                # project midpoint to lane boundaries
                p_left = left_bound.interpolate(left_bound.project(midpoint))
                p_right = right_bound.interpolate(right_bound.project(midpoint))

                cropped_stopline = np.array([
                    np.array(p_left.coords[0]),
                    np.array(p_right.coords[0])
                ])

                stop_line = StopLine(
                    cropped_stopline[0],
                    cropped_stopline[-1],
                    LineMarking.SOLID_SOLID,
                    traffic_light_ref = set([tl_id])
                )

                lanelet.stop_line = stop_line
                # lanelet.traffic_lights = set([tl_id])
        
        # Retrieve the map traffic lights related to the current signal
        related_signals = mp.get_signals_wrt(signal_id)
        signals_relations_map[signal_id] = related_signals

    # Save to JSON
    with open(output_signal_ids_map_file_path, "w") as f:
        json.dump(signals_id_map, f, indent = 2)

    # Save to JSON
    with open(output_signal_relations_map_file_path, "w") as f:
        json.dump(signals_relations_map, f, indent = 2)
    
    print("Signal ids and relations maps saved")
        
    # Export CommonRoad converted scenario to .xml
    writer = CommonRoadFileWriter(
        scenario = scenario,
        planning_problem_set = PlanningProblemSet()
    )

    from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
    writer.write_to_file(output_map_file_path, OverwriteExistingFile.ALWAYS)

    #
    # ----- Extract lanes not in junction -----
    #

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
    for lane_id in mp.get_lanes():
        reachable_paths = mp._MapParser__get_reachable_from(lane_id, 10)
        
        reachable_set = set()
        for path in reachable_paths:
            reachable_set.update(path)

        converted_reachable_lanes = [lane_id_map[l_id] for l_id in reachable_set]

        lanelets_reachable[lane_id_map[lane_id]] = converted_reachable_lanes
        
        converted_feasible_paths = [[lane_id_map[lane_id]]]
        for path in reachable_paths:
            converted_path = []
            for lane in path:
                converted_path.append(lane_id_map[lane])

            converted_feasible_paths.append(converted_path)

        feasible_paths[lane_id_map[lane_id]] = converted_feasible_paths

    ''' 
    ===== Manual Approach =====
    
    # Import map file (I manually added some missing lanes link)
    scenario, _ = XMLFileReader(output_map_file_path).open(lanelet_assignment = True)
    
    lanelets_reachable = {}
    for current_lanelet in scenario.lanelet_network.lanelets:

        visited = set()
        queue = deque([current_lanelet.lanelet_id])

        while queue:
            current_id = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            lanelet = scenario.lanelet_network.find_lanelet_by_id(current_id)
            if lanelet is None:
                continue

            to_check = lanelet.successor

            if lanelet.adj_left_same_direction:
                to_check.append(lanelet.adj_left)
            if lanelet.adj_right_same_direction:
                to_check.append(lanelet.adj_right)

            for succ_id in to_check:
                if succ_id not in visited:
                    queue.append(succ_id)

        lanelets_reachable[current_lanelet.lanelet_id] = list(visited)
    '''

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
            fontsize=2,
            color="black",
            ha="center",
            va="center",
            zorder=100,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.5
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