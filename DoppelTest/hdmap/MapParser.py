from collections import defaultdict
from typing import List, Set, Tuple

import networkx as nx
import numpy as np
import logging

from shapely.geometry import LineString, Point
from shapely.ops import split

from ..modules.common.proto.geometry_pb2 import PointENU
from ..hdmap import load_hd_map
from ..modules.map.proto.map_pb2 import Map
from ..modules.map.proto.map_junction_pb2 import Junction
from ..modules.map.proto.map_lane_pb2 import Lane
from ..modules.map.proto.map_crosswalk_pb2 import Crosswalk
from ..modules.map.proto.map_signal_pb2 import Signal
from ..modules.map.proto.map_stop_sign_pb2 import StopSign
# Causes a Circular Import
# from ..framework.scenario.ad_agents import ADAgent

import math

from typing import Dict
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.planning.planning_problem import PlanningProblem, GoalRegion, InitialState

from commonroad_route_planner.lane_changing.lane_change_methods.method_interface import LaneChangeMethod

from cr_scenario_handler.utils.utils_coordinate_system import smooth_ref_path, extend_ref_path_both_ends
from evita.generators import utils as generation_utils
from evita.config import ScenarioGeneratorConfig

class MapParser:
    """
    Reimplementation of the original MapParser for CommonRoad.
    This is mostly a dutch-tape implementation, since CommonRoad has not yet full HD Map (they are getting there).
    Uses a cache to speed up the computation of the routes/scenarios

    This class has the same interface of the original implementation for compatibility reasons
    """
    
    # __map: Map
    __map: LaneletNetwork
    __junctions: dict
    __signals: dict
    __stop_signs: dict
    __lanes: Dict[int, Lanelet]
    __crosswalk: dict

    __signals_at_junction: dict
    __lanes_at_junction: dict
    __lanes_controlled_by_signal: dict

    __signal_relations: nx.Graph
    __lane_nx: nx.DiGraph

    __route_cache: dict

    __instance = None

    
    def __init__(self, filename: str) -> None:
        """
        Constructor
        """
        # self.__map = load_hd_map(filename)
        crfr = CommonRoadFileReader(filename)
        self.scenario, _ = crfr.open()
        self.__map = self.scenario.lanelet_network

        self.load_junctions()
        self.load_signals()
        self.load_stop_signs()
        self.load_lanes()
        self.load_crosswalks()

        self.parse_relations()
        self.parse_signal_relations()
        self.parse_lane_relations()

        # Book keep computed routes.
        # Tuple(Position, set(lanelets)) 
        self.__route_cache = dict()

        MapParser.__instance = self

    def clear_cache(self):
        """
        Clear the computed routes
        """
        self.__route_cache.clear()

    def lookup_cache(self, adagent):
        candidate_key = (adagent.start_s, frozenset(adagent.routing))
        if candidate_key in self.__route_cache:
            return self.__route_cache[candidate_key]
        else:
            return None
    
    def store_cache(self, adagent, route) -> None:
        candidate_key = (adagent.start_s, frozenset(adagent.routing))
        self.__route_cache[candidate_key] = route
    
    @staticmethod
    def get_instance():
        """
        Get the singleton instance of MapParser
        """
        assert not MapParser.__instance is None
        return MapParser.__instance

    def load_junctions(self):
        """
        Load junctions on the CommonRoad Map (they are colled intersections)
        TODO: Not considered for the moment. Are any overlapping of lanelets to be considered a junction? or juction is an intersection (3-way, 4-way, etc.)
        """
        self.__junctions = dict()
        # for junc in self.__map.intersections:
        #     self.__junctions[junc.intersection_id] = junc

    def load_signals(self):
        """
        Load traffic signals on the CommonRoad Map
        TODO: Not considered for the moment
        """
        self.__signals = dict()
        # for sig in self.__map.traffic_lights:
        #     self.__signals[sig.traffic_light_id] = sig

    def load_stop_signs(self):
        """
        Load stop signs on the CommonRoad Map
        TODO: Not considered for the moment
        """
        self.__stop_signs = dict()
        # for ss in self.__map.traffic_signs:
        #     self.__stop_signs[ss.traffic_sign_id] = ss

    def load_lanes(self):
        """
        Load lanes on the HD Map
        TODO: Are those lanes in a road or lanelets? 
        
        For us, those are lanelets

        """
        self.__lanes = dict()
        for l in self.__map.lanelets:
            self.__lanes[l.lanelet_id] = l

    def load_crosswalks(self):
        """
        Load crosswalks on the HD Map

        TODO: Not considered for the moment
        """
        self.__crosswalk = dict()
        # for cw in self.__map:
        #     self.__crosswalk[cw.id.id] = cw



    def parse_relations(self):
        """
        Parse relations between signals and junctions,
        lanes and junctions, and lanes and signals

        TODO: Not considered for the moment

        """
        # load signals at junction
        self.__signals_at_junction = defaultdict(lambda: list())
        for sigk, sigv in self.__signals.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(sigv, junv):
                    self.__signals_at_junction[junk].append(sigk)

        # load lanes at junction
        self.__lanes_at_junction = defaultdict(lambda: list())
        for lank, lanv in self.__lanes.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(lanv, junv):
                    self.__lanes_at_junction[junk].append(lank)

        # load lanes controlled by signal
        self.__lanes_controlled_by_signal = defaultdict(lambda: list())
        for junk, junv in self.__junctions.items():
            signal_ids = self.__signals_at_junction[junk]
            lane_ids = self.__lanes_at_junction[junk]
            for sid in signal_ids:
                for lid in lane_ids:
                    if self.__is_overlap(self.__signals[sid], self.__lanes[lid]):
                        self.__lanes_controlled_by_signal[sid].append(lid)

    def parse_signal_relations(self):
        """
        Analyze the relation between signals (e.g., signals that
        cannot be green at the same time)
        """
        g = nx.Graph()
        
        # TODO: Not sure what this should represent, but we do not allow Traffic Signal for the moment

        # for junk, junv in self.__junctions.items():
        #     signal_ids = self.__signals_at_junction[junk]
        #     for sid1 in signal_ids:
        #         g.add_node(sid1)
        #         for sid2 in signal_ids:
        #             if sid1 == sid2:
        #                 continue
        #             lg1 = self.__lanes_controlled_by_signal[sid1]
        #             lg2 = self.__lanes_controlled_by_signal[sid2]
        #             if lg1 == lg2:
        #                 g.add_edge(sid1, sid2, v='EQ')
        #             elif self.is_conflict_lanes(lg1, lg2):
        #                 g.add_edge(sid1, sid2, v='NE')
        self.__signal_relations = g

    def parse_lane_relations(self):
        """
        Analyze the relation between lanes (e.g., which lane is connected 
        to which lane)

        :note: the relation is supposed to be included in the HD Map
          via predecessor and successor relation, but experimentally
          we found HD Map may be buggy and leave out some information,
          causing Routing module to fail.
        """
        dg = nx.DiGraph()
        
        # Go over all the lanelets
        lane1: Lanelet
        for lane1_id, lane1 in self.__lanes.items():
            dg.add_node(lane1_id)

            
            # Go over all the other lanelets
            for lane2_id, lane2 in self.__lanes.items():
                
                if lane1_id == lane2_id:
                    continue

                # If lane 1 follows lane 2, add an edge lane 2 -> lane 1
                if lane2_id in lane1.predecessor:
                    dg.add_edge(lane2_id, lane1_id)

                # If lane 2 follows lane 1, add an edge lane 1 -> lane 2
                if lane2_id in lane1.successor:
                    dg.add_edge(lane1_id, lane2_id)
                # If lane 2 is on the left/right of lane 1 and they have the same direction, lane 1 -> lane 2
                if lane2_id == lane1.adj_left and lane1.adj_left_same_direction:
                    dg.add_edge(lane1_id, lane2_id)

                if lane2_id == lane1.adj_right and lane1.adj_right_same_direction:
                    dg.add_edge(lane1_id, lane2_id)

                # line1 = self.get_lane_central_curve(lane1)
                # line2 = self.get_lane_central_curve(lane2)

                # s1, e1 = Point(line1.coords[0]), Point(line1.coords[-1])
                # s2, e2 = Point(line2.coords[0]), Point(line2.coords[-1])

                # if s1.distance(e2) < 0.001:
                #     dg.add_edge(lane2, lane1)
                # elif e1.distance(s2) < 0.001:
                #     dg.add_edge(lane1, lane2)


        self.__lane_nx = dg




    def __is_overlap(self, obj1, obj2):
        """
        Check if 2 objects (e.g., lanes, junctions) have overlap

        :param any obj1: left hand side
        :param any obj2: right hand side
        """
        # Make sure we do not use it anywhere
        raise NotImplementedError()
        # oid1 = set([x.id for x in obj1.overlap_id])
        # oid2 = set([x.id for x in obj2.overlap_id])
        # return oid1 & oid2 != set()

    def get_signals_wrt(self, signal_id: str) -> List[Tuple[str, str]]:
        """
        Get signals that have constraint with the specified signal

        :param str signal_id: ID of the signal interested in

        :returns: list of tuple each indicates the signal and the constraint
        :rtype: List[Tuple[str, str]]

        :example: ``[('signal_5', 'EQ'), ('signal_6', 'NE')]``
          indicates ``signal_5`` should have the same color, ``signal_6`` 
          cannot be green if the signal passed in is green.
        """
        result = list()
        for u, v, data in self.__signal_relations.edges(signal_id, data=True):
            result.append((v, data['v']))
        return result

    def _compute_reference_path(self, list_of_lanelet_ids: list):
        # Start of the lanelet
        # 1.0 inside the lanelet, in the middle
        initial_position = self.__map.find_lanelet_by_id(list_of_lanelet_ids[0]).interpolate_position(1.0)[0]
        orientation = self.__map.find_lanelet_by_id(list_of_lanelet_ids[0]).orientation_by_position(initial_position)
        initial_state = InitialState(time_step=0, position=initial_position, orientation=orientation, velocity=0.0, acceleration=0.0, yaw_rate=0.0, slip_angle=0.0)
        config = ScenarioGeneratorConfig()
        
        goal_lanelet = self.__map.find_lanelet_by_id(list_of_lanelet_ids[-1])
        goal_area = generation_utils.get_possible_goal_for_lanelet(config, goal_lanelet)
        goal_region = generation_utils.generate_goal(config, goal_area)
        planning_problem = PlanningProblem(1, initial_state, goal_region)

        # Setting Extended Search to True ensures that a solution is possible, however, it takes a lot to compute it...
        route_planner = RoutePlanner(scenario=self.scenario, planning_problem=planning_problem, extended_search=True, logging_level=logging.CRITICAL)
        # Find the route corresponding including the given lanelets
        # target_routes, _ = route_planner.plan_routes().retrieve_all_routes()
        # for target_route in target_routes:
        #     if target_route.lanelet_ids == list_of_lanelet_ids:
        #         reference_path = extend_ref_path_both_ends(target_route.reference_path)
        #         reference_path = smooth_ref_path(reference_path)
        #         return reference_path
        # raise Exception(f"Cannot find a reference path for {list_of_lanelet_ids}")

        target_route = route_planner.plan_routes(lane_change_method=LaneChangeMethod.CUBIC_SPLINE).retrieve_first_route(list_of_lanelet_ids)
        assert target_route, (f"Cannot find a reference path for {list_of_lanelet_ids}")
        reference_path = extend_ref_path_both_ends(target_route.reference_path)
        reference_path = smooth_ref_path(reference_path)
        return reference_path

    def _cut_from_initial_position(self, initial_lanelet, distance_from_beginning, reference_path):
        # https://gis.stackexchange.com/questions/428955/shapely-split-function-return-the-line-to-split
        # Get the position of the agent on the center of the lane
        initial_position_of_agent = self.__map.find_lanelet_by_id(initial_lanelet).interpolate_position(distance_from_beginning)[0]
        
        initial_point = Point(initial_position_of_agent)
        reference_line = LineString(reference_path)
        
        # The reference path is smoothed, so it might not pass on the center of the lanlet, so we need to project the point on it before cutting
        projected_position = reference_line.interpolate(reference_line.project(initial_point))
        # Floating point errors might prevent splitting the linestring 
        buffer = projected_position.buffer(0.0001)
        ### Split the line on the buffer
        segments = split(reference_line, buffer)
        ### Stitch together the first segment, the interpolated point, and the last segment 
        
        line = LineString(list(segments.geoms[0].coords) + list(projected_position.coords) + list(segments.geoms[-1].coords))
        # Do the split
        res = split(line,projected_position)
        
        # Return the "second piece of the reference_path" as numpy array
        cut_reference_line = res.geoms[1]  # return point
        cut_reference_path = np.array(cut_reference_line.coords)
        # DEBUG: Plot this !!
        def debug_plot():
            import matplotlib.pyplot as plt

            plt.axis('equal')
            plt.plot([ p[0] for p in reference_path], [p[1] for p in reference_path], "-.", color = 'red')
            plt.plot(initial_position_of_agent[0], initial_position_of_agent[1], "o", color = "red")
            plt.plot(projected_position.x, projected_position.y, "*", color = 'blue')
            plt.plot([ p[0] for p in cut_reference_path], [p[1] for p in cut_reference_path], "-.", color = 'green')
            plt.show()
        
        # TODO Enable this only during debug
        # debug_plot()
        

        return cut_reference_path

    # def is_conflict_lanes(self, ad1: "ADAgent", ad2: "ADAgent") -> bool:
    def is_conflict_lanes(self, ad1, ad2) -> bool:
        """
        Check if 2 groups of lanes intersect with each other
        
        
        EVITA: This method checks whether a Route build with lane_id1 will crosses the one from the otherlane_id2.

        :param List[str] lane_id1: list of lane ids
        :param List[str] lane_id2: another list of lane ids

        :returns: True if at least 1 lane from lhs intersects with another from rhs,
            False otherwise.
        :rtype: bool
        """
        lane_id1: List[str]
        lane_id1 = ad1.routing

        lane_id2: List[str]
        lane_id2 = ad2.routing

        # This approach is more accurate, as it computes the actual path followed by the agent from its initial position
        # However, it is complex and time consuming, especially when the agent needs to cross multiple lanes to get to the goal area
        # Thefore we adopt a simplified approach. If the routing, i.e., list of lanelets, have a common segment, they count as conflicting
        
        # Complex/accurate
        # # Build the Route for the lane_id1
        # reference_path_1 = self.lookup_cache(ad1)
        # reference_path_1 = reference_path_1 if reference_path_1 is not None else self._compute_reference_path(lane_id1)
        # self.store_cache(ad1, reference_path_1)

        # # Build the Route for the lane_id2
        # reference_path_2 = self.lookup_cache(ad2)
        # reference_path_2 = reference_path_2 if reference_path_2 is not None else self._compute_reference_path(lane_id2)
        # self.store_cache(ad2, reference_path_2)

        # # Cut the reference path from the initial position of the vehicle, expressed as distance from the beginning of the reference path?
        # reduced_path_1 = self._cut_from_initial_position(ad1.routing[0], ad1.start_s, reference_path_1)

        # reduced_path_2 = self._cut_from_initial_position(ad2.routing[0], ad2.start_s, reference_path_2)
       
        # DEBUG: Plot this !!
        # def debug_plot():
        #     import matplotlib.pyplot as plt

        #     plt.axis('equal')
        #     plt.plot([ p[0] for p in reference_path_1], [p[1] for p in reference_path_1], "-.", color = 'red')
        #     plt.plot([ p[0] for p in reference_path_2], [p[1] for p in reference_path_2], "-.", color = 'green')
            
        #     plt.plot([ p[0] for p in reduced_path_1], [p[1] for p in reduced_path_1], color = 'blue')
        #     plt.plot([ p[0] for p in reduced_path_2], [p[1] for p in reduced_path_2], color = 'black')
        #     plt.show()

        # return LineString(reduced_path_1).intersects(LineString(reduced_path_2))

        return len(set(lane_id1).intersection(set(lane_id2))) > 0
        # Original code
        # for lid1 in lane_id1:
        #     for lid2 in lane_id2:
        #         if lid1 == lid2:
        #             continue
        #         l1: Lanelet
        #         l1 = self.__map.find_lanelet_by_id(lid1)

        #         l2: Lanelet
        #         l2 = self.__map.find_lanelet_by_id(lid2)

        #         # lane1 = self.get_lane_central_curve(lid1)
        #         # lane2 = self.get_lane_central_curve(lid2)
                
        #         # if the lanes poligons overlap, not just their central line!
                # if l1.polygon.intersects(l2.polygon):
                    # return True
       # return False

    def get_lane_central_curve(self, lane_id: str) -> LineString:
        """
        Gets the central curve of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: an object representing the lane's central curve
        :rypte: LineString
        """
        lane = self.__lanes[lane_id]
        # points = lane.central_curve.segment[0].line_segment
        points = lane.center_vertices
        line = LineString(points)
        
        return line

    def get_lane_length(self, lane_id: str) -> float:
        """
        Gets the length of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: length of the lane
        :rtype: float
        """
        return self.get_lane_central_curve(lane_id).length

    def get_coordinate_and_heading(self, lane_id: str, s: float) -> Tuple[PointENU, float]:
        """
        Given a lane_id and a point on the lane, get the actual coordinate and the heading
        at that point.

        :param str lane_id: ID of the lane intereted in
        :param float s: meters away from the start of the lane

        :returns: coordinate and heading in a tuple
        :rtype: Tuple[PointENU, float]
        """
        lst = self.get_lane_central_curve(lane_id)
        ip = lst.interpolate(s)

        segments = list(map(LineString, zip(lst.coords[:-1], lst.coords[1:])))
        segments.sort(key=lambda x: ip.distance(x))
        line = segments[0]
        x1, x2 = line.xy[0]
        y1, y2 = line.xy[1]

        # ALESSIO: not sure what Datastructure is this one
        return (PointENU(x=ip.x, y=ip.y), math.atan2(y2-y1, x2-x1))

    def get_junctions(self) -> List[str]:
        """
        Get a list of all junction IDs on the HD Map

        :returns: list of junction IDs
        :rtype: List[str]
        """
        return list(self.__junctions.keys())

    def get_junction_by_id(self, j_id: str) -> Junction:
        """
        Get a specific junction object based on ID

        :param str j_id: ID of the junction interested in

        :returns: junction object
        :rtype: Junction
        """
        return self.__junctions[j_id]

    def get_lanes(self) -> List[str]:
        """
        Get a list of all lane IDs on the HD Map

        :returns: list of lane IDs
        :rtype: List[str]
        """
        return list(self.__lanes.keys())

    def get_lane_by_id(self, l_id: str) -> Lane:
        """
        Get a specific junction object based on ID

        :param str l_id: ID of the lane interested in

        :returns: lane object
        :rtype: Lane
        """
        return self.__lanes[l_id]

    def get_crosswalks(self) -> List[str]:
        """
        Get a list of all crosswalk IDs on the HD Map

        :returns: list of crosswalk IDs
        :rtype: List[str]
        """
        return list(self.__crosswalk.keys())

    def get_crosswalk_by_id(self, cw_id: str) -> Crosswalk:
        """
        Get a specific crosswalk object based on ID

        :param str cw_id: ID of the crosswalk interested in

        :returns: crosswalk object
        :rtype: Crosswalk
        """
        return self.__crosswalk[cw_id]

    def get_signals(self) -> List[str]:
        """
        Get a list of all signal IDs on the HD Map

        :returns: list of signal IDs
        :rtype: List[str]
        """
        return list(self.__signals.keys())

    def get_signal_by_id(self, s_id: str) -> Signal:
        """
        Get a specific signal object based on ID

        :param str s_id: ID of the signal interested in

        :returns: signal object
        :rtype: Signal
        """
        return self.__signals[s_id]

    def get_stop_signs(self) -> List[str]:
        """
        Get a list of all stop sign IDs on the HD Map

        :returns: list of stop sign IDs
        :rtype: List[str]
        """
        return list(self.__stop_signs.keys())

    def get_stop_sign_by_id(self, ss_id: str) -> StopSign:
        """
        Get a specific stop sign object based on ID

        :param str ss_id: ID of the stop sign interested in

        :returns: stop sign object
        :rtype: StopSign
        """
        return self.__stop_signs[ss_id]

    def get_lanes_not_in_junction(self) -> Set[str]:
        """
        Get the set of all lanes that are not in the junction.

        :returns: ID of lanes who is not in a junction
        :rtype: Set[str]
        """
        lanes = set(self.get_lanes())
        for junc in self.__lanes_at_junction:
            jlanes = set(self.__lanes_at_junction[junc])
            lanes = lanes - jlanes
        return lanes

    def get_path_from(self, lane_id: str) -> List[List[str]]:
        """
        Get possible paths starting from a specified lane

        EVITA: Does not handle cycles, so we need to limit ourselves to "simple" paths

        :param str lane_id: ID of the starting lane

        :returns: list of possible paths, each consisten multiple lanes
        :rtype: List[List[str]]
        """
        target_lanes = self.get_lanes_not_in_junction()

        # Target lanes must be outgoing lanes
        outgoing_lanes = [ l_id for l_id in target_lanes if len(self.__lanes[l_id].successor) == 0 ]

        # TODO Well a lane is always reacheable from itself!

        reachable = self.__get_reachable_from(lane_id)
        all_paths = [p for p in reachable if p[-1] in outgoing_lanes]
        simple_paths = [ p for p in all_paths if sorted(list(set(p))) == sorted(p)]
        
        return simple_paths

    def __get_reachable_from(self, lane_id: str, depth=5) -> List[List[str]]:
        """
        Recursive method to compute paths no more than ``depth`` lanes from
        the starting lane.

        EVITA: Does not handle cycles, so we need to limit ourselves to "simple" paths

        :param str lane_id: ID of the starting lane
        :param int depth: maximum number of lanes traveled

        :returns: list of possible paths
        :rtype: List[List[str]]
        """
        if depth == 1:
            return [[lane_id]]
        result = list()
        for u, v in self.__lane_nx.edges(lane_id):
            result.append([u, v])
            for rp in self.__get_reachable_from(v, depth-1):
                result.append([u] + rp)
        return result


class ApolloMapParser:
    """
    Class to load and parse HD Map

    :param str filename: filename of the HD Map
    """

    __map: Map
    __junctions: dict
    __signals: dict
    __stop_signs: dict
    __lanes: dict
    __crosswalk: dict

    __signals_at_junction: dict
    __lanes_at_junction: dict
    __lanes_controlled_by_signal: dict

    __signal_relations: nx.Graph
    __lane_nx: nx.DiGraph

    __instance = None

    def __init__(self, filename: str) -> None:
        """
        Constructor
        """
        self.__map = load_hd_map(filename)
        self.load_junctions()
        self.load_signals()
        self.load_stop_signs()
        self.load_lanes()
        self.load_crosswalks()
        self.parse_relations()
        self.parse_signal_relations()
        self.parse_lane_relations()
        MapParser.__instance = self

    @staticmethod
    def get_instance():
        """
        Get the singleton instance of MapParser
        """
        assert not MapParser.__instance is None
        return MapParser.__instance

    def load_junctions(self):
        """
        Load junctions on the HD Map
        """
        self.__junctions = dict()
        for junc in self.__map.junction:
            self.__junctions[junc.id.id] = junc

    def load_signals(self):
        """
        Load traffic signals on the HD Map
        """
        self.__signals = dict()
        for sig in self.__map.signal:
            self.__signals[sig.id.id] = sig

    def load_stop_signs(self):
        """
        Load stop signs on the HD Map
        """
        self.__stop_signs = dict()
        for ss in self.__map.stop_sign:
            self.__stop_signs[ss.id.id] = ss

    def load_lanes(self):
        """
        Load lanes on the HD Map
        """
        self.__lanes = dict()
        for l in self.__map.lane:
            self.__lanes[l.id.id] = l

    def load_crosswalks(self):
        """
        Load crosswalks on the HD Map
        """
        self.__crosswalk = dict()
        for cw in self.__map.crosswalk:
            self.__crosswalk[cw.id.id] = cw

    def parse_relations(self):
        """
        Parse relations between signals and junctions,
        lanes and junctions, and lanes and signals
        """
        # load signals at junction
        self.__signals_at_junction = defaultdict(lambda: list())
        for sigk, sigv in self.__signals.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(sigv, junv):
                    self.__signals_at_junction[junk].append(sigk)

        # load lanes at junction
        self.__lanes_at_junction = defaultdict(lambda: list())
        for lank, lanv in self.__lanes.items():
            for junk, junv in self.__junctions.items():
                if self.__is_overlap(lanv, junv):
                    self.__lanes_at_junction[junk].append(lank)

        # load lanes controlled by signal
        self.__lanes_controlled_by_signal = defaultdict(lambda: list())
        for junk, junv in self.__junctions.items():
            signal_ids = self.__signals_at_junction[junk]
            lane_ids = self.__lanes_at_junction[junk]
            for sid in signal_ids:
                for lid in lane_ids:
                    if self.__is_overlap(self.__signals[sid], self.__lanes[lid]):
                        self.__lanes_controlled_by_signal[sid].append(lid)

    def parse_signal_relations(self):
        """
        Analyze the relation between signals (e.g., signals that
        cannot be green at the same time)
        """
        g = nx.Graph()
        for junk, junv in self.__junctions.items():
            signal_ids = self.__signals_at_junction[junk]
            for sid1 in signal_ids:
                g.add_node(sid1)
                for sid2 in signal_ids:
                    if sid1 == sid2:
                        continue
                    lg1 = self.__lanes_controlled_by_signal[sid1]
                    lg2 = self.__lanes_controlled_by_signal[sid2]
                    if lg1 == lg2:
                        g.add_edge(sid1, sid2, v='EQ')
                    elif self.is_conflict_lanes(lg1, lg2):
                        g.add_edge(sid1, sid2, v='NE')
        self.__signal_relations = g

    def parse_lane_relations(self):
        """
        Analyze the relation between lanes (e.g., which lane is connected 
        to which lane)

        :note: the relation is supposed to be included in the HD Map
          via predecessor and successor relation, but experimentally
          we found HD Map may be buggy and leave out some information,
          causing Routing module to fail.
        """
        dg = nx.DiGraph()
        for lane1 in self.__lanes:
            dg.add_node(lane1)
            for lane2 in self.__lanes:
                if lane1 == lane2:
                    continue
                line1 = self.get_lane_central_curve(lane1)
                line2 = self.get_lane_central_curve(lane2)
                s1, e1 = Point(line1.coords[0]), Point(line1.coords[-1])
                s2, e2 = Point(line2.coords[0]), Point(line2.coords[-1])

                if s1.distance(e2) < 0.001:
                    dg.add_edge(lane2, lane1)
                elif e1.distance(s2) < 0.001:
                    dg.add_edge(lane1, lane2)
        self.__lane_nx = dg

    def __is_overlap(self, obj1, obj2):
        """
        Check if 2 objects (e.g., lanes, junctions) have overlap

        :param any obj1: left hand side
        :param any obj2: right hand side
        """
        oid1 = set([x.id for x in obj1.overlap_id])
        oid2 = set([x.id for x in obj2.overlap_id])
        return oid1 & oid2 != set()

    def get_signals_wrt(self, signal_id: str) -> List[Tuple[str, str]]:
        """
        Get signals that have constraint with the specified signal

        :param str signal_id: ID of the signal interested in

        :returns: list of tuple each indicates the signal and the constraint
        :rtype: List[Tuple[str, str]]

        :example: ``[('signal_5', 'EQ'), ('signal_6', 'NE')]``
          indicates ``signal_5`` should have the same color, ``signal_6`` 
          cannot be green if the signal passed in is green.
        """
        result = list()
        for u, v, data in self.__signal_relations.edges(signal_id, data=True):
            result.append((v, data['v']))
        return result

    def is_conflict_lanes(self, lane_id1: List[str], lane_id2: List[str]) -> bool:
        """
        Check if 2 groups of lanes intersect with each other

        :param List[str] lane_id1: list of lane ids
        :param List[str] lane_id2: another list of lane ids

        :returns: True if at least 1 lane from lhs intersects with another from rhs,
            False otherwise.
        :rtype: bool
        """
        for lid1 in lane_id1:
            for lid2 in lane_id2:
                if lid1 == lid2:
                    continue
                lane1 = self.get_lane_central_curve(lid1)
                lane2 = self.get_lane_central_curve(lid2)
                if lane1.intersects(lane2):
                    return True
        return False

    def get_lane_central_curve(self, lane_id: str) -> LineString:
        """
        Gets the central curve of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: an object representing the lane's central curve
        :rypte: LineString
        """
        lane = self.__lanes[lane_id]
        points = lane.central_curve.segment[0].line_segment
        line = LineString([[x.x, x.y] for x in points.point])
        return line

    def get_lane_length(self, lane_id: str) -> float:
        """
        Gets the length of the lane.

        :param str lane_id: ID of the lane interested in

        :returns: length of the lane
        :rtype: float
        """
        return self.get_lane_central_curve(lane_id).length

    def get_coordinate_and_heading(self, lane_id: str, s: float) -> Tuple[PointENU, float]:
        """
        Given a lane_id and a point on the lane, get the actual coordinate and the heading
        at that point.

        :param str lane_id: ID of the lane intereted in
        :param float s: meters away from the start of the lane

        :returns: coordinate and heading in a tuple
        :rtype: Tuple[PointENU, float]
        """
        lst = self.get_lane_central_curve(lane_id)
        ip = lst.interpolate(s)

        segments = list(map(LineString, zip(lst.coords[:-1], lst.coords[1:])))
        segments.sort(key=lambda x: ip.distance(x))
        line = segments[0]
        x1, x2 = line.xy[0]
        y1, y2 = line.xy[1]

        return (PointENU(x=ip.x, y=ip.y), math.atan2(y2-y1, x2-x1))

    def get_junctions(self) -> List[str]:
        """
        Get a list of all junction IDs on the HD Map

        :returns: list of junction IDs
        :rtype: List[str]
        """
        return list(self.__junctions.keys())

    def get_junction_by_id(self, j_id: str) -> Junction:
        """
        Get a specific junction object based on ID

        :param str j_id: ID of the junction interested in

        :returns: junction object
        :rtype: Junction
        """
        return self.__junctions[j_id]

    def get_lanes(self) -> List[str]:
        """
        Get a list of all lane IDs on the HD Map

        :returns: list of lane IDs
        :rtype: List[str]
        """
        return list(self.__lanes.keys())

    def get_lane_by_id(self, l_id: str) -> Lane:
        """
        Get a specific junction object based on ID

        :param str l_id: ID of the lane interested in

        :returns: lane object
        :rtype: Lane
        """
        return self.__lanes[l_id]

    def get_crosswalks(self) -> List[str]:
        """
        Get a list of all crosswalk IDs on the HD Map

        :returns: list of crosswalk IDs
        :rtype: List[str]
        """
        return list(self.__crosswalk.keys())

    def get_crosswalk_by_id(self, cw_id: str) -> Crosswalk:
        """
        Get a specific crosswalk object based on ID

        :param str cw_id: ID of the crosswalk interested in

        :returns: crosswalk object
        :rtype: Crosswalk
        """
        return self.__crosswalk[cw_id]

    def get_signals(self) -> List[str]:
        """
        Get a list of all signal IDs on the HD Map

        :returns: list of signal IDs
        :rtype: List[str]
        """
        return list(self.__signals.keys())

    def get_signal_by_id(self, s_id: str) -> Signal:
        """
        Get a specific signal object based on ID

        :param str s_id: ID of the signal interested in

        :returns: signal object
        :rtype: Signal
        """
        return self.__signals[s_id]

    def get_stop_signs(self) -> List[str]:
        """
        Get a list of all stop sign IDs on the HD Map

        :returns: list of stop sign IDs
        :rtype: List[str]
        """
        return list(self.__stop_signs.keys())

    def get_stop_sign_by_id(self, ss_id: str) -> StopSign:
        """
        Get a specific stop sign object based on ID

        :param str ss_id: ID of the stop sign interested in

        :returns: stop sign object
        :rtype: StopSign
        """
        return self.__stop_signs[ss_id]

    def get_lanes_not_in_junction(self) -> Set[str]:
        """
        Get the set of all lanes that are not in the junction.

        :returns: ID of lanes who is not in a junction
        :rtype: Set[str]
        """
        lanes = set(self.get_lanes())
        for junc in self.__lanes_at_junction:
            jlanes = set(self.__lanes_at_junction[junc])
            lanes = lanes - jlanes
        return lanes

    def get_path_from(self, lane_id: str) -> List[List[str]]:
        """
        Get possible paths starting from a specified lane

        :param str lane_id: ID of the starting lane

        :returns: list of possible paths, each consisten multiple lanes
        :rtype: List[List[str]]
        """
        target_lanes = self.get_lanes_not_in_junction()
        reachable = self.__get_reachable_from(lane_id)
        return [p for p in reachable if p[-1] in target_lanes]

    def __get_reachable_from(self, lane_id: str, depth=5) -> List[List[str]]:
        """
        Recursive method to compute paths no more than ``depth`` lanes from
        the starting lane.

        :param str lane_id: ID of the starting lane
        :param int depth: maximum number of lanes traveled

        :returns: list of possible paths
        :rtype: List[List[str]]
        """
        if depth == 1:
            return [[lane_id]]
        result = list()
        for u, v in self.__lane_nx.edges(lane_id):
            result.append([u, v])
            for rp in self.__get_reachable_from(v, depth-1):
                result.append([u] + rp)
        return result
