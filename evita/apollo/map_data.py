"""
Correspondence between an Apollo HD map and its CommonRoad conversion.

The map converter (scripts/apollo/map_converter.py) writes three files next to the converted map <name>.xml:

    <name>_lane_ids_map.json          Apollo lane id -> CommonRoad lanelet id         {"lane_0": 1, ...}
    <name>_signals_ids_map.json       Apollo signal id -> CommonRoad traffic light id  {"signal_0": 10000, ...}
    <name>_signals_relations_map.json Apollo signal id -> [[other signal, "EQ" | "NE"], ...]
                                      EQ: the two signals always show the same color; NE: never green together

Everything that needs these correspondences (scenario conversion, traffic lights, record conversion) reads them
through ApolloMapData, from the map file that is actually in use.
"""

import os
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

from commonroad.scenario.lanelet import LaneletNetwork

SAME_COLOR = "EQ"
NEVER_BOTH_GREEN = "NE"

class MapDataError(Exception):
    """
    The id files of a converted map are missing or do not match the map
    """

@dataclass(frozen = True)
class ApolloMapData:
    """
    Id correspondences of one converted map
    """
    map_file: str
    lanelet_by_lane: Dict[str, int]
    lane_by_lanelet: Dict[int, str]
    traffic_light_by_signal: Dict[str, int]
    signal_by_traffic_light: Dict[int, str]
    relations_by_traffic_light: Dict[int, Tuple[Tuple[int, str], ...]]

    @staticmethod
    def files_of(map_file: str) -> Dict[str, str]:
        """
        Paths of the three id files of a converted map
        """
        base = map_file[:-len(".xml")] if map_file.endswith(".xml") else map_file
        return {"lanes": f"{base}_lane_ids_map.json",
                "signals": f"{base}_signals_ids_map.json",
                "relations": f"{base}_signals_relations_map.json"}

    @classmethod
    def load(cls, map_file: str) -> "ApolloMapData":
        """
        Read the id files of the map (cached: each map is read once per process)
        """
        return _load(os.path.abspath(map_file))

    # ---------- Lanes ----------

    def lanelet_id(self, lane_id: str) -> int:
        """
        CommonRoad lanelet of an Apollo lane
        """
        try:
            return self.lanelet_by_lane[lane_id]
        except KeyError:
            raise MapDataError(f"Apollo lane {lane_id} is not in {self.map_file}") from None

    def lane_id(self, lanelet_id: int) -> str:
        """
        Apollo lane of a CommonRoad lanelet
        """
        try:
            return self.lane_by_lanelet[int(lanelet_id)]
        except KeyError:
            raise MapDataError(f"Lanelet {lanelet_id} of {self.map_file} has no Apollo lane") from None

    # ---------- Traffic lights ----------

    def traffic_light_id(self, signal_id: str) -> int:
        """
        CommonRoad traffic light of an Apollo signal
        """
        try:
            return self.traffic_light_by_signal[signal_id]
        except KeyError:
            raise MapDataError(f"Apollo signal {signal_id} is not in {self.map_file}") from None

    def signal_id(self, traffic_light_id: int) -> str:
        """
        Apollo signal of a CommonRoad traffic light
        """
        try:
            return self.signal_by_traffic_light[int(traffic_light_id)]
        except KeyError:
            raise MapDataError(f"Traffic light {traffic_light_id} of {self.map_file} has no Apollo signal") from None

    def related_traffic_lights(self, traffic_light_id: int) -> Tuple[Tuple[int, str], ...]:
        """
        Traffic lights constrained by this one, each with its relation (SAME_COLOR or NEVER_BOTH_GREEN)
        """
        return self.relations_by_traffic_light.get(int(traffic_light_id), ())

    # ---------- Consistency ----------

    def check(self, lanelet_network: LaneletNetwork) -> None:
        """
        Raise MapDataError if the id files do not match the lanelet network (e.g. id files of another map version)
        """
        # Extract scenario lanelet and traffic light ids from the map
        lanelets = {lanelet.lanelet_id for lanelet in lanelet_network.lanelets}
        traffic_lights = {light.traffic_light_id for light in lanelet_network.traffic_lights}

        # Check that the id files contain all lanelets and traffic lights of the map, and that all traffic lights of the map have an Apollo signal
        missing_lanelets = sorted(set(self.lane_by_lanelet) - lanelets)
        missing_lights = sorted(set(self.signal_by_traffic_light) - traffic_lights)
        unmapped_lights = sorted(traffic_lights - set(self.signal_by_traffic_light))

        problems = []
        if missing_lanelets:
            problems.append(f"lanelets {missing_lanelets} are in the id files but not in the map")

        if missing_lights:
            problems.append(f"traffic lights {missing_lights} are in the id files but not in the map")
        
        if unmapped_lights:
            problems.append(f"traffic lights {unmapped_lights} of the map have no Apollo signal")
        
        if problems:
            raise MapDataError(f"{self.map_file}: " + "; ".join(problems))

@lru_cache(maxsize = None)
def _load(map_file: str) -> ApolloMapData:
    """
    Read the id files of the map (cached: each map is read once per process)
    """
    files = ApolloMapData.files_of(map_file)

    # Load the three id files of the map
    data = {}
    for name, path in files.items():
        try:
            with open(path, "r") as f:
                data[name] = json.load(f)
        except FileNotFoundError:
            raise MapDataError(f"Missing id file of {map_file}: {path}") from None

    # Extract the correspondences from the id files, checking that they are valid
    lanelet_by_lane = {lane: int(lanelet) for lane, lanelet in data["lanes"].items()}
    traffic_light_by_signal = {signal: int(light) for signal, light in data["signals"].items()}

    # Extract the relations between traffic lights, checking that they are valid
    relations: Dict[int, Tuple[Tuple[int, str], ...]] = {}
    for signal, related in data["relations"].items():
        if signal not in traffic_light_by_signal:
            raise MapDataError(f"{files['relations']}: signal {signal} is not in {files['signals']}")

        constrained: List[Tuple[int, str]] = []
        for other, relation in related:
            if relation not in (SAME_COLOR, NEVER_BOTH_GREEN) or other not in traffic_light_by_signal:
                raise MapDataError(f"{files['relations']}: invalid relation {signal} -> {other} ({relation})")
            
            constrained.append((traffic_light_by_signal[other], relation))

        relations[traffic_light_by_signal[signal]] = tuple(constrained)

    return ApolloMapData(
        map_file = map_file,
        lanelet_by_lane = lanelet_by_lane,
        lane_by_lanelet = {lanelet: lane for lane, lanelet in lanelet_by_lane.items()},
        traffic_light_by_signal = traffic_light_by_signal,
        signal_by_traffic_light = {light: signal for signal, light in traffic_light_by_signal.items()},
        relations_by_traffic_light = relations
    )