"""
Segmentation of a driven trajectory into straight parts and turns, used by the DoppelTest generator to count a vehicle's turns ("local decisions").

Adapted from DeepAtash:
https://github.com/testingautomated-usi/deepatash/blob/96ea0151f508b9155ee62fdeeb9760431f6b867b/BNG/utils.py#L121
"""
from typing import Iterable, Iterator, List, Tuple

import numpy as np

# A turn shorter than this many points is merged into the neighbouring segment
SEGMENT_THRESHOLD = 15
# A straight part shorter than this many points is merged into the neighbouring turn
SEGMENT_THRESHOLD2 = 10
# Minimum heading change (radians) between consecutive steps to count as turning
ANGLE_THRESHOLD = 0.005

def _calc_angle_distance(v0: np.ndarray, v1: np.ndarray) -> float:
    """
    Heading change (radians, not normalised) from direction v0 to direction v1.
    """
    return np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])

def _calc_dist_angle(points: List[np.ndarray]) -> List[Tuple[float, float, List[np.ndarray]]]:
    """
    For each step between consecutive points: (heading change from the previous step, step length, [end, start]).
    """
    def vector(idx: int) -> np.ndarray:
        return np.subtract(points[idx + 1], points[idx])

    result = []
    previous = vector(0)
    for i in range(len(points) - 1):
        current = vector(i)
        result.append((_calc_angle_distance(previous, current), np.linalg.norm(current), [points[i + 1], points[i]]))
        previous = current

    return result

def _group_equal(labels: Iterable[str]) -> Iterator[List[str]]:
    """
    Group consecutive equal labels ("l", "r", "s").
    """
    previous, group = None, []
    for label in labels:
        if previous is None or label == previous:
            group.append(label)
        else:
            yield group
            group = [label]
        previous = label
        
    if group:
        yield group

def _merge_short_straights_and_turns(groups: Iterable[List[str]]) -> Iterator[List[str]]:
    """
    Merge into the previous group: straight parts shorter than SEGMENT_THRESHOLD2 points, and turns shorter than
    SEGMENT_THRESHOLD points in the same direction as the previous group.
    """
    previous, group = None, []
    for item in groups:
        if previous is None:
            group.extend(item)
        elif len(item) < SEGMENT_THRESHOLD2 and item[0] == "s":
            item = [previous[-1]] * len(item)
            group.extend(item)
        elif len(item) < SEGMENT_THRESHOLD and item[0] != "s" and previous[-1] == item[0]:
            item = [previous[-1]] * len(item)
            group.extend(item)
        else:
            yield group
            group = item
        previous = item

    if group:
        yield group

def _merge_short_groups(groups: Iterable[List[str]]) -> Iterator[List[str]]:
    """
    Merge into the previous group every group shorter than SEGMENT_THRESHOLD points.
    """
    previous, group = None, []
    for item in groups:
        if previous is None:
            group.extend(item)
        elif len(item) < SEGMENT_THRESHOLD:
            item = [previous[-1]] * len(item)
            group.extend(item)
        else:
            yield group
            group = item
        previous = item

    if group:
        yield group

def identify_segments(agent) -> Tuple[int, List[List]]:
    """
    Split the trajectory driven by `agent` (its recorded states) into straight parts and turns.

    Returns the number of turns and the segments, each a list of [point, heading change] pairs.
    Steps where the vehicle did not move are skipped: they have no direction, and would register as heading changes.
    """
    positions = [np.asarray(state.position, dtype = float) for state in sorted(agent.record_state_list, key = lambda s: s.time_step)]
    nodes = [p for i, p in enumerate(positions) if i == 0 or not np.allclose(p, positions[i - 1])]
    
    if len(nodes) < 2:
        return 0, []

    # (heading change, step length, [end point, start point]) for each step
    steps = _calc_dist_angle(nodes)

    # Label each step as left turn ("l"), right turn ("r") or straight ("s")
    turns = []
    for angle, _, _ in steps:
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        if np.abs(angle) > ANGLE_THRESHOLD:
            turns.append("l" if angle > 0 else "r")
        else:
            turns.append("s")

    # Group the labels, then merge short groups into their neighbours
    groups = _merge_short_groups(_merge_short_straights_and_turns(_group_equal(turns)))

    # Count the turns and find where each group ends
    segment_count, segment_ends, count = 0, [], 0
    for group in groups:
        if group[-1] != "s":
            segment_count += 1
        count += len(group)
        segment_ends.append(count)

    # Build the segments: the points of each group with their heading changes
    segments, segment_begin = [], 0
    for segment_end in segment_ends:
        segment = []
        for j in range(segment_begin, segment_end):
            if j == 0:
                segment.append([steps[j][2][0], steps[j][0]])
            segment.append([steps[j][2][1], steps[j][0]])
        segments.append(segment)
        segment_begin = segment_end

    return segment_count, segments