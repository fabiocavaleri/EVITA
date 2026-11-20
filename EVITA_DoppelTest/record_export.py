"""
Export what Apollo recorded during a scenario to a neutral trace file.

This is the Python 3.8 side of the Apollo pipeline: it is the only code that reads Apollo's .record files and protobuf messages.

Trace file structure (gzip-compressed JSON):

    {
      "format_version": 1,
      "scenario": {...},                  # set by the runner: names, status, traffic control, ...
      "vehicles": [                       # one entry per vehicle that ran, in the order of the scenario JSON
        {
          "container": "ROUTE_0",
          "apollo_id": 12345,             # id under which the other vehicles perceive this one
          "json_index": 0,                # position of the vehicle in the scenario JSON
          "evita_id": 20000,              # EVITA's vehicle id (planning problem id), if the JSON has it
          "record_files": [...],
          "localization": {"columns": [...], "rows": [[...], ...]},
          "planning": [{...}, ...],       # every planning message
          "obstacles": [{...}, ...],      # every perception message
          "traffic_lights": [{...}, ...], # every traffic light message
          "oracles": {"collisions": [...], "stop_sign_violations": [...], ...}
        }
      ]
    }

All times are integers in nanoseconds, from the clock of the messages' headers (not the recorder's write time).
"""

import os
import json
import gzip
import math
from typing import Any, Dict, List, Optional

from google.protobuf.json_format import MessageToDict

from framework.oracles import Record
from framework.oracles.impl.SpeedingOracle import SpeedingOracle
from framework.oracles.impl.StopSignOracle import StopSignOracle
from framework.oracles.impl.CollisionOracle import CollisionOracle
from framework.oracles.impl.TrafficSignalOracle import TrafficSignalOracle
from modules.common.proto.drive_state_pb2 import EngageAdvice
from modules.planning.proto.planning_pb2 import ADCTrajectory
from modules.perception.proto.perception_obstacle_pb2 import PerceptionObstacle
from modules.perception.proto.traffic_light_detection_pb2 import TrafficLight

TRACE_FORMAT_VERSION = 1

TOPIC_PLANNING = "/apollo/planning"
TOPIC_LOCALIZATION = "/apollo/localization/pose"
TOPIC_OBSTACLES = "/apollo/perception/obstacles"
TOPIC_TRAFFIC_LIGHTS = "/apollo/perception/traffic_light"

LOCALIZATION_COLUMNS = ["t", "x", "y", "z", "heading", "vx", "vy", "ax", "ay", "yaw_rate"]
TRAJECTORY_COLUMNS = ["relative_time", "x", "y", "theta", "kappa", "s", "v", "a"]

class RecordExportError(Exception):
    """
    The record files of a scenario are missing or cannot be read
    """

# ---------- Helpers ----------

def seconds_to_ns(seconds: float) -> int:
    """
    Seconds (float, as in the message headers) to integer nanoseconds
    """
    return int(round(seconds * 1e9))

def heading_of(pose) -> float:
    """
    Vehicle heading (radians): Apollo's own heading field when set, otherwise from the orientation quaternion.
    Apollo's quaternion is rotated by -pi/2 with respect to the heading (vehicle frame: x to the right, y forward).
    """
    if pose.HasField("heading"):
        return pose.heading

    q = pose.orientation
    yaw = math.atan2(2.0 * (q.qw * q.qz + q.qx * q.qy), 1.0 - 2.0 * (q.qy * q.qy + q.qz * q.qz))

    return math.atan2(math.sin(yaw + math.pi / 2), math.cos(yaw + math.pi / 2))

def message_payload(message) -> Dict[str, Any]:
    """
    All the fields of a protobuf message as a JSON-compatible dict (enum values as names, field names as in the .proto)
    """
    return MessageToDict(message, preserving_proto_field_name = True)

# ---------- One message to JSON ----------

def export_localization(message) -> Optional[List[float]]:
    """
    Export a localization message as a row of LOCALIZATION_COLUMNS, or None without a header timestamp
    """
    if not message.header.HasField("timestamp_sec"):
        return None

    pose = message.pose
    return [seconds_to_ns(message.header.timestamp_sec),
            pose.position.x, pose.position.y, pose.position.z,
            heading_of(pose),
            pose.linear_velocity.x, pose.linear_velocity.y,
            pose.linear_acceleration.x, pose.linear_acceleration.y,
            pose.angular_velocity.z]

def export_object_decisions(decision_result) -> List[Dict[str, Any]]:
    """
    The decision of the planner for each object (vehicle, stop sign, traffic light, ...) in a planning message.
    Each object can have several decisions (e.g. a lateral nudge and a longitudinal follow), each written as
    {"decision": <follow, yield, stop, ...>, "details": <the decision's fields>}.
    """
    objects = []
    # Iterate over the objects in the decision result, and for each object iterate over its decisions
    for object_decision in decision_result.object_decision.decision:
        decisions = []
        for decision_type in object_decision.object_decision:
            tag = decision_type.WhichOneof("object_tag")
            if tag is None:
                continue
            decisions.append({"decision": tag, "details": message_payload(getattr(decision_type, tag))})

        objects.append({"id": object_decision.id, "perception_id": object_decision.perception_id, "decisions": decisions})

    return objects

def export_planning(message, record_time_ns: int) -> Dict[str, Any]:
    """
    One planning message: the planned trajectory (times relative to the message header), the main decision, the
    decision for each object, emergency stop and engage advice
    """
    main_decision = message.decision.main_decision
    main_tag = main_decision.WhichOneof("task")

    return {
        "t": seconds_to_ns(message.header.timestamp_sec),   # Time of the planning (header timestamp, ns)
        "t_record": record_time_ns,     # Time the recorder wrote the message (ns)
        "sequence_num": message.header.sequence_num,    # Number of the planning message
        "is_replan": message.is_replan,     # True when the planner discarded its previous trajectory and planned from the vehicle's current state
        "replan_reason": message.replan_reason,     # Reason for the replan
        "gear": message.gear,       # Requested gear (canbus Chassis.GearPosition value: 0 neutral, 1 drive, 2 reverse, 3 parking, ...)
        # Kind of trajectory: NORMAL; PATH_REUSED (previous path kept); PATH_FALLBACK or SPEED_FALLBACK (the planner could not plan normally); UNKNOWN
        "trajectory_type": ADCTrajectory.TrajectoryType.Name(message.trajectory_type),
        # Planned trajectory, one row per point:
        # time relative to "t" (s), position (m), heading (rad), curvature (1/m), distance along the path (m), speed (m/s), acceleration (m/s^2)
        "trajectory": {
            "columns": TRAJECTORY_COLUMNS,
            "rows": [[p.relative_time, p.path_point.x, p.path_point.y, p.path_point.theta, p.path_point.kappa,
                      p.path_point.s, p.v, p.a] for p in message.trajectory_point]
        },
        # What the vehicle does as a whole: cruise, stop (with reason), estop, change_lane, mission_complete, ...
        "main_decision": {"decision": main_tag, "details": message_payload(getattr(main_decision, main_tag)) if main_tag else {}},
        # What the vehicle decided for each object (vehicle, stop sign, traffic light, destination, ...):
        # follow, yield, overtake, nudge, stop or ignore, with each decision's fields
        "object_decisions": export_object_decisions(message.decision),
        # Emergency stop requested by the planner, and why
        "estop": {"is_estop": message.estop.is_estop, "reason": message.estop.reason},
        # Whether the planner advises engaging autonomous driving (UNKNOWN, DISALLOW_ENGAGE, READY_TO_ENGAGE,
        # KEEP_ENGAGED, PREPARE_DISENGAGE), and why
        "engage_advice": {"advice": EngageAdvice.Advice.Name(message.engage_advice.advice),
                          "reason": message.engage_advice.reason}
    }

def export_obstacles(message) -> Dict[str, Any]:
    """
    One perception message: every obstacle with its pose, size and polygon
    """
    return {
        "t": seconds_to_ns(message.header.timestamp_sec),
        "obstacles": [{
            "id": obstacle.id,
            "type": PerceptionObstacle.Type.Name(obstacle.type),
            "x": obstacle.position.x,
            "y": obstacle.position.y,
            "theta": obstacle.theta,
            "vx": obstacle.velocity.x,
            "vy": obstacle.velocity.y,
            "length": obstacle.length,
            "width": obstacle.width,
            "polygon": [[p.x, p.y] for p in obstacle.polygon_point]
        } for obstacle in message.perception_obstacle]
    }

def export_traffic_lights(message) -> Dict[str, Any]:
    """
    One traffic light message: the color of every signal, as published by DoppelTest's traffic control manager
    """
    return {
        "t": seconds_to_ns(message.header.timestamp_sec),
        "lights": [{"id": light.id, "color": TrafficLight.Color.Name(light.color), "confidence": light.confidence}
                   for light in message.traffic_light]
    }

# ---------- DoppelTest's oracles ----------

def json_safe(value):
    """
    Oracle results (tuples, numpy numbers) as JSON-compatible values
    """
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if hasattr(value, "item"):
        return value.item()

    return value

class Oracles:
    """
    DoppelTest's collision and violation oracles, fed with every message of a vehicle's record
    """
    def __init__(self) -> None:
        self.oracles = {
            "collisions": CollisionOracle(),
            "stop_sign_violations": StopSignOracle(),
            "traffic_light_violations": TrafficSignalOracle(),
            "speeding_violations": SpeedingOracle()
        }
        self.topics = {name: set(oracle.get_interested_topics()) for name, oracle in self.oracles.items()}

    def feed(self, topic: str, message, record_time_ns: int) -> None:
        for name, oracle in self.oracles.items():
            if topic in self.topics[name]:
                oracle.on_new_message(topic, message, record_time_ns)

    def results(self) -> Dict[str, Any]:
        """
        Each oracle's result. Times in these results are the recorder's write times (as DoppelTest passes them).
        """
        return {name: json_safe(oracle.get_result()) for name, oracle in self.oracles.items()}

# ---------- One vehicle ----------

def vehicle_record_files(records_dir: str, container: str, scenario_name: str) -> List[str]:
    """
    Record files of one vehicle, in order: <container>.<scenario>, possibly split by Cyber into numbered chunks
    """
    prefix = f"{container}.{scenario_name}"

    files = [filename for filename in os.listdir(records_dir) if os.path.isfile(os.path.join(records_dir, filename)) and (filename == prefix or filename.startswith(prefix + "."))]
    
    return [os.path.join(records_dir, file) for file in sorted(files)]

def export_vehicle(record_files: List[str]) -> Dict[str, Any]:
    """
    Extract every localization, planning, perception and traffic light message of one vehicle, and its oracles' results
    """
    if not record_files:
        raise RecordExportError("No record file")

    # Initialize the oracles and the lists of messages
    localization, planning, obstacles, traffic_lights = [], [], [], []
    oracles = Oracles()

    # Iterate over each vehicle's record file, read every message and feed the oracles
    for record_file in record_files:
        # Read the record file and get its messages (topic, message, record_time_ns)
        try:
            record = Record(record_file)
            messages = list(record.read_messages())
        except Exception as exc:
            raise RecordExportError(f"Cannot read record file {record_file}: {exc}") from exc

        # Iterate over the messages in the order of the record, feed the oracles and export the messages of interest
        # (the exported messages are sorted by their header time afterwards)
        for topic, message, record_time_ns in messages:
            oracles.feed(topic, message, record_time_ns)

            if topic == TOPIC_LOCALIZATION:
                row = export_localization(message)
                if row is not None:
                    localization.append(row)
            elif topic == TOPIC_PLANNING:
                planning.append(export_planning(message, record_time_ns))
            elif topic == TOPIC_OBSTACLES:
                obstacles.append(export_obstacles(message))
            elif topic == TOPIC_TRAFFIC_LIGHTS:
                traffic_lights.append(export_traffic_lights(message))

    # Messages in time order, without repeated localization samples
    localization.sort(key = lambda row: row[0])
    localization = [row for i, row in enumerate(localization) if i == 0 or row[0] != localization[i - 1][0]]
    for messages in (planning, obstacles, traffic_lights):
        messages.sort(key = lambda m: m["t"])

    return {
        "record_files": [os.path.basename(f) for f in record_files],
        "localization": {"columns": LOCALIZATION_COLUMNS, "rows": localization},
        "planning": planning,
        "obstacles": obstacles,
        "traffic_lights": traffic_lights,
        "oracles": oracles.results()
    }

# ---------- Whole scenario ----------

def build_trace(records_dir: str, scenario_name: str, scenario: Dict[str, Any], vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the trace of a scenario.
    `vehicles` describes each vehicle that ran (container, apollo_id, json_index, evita_id, ...); the recorded data of each vehicle is added to its entry.
    DoppelTest's stop sign oracle reads the HD map: the MapParser must be loaded first.
    """
    exported = []

    # Each vehicle's record files are read and exported to JSON, and the oracles are fed with every message
    for vehicle in vehicles:
        files = vehicle_record_files(records_dir, vehicle["container"], scenario_name)

        try:
            data = export_vehicle(files)
        except RecordExportError as exc:
            raise RecordExportError(f"{vehicle['container']}: {exc}") from exc

        exported.append({**vehicle, **data})

    return {"format_version": TRACE_FORMAT_VERSION, "scenario": scenario, "vehicles": exported}

def write_trace(trace: Dict[str, Any], trace_path: str) -> None:
    """
    Write the trace as gzip-compressed JSON (written to a temporary file first, so a trace file is always complete)
    """
    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok = True)
    temporary_path = trace_path + ".part"

    with gzip.open(temporary_path, "wt", encoding = "utf-8") as f:
        json.dump(trace, f)

    os.replace(temporary_path, trace_path)