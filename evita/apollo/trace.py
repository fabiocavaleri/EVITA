"""
Apollo traces on EVITA's side: reading the file written by record_export.py and converting it to CommonRoad.

Time: timestep 0 is the first traffic light message. DoppelTest's runner publishes the lights from the start of the
scenario, right after the recorders start, and every vehicle records them. Timestep k is t_0 + k * dt.

Positions: Apollo's position is a reference point at back_edge_to_center from the rear edge; CommonRoad's is the center of the vehicle.
Every position is moved forward by length / 2 - back_edge_to_center along the vehicle's heading; the dimensions come from the trace.

The conversion gives:
- the executed scenario: the generated scenario's lanelet network, with each vehicle as a dynamic obstacle driving its localization, sampled at every time step,
    and the traffic lights as Apollo received them;
- every plan of every vehicle, with its trajectory and decisions;
- per vehicle, the facts the executor needs to decide its outcome (start of motion, mission complete, collisions, emergency stops, disengagement advice).
"""

import copy
import gzip
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.traffic_light import TrafficLightCycle, TrafficLightCycleElement

from .map_data import ApolloMapData
from .scenario_spec import ApolloVehicle
from .traffic_control import COMMONROAD_STATE, LONG_DURATION

module_logger = logging.getLogger(__name__)

SUPPORTED_FORMAT_VERSIONS = (1,)

# Status of the run (trace["scenario"]["status"])
EXECUTED = "executed"
REJECTED = "rejected"

# A localization gap longer than this (s) is reported: states inside it are interpolated
MAX_LOCALIZATION_GAP = 0.5

# A vehicle whose localization starts more than this (s) from time step 0 suggests that the traffic light messages and
# the localization do not share a clock
MAX_CLOCK_OFFSET = 5.0

# Speed (m/s) above which a vehicle is moving
MOVING_SPEED = 0.1

# Engage advices: the vehicle is engaged, or the planner asks to stop autonomous driving
ENGAGED_ADVICES = {"READY_TO_ENGAGE", "KEEP_ENGAGED"}
DISENGAGE_ADVICES = {"PREPARE_DISENGAGE", "DISALLOW_ENGAGE"}

class TraceError(Exception):
    """
    A trace cannot be read or converted
    """

# ---------- Reading ----------

@dataclass
class VehicleTrace:
    """
    What one Apollo instance recorded
    """
    container: str
    apollo_id: int
    json_index: int
    evita_id: Optional[int]
    routing: List[str]
    start_s: float
    dest_s: float
    start_t: float
    localization: np.ndarray
    localization_columns: List[str]
    planning: List[Dict]
    obstacles: List[Dict]
    traffic_lights: List[Dict]
    oracles: Dict[str, List]

    @property
    def vehicle_id(self) -> int:
        """
        EVITA's id of the vehicle (the planning problem id), or a stand-in when the scenario JSON had no ids
        """
        return self.evita_id if self.evita_id is not None else 30000 + self.json_index

    def column(self, name: str) -> np.ndarray:
        return self.localization[:, self.localization_columns.index(name)]

@dataclass
class Trace:
    """
    A whole trace: the run's description and every vehicle's recording
    """
    scenario: Dict
    vehicles: List[VehicleTrace]

    @property
    def status(self) -> str:
        return self.scenario.get("status", "")

    @property
    def reason(self) -> str:
        return self.scenario.get("reason", "")

    @property
    def vehicle(self) -> ApolloVehicle:
        """
        Apollo's vehicle dimensions recorded by the runner
        """
        dimensions = self.scenario.get("vehicle")

        if not dimensions:
            raise TraceError("The trace has no vehicle dimensions")
        
        return ApolloVehicle(dimensions["length"], dimensions["width"], dimensions["back_edge_to_center"])

    def vehicle_ids_by_apollo_id(self) -> Dict[int, int]:
        """
        EVITA's vehicle id of each Apollo id (the id under which the other vehicles perceive it)
        """
        return {vehicle.apollo_id: vehicle.vehicle_id for vehicle in self.vehicles}

def read_trace(path: str) -> Trace:
    """
    Read a trace file
    """
    # Open the gzip file and read the JSON data
    try:
        with gzip.open(path, "rt", encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceError(f"Cannot read the trace {path}: {exc}") from exc

    # Check that the trace format is supported
    if data.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise TraceError(f"{path}: trace format {data.get('format_version')} is not supported")

    vehicles = []
    # For each vehicle, reshape the localization rows and create a VehicleTrace object
    for vehicle in data["vehicles"]:
        localization = vehicle["localization"]
        # Reshape the localization rows into a 2D numpy array with the correct number of columns
        rows = np.array(localization["rows"], dtype = float).reshape(-1, len(localization["columns"]))

        vehicles.append(VehicleTrace(
            container = vehicle["container"],
            apollo_id = vehicle["apollo_id"],
            json_index = vehicle["json_index"],
            evita_id = vehicle.get("evita_id"),
            routing = vehicle.get("routing", []),
            start_s = vehicle.get("start_s"),
            dest_s = vehicle.get("dest_s"),
            start_t = vehicle.get("start_t"),
            localization = rows,
            localization_columns = list(localization["columns"]),
            planning = vehicle["planning"],
            obstacles = vehicle["obstacles"],
            traffic_lights = vehicle["traffic_lights"],
            oracles = vehicle["oracles"])
        )

    return Trace(scenario = data["scenario"], vehicles = vehicles)

# ---------- Time ----------

def time_zero(trace: Trace) -> int:
    """
    Time (ns) of time step 0: the first traffic light message, or the first localization sample without lights
    """
    lights = [message["t"] for vehicle in trace.vehicles for message in vehicle.traffic_lights]
    if lights:
        return min(lights)

    module_logger.warning("The trace has no traffic light message: time step 0 is the first localization sample")

    samples = [int(vehicle.localization[0, 0]) for vehicle in trace.vehicles if len(vehicle.localization)]
    if not samples:
        raise TraceError("The trace has neither traffic light messages nor localization")
    
    return min(samples)

def nearest_step(t_ns: int, t0: int, dt: float) -> int:
    """
    Time step closest to a time (ns)
    """
    return int(round((t_ns - t0) / (dt * 1e9)))

# ---------- Conversion results ----------

@dataclass
class Plan:
    """
    One plan of a vehicle.
    Trajectory rows: time from time step 0 (s), center x, center y, heading, speed, acceleration, curvature, distance along the path.
    """
    t: int                          # header time of the plan (ns)
    time_step: float                # (t - t0) / dt, not rounded
    trajectory: np.ndarray
    main_decision: Dict
    object_decisions: List[Dict]
    estop: Dict
    engage_advice: Dict
    is_replan: bool
    trajectory_type: str

    TRAJECTORY_COLUMNS = ("time", "x", "y", "theta", "v", "a", "kappa", "s")

@dataclass
class ExecutedVehicle:
    """
    One vehicle after conversion
    """
    vehicle_id: int
    apollo_id: int
    container: str
    obstacle: Optional[DynamicObstacle]         # None without localization
    plans: List[Plan]
    first_motion_step: Optional[int]            # first time step with speed > MOVING_SPEED
    mission_complete_step: Optional[int]        # first plan whose main decision is mission_complete
    # DoppelTest's collision oracle (recorder's clock): one entry per message in which the two outlines touch
    collision_steps: List[int]
    collision_partners: List[Optional[int]]     # EVITA id of the other vehicle of each entry (None: not a vehicle)
    emergency_steps: Set[int]                   # plans with an emergency stop
    disengagement_steps: Set[int]               # plans advising to stop autonomous driving after being engaged
    localization_gaps: List[float] = field(default_factory = list)   # gaps longer than MAX_LOCALIZATION_GAP (s)

    @property
    def initial_time_step(self) -> Optional[int]:
        return self.obstacle.initial_state.time_step if self.obstacle else None

    @property
    def last_time_step(self) -> Optional[int]:
        if self.obstacle is None:
            return None

        prediction = self.obstacle.prediction
        return prediction.final_time_step if prediction else self.obstacle.initial_state.time_step

    @property
    def driven_states(self) -> List:
        if self.obstacle is None:
            return []

        states = [self.obstacle.initial_state]
        if self.obstacle.prediction:
            states += self.obstacle.prediction.trajectory.state_list

        return states

@dataclass
class TraceConversion:
    """
    Everything converted from one trace
    """
    executed_scenario: Scenario
    vehicles: Dict[int, ExecutedVehicle]        # by EVITA's vehicle id
    t0: int
    dt: float
    warnings: List[str]

# ---------- Driven trajectories ----------

def sample_localization(vehicle: VehicleTrace, t_0: int, dt: float, offset: float) -> List[Dict]:
    """
    Vehicle center, heading, speed, acceleration and yaw rate at every time step between its first and last sample
    """
    times = vehicle.column("t")
    if len(times) == 0:
        return []

    # Compute vehicle's initial and final timesteps, and the timesteps to sample
    dt_ns = dt * 1e9
    first_timestep = math.ceil((times[0] - t_0) / dt_ns - 1e-9)
    last_timestep = math.floor((times[-1] - t_0) / dt_ns + 1e-9)
    steps = np.arange(max(first_timestep, 0), last_timestep + 1)

    if len(steps) == 0:
        return []
    
    # Sample the vehicle's states at the computed timesteps, interpolating the localization data
    sample_times = t_0 + steps * dt_ns

    def at(name):
        return np.interp(sample_times, times, vehicle.column(name))

    # Interpolate the heading
    heading = np.interp(sample_times, times, np.unwrap(vehicle.column("heading")))
    vx, vy, ax, ay = at("vx"), at("vy"), at("ax"), at("ay")
    direction_x, direction_y = np.cos(heading), np.sin(heading)

    # Signed speed and acceleration along the heading (negative when reversing or braking)
    speed = np.hypot(vx, vy) * np.where(vx * direction_x + vy * direction_y < 0, -1.0, 1.0)
    acceleration = ax * direction_x + ay * direction_y

    # Move the position from the reference point to the vehicle center along the heading
    x = at("x") + offset * direction_x
    y = at("y") + offset * direction_y
    yaw_rate = at("yaw_rate")
    wrapped = np.arctan2(np.sin(heading), np.cos(heading))

    return [{"time_step": int(k), "position": np.array([x[i], y[i]]), "orientation": float(wrapped[i]),
             "velocity": float(speed[i]), "acceleration": float(acceleration[i]), "yaw_rate": float(yaw_rate[i])}
            for i, k in enumerate(steps)]

def localization_gaps(vehicle: VehicleTrace) -> List[float]:
    """
    Gaps longer than MAX_LOCALIZATION_GAP (s) in the vehicle's localization samples
    """
    times = vehicle.column("t")
    gaps = np.diff(times) / 1e9

    return [float(g) for g in gaps if g > MAX_LOCALIZATION_GAP]

def build_obstacle(vehicle_id: int, samples: List[Dict], dimensions: ApolloVehicle) -> Optional[DynamicObstacle]:
    """
    Dynamic obstacle driving the sampled states
    """
    if not samples:
        return None

    shape = Rectangle(dimensions.length, dimensions.width)
    initial = InitialState(slip_angle = 0.0, **samples[0])

    prediction = None
    if len(samples) > 1:
        states = [CustomState(**sample) for sample in samples[1:]]
        prediction = TrajectoryPrediction(Trajectory(states[0].time_step, states), shape)

    return DynamicObstacle(vehicle_id, ObstacleType.CAR, shape, initial, prediction)

# ---------- Plans ----------

def convert_plan(message: Dict, t0: int, dt: float, offset: float) -> Plan:
    """
    One plan of a vehicle, with its trajectory and decisions
    """
    trajectory = message["trajectory"]
    columns = trajectory["columns"]
    rows = np.array(trajectory["rows"], dtype = float).reshape(-1, len(columns))

    def column(name):
        return rows[:, columns.index(name)]

    # Compute the time of the plan (s) and convert the trajectory to CommonRoad's reference point
    plan_time = (message["t"] - t0) / 1e9
    theta = column("theta")
    converted = np.column_stack([plan_time + column("relative_time"),
                                 column("x") + offset * np.cos(theta), column("y") + offset * np.sin(theta),
                                 theta, column("v"), column("a"), column("kappa"), column("s")]) if len(rows) else np.empty((0, 8))

    return Plan(
        t = message["t"],
        time_step = plan_time / dt,
        trajectory = converted,
        main_decision = message["main_decision"],
        object_decisions = message["object_decisions"],
        estop = message["estop"],
        engage_advice = message["engage_advice"],
        is_replan = message["is_replan"],
        trajectory_type = message["trajectory_type"]
    )

def plan_facts(plans: List[Plan]):
    """
    Mission complete, emergency and disengagement time steps from a vehicle's plans
    """
    mission_complete, emergencies, disengagements = None, set(), set()
    engaged = False

    # Iterate over the plans in time order, recording the first mission_complete, all emergency stops and disengagements
    for plan in plans:
        step = int(round(plan.time_step))

        if mission_complete is None and plan.main_decision.get("decision") == "mission_complete":
            mission_complete = step
        
        if plan.estop.get("is_estop"):
            emergencies.add(step)
        
        advice = plan.engage_advice.get("advice")
        if advice in ENGAGED_ADVICES:
            engaged = True
        elif engaged and advice in DISENGAGE_ADVICES:
            disengagements.add(step)

    return mission_complete, emergencies, disengagements

# ---------- Traffic lights ----------

def light_timelines(trace: Trace, t_0: int, dt: float, last_step: int, map_data: ApolloMapData) -> Dict[int, List[str]]:
    """
    Color of every traffic light (CommonRoad id) at every time step.
    At each time step, the last message published at or before it (messages of all vehicles merged; they carry the same broadcast)
    """
    # Extract all traffic light messages from all vehicles, indexed by their time (ns)
    messages = {}
    for vehicle in trace.vehicles:
        for message in vehicle.traffic_lights:
            messages[message["t"]] = message["lights"]

    # Sort the messages by time
    times = sorted(messages)

    if not times:
        return {}

    dt_ns = dt * 1e9
    timelines: Dict[int, List[str]] = {}
    
    # Iterate over the time steps, finding the last message at or before each step and recording the colors of the lights
    index = 0
    for step in range(last_step + 1):
        step_time = t_0 + step * dt_ns + 1e6           # 1 ms tolerance on the publishing time
        
        while index + 1 < len(times) and times[index + 1] <= step_time:
            index += 1
        
        for light in messages[times[index]]:
            timelines.setdefault(map_data.traffic_light_id(light["id"]), []).append(light["color"])
    
    return timelines

def timeline_to_cycle(colors: List[str]) -> TrafficLightCycle:
    """
    CommonRoad cycle reproducing a timeline once (consecutive equal colors merged; the last color held long enough
    that the cycle never repeats within a scenario)
    """
    elements = []
    for color in colors:
        if elements and elements[-1][0] == color:
            elements[-1][1] += 1
        else:
            elements.append([color, 1])

    elements[-1][1] = LONG_DURATION
    
    return TrafficLightCycle([TrafficLightCycleElement(COMMONROAD_STATE[color], duration) for color, duration in elements])

# ---------- Whole trace ----------

def convert_trace(trace: Trace, generated_scenario: Scenario, map_data: ApolloMapData) -> TraceConversion:
    """
    Executed scenario, plans and outcome facts of an executed trace.
    The executed scenario is a copy of the generated scenario (same map) with every vehicle as a dynamic obstacle and the traffic lights as received.
    """
    if trace.status != EXECUTED:
        raise TraceError(f"The scenario was not executed ({trace.status}: {trace.reason})")

    dt = generated_scenario.dt
    dimensions = trace.vehicle
    offset = dimensions.center_offset
    t_0 = time_zero(trace)
    warnings: List[str] = []
    apollo_to_evita = trace.vehicle_ids_by_apollo_id()

    # Remove the generated dynamic obstacles from the scenario: they will be replaced by the executed ones
    executed = copy.deepcopy(generated_scenario)
    for obstacle in list(executed.dynamic_obstacles):
        executed.remove_obstacle(obstacle)

    # Convert every vehicle's localization to a dynamic obstacle, and its plans to CommonRoad
    vehicles: Dict[int, ExecutedVehicle] = {}
    for vehicle in trace.vehicles:
        samples = sample_localization(vehicle, t_0, dt, offset)
        obstacle = build_obstacle(vehicle.vehicle_id, samples, dimensions)

        if obstacle is None:
            warnings.append(f"Vehicle {vehicle.vehicle_id} ({vehicle.container}) has no localization")
        else:
            executed.add_objects(obstacle)

        # Check that the vehicle's localization starts close enough to time step 0
        if len(vehicle.localization) and abs(vehicle.localization[0, 0] - t_0) / 1e9 > MAX_CLOCK_OFFSET:
            warnings.append(f"Vehicle {vehicle.vehicle_id}: localization starts {(vehicle.localization[0, 0] - t_0) / 1e9:.1f} s from the first traffic light message: do the two use the same clock?")

        # Check for gaps in the vehicle's localization samples
        gaps = localization_gaps(vehicle)
        if gaps:
            warnings.append(f"Vehicle {vehicle.vehicle_id}: {len(gaps)} localization gaps, longest {max(gaps):.2f} s")

        plans = [convert_plan(message, t_0, dt, offset) for message in vehicle.planning]
        mission_complete, emergencies, disengagements = plan_facts(plans)
        moving = [sample["time_step"] for sample in samples if abs(sample["velocity"]) > MOVING_SPEED]

        collisions = vehicle.oracles.get("collisions", [])

        vehicles[vehicle.vehicle_id] = ExecutedVehicle(
            vehicle_id = vehicle.vehicle_id,
            apollo_id = vehicle.apollo_id,
            container = vehicle.container,
            obstacle = obstacle,
            plans = plans,
            first_motion_step = moving[0] if moving else None,
            mission_complete_step = mission_complete,
            collision_steps = [nearest_step(collision[2], t_0, dt) for collision in collisions],
            collision_partners = [apollo_to_evita.get(collision[1]) for collision in collisions],
            emergency_steps = emergencies,
            disengagement_steps = disengagements,
            localization_gaps = gaps)

    # Traffic lights as received, up to the last time step of any vehicle
    last_step = max((v.last_time_step for v in vehicles.values() if v.last_time_step is not None), default = 0)
    timelines = light_timelines(trace, t_0, dt, last_step, map_data)

    if not timelines:
        warnings.append("No traffic light message: the executed scenario keeps the generated traffic lights")
    
    # Convert the traffic light timelines to CommonRoad cycles and assign them to the executed scenario's traffic lights
    for light in executed.lanelet_network.traffic_lights:
        colors = timelines.get(light.traffic_light_id)
        if colors:
            light.traffic_light_cycle = timeline_to_cycle(colors)
        elif timelines:
            warnings.append(f"Traffic light {light.traffic_light_id} is not in the recorded messages: generated cycle kept")

    for warning in warnings:
        module_logger.warning(warning)

    return TraceConversion(executed_scenario = executed, vehicles = vehicles, t0 = t_0, dt = dt, warnings = warnings)