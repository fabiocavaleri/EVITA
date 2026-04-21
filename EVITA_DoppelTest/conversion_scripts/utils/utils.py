import os
import math
import json
import copy
import numpy as np
from typing import Union, Dict, List, Tuple
from scipy.spatial.transform import Rotation as R

from framework.oracles import Record
from framework.oracles.impl.CollisionOracle import CollisionOracle
from framework.oracles.impl.StopSignOracle import StopSignOracle
from framework.oracles.impl.TrafficSignalOracle import TrafficSignalOracle
from framework.oracles.impl.SpeedingOracle import SpeedingOracle
from modules.planning.proto.decision_pb2 import StopReasonCode

from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.state import CustomState, InitialState
from commonroad.scenario.scenario import Scenario, DynamicObstacle, ObstacleType
from commonroad.scenario.traffic_light import TrafficLightCycleElement, TrafficLightCycle, TrafficLightState
from commonroad.prediction.prediction import Trajectory, TrajectoryPrediction
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad_route_planner.route_planner import RoutePlanner

def safe_scalar(value, default: float = 0.0) -> float:
    """
    Safely convert a value to a scalar float, returning a default value if the conversion fails or results in NaN/Inf.
    """
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default
    
def convert_history_state(state_index: int, json_state: dict, scenario_initial_timestep: int) ->  Union[InitialState, CustomState]:
        """
        Method to convert a vehicle history state from a .record file to CommonRoad state format

        :param json_state: Dictionary with keys: relative_time, x, y, theta, v, a, kappa
        :param scenario_initial_timestep: Scenario initial timestep (in nanoseconds) to normalize the state timestep
        :return: InitialState or CustomState object
        """
        # Timestep normalized with the scenario initial timestep
        timestep = json_state["timestamp"] - scenario_initial_timestep

        # Position
        position = np.array([json_state["x"], json_state["y"]])

        # Orientation
        quat = [json_state["qx"], json_state["qy"], json_state["qz"], json_state["qw"]]
        orientation = safe_scalar(R.from_quat(quat).as_euler('zyx')[0])

        # Velocity
        vx = safe_scalar(json_state["linear_velocity"]["x"])
        vy = safe_scalar(json_state["linear_velocity"]["y"])
        velocity = safe_scalar(np.hypot(vx, vy))

        # Acceleration
        ax = safe_scalar(json_state["linear_acceleration"]["x"])
        ay = safe_scalar(json_state["linear_acceleration"]["y"])
        acceleration = safe_scalar(np.hypot(ax, ay))

        # Yaw rate from angular velocity
        yaw_rate = safe_scalar(json_state["angular_velocity"]["z"])
        
        # If the input json state is the first of the state list
        # return an InitialState, otherwise return a CustomState
        if state_index == 0:
            return InitialState(time_step = timestep, position = position, orientation = orientation, velocity = velocity, acceleration = acceleration, yaw_rate = yaw_rate, slip_angle = 0)
        else:
            return CustomState(time_step = timestep, position = position, orientation = orientation, velocity = velocity, acceleration = acceleration, yaw_rate = yaw_rate, slip_angle = 0)

def convert_trajectory_state(json_state: dict, scenario_initial_timestep: int) -> CustomState:
    """
    Method to convert a trajectory state from a .record file to CommonRoad state format

    :param json_state: Dictionary with keys: relative_time, x, y, theta, v, a, kappa
    :return: CustomState object
    """
    # Normalize state timestep
    timestep = json_state["trajectory_timestep"] - scenario_initial_timestep
    position = np.array([json_state["x"], json_state["y"]])
    orientation = safe_scalar(json_state["theta"])
    velocity = safe_scalar(json_state["v"])
    acceleration = safe_scalar(json_state["a"])
    yaw_rate = safe_scalar(json_state["v"] * json_state.get("kappa", 0.0))

    return CustomState(time_step = timestep, position = position, orientation = orientation, velocity = velocity, acceleration = acceleration, yaw_rate = yaw_rate, slip_angle = 0)

def interpolate_state_backwards(first_state: CustomState, target_timestep: int) -> CustomState:
    """
    Create a new CustomState at `target_timestep` by backward integrating the first known trajectory state.
    This is used only when the planned trajectory starts after the planning timestep.
    """

    # Compute time difference between the first known state and the target timestep
    dt = (target_timestep - first_state.time_step)

    # Backward interpolate position
    # I assume velocity and yaw do not change drastically over small dt.
    dx = first_state.velocity * np.cos(first_state.orientation) * dt
    dy = first_state.velocity * np.sin(first_state.orientation) * dt
    new_position = first_state.position + np.array([dx, dy])

    # Backward interpolate orientation
    new_orientation = first_state.orientation + first_state.yaw_rate * dt

    # Backward interpolate velocity (assuming constant acceleration)
    new_velocity = first_state.velocity + first_state.acceleration * dt
    new_velocity = max(new_velocity, 0.0)   # avoid negative velocity

    # Acceleration, yaw_rate, slip_angle assumed constant
    new_acceleration = first_state.acceleration
    new_yaw_rate = first_state.yaw_rate

    # Build new state
    new_state = CustomState(
        time_step = target_timestep,
        position = new_position,
        orientation = new_orientation,
        velocity = new_velocity,
        acceleration = new_acceleration,
        yaw_rate = new_yaw_rate,
        slip_angle = 0
    )

    return new_state

def interpolate_state(first_state: CustomState, second_state: CustomState, target_timestep: int, convert_timesteps: bool = False) -> CustomState:
    """
    Linearly interpolate between two CustomState objects to create a new CustomState at target_timestep.

    :param first_state: Vehicle history state at time t1
    :param second_state: Vehicle history state at time t2
    :param target_timestep: The timestep at which to create the new state
    :param convert_timesteps: Whether to convert timesteps to CommonRoad format
    """

    # Extract states timesteps
    t1 = first_state.time_step
    t2 = second_state.time_step

    # Compute interpolation factor
    alpha = (target_timestep - t1) / (t2 - t1)

    # Linear interpolation helper
    def interpolation_helper(a, b):
        return a + alpha * (b - a)

    # Interpolate each field
    new_position = interpolation_helper(first_state.position, second_state.position)
    # Add 90° to the orientation to fix it in CommonRoad
    new_orientation = interpolation_helper(first_state.orientation, second_state.orientation) + math.pi / 2
    new_velocity = interpolation_helper(first_state.velocity, second_state.velocity)
    new_acceleration = interpolation_helper(first_state.acceleration, second_state.acceleration)
    new_yaw_rate = interpolation_helper(first_state.yaw_rate, second_state.yaw_rate)

    # Build new state
    new_state = CustomState(
        time_step = target_timestep,
        position = new_position,
        orientation = new_orientation,
        velocity = new_velocity,
        acceleration = new_acceleration,
        yaw_rate = new_yaw_rate, 
        slip_angle = 0
    )

    # If requested, convert Apollo timestep format (nanoseconds) to CommonRoad format (tenths of second)
    if convert_timesteps:
        conversion_factor = 1e8 # nanoseconds -> tenths of second
        new_state.time_step = int(new_state.time_step / conversion_factor)

    return new_state

def extract_from_record(record: Record) -> Tuple[Dict, Dict, List, List, List]:
    """
    Extract vehicle's simulation data from the DoppelTest .record file.
    
    :param record: DoppelTest vehicle's .record file
    :type record: Record
    :return Dicts or Lists containing: vehicle history, planned trajectories, vehicle collisions, vehicle's obstacle perception and traffic light perception
    """
    planned_trajectories: Dict[int, List[Dict]] = {}
    vehicle_states: Dict = {}
    vehicle_perception = []
    traffic_light_perception = []
    collision_oracle = CollisionOracle()
    stop_signs_oracle = StopSignOracle()
    traffic_lights_oracle = TrafficSignalOracle()
    speeding_oracle = SpeedingOracle()
    
    # Iterate over the recorded messages in the .record file    
    for topic, message, t in record.read_messages():
        # Send message to DoppelTest collision oracle
        if topic in collision_oracle.get_interested_topics():
            collision_oracle.on_new_message(topic, message, t)

        # Send message to DoppelTest stop signs violation oracle
        if topic in stop_signs_oracle.get_interested_topics():
            stop_signs_oracle.on_new_message(topic, message, t)

        # Send message to DoppelTest traffic lights violation oracle
        if topic in traffic_lights_oracle.get_interested_topics():
            traffic_lights_oracle.on_new_message(topic, message, t)

            # Send message to DoppelTest speeding violation oracle
        if topic in speeding_oracle.get_interested_topics():
            speeding_oracle.on_new_message(topic, message, t)

        # Extract vehicle's obstacle perception
        if topic == '/apollo/perception/obstacles':
            vehicle_perception.append(message)
        
        # Extract vehicle's traffic light perception
        if topic == '/apollo/perception/traffic_light':
            traffic_light_perception.append(message)

        # Extract vehicle's planned trajectory
        if topic == '/apollo/planning':
            # Check if planned trajectory is available
            if message.trajectory_point:
                trajectory_data = []
                # Iterate over the planned trajectory "states"
                for point in message.trajectory_point:
                    trajectory_timestep = t + round(point.relative_time * 1e9)

                    # Append extracted trajectory "state"
                    trajectory_data.append({
                        'trajectory_timestep': trajectory_timestep,
                        'relative_time': point.relative_time,
                        'x': point.path_point.x,
                        'y': point.path_point.y,
                        'theta': point.path_point.theta,
                        'kappa': point.path_point.kappa,
                        's': point.path_point.s,
                        'dkappa': point.path_point.dkappa,
                        'ddkappa': point.path_point.ddkappa,
                        'v': point.v, 
                        'a': point.a,
                    })

                decision_type = str(message.decision.main_decision.WhichOneof("task"))

                if decision_type == "stop":
                    # Extract replanning decision
                    decision_reason = StopReasonCode.Name(message.decision.main_decision.stop.reason_code)
                    decision_object = message.decision.main_decision.stop.reason.split(" ")[2]
                    stop_position = [message.decision.main_decision.stop.stop_point.x, message.decision.main_decision.stop.stop_point.y]
                elif decision_type == "cruise":
                    decision_reason = "cruise"
                    decision_object = None
                    stop_position = None
                elif decision_type == "mission_complete":
                    decision_reason = "mission_complete"
                    decision_object = None
                    stop_position = [message.decision.main_decision.stop.stop_point.x, message.decision.main_decision.stop.stop_point.y]
                elif decision_type == "not_ready":
                    continue
                else:
                    print(message.decision.main_decision.WhichOneof("task"))
                    print(message.decision.main_decision)
                    continue

                # Save planned trajectory
                planned_trajectories[t] = {
                    "trajectory": trajectory_data,
                    "decision": decision_reason,
                    "decision_object": decision_object,
                    "stop_position": stop_position
                }
            
        # Extract vehicles history states
        if topic == '/apollo/localization/pose':
            header = message.header
            # Retrieve history state timestep
            time = round(header.timestamp_sec * 1e9)

            # Retrieve history state object
            pose = message.pose
    
            # Avoid duplicated timesteps
            if time not in vehicle_states:
                # Save history state data
                vehicle_states[time] = {
                    'timestamp': time,
                    'x': pose.position.x,
                    'y': pose.position.y,
                    'z': pose.position.z,

                    'qx': pose.orientation.qx,
                    'qy': pose.orientation.qy,
                    'qz': pose.orientation.qz,
                    'qw': pose.orientation.qw,

                    'linear_velocity': {
                        'x': pose.linear_velocity.x,
                        'y': pose.linear_velocity.y,
                        'z': pose.linear_velocity.z,
                    },

                    'angular_velocity': {
                        'x': pose.angular_velocity.x,
                        'y': pose.angular_velocity.y,
                        'z': pose.angular_velocity.z,
                    },

                    'linear_acceleration': {
                        'x': pose.linear_acceleration.x,
                        'y': pose.linear_acceleration.y,
                        'z': pose.linear_acceleration.z,
                    }
                }
    
    # Sorting history states and planned trajectory states by timestep
    sorted_vehicle_states = dict(sorted(vehicle_states.items(), key = lambda item: item[0]))
    planned_trajectories_sorted = dict(sorted(planned_trajectories.items(), key = lambda item: item[0]))

    # Get detected collisions from the collision oracle
    detected_collisions = collision_oracle.get_result()
    detected_stop_signs_violations = stop_signs_oracle.get_result()
    detected_traffic_lights_violations = traffic_lights_oracle.get_result()
    detected_speeding_violations = speeding_oracle.get_result()

    return sorted_vehicle_states, planned_trajectories_sorted, detected_collisions, vehicle_perception, traffic_light_perception, detected_stop_signs_violations, detected_traffic_lights_violations, detected_speeding_violations

def downsample_state_list(state_list: dict, timestep_interval: int, convert_timestep: bool = False) -> List[CustomState]:
    """
    Downsample a trajectory state list to have a uniform time distance between each state.
    
    :param state_list: State list to downsample
    :type state_list: dict
    :param timestep_interval: Time distance between consecutive states
    :type timestep_interval: int
    :param convert_timestep: Boolean value determining if the states timesteps need to be converted to CommonRoad format
    :type convert_timestep: bool
    """

    # Initialize the downsampled state list and save the first state in the list
    initial_state = copy.deepcopy(state_list[0])
    # Adjust the state orientation (add 90°)
    initial_state.orientation = initial_state.orientation + math.pi / 2
    # If requested, adjust the timesteps to CommonRoad format
    if convert_timestep:
        conversion_factor = 1e8 # nanoseconds -> tenths of second
        initial_state.time_step = int(initial_state.time_step / conversion_factor)

    downsampled_state_list = [initial_state]

    # Retrieve the state list initial and final timestep
    initial_timestep = state_list[0].time_step
    final_timestep = state_list[-1].time_step
    # Define the timestep of the second state in the downsampled list
    target_timestep = initial_timestep + timestep_interval
    k = 0

    # The downsampled state list must also have the same number of states as the original one
    while target_timestep <= final_timestep and k < len(state_list) - 1:
        state_a = state_list[k]
        state_b = state_list[k + 1]

        # Move forward until target_time is between state_a and state_b
        if state_b.time_step < target_timestep:
            k += 1
            continue

        # Create a new state with the correct timestep by interpolating two consecutive states
        interpolated_state = interpolate_state(state_a, state_b, target_timestep, convert_timestep)

        # Add the interpolated state to the donwsampled state list
        downsampled_state_list.append(interpolated_state)

        # Increment the target timestep to the next state timestep
        target_timestep += timestep_interval

    # Check that the states in the downsampled list are in the right time order
    checked_downsampled_state_list = []
    last_time_step = None

    for state in downsampled_state_list:
        # Extract state timestep
        time_step = int(state.time_step)

        # Skip states with negative or duplicated timesteps
        if (time_step < 0) or (last_time_step is not None and time_step <= last_time_step):
            continue

        # Add state to final list and update last_time_step
        checked_downsampled_state_list.append(state)
        last_time_step = time_step

    return checked_downsampled_state_list

def dynamic_obstacle_from_record(vehicle_id: int, extracted_vehicle_history: dict, extracted_planned_trajectories: dict, vehicle_shape: Rectangle, scenario_initial_timestep: int) -> Tuple[DynamicObstacle, Dict]:
    """
    Convert the data extracted from the record file to CommonRoad format and create a DynamicObstacle object
    
    :param vehicle_id: Vehicle identifier
    :type vehicle_id: int
    :param extracted_vehicle_history: Vehicle history in json format
    :type extracted_vehicle_history: dict
    :param extracted_planned_trajectories: Planned trajectories in json format
    :type extracted_planned_trajectories: dict
    :param vehicle_shape: Vehicle shape
    :type vehicle_shape: Rectangle
    :param scenario_initial_timestep: Scenario initial timestep (in nanoseconds)
    :type scenario_initial_timestep: int
    """

    #
    # ---------- Convert Vehicle History ----------
    #

    # Initialize vehicle's history state list
    state_list = []

    # Iterate over the .json like dict containing the vehicle history extracted from the .record file
    for i, (_, history_state) in enumerate(extracted_vehicle_history.items()):
        # Convert each state to CommonRoad state format and append to the new state list
        state_list.append(convert_history_state(i, history_state, scenario_initial_timestep))

    # Downsample the converted state list and convert the timestep to CommonRoad format (tenths of second)
    timestep_interval = int(0.1 * 1e9)
    downsampled_state_list = downsample_state_list(state_list, timestep_interval, True)

    # Get the vehicle history initial state
    initial_state = downsampled_state_list[0]

    # Create the CommonRoad object to represent the vehicle history
    vehicle_history = TrajectoryPrediction(Trajectory(initial_time_step = downsampled_state_list[0].time_step, state_list = downsampled_state_list), vehicle_shape)

    #
    # ---------- Convert Vehicle Planned Trajectories ----------
    #

    planned_trajectories = {}

    # Iterate over each planned trajectory
    for original_initial_timestep, replanning_data in extracted_planned_trajectories.items():
        trajectory_states = replanning_data["trajectory"]
        decision_reason = replanning_data["decision"]
        decision_object = replanning_data["decision_object"]
        stop_position = replanning_data["stop_position"]

        # Normalize planned trajectory initial timestep (avoid negative values)
        normalized_original_initial_timestep = max(0, int(original_initial_timestep - scenario_initial_timestep))

        # Convert all the trajectory states to CommonRoad state format
        state_list = [convert_trajectory_state(state, scenario_initial_timestep) for state in trajectory_states]

        if len(state_list) < 2:
            # Skip the conversion of planned trajectories with only one state
            continue

        # Get planned trajectory initial timestep
        initial_timestep = state_list[0].time_step

        # The timestep of the first state is basically never equal to the planning timestep.
        # If it is lower than the planning timestep I interpolate the first and the second states
        # If it is greater than the planning timestep create a new state as an interpolation from the first one in state_list.
        if initial_timestep > normalized_original_initial_timestep:
            # Create new state interpolating backwards
            new_initial_state = interpolate_state_backwards(state_list[0], normalized_original_initial_timestep)
            # Set planned trajectory initial state
            state_list.insert(0, new_initial_state)
            # Set planned trajectory initial timestep
            initial_timestep = new_initial_state.time_step
        elif initial_timestep < normalized_original_initial_timestep:
                # Get two states such that that the target timestep is between them
                i = 0
                found = False
                while i < len(state_list) - 1:
                    first_state = state_list[i]
                    second_state = state_list[i + 1]

                    # Check if the initial timestep is between the two states
                    if first_state.time_step < normalized_original_initial_timestep <= second_state.time_step:
                        found = True
                        break
                    else:
                        i = i + 1

                if not found:
                    # Skip trajectories for which is impossible to find two states to interpolate
                    continue

                # Compute new initial state by interpolating the two states
                new_initial_state = interpolate_state(first_state, second_state, normalized_original_initial_timestep)
                # Remove the first state and insert the new interpolated one
                state_list.pop(0)
                state_list.insert(0, new_initial_state)
                initial_timestep = state_list[0].time_step

        # Downsample adjusted state list
        downsampled_state_list = downsample_state_list(state_list, timestep_interval, True)

        # Compute planned trajectorygeometrical length
        # trajectory_length = compute_trajectory_length(downsampled_state_list)

        # if trajectory_length < 5.0:
        #     continue

        # Create the trajectory object
        planned_trajectory = Trajectory(initial_time_step = downsampled_state_list[0].time_step, state_list = downsampled_state_list)

        # Add trajectory object at planning timestep
        planned_trajectories[downsampled_state_list[0].time_step] = {
            "planned_trajectory": planned_trajectory,
            "decision": decision_reason,
            "decision_object": decision_object,
            "stop_position": stop_position
        }

    # Create dynamic obstacle object
    dynamic_obstacle = DynamicObstacle(vehicle_id, ObstacleType.CAR, vehicle_shape, initial_state, prediction = vehicle_history)

    return dynamic_obstacle, planned_trajectories

def compute_trajectory_length(state_list: List) -> float:
    """
    From the planned trajectory state list compute geometrical length
    """
    total_length = 0.0

    # Iterate over state list
    for i in range(len(state_list) - 1):
        # Get planned position from two consecutive states
        p1 = state_list[i].position
        p2 = state_list[i + 1].position

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        total_length += math.hypot(dx, dy)

    return total_length

def export_planned_trajectories(scenario_planned_trajectories: Dict, executed_scenario_folder: str):
    """
    Export planned trajectories to a JSON file
    """
    planned_trajectories = {}

    # Iterate over each scenario vehicle
    for vehicle_id, vehicle_planned_trajectories in scenario_planned_trajectories.items():
        # Initialize nested dict
        planned_trajectories[vehicle_id] = {}

        # Iterate over each planned trajectory
        for trajectory_timestep, replanning_data in vehicle_planned_trajectories.items():
            trajectory = replanning_data["planned_trajectory"]
            decision_reason = replanning_data["decision"]
            decision_object = replanning_data["decision_object"]
            stop_position = replanning_data["stop_position"]

            planned_trajectories[vehicle_id][trajectory_timestep] = {
                "planned_trajectory": trajectory_to_dict(trajectory),
                "decision": decision_reason,
                "decision_object": decision_object,
                "stop_position": stop_position
            }

    # Output path
    out_path = os.path.join(executed_scenario_folder, "planned_trajectories.json")

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(planned_trajectories, f, indent = 4)

def trajectory_to_dict(trajectory: Trajectory) -> Dict:
    """
    Convert CommonRoad trajectory object to a .json like representation
    
    :param trajectory: CommonRoad trajectory object
    :type trajectory: Trajectory
    """

    return {
        "initial_time_step": trajectory.initial_time_step,
        "state_list": [state_to_dict(s) for s in trajectory.state_list],
    }

def state_to_dict(state: CustomState) -> Dict:
    """
    Convert CommonRoad state object to a .json like representation
    
    :param state: CommonRoad state object
    :type state: CustomState
    """

    return {
        "time_step": state.time_step,
        "position": [state.position[0], state.position[1]],
        "orientation": state.orientation,
        "velocity": state.velocity,
        "acceleration": state.acceleration,
        "yaw_rate": state.yaw_rate,
    }

def get_scenario_planning_problems(scenario: Scenario, scenario_id_base_counter: int, routing_data: dict) -> Dict:
    """
    Build a planning problem for each vehicle and compute its reference path
    
    :param scenario: Scenario
    :type scenario: Scenario
    :param scenario_id_base_counter: Vehicles id base counter
    :type scenario_id_base_counter: int
    :param routing_data: Goals information from DoppelTest scenario specification
    :type routing_data: dict
    """
    
    reference_paths = {}

    # Get scenario's lanelet network
    lanelet_network: LaneletNetwork = scenario.lanelet_network

    # Build fast lookup for obstacles
    obstacles_by_id = {o.obstacle_id: o for o in scenario.obstacles}

    for i, entry in enumerate(routing_data):
        # Compute planning problem (and vehicle) id
        planning_problem_id = scenario_id_base_counter + i

        # Lookup scenario obstacle from its id
        obstacle = obstacles_by_id.get(planning_problem_id)

        # Get the lanes the vehicle is supposed to drive through
        lane_sequence = entry["routing"]
        # Get destination position (on last lane)
        dest_s = float(entry["dest_s"])

        # Retrieve vehicle's initial state
        initial_state = obstacle.initial_state

        # Compute goal region position
        last_lanelet_id = int(lane_sequence[-1].split("_")[1])
        last_lanelet = lanelet_network.find_lanelet_by_id(last_lanelet_id)
        goal_center, _, _, _ = last_lanelet.interpolate_position(dest_s)

        # Goal rectangle parameters
        goal_width = 10.0
        goal_height = 5.0

        goal_area = Rectangle(
            length = goal_width,
            width = goal_height,
            center = goal_center
        )

        # Allowed goal time interval
        final_timestep = obstacle.prediction.final_time_step
        
        # Build goal state object
        goal_state = CustomState(
            time_step=Interval(final_timestep - 25, final_timestep + 25),
            position = goal_area
        )

        goal_region = GoalRegion([goal_state])

        # Build PlanningProblem object
        planning_problem = PlanningProblem(
            planning_problem_id = planning_problem_id,
            initial_state = initial_state,
            goal_region = goal_region
        )

        # Compute reference path
        route_planner = RoutePlanner(
                                    lanelet_network = scenario.lanelet_network,
                                    planning_problem = planning_problem,
                                    scenario = scenario,
                                    extended_search = False
                        )
        reference_paths[planning_problem_id] = route_planner.plan_routes().retrieve_shortest_route(retrieve_shortest = True)

    return reference_paths

def set_traffic_lights_cycles (executed_scenario: Scenario, traffic_lights_initial_states: dict, traffic_light_durations: dict) -> Scenario:
    """
    Update the scenario's traffic lights initial states and light color cycles based on the scenario specifications
    
    :param executed_scenario: CommonRoad scenario with initialized traffic light objects
    :type executed_scenario: Scenario
    :param traffic_lights_final_states: Traffic lights initial color from scenario specification
    :type traffic_lights_final_states: dict
    :param traffic_light_durations: Traffic lights color cycles from scenario specification
    :type traffic_light_durations: dict
    """

    # Iterate over scenario's traffic lights
    for traffic_light in executed_scenario.lanelet_network.traffic_lights:
        # Retrive Apollo original traffic light identifier
        traffic_light_id = traffic_light.traffic_light_id - 10000
        # Retrieve traffic light initial state
        traffic_light_initial_state = traffic_lights_initial_states[f"signal_{str(traffic_light_id)}"]

        # Reconstruct traffic light color elements and cycle based on the scenario definition durations
        red_cycle_element = TrafficLightCycleElement(
                state = TrafficLightState.RED,
                duration = int((traffic_light_durations["red"] + traffic_light_durations["yellow"] + traffic_light_durations["green"]) / executed_scenario.dt)
            )
        
        yellow_cycle_element = TrafficLightCycleElement(
                state = TrafficLightState.YELLOW,
                duration = int(traffic_light_durations["yellow"] / executed_scenario.dt)
            )
        
        green_cycle_element = TrafficLightCycleElement(
                state = TrafficLightState.GREEN,
                duration = int(traffic_light_durations["green"] / executed_scenario.dt)
            )
        
        # Build traffic light cycle based on the initial state
        if traffic_light_initial_state == "GREEN":
            new_cycle = [green_cycle_element, yellow_cycle_element, red_cycle_element]
        elif traffic_light_initial_state == "YELLOW":
            new_cycle = [yellow_cycle_element, red_cycle_element, green_cycle_element]
        else:
            new_cycle = [red_cycle_element, green_cycle_element, yellow_cycle_element]
        
        traffic_light.traffic_light_cycle = TrafficLightCycle(cycle_elements = new_cycle)

    return executed_scenario

def extract_planning_problems(scenario: Scenario, pp_ids: List[int], scenario_specs: dict, map_lane_id_map: dict) -> PlanningProblemSet:
    """
    
    """
    # Initialize the planning problem set
    problem_set = PlanningProblemSet()
    
    # Extract planning problems from Apollo .json scenario specifications
    apollo_planning_problems = scenario_specs["ad_section"]["adcs"]

    # Iterate over the extracted planning problems
    for i, pp_id in enumerate(pp_ids):
        # Extract initial lane and position
        apollo_pp = apollo_planning_problems[i]
        start_lane = map_lane_id_map[apollo_pp["routing"][0]]
        start_position = float(apollo_pp["start_s"])
        
        # Extract destination lane and position
        dest_lane = map_lane_id_map[apollo_pp["routing"][-1]]
        dest_position = float(apollo_pp["dest_s"])
        
        # Extract and convert vehicle initial timestep
        initial_timestep = 10 * int(apollo_pp["start_t"])

        # Retrive CommonRoad initial lanelet and compute initial state position
        try:
            initial_lane = scenario.lanelet_network.find_lanelet_by_id(start_lane)
            starting_point = initial_lane.interpolate_position(start_position)[0]
        
            # Build initial state object
            # Simplification: orientation, velocity, acceleration, yaw_rate and slip_angle are set to 0
            init_state = InitialState(time_step = initial_timestep, 
                                    position = starting_point,
                                    orientation = 0.0, 
                                    velocity = 0.0, 
                                    acceleration = 0.0,
                                    yaw_rate = 0.0,
                                    slip_angle = 0.0)
            
            # Retrive CommonRoad destination lanelet
            destination_lane = scenario.lanelet_network.find_lanelet_by_id(dest_lane)
            # Get two consecutive points on the destination positionto compute the goal region orientation and position
            goal_point = destination_lane.interpolate_position(dest_position)[0]
            goal_next_point = destination_lane.interpolate_position(dest_position + 1.0)[0]

            orientation = math.atan2(goal_point[1] - goal_next_point[1], goal_point[0] - goal_next_point[0])
        except:
            continue

        # Build goal region object
        goal_area = Rectangle(7, 3, np.array([goal_point[0], goal_point[1]]), orientation)
        goal_state_list = [CustomState(position = goal_area, time_step = Interval(0, 300))]
        goal = GoalRegion(goal_state_list)

        # Build planning problem object and add it to the problem set
        pp = PlanningProblem(pp_id, init_state, goal)
        problem_set.add_planning_problem(pp)

    return problem_set

def convert_vehicle_perceptions(vehicle_perception: dict, ids_map: dict = None) -> Dict[int, List[Dict]]:
    """
    Convert vehicle perceptions from .record files message format to a standardized format.
    """
    perceptions = {}
    for perception in vehicle_perception:
        for obs in perception.perception_obstacle:
            if ids_map is None:
                obstacle_id = obs.id
            else:
                obstacle_id = ids_map[obs.id]
            obstacle_position = [obs.position.x, obs.position.y]
            obstacle_orientation = obs.theta
            obstacle_velocity = np.hypot(obs.velocity.x, obs.velocity.y)
            obstacle_length = obs.length
            obstacle_width = obs.width
            obstacle_height = obs.height
            perception_timestep = obs.timestamp

            entry = {
                "obstacle_id": obstacle_id,
                "obstacle_position": obstacle_position,
                "obstacle_orientation": obstacle_orientation,
                "obstacle_velocity": obstacle_velocity,
                "obstacle_length": obstacle_length,
                "obstacle_width": obstacle_width,
                "obstacle_height": obstacle_height
            } 

            if perception_timestep not in perceptions:
                perceptions[perception_timestep] = []

            perceptions[perception_timestep].append(entry)

    return perceptions

def adjust_perceptions(vehicles_perception: dict, scenario_start_timestep: int, id_label: str) -> dict:
    """
    Check scenario vehicles perception messages and adjust their format

    :param vehicles_perception: Dict containing the vehicles perceptions (obstacles or traffic lights) at each timestep
    :type vehicles_perception: dict
    :param scenario_start_timestep: Scenario initial timestep (in nanoseconds) to normalize the perception timesteps
    :type scenario_start_timestep: int
    :param id_label: Label used in the perception dict to identify the perceived obstacles
    :type id_label: str
    :return: Dict containing the adjusted vehicles perceptions at each timestep
    """
    adjusted_vehicles_perceptions = {}

    # Get each scenario vehicle obstacles perception
    for vehicle_id, vehicle_perceptions in vehicles_perception.items():
        # Iterate over the vehicle perceptions at each timesteps
        for timestep, perceptions in vehicle_perceptions.items():
            # Normalize and convert perception timestep to CommonRoad format (tenths of second)
            adjusted_timestep = round((timestep - (scenario_start_timestep / 1e9)) * 10)
            
            # Vehicle perception may start before vehicle's first history state
            if adjusted_timestep < 0:
                continue

            # Initialize vehicles perception dict
            if vehicle_id not in adjusted_vehicles_perceptions:
                adjusted_vehicles_perceptions[vehicle_id] = {}

            # Check if there are already perceptions at the adjusted timestep
            if adjusted_timestep not in adjusted_vehicles_perceptions[vehicle_id]:
                adjusted_vehicles_perceptions[vehicle_id][adjusted_timestep] = []
                
            # Get perceived obstacles ids
            existing = adjusted_vehicles_perceptions[vehicle_id][adjusted_timestep]
            existing_ids = {p[id_label] for p in existing}

            # Add only new obstacles
            for p in perceptions:
                if p[id_label] not in existing_ids:
                    existing.append(p)
                    existing_ids.add(p[id_label])

    return adjusted_vehicles_perceptions

def convert_traffic_light_perception(traffic_light_perception: List) -> Dict[int, List[Dict]]:
    """
    Convert traffic light perceptions from .record files message format to a standardized format.
    """
    traffic_light_perceptions = {}

    # Iterate over each traffic light perception message
    for tl_perception in traffic_light_perception:
        # Extract perception timestep
        perception_timestep = tl_perception.header.timestamp_sec
        # Iterate over each perceived traffic light
        for tl in tl_perception.traffic_light:
            entry = {
                "id": (int(tl.id.split("_")[1]) + 10000),
                "color": tl.color,
                "confidence": tl.confidence
            }

            if perception_timestep not in traffic_light_perceptions:
                traffic_light_perceptions[perception_timestep] = []

            traffic_light_perceptions[perception_timestep].append(entry)
        
    return traffic_light_perceptions

def convert_planned_trajectories(planned_trajectories: Dict, ids_map: Dict) -> Dict:
    """
    Convert interacting obstacle id when a new trajectory is planned
    """

    # Iterate over each planned trajectory
    for _, replanning_data in planned_trajectories.items():
        decision_reason = replanning_data["decision"]
        decision_object = replanning_data["decision_object"]

        # If replanning decision contains a vehicle id convert it
        if decision_reason in ["cruise", "STOP_REASON_OBSTACLE", "STOP_REASON_SIDE", "STOP_REASON_SIDEPASS_SAFETY"] and \
            decision_object not in [None, "Side_Pass_Stop"]:

            replanning_data["decision_object"] = ids_map[int(decision_object)]

    return planned_trajectories