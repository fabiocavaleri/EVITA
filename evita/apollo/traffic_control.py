"""
Traffic lights with DoppelTest's rules, on CommonRoad scenarios.

DoppelTest (framework/scenario/tc_config.py and TrafficControlManager.py) does not cycle the lights.
A scenario's traffic control is two configurations of all the map's signals (each GREEN or RED) and three durations, in whole seconds:

    0 < t <= g              initial configuration
    g < t <= g + y          transition: signals going from GREEN to RED are YELLOW, the others keep their initial color
    g + y < t <= g + y + b  every signal RED
    t > g + y + b           final configuration, until the end of the scenario
    (if initial == final, the initial configuration holds for the whole scenario)

A configuration respects the map's signal relations: related signals with "EQ" show the same color, signals with "NE" are never green together.

On the CommonRoad side, each traffic light gets four cycle elements (initial, transition, all RED, final).
All lights with the same durations; the last element lasts LONG_DURATION time steps, so the cycle never repeats within a scenario.
The first element is one time step longer than g, because DoppelTest's intervals include their end:
with time steps of 0.1 s, CommonRoad then shows at every time step exactly the color DoppelTest publishes at that time.

TrafficControl.from_commonroad reads the configuration back from a scenario written this way.
"""

import random
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.traffic_light import TrafficLightCycle, TrafficLightCycleElement, TrafficLightState

from .map_data import ApolloMapData, SAME_COLOR

GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

# DoppelTest's durations (tc_config.TCSection.get_one): yellow and all-red intervals, and the shortest green
DURATION_YELLOW = 3
DURATION_ALL_RED = 2
MIN_DURATION_GREEN = 5

# Duration (time steps) of the final element: the cycle never repeats within a scenario
LONG_DURATION = 10 ** 6

COMMONROAD_STATE = {GREEN: TrafficLightState.GREEN, YELLOW: TrafficLightState.YELLOW, RED: TrafficLightState.RED}
COLOR_OF_STATE = {state: color for color, state in COMMONROAD_STATE.items()}

class TrafficControlError(Exception):
    """
    A traffic control is invalid, or a scenario's traffic lights were not written by this module
    """

@dataclass(frozen = True)
class TrafficControl:
    """
    DoppelTest's traffic control of a scenario.
    Keys are CommonRoad traffic light ids; durations are whole seconds.
    """
    initial: Dict[int, str]
    final: Dict[int, str]
    duration_g: int
    duration_y: int = DURATION_YELLOW
    duration_b: int = DURATION_ALL_RED

    def __post_init__(self):
        # Check that the initial and final configurations cover the same traffic lights
        if set(self.initial) != set(self.final):
            raise TrafficControlError("The initial and final configurations do not cover the same traffic lights")

        # Check that the configurations only use GREEN and RED
        colors = set(self.initial.values()) | set(self.final.values())
        if not colors <= {GREEN, RED}:
            raise TrafficControlError(f"Configurations can only be GREEN or RED, not {sorted(colors - {GREEN, RED})}")
        
        # Check that the durations are positive whole seconds
        for name in ("duration_g", "duration_y", "duration_b"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise TrafficControlError(f"{name} must be a positive whole number of seconds, not {value!r}")

    # ---------- Colors over time (DoppelTest's TrafficControlManager) ----------

    def transition(self) -> Dict[int, str]:
        """
        Colors between g and g + y: YELLOW for the signals going from GREEN to RED
        """
        return {light: YELLOW if self.initial[light] == GREEN and self.final[light] == RED else self.initial[light]
                for light in self.initial}

    def colors_at(self, time: float) -> Dict[int, str]:
        """
        Color of every traffic light at a time (seconds from the scenario start), as DoppelTest publishes it
        """
        if self.initial == self.final or time <= self.duration_g:
            return dict(self.initial)

        if time <= self.duration_g + self.duration_y:
            return self.transition()

        if time <= self.duration_g + self.duration_y + self.duration_b:
            return {light: RED for light in self.initial}

        return dict(self.final)

    # ---------- CommonRoad ----------

    def _phases(self, light: int) -> List[str]:
        """
        Colors of one light in the four phases (initial, transition, all RED, final)
        """
        if self.initial == self.final:
            return [self.initial[light]] * 4

        return [self.initial[light], self.transition()[light], RED, self.final[light]]

    def _durations(self, dt: float) -> List[int]:
        """
        Durations (time steps) of the four phases; the first includes the time step at t = g
        """

        steps_per_second = round(1 / dt)
        
        if abs(steps_per_second * dt - 1) > 1e-9:
            raise TrafficControlError(f"Time step {dt} s does not divide one second")
        
        return [self.duration_g * steps_per_second + 1, self.duration_y * steps_per_second,
                self.duration_b * steps_per_second, LONG_DURATION]

    def to_commonroad(self, lanelet_network: LaneletNetwork, dt: float) -> None:
        """
        Write the traffic control into the network's traffic lights (every light of the network must be configured)
        """
        lights = {light.traffic_light_id: light for light in lanelet_network.traffic_lights}

        if set(lights) != set(self.initial):
            raise TrafficControlError(f"The traffic control configures lights {sorted(self.initial)}, the network has {sorted(lights)}")

        durations = self._durations(dt)
        for light_id, light in lights.items():
            light.traffic_light_cycle = TrafficLightCycle([TrafficLightCycleElement(COMMONROAD_STATE[color], duration) for color, duration in zip(self._phases(light_id), durations)])
            light.active = True

    @classmethod
    def from_commonroad(cls, lanelet_network: LaneletNetwork, dt: float) -> "TrafficControl":
        """
        Read back the traffic control written by to_commonroad
        """
        steps_per_second = round(1 / dt)
        initial, final, durations = {}, {}, None

        # Iterate over scenario's traffic lights
        for light in lanelet_network.traffic_lights:
            # Extract traffic light cycle and elements' durations
            cycle = light.traffic_light_cycle
            elements = cycle.cycle_elements if cycle is not None else []
            light_durations = [element.duration for element in elements]

            # Check that the light was written by to_commonroad, and that all lights have the same durations
            if len(elements) != 4 or light_durations[3] != LONG_DURATION or (cycle.time_offset or 0) != 0:
                raise TrafficControlError(f"Traffic light {light.traffic_light_id} was not written by traffic_control")

            if durations is not None and light_durations != durations:
                raise TrafficControlError(f"Traffic light {light.traffic_light_id} has different durations than the others")
                
            durations = light_durations

            colors = [COLOR_OF_STATE.get(element.state) for element in elements]
            initial[light.traffic_light_id] = colors[0]
            final[light.traffic_light_id] = colors[3]

        if durations is None:
            raise TrafficControlError("The network has no traffic light")

        green_steps, yellow_steps, red_steps = durations[0] - 1, durations[1], durations[2]
        if any(steps % steps_per_second for steps in (green_steps, yellow_steps, red_steps)):
            raise TrafficControlError(f"Durations {durations[:3]} are not whole seconds")

        return cls(initial, final, green_steps // steps_per_second, yellow_steps // steps_per_second, red_steps // steps_per_second)

    # ---------- DoppelTest's JSON ----------

    def to_doppeltest(self, map_data: ApolloMapData) -> Dict:
        """
        The tc_section of DoppelTest's scenario JSON (Apollo signal ids)
        """
        return {
            "initial": {map_data.signal_id(light): color for light, color in self.initial.items()},
            "final": {map_data.signal_id(light): color for light, color in self.final.items()},
            "duration_g": self.duration_g,
            "duration_y": self.duration_y,
            "duration_b": self.duration_b
        }

    @classmethod
    def from_doppeltest(cls, tc_section: Dict, map_data: ApolloMapData) -> "TrafficControl":
        """
        Traffic control from the tc_section of DoppelTest's scenario JSON
        """
        return cls(
            initial = {map_data.traffic_light_id(signal): color for signal, color in tc_section["initial"].items()},
            final = {map_data.traffic_light_id(signal): color for signal, color in tc_section["final"].items()},
            duration_g = int(tc_section["duration_g"]),
            duration_y = int(tc_section["duration_y"]),
            duration_b = int(tc_section["duration_b"])
        )

# ---------- Random generation and mutation (DoppelTest's tc_config) ----------

def random_configuration(map_data: ApolloMapData, traffic_lights: Iterable[int], preferred_green: Sequence[int] = (), random_generator = random) -> Dict[int, str]:
    """
    Generate a configuration respecting the signal relations.
    Signals are taken in random order, preferred ones first; each one taken becomes GREEN, and fixes the color of its related signals.
    """
    remaining = list(traffic_lights)
    random_generator.shuffle(remaining)
    preferred = [light for light in preferred_green if light in remaining]
    configuration = {}

    # Keep iterating until all lights are configured, taking preferred ones first
    while remaining:
        # Assign first the prioritized traffic lights
        if preferred:
            light = preferred.pop()
            remaining.remove(light)
        else:
            light = remaining.pop()

        # Assign green light color to the traffic light
        configuration[light] = GREEN

        # Assign colors to the related traffic lights, removing them from the remaining and preferred lists
        for related, relation in map_data.related_traffic_lights(light):
            if related in preferred:
                preferred.remove(related)
            if related in remaining:
                remaining.remove(related)
                configuration[related] = GREEN if relation == SAME_COLOR else RED

    return configuration

def random_traffic_control(map_data: ApolloMapData, traffic_lights: Iterable[int], max_duration_g: int, random_generator = random) -> TrafficControl:
    """
    A random traffic control: the final configuration prefers green for the signals that are red at the start, and green lasts between MIN_DURATION_GREEN and max_duration_g seconds
    """
    traffic_lights = list(traffic_lights)
    initial = random_configuration(map_data, traffic_lights, random_generator = random_generator)
    final = random_configuration(map_data, traffic_lights, [light for light, color in initial.items() if color == RED], random_generator)
    
    return TrafficControl(initial, final, random_generator.randint(MIN_DURATION_GREEN, max(MIN_DURATION_GREEN, max_duration_g)))

def mutate_configurations(traffic_control: TrafficControl, map_data: ApolloMapData, random_generator = random) -> TrafficControl:
    """
    New initial and final configurations, same durations
    """
    lights = list(traffic_control.initial)
    initial = random_configuration(map_data, lights, random_generator = random_generator)
    final = random_configuration(map_data, lights, [light for light, color in initial.items() if color == RED], random_generator)

    return replace(traffic_control, initial = initial, final = final)

def mutate_green_duration(traffic_control: TrafficControl, max_duration_g: int, random_generator = random) -> TrafficControl:
    """
    New green duration, same configurations
    """
    return replace(traffic_control, duration_g = random_generator.randint(MIN_DURATION_GREEN, max(MIN_DURATION_GREEN, max_duration_g)))

def max_duration_green(scenario_duration_steps: int, dt: float) -> int:
    """
    DoppelTest's longest green: half the scenario duration, in whole seconds
    """
    return int(scenario_duration_steps * dt / 2)