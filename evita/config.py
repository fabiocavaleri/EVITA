
"""
Configuration of scenario generation and validation
"""
from commonroad.scenario.lanelet import LaneletType
from cr_scenario_handler.utils.configuration import VehicleParameterMapping, VehicleType

class ValidationConfig:
    """
    Parameters of the scenario validation (paper, Section 4.5).
    """
    # Lanelet types on which vehicles may be placed (checked only if validate_lanelet_type_and_direction is set)
    lanelet_types = {LaneletType.HIGHWAY, LaneletType.ACCESS_RAMP, LaneletType.EXIT_RAMP, LaneletType.URBAN}

    # Lanelet type and direction checks: not needed for the maps of the CommonRoad dataset, which are already valid
    validate_lanelet_type_and_direction = False
    lanelet_same_direction_only = False

    # Minimum distance (metres, along the route) between a vehicle's start and its goal area
    min_dist_goal = 20

    # Minimum distance (metres) between a vehicle's start and the beginning of an entry lanelet,
    # so that the whole vehicle starts on the road
    min_dist_init = 5

    # Maximum number of attempts to generate a valid scenario
    retry_generation = 2000

    # Skip validation altogether (debugging only)
    ignore_validation = False

    def __init__(self, safety_time: float = 2.0) -> None:
        # Time headway (seconds) defining the safety distance between two vehicles in the same lane
        self.safety_time = safety_time

    def car_distance_formula(self, speed: float) -> float:
        """
        Minimum distance (metres) that a vehicle driving at `speed` (m/s) must keep from the vehicle ahead:
        the distance covered in `safety_time` seconds, and at least 2 m so that standing vehicles do not overlap.
        """
        if speed > 1.0:
            return speed * self.safety_time
        else:
            return 2.0

class ScenarioGeneratorConfig:
    """
    Parameters of the scenario generation and of the interaction maps.
    """

    def __init__(self, scenario_duration: int = 300, max_planning_problems: int = 8) -> None:
        # Distance (metres) between the goal area and the end of its lanelet (dist_to_end)
        self.dist_to_end = 10.0

        # CommonRoad vehicle model: 1 = Ford Escort, 2 = BMW 320i, 3 = VW Vanagon
        self.cr_vehicle_id = 2
        vehicle_parameters = VehicleParameterMapping.from_vehicle_type(VehicleType(self.cr_vehicle_id))

        # Vehicle dimensions (metres): used for goal areas and for the longitudinal range of the interaction map
        self.length = vehicle_parameters.l
        self.width = vehicle_parameters.w

        # Number of vehicles (planning problems) per scenario (min_AVs, max_AVs)
        self.min_amount_pp = 2
        self.max_amount_pp = max_planning_problems

        # Range (metres) of the position mutation: [-max_position_modifier, +max_position_modifier]
        self.max_position_modifier = 50

        # Range (m/s) of the initial velocity of new vehicles, also used as the range of velocity mutations
        self.min_init_velocity = 5
        self.max_init_velocity = 10

        # Safety distance between vehicles in the same lane
        self.car_distance_formula = ValidationConfig().car_distance_formula

        # Maximum scenario duration in simulation steps of 0.1 s (max_time_step)
        self.max_time_step = scenario_duration

        # Identifiers of the generated vehicles start from this number
        self.av_base_id = 10000

        # Number of bin edges per axis of the interaction map (n_bins): n_bins - 1 bins plus one bucket on each side
        self.n_bins = 10