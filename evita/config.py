from commonroad.scenario.lanelet import LaneletType

class ValidationConfig:
    """
    This class contains the configuration parameters for the scenario validation process.
    """
    lanelet_types = {LaneletType.HIGHWAY, LaneletType.ACCESS_RAMP, LaneletType.EXIT_RAMP, LaneletType.URBAN}

    # This validation is not needed when taking valid scenarios from the online dataset
    validate_lanelet_type_and_direction = False
    lanelet_same_direction_only = False

    # How close the vehicle can be to its goal area
    min_dist_goal = 20
    
    # The vehicle must be at least 5 meters inside any of the initial lanelets
    min_dist_init = 5
    retry_generation = 2000

    ignore_validation = False  # for debugging purposes

    def __init__(self, safety_time = 2) -> None:
        # Time in seconds for a safe distance between two cars
        self.safety_time = safety_time

    # Min distance of 2m seems reasonable. It cannot be 0.0 otherwise they overlap!
    def car_distance_formula(self, speed):
        if speed > 1.0:
            return speed * self.safety_time
        else:
            return 2.0


from cr_scenario_handler.utils.configuration import VehicleParameterMapping, VehicleType

class ScenarioGeneratorConfig:
    """
    This class contains the configuration parameters for the scenario generation process.
    """
    def __init__(self, scenario_duration: int = 300, max_planning_problems: int = 8):
        # Length of goal area at the end of a lanelet
        self.dist_to_end = 10.0

        # CR vehicle type (see commonroad-vehicle-models): #1: Ford Escort, #2: BMW 320i, 3 = VW Vanagon
        self.cr_vehicle_id = 2

        # Get vehicle parameters from CommonRoad vehicle models given cr_vehicle_id
        vehicle_parameters = VehicleParameterMapping.from_vehicle_type(VehicleType(self.cr_vehicle_id))

        # Get dimensions from given vehicle ID
        self.length = vehicle_parameters.l
        self.width = vehicle_parameters.w
        self.wb_front_axle = vehicle_parameters.a
        self.wb_rear_axle = vehicle_parameters.b
        self.wheelbase = vehicle_parameters.a + vehicle_parameters.b
        self.mass = vehicle_parameters.m

        self.amount_pp = 1  # initially added planning problems
        
        self.min_amount_pp = 2 # min amount of planning problems in a scenario
        self.max_amount_pp = max_planning_problems # max amount of planning problems in a scenario
        
        self.do_mutations = True  # for debugging
        self.max_position_modifier = 50  # max amount for position mutations; min amount is -max_position_modifier
        self.max_velocity = 25  # max velocity of an ego by mutation
        self.min_velocity = 0  # min velocity of an ego by mutation
        self.max_init_velocity = 10  # max init velocity for new egos; also max velocity modifier for velocity mutations
        self.min_init_velocity = 5  # see max_init_velocity
        self.car_distance_formula = ValidationConfig().car_distance_formula
        #
        self.max_time_step = scenario_duration  # maximum scenario duration in simulation steps (default: 300 -> 30 seconds)
        self.emergency_slow_down_amount = 5  # how much a car is slowing down for each step it spends in emergency mode
        self.max_emergency_steps = 4
        #
        self.av_base_id = 10000 # AV IDs start from this number
        #
        # Interaction maps definition
        self.n_bins = 10
        
