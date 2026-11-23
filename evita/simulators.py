"""
The simulators EVITA can run scenarios with, selected by `--simulator` in both entry points (nsgaII_search_no_generator.py, and the orchestrator.py).

For each simulator: the executor and the evaluator, the scenario generation configuration (number of vehicles, scenario length, vehicle model) and the validator's mode.
Apollo's congifurations are taken from DoppelTest's own configuration (EVITA_DoppelTest/config.py).
"""
import os
import yaml
from typing import Optional
from dataclasses import dataclass

from cr_scenario_handler.utils.vehicle_models import APOLLO_LINCOLN_MKZ, vehicle_parameters

from .utils import FRENETIX_ROOT
from .validator import ScenarioValidator
from .config import ScenarioGeneratorConfig, ValidationConfig

FRENETIX = "frenetix"
APOLLO = "apollo"

@dataclass(frozen = True)
class Simulator:
    name: str
    executor_path: str
    executor_module: str
    executor_class: str
    evaluator_module: str
    evaluator_class: str
    apollo_validation: bool                # the validator's Apollo mode
    max_vehicles: Optional[int] = None     # None: ScenarioGeneratorConfig's default
    scenario_duration: Optional[int] = None  # timesteps; None: ScenarioGeneratorConfig's default
    vehicle_model: Optional[int] = None    # cr_vehicle_id; None: ScenarioGeneratorConfig's default

# FrenetiX's planning configuration: its time step (dt, seconds) is the time step of EVITA's scenarios
FRENETIX_PLANNING_CONFIGURATION = os.path.join(FRENETIX_ROOT, "configurations", "frenetix_motion_planner", "planning.yaml")

def frenetix_time_step() -> float:
    """
    FrenetiX's time step in seconds (planning.yaml, dt)
    """
    with open(FRENETIX_PLANNING_CONFIGURATION) as f:
        return float(yaml.safe_load(f)["dt"])

def _apollo_limits():
    """
    DoppelTest's limits: Apollo instances (vehicles) per scenario, and the scenario length (SCENARIO_UPPER_LIMIT, in
    seconds) converted into time steps with FrenetiX's time step
    """
    from EVITA_DoppelTest.config import MAX_ADC_COUNT, SCENARIO_UPPER_LIMIT
    return MAX_ADC_COUNT, int(round(SCENARIO_UPPER_LIMIT / frenetix_time_step()))

SIMULATORS = {
    FRENETIX: Simulator(name = FRENETIX, executor_path = FRENETIX_ROOT,
                        executor_module = "evita.executors.frenetix_executor", executor_class = "FrenetixScenarioExecution",
                        evaluator_module = "evita.evaluators.frenetix_evaluator", evaluator_class = "FrenetixScenarioEvaluator",
                        apollo_validation = False),
    APOLLO: Simulator(name = APOLLO, executor_path = ".",
                      executor_module = "evita.executors.apollo_executor", executor_class = "ApolloScenarioExecution",
                      evaluator_module = "evita.evaluators.scenario_evaluator", evaluator_class = "ScenarioEvaluator",
                      apollo_validation = True, vehicle_model = APOLLO_LINCOLN_MKZ),
}

def get_simulator(name: str) -> Simulator:
    if name not in SIMULATORS:
        raise ValueError(f"Unknown simulator {name}: one of {', '.join(SIMULATORS)}")
    
    return SIMULATORS[name]

def generation_configuration(simulator: Simulator) -> ScenarioGeneratorConfig:
    """
    The scenario generation configuration for the simulator (FrenetiX: the default one, as before)
    """
    if simulator.name == APOLLO:
        max_vehicles, duration = _apollo_limits()
        config = ScenarioGeneratorConfig(duration, max_vehicles)
    else:
        config = ScenarioGeneratorConfig()

    if simulator.vehicle_model is not None:
        config.cr_vehicle_id = simulator.vehicle_model
        parameters = vehicle_parameters(simulator.vehicle_model)
        config.length, config.width = parameters.l, parameters.w

    return config

def scenario_validator(simulator: Simulator, validation_configuration: ValidationConfig, generation_config: Optional[ScenarioGeneratorConfig] = None) -> ScenarioValidator:
    """
    The validator for the simulator (Apollo: the validator's Apollo mode)
    """
    return ScenarioValidator(validation_configuration, generation_config, apollo = simulator.apollo_validation)
