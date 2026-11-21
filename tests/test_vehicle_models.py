"""
Tests of the custom vehicle models in FrenetiX (cr_scenario_handler/utils/vehicle_models.py) and their use in FrenetiX.
Run from the project root:  python3 tests/test_vehicle_models.py [Frenetix-Motion-Planner]
"""
import os, sys
import numpy as np

frenetix = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "Frenetix-Motion-Planner")
sys.path.insert(0, frenetix)
results = []
def check(name, ok):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name)

from omegaconf import OmegaConf
from commonroad.common.solution import VehicleType, VehicleModel
from commonroad.scenario.state import KSState, InputState
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
from cr_scenario_handler.utils import vehicle_models as vm
from cr_scenario_handler.utils.configuration import VehicleConfiguration
from cr_scenario_handler.evaluation import agent_evaluation as ae
from types import SimpleNamespace

check("ids 1-3: CommonRoad's own parameters", all(vm.vehicle_parameters(i) is VehicleParameterMapping.from_vehicle_type(VehicleType(i)) for i in (1, 2, 3)))
yaml = OmegaConf.load(os.path.join(frenetix, "configurations/simulation/vehicle.yaml"))
yaml.cr_vehicle_id = 2; v2 = VehicleConfiguration(yaml)
check("FrenetiX's vehicle configuration with 2: BMW 320i as before", (v2.length, v2.width) == (4.508, 1.61))
yaml.cr_vehicle_id = 101; v = VehicleConfiguration(yaml)
check("FrenetiX's vehicle configuration with 101: Apollo's size and wheelbase", (v.length, v.width, round(v.wheelbase, 4)) == (4.933, 2.11, 2.8448))
check("rear axle = Apollo's reference point (1.043 m from the rear edge)", abs(vm.back_edge_to_center(101) - 1.043) < 1e-9)
check("BMW 320i's shared parameters untouched", VehicleParameterMapping.from_vehicle_type(VehicleType(2)).l == 4.508)
dyn = vm.vehicle_dynamics(VehicleModel.KS, 101)
check("KS dynamics with 101: shape and steering-rate bound", (dyn.shape.length, dyn.shape.width) == (4.933, 2.11) and abs(dyn.input_bounds.ub[0] - 6.98131700798 / 16) < 1e-9)
check("solution files use the base type", vm.commonroad_vehicle_type(101) == VehicleType.BMW_320i)
config = SimpleNamespace(vehicle = SimpleNamespace(cr_vehicle_id = 101, wb_rear_axle = v.wb_rear_axle))
states = [KSState(time_step = 0, position = np.array([0.0, 0.0]), steering_angle = 0.0, velocity = 10.0, orientation = 0.0)]
rebuilt = ae.reconstruct_states(config, SimpleNamespace(dt = 0.1), states, [InputState(time_step = 0, steering_angle_speed = 0.0, acceleration = 1.0)])
check("FrenetiX's agent evaluation with 101 (state reconstruction)", len(rebuilt) == 2 and abs(rebuilt[1].velocity - 10.1) < 1e-6)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
