"""
Health check of an Apollo execution: verify whether Apollo ran the scenario as specified, so that its evaluation can be used.

An execution is unhealthy when a vehicle:
- did not run: no localization (its Apollo instance did not start);
- never moved (localized, but never left its start; FrenetiX's rule for agents that do not start);
- lost its planning: no plan at all for at least PLANNING_STOPPED_LIMIT seconds or more of its run, without having
  completed its mission.
  A localized vehicle without any plan has lost its planning too.

A collision ends a vehicle's evaluation, so what happens after its first collision does not count.
"""
from typing import List

from .trace import TraceConversion

# Seconds without any plan at the end of a vehicle's run, without its mission completed, that mean its planner died.
PLANNING_STOPPED_LIMIT = 5.0

def health_problems(conversion: TraceConversion, collision_steps = None) -> List[str]:
    """
    Verify if the execution can be used.
    Empty if healthy.
    `collision_steps`: each vehicle's first collision step, where its evaluation ends.
    """
    problems = []
    collision_steps = collision_steps or {}

    for vehicle_id, vehicle in conversion.vehicles.items():
        name = f"vehicle {vehicle_id} ({vehicle.container})"
        if vehicle.obstacle is None:
            problems.append(f"{name} did not run (no localization)")
            continue

        collision = collision_steps.get(vehicle_id)
        if vehicle.first_motion_step is None and collision is None:
            problems.append(f"{name} never moved")
            continue

        if vehicle.mission_complete_step is not None:
            continue

        if collision is None:
            last_step = vehicle.last_time_step
        else:
            last_step = min(vehicle.last_time_step, collision)

        plan_steps = [p.time_step for p in vehicle.plans]
        if not plan_steps:
            problems.append(f"{name} received no plan at all")
            continue

        idle = (last_step - max(plan_steps)) * conversion.dt
        if idle >= PLANNING_STOPPED_LIMIT:
            problems.append(f"{name} lost its planning: no plan during the last {idle:.1f} s of its run, mission not complete")
            
    return problems