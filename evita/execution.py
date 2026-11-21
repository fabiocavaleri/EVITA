"""
Simulator-neutral result of a scenario execution.

Executors return an ExecutionResult
The orchestrator, EVITA's NSGA-II problem, the generators and the fitness functions only use this result, never the simulator's own objects.
The simulator-specific handle (`native`) is meant for the evaluator and for releasing the simulator's resources only.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory

# Final status of an agent (names of FrenetiX's AgentStatus); other simulators map their outcomes onto these names
STATUS_COMPLETED = "COMPLETED_SUCCESS"
STATUS_COLLISION = "COLLISION"
STATUS_TIMELIMIT = "TIMELIMIT"
STATUS_MISSED_GOAL = "MAX_S_POSITION"
STATUS_ERROR = "ERROR"

@dataclass(frozen = True)
class ObjectDecision:
    """
    A planner's decision about one object in one plan, for simulators that report them (Apollo)
    """
    decision: str                    # follow, yield, overtake, nudge, stop, avoid, side_pass, ignore
    object_kind: str                 # vehicle, traffic_light, stop_sign, pedestrian, crosswalk, destination, ...
    object_id: Optional[int]         # EVITA's id of the object (vehicle id, traffic light id, ...); None if not mapped
    raw_object_id: str               # the simulator's own id (e.g. Apollo's "TL_signal_3", "212397_0")
    reason: Optional[str] = None     # stop reason (e.g. "STOP_REASON_SIGNAL")

@dataclass
class Plan:
    """
    One plan of an agent: its planned trajectory, and the planner's decisions when the simulator reports them
    """
    time_step: int                                          # time step at which the plan was made
    trajectory: Trajectory                                  # planned trajectory (CommonRoad states)
    object_decisions: Optional[List[ObjectDecision]] = None # None: the simulator reports no decisions (FrenetiX)
    main_decision: Optional[str] = None                     # whole-vehicle decision (Apollo: cruise, stop, ...)
    main_reason: Optional[str] = None                       # its stop reason, if any

@dataclass
class AgentOutcome:
    """
    Represents the outcome of a single agent in a simulation scenario.
    """
    id: int
    initial_time_step: int
    status: str                  # "COMPLETED_SUCCESS", "COLLISION", "TIMELIMIT", "MAX_S_POSITION", "ERROR", ...
    last_timestep: int
    driven_states: List = field(default_factory = list)
    emergency_steps: Set[int] = field(default_factory = set)
    disengagement_steps: Set[int] = field(default_factory = set)
    plans: Optional[List[Plan]] = None 

    @property
    def started(self) -> bool:
        failed_at_start = self.status == STATUS_ERROR and self.last_timestep == 0
        return not (failed_at_start or self.initial_time_step == self.last_timestep)

@dataclass
class ExecutionResult:
    """
    Represents the outcome of a simulation scenario execution.
    """
    scenario_id: object
    output_folder: str
    executed_scenario: Scenario      # every vehicle as an obstacle with its driven trajectory
    agents: Dict[int, AgentOutcome]
    native: Optional[object] = None  # FrenetiX: the Simulation (for the evaluator and release only)
    simulator: str = "frenetix"      # "frenetix" or "apollo": selects the interaction rule
    # Loads the plans of every agent ({agent id: plans}); called by load_plans, so that reading them counts as evaluation time.
    # None: the plans are already set.
    plan_loader: Optional[Callable[[], Dict[int, List[Plan]]]] = field(default = None, repr = False)
    _loaded_plans: Optional[Dict[int, List[Plan]]] = field(default = None, init = False, repr = False)

    def load_plans(self) -> Dict[int, List[Plan]]:
        """
        Plans of every agent ({agent id: plans}, in the loader's order), loaded once; also set on each AgentOutcome
        """
        if self._loaded_plans is None:
            if self.plan_loader is not None:
                self._loaded_plans = self.plan_loader()
            else:
                self._loaded_plans = {agent_id: agent.plans for agent_id, agent in self.agents.items() if agent.plans is not None}

            for agent_id, plans in self._loaded_plans.items():
                if int(agent_id) in self.agents:
                    self.agents[int(agent_id)].plans = plans
                    
        return self._loaded_plans

    @property
    def not_started(self) -> List[int]:
        return [agent_id for agent_id, agent in self.agents.items() if not agent.started]