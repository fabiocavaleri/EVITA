"""
Simulator-neutral result of a scenario execution.

Executors return an ExecutionResult
The orchestrator, EVITA's NSGA-II problem, the generators and the fitness functions only use this result, never the simulator's own objects.
The simulator-specific handle (`native`) is meant for the evaluator and for releasing the simulator's resources only.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from commonroad.scenario.scenario import Scenario

# Final status of an agent (names of FrenetiX's AgentStatus); other simulators map their outcomes onto these names
STATUS_COMPLETED = "COMPLETED_SUCCESS"
STATUS_COLLISION = "COLLISION"
STATUS_TIMELIMIT = "TIMELIMIT"
STATUS_MISSED_GOAL = "MAX_S_POSITION"
STATUS_ERROR = "ERROR"

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

    @property
    def not_started(self) -> List[int]:
        return [agent_id for agent_id, agent in self.agents.items() if not agent.started]