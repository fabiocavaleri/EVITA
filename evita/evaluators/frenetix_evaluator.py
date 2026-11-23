"""
Evaluator of FrenetiX simulations. The evaluation is common to every simulator (evita/evaluators/scenario_evaluator.py):
this class keeps the name the orchestrator and the scripts use, and the reading of a Simulation's plans for scripts
that pass a Simulation (e.g. scripts/false_positive_analysis.py).
"""
import pandas as pd

from cr_scenario_handler.simulation.simulation import Simulation

from ..executors.frenetix_executor import read_frenetix_plans
from .scenario_evaluator import ScenarioEvaluator


class FrenetixScenarioEvaluator(ScenarioEvaluator):
    """
    The common evaluator (ScenarioEvaluator); with a FrenetiX ExecutionResult it evaluates exactly as before: FrenetiX's
    criticality metrics, EVITA's metrics, FrenetiX's database.
    """

    @staticmethod
    def _planned_trajectories(simulation: Simulation) -> pd.DataFrame:
        """
        Planned trajectory of each agent at each time step, read from the agents' logs.csv (kept for scripts that pass
        a Simulation, e.g. scripts/false_positive_analysis.py; the evaluation reads the plans of the ExecutionResult)
        """
        return ScenarioEvaluator._plans_table(read_frenetix_plans(simulation.config.simulation.log_path, simulation.agent_id_list))