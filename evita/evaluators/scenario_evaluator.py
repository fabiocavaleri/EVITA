"""
Evaluator of executed scenarios, for every simulator (FrenetiX, Apollo).
Creates one table per scenario, one row per agent and time step, with EVITA's metrics (danger, interactions, collisions, missed goals and timeouts, emergencies and disengagements),
stored in the scenario's database (table scenario_evaluation).

Common to both simulators: danger, collisions, missed goals, timeouts, emergencies, disengagements.
Per simulator:
- FrenetiX: FrenetiX's own criticality metrics; interactions are detected checking C1 and C2.
- Apollo: Interactions come from Apollo's decisions; the database complies with the tables and columns of FrenetiX's.
"""
import os
import sys
import json
import sqlite3
import logging
import pandas as pd

from cr_scenario_handler.evaluation.simulation_evaluation import evaluate_simulation

from ..timing import TASK_TIMER
from ..execution import ExecutionResult
from ..interfaces import AbstractScenarioEvaluator
from ..utils import EVITA_MAIN_LOGGER_NAME, sanitize_dataframe
from ..generators.utils import (AGENT_COLUMN, TIME_COLUMN, augment_with_collisions, augment_with_danger,
                                augment_with_decision_interactions, augment_with_emergencies_and_disengagements,
                                augment_with_interactions, augment_with_missed_goals)

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

# FrenetiX's database stores scenario_evaluation.agent_id as the vehicle id + 10;
# the Apollo database follows the same convention, so that the analysis scripts read both alike
AGENT_ID_OFFSET_IN_DATABASE = 10


def apollo_base_table(execution: ExecutionResult) -> pd.DataFrame:
    """
    Create a DataFrame with one row per agent and timestep of its executed trajectory
    """
    rows = []

    for agent_id in sorted(execution.agents):
        driven = execution.agents[agent_id].driven_states
        if driven:
            rows += [(agent_id, t) for t in range(driven[0].time_step, driven[-1].time_step + 1)]

    return pd.DataFrame(rows, columns = [AGENT_COLUMN, TIME_COLUMN])

def write_evaluation_database(database: str, scenario_name: str, evaluation: pd.DataFrame, agent_ids) -> None:
    """
    Write the evaluation (index: agent, time_step) to a database with FrenetiX's tables meta and scenario_evaluation
    """
    os.makedirs(os.path.dirname(os.path.abspath(database)), exist_ok = True)

    con = sqlite3.connect(database)
    try:
        # Create and populate the `meta` table with the scenario's metadata (one row per scenario)
        con.execute("""
            CREATE TABLE IF NOT EXISTS meta(
                scenario TEXT NOT NULL ,
                num_agents INT Not NULL,
                agent_ids ANY,
                agent_batches ANY,
                duration_init REAL NOT NULL,
                sim_duration REAL,
                post_duration REAL,
                simulation_config ANY NOT NULL,
                planner_config TEXT NOT NULL,
                PRIMARY KEY(scenario))""")
        
        con.execute("INSERT OR REPLACE INTO meta VALUES(?,?,?,?,?,?,?,?,?)",
                    [scenario_name, len(agent_ids), json.dumps(list(agent_ids)), json.dumps([]), 0.0, None, None,
                     json.dumps({"simulator": "apollo"}), json.dumps({})])

        # Create and populate the `scenario_evaluation` table with the evaluation (one row per agent and timestep)
        columns = list(evaluation.columns)
        definitions = "".join(f'"{column}" ANY,\n' for column in columns)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS scenario_evaluation(
                scenario TEXT NOT NULL,
                agent_id INT NOT NULL,
                timestep INT NOT NULL,
                {definitions}PRIMARY KEY(scenario, agent_id, timestep))""")
        
        existing = {row[1] for row in con.execute("PRAGMA table_info(scenario_evaluation)")}
        for column in columns:
            if column not in existing:
                con.execute(f'ALTER TABLE scenario_evaluation ADD COLUMN "{column}" ANY')

        con.execute("DELETE FROM scenario_evaluation WHERE scenario=?", (scenario_name,))

        data = [[scenario_name, agent_id + AGENT_ID_OFFSET_IN_DATABASE, t] + [str(v) for v in evaluation.loc[(agent_id, t)]] for agent_id, t in evaluation.index]
        
        if data:
            names = ", ".join(["scenario", "agent_id", "timestep"] + [f'"{c}"' for c in columns])
            con.executemany(f"INSERT INTO scenario_evaluation ({names}) VALUES ({', '.join(['?'] * (len(columns) + 3))})", data)
        
        con.commit()
    finally:
        con.close()

class ScenarioEvaluator(AbstractScenarioEvaluator):
    """
    Evaluates an executed scenario (ExecutionResult)
    """
    def __init__(self, output_folder: str, **kwargs) -> None:
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        self.output_folder = output_folder

    @staticmethod
    def _plans_table(plans_by_agent) -> pd.DataFrame:
        """
        Planned trajectory of each agent at each time step ({agent id: plans} -> one row per plan)
        """
        rows = [[agent_id, plan.time_step, plan.trajectory] for agent_id, plans in plans_by_agent.items() for plan in plans]

        return pd.DataFrame(rows, columns = ["vehicle_id", "timestep", "trajectory"])

    def evaluate(self, execution: ExecutionResult, trajectory_difference_threshold, on_step = None) -> pd.DataFrame:
        """
        Evaluation table of the executed scenario (index: agent, time_step).
        `trajectory_difference_threshold` is Th_ED of FrenetiX's interaction detection (not used for Apollo);
        `on_step` is called after each phase to refresh a progress bar.
        """
        def sync_progress_bar():
            if on_step is not None:
                on_step()

        frenetix = execution.simulator == "frenetix"

        # Extract the planned trajectories of each agent at each time step (only for FrenetiX)
        with TASK_TIMER.measure("evaluation.planned_trajectories"):
            plans = execution.load_plans()
            if frenetix:
                trajectories_df = self._plans_table(plans)
            else:
                trajectories_df = None

        sync_progress_bar()

        # Create the base evaluation table (one row per agent and time step) with FrenetiX's or Apollo's metrics
        if frenetix:
            simulation = execution.native

            for obstacle in simulation.scenario.obstacles:
                module_logger.debug(f"Obstacle {obstacle.obstacle_id} has initial state at timestep {obstacle.initial_state.time_step} and final state at timestep {obstacle.prediction.trajectory.state_list[-1].time_step}")

            # FrenetiX's own evaluation (it also stores the database)
            module_logger.info("Evaluating simulation...")
            with TASK_TIMER.measure("evaluation.frenetix_metrics"):
                criticality_dataframe = evaluate_simulation(simulation).criticality_dataframe

            # level_0 -> agent_id, level_1 -> time_step; ordered by level_0 and level_1
            criticality_dataframe.reset_index(inplace = True)
        else:
            with TASK_TIMER.measure("evaluation.base_table"):
                criticality_dataframe = apollo_base_table(execution)

        sync_progress_bar()

        module_logger.info("Computing danger criticality metric...")
        with TASK_TIMER.measure("evaluation.danger"):
            criticality_dataframe = augment_with_danger(execution, criticality_dataframe)

        sync_progress_bar()

        module_logger.info("Computing interactions...")
        with TASK_TIMER.measure("evaluation.interactions"):
            if frenetix:
                criticality_dataframe = augment_with_interactions(execution, trajectories_df, criticality_dataframe, trajectory_difference_threshold)
            else:
                criticality_dataframe = augment_with_decision_interactions(execution, criticality_dataframe)

        sync_progress_bar()

        module_logger.info("Computing collisions...")
        with TASK_TIMER.measure("evaluation.collisions"):
            criticality_dataframe = augment_with_collisions(execution, criticality_dataframe)

        sync_progress_bar()

        module_logger.info("Computing missed goals and timeouts...")
        with TASK_TIMER.measure("evaluation.missed_goals"):
            criticality_dataframe = augment_with_missed_goals(execution, criticality_dataframe)

        sync_progress_bar()

        module_logger.info("Computing emergencies and disengagements...")
        with TASK_TIMER.measure("evaluation.emergencies"):
            criticality_dataframe = augment_with_emergencies_and_disengagements(execution, criticality_dataframe)

        sync_progress_bar()

        # Replace NaN with None otherwise SQLITE Complains
        criticality_dataframe = criticality_dataframe.replace({float('nan'): None})

        criticality_dataframe.rename(columns = {'level_0': 'agent', 'level_1': 'time_step'}, inplace = True)
        criticality_dataframe.set_index(["agent", "time_step"], inplace = True)

        criticality_dataframe = sanitize_dataframe(criticality_dataframe)

        with TASK_TIMER.measure("evaluation.database"):
            if frenetix:
                self._store_frenetix(simulation, criticality_dataframe)
            else:
                database = os.path.join(execution.output_folder, "logs", "simulation.db")
                write_evaluation_database(database, str(execution.scenario_id), criticality_dataframe, sorted(execution.agents))
                
        sync_progress_bar()

        return criticality_dataframe

    @staticmethod
    def _store_frenetix(simulation, criticality_dataframe: pd.DataFrame) -> None:
        """
        Store the evaluation in FrenetiX's database (adding EVITA's columns to its scenario_evaluation table)
        """
        con = simulation.sim_logger.con
        existing = {row[1] for row in con.execute("PRAGMA table_info(scenario_evaluation)")}

        for col in criticality_dataframe.columns:
            if col not in existing:
                con.execute(f'ALTER TABLE scenario_evaluation ADD COLUMN "{col}" ANY')

        con.execute("DELETE FROM scenario_evaluation WHERE scenario=?", (simulation.config.simulation.name_scenario,))
        con.commit()

        # Make sure we log everything inside the dataframe
        simulation.sim_logger.log_evaluation(criticality_dataframe)
