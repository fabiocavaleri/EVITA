"""
Evaluator of FrenetiX simulations: FrenetiX's own criticality metrics, extended with EVITA's metrics
(danger, interactions, collisions, missed goals, emergencies), stored in the simulation's database (table scenario_evaluation).
"""
import os
import sys
import logging
import pandas as pd

from cr_scenario_handler.evaluation.simulation_evaluation import evaluate_simulation
from cr_scenario_handler.simulation.simulation import Simulation

from ..data_analysis import _rebuild_trajectory
from ..generators.utils import (augment_with_collisions, augment_with_danger, augment_with_emergencies_and_disengagements,
                                augment_with_interactions, augment_with_missed_goals)
from ..interfaces import AbstractScenarioEvaluator
from ..timing import TASK_TIMER
from ..utils import EVITA_MAIN_LOGGER_NAME, sanitize_dataframe

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

class FrenetixScenarioEvaluator(AbstractScenarioEvaluator):
    """
    Evaluates a simulation with FrenetiX's criticality metrics, then adds EVITA's metrics.
    The result, one row per agent and time step, is written to the simulation's database and returned.
    """
    def __init__(self, output_folder: str, **kwargs) -> None:           
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
        self.output_folder = output_folder
    
    @staticmethod
    def _planned_trajectories(simulation: Simulation) -> pd.DataFrame:
        """
        Planned trajectory of each agent at each time step, read from the agents' logs.csv
        """
        rows = []

        for agent_id in simulation.agent_id_list:
            agent_log = pd.read_csv(os.path.join(simulation.config.simulation.log_path, str(agent_id), "logs.csv"), sep = ';')
            
            for record in sorted(agent_log.to_dict(orient = "records"), key = lambda r: int(r["trajectory_number"])):
                timestep = int(record["trajectory_number"])
                rows.append([agent_id, timestep, _rebuild_trajectory(timestep, record["x_positions_m"], record["y_positions_m"], record["theta_orientations_rad"], record["velocities_mps"])])

        return pd.DataFrame(rows, columns = ["vehicle_id", "timestep", "trajectory"])
    
    def evaluate(self, simulation: Simulation, trajectory_difference_threshold = 1.0, on_step = None) -> pd.DataFrame:
        """
        Evaluation table of the simulation (index: agent, time_step). `trajectory_difference_threshold` is Th_ED of the interaction detection;
        `on_step` is called after each phase to refresh a progress bar.
        """
        def sync_progress_bar():
            if on_step is not None:
                on_step()

        with TASK_TIMER.measure("evaluation.planned_trajectories"):
            trajectories_df = self._planned_trajectories(simulation)
        sync_progress_bar()

        for obstacle in simulation.scenario.obstacles:
            module_logger.debug(f"Obstacle {obstacle.obstacle_id} has initial state at timestep {obstacle.initial_state.time_step} and final state at timestep {obstacle.prediction.trajectory.state_list[-1].time_step}")

        # FrenetiX's own evaluation (it also stores the database)
        module_logger.info("Evaluating simulation...")
        with TASK_TIMER.measure("evaluation.frenetix_metrics"):
            criticality_dataframe = evaluate_simulation(simulation).criticality_dataframe

        sync_progress_bar()

        # Extend evaluation with additional metrics for each agent:
        #  - max_danger: at each time, what's the max_danger for each agent
        #  - interaction: at each time, was the agent under some interaction
        #  - collision: at each time, was the agent involved in a collision
        #  - missed_goal: at each time, did the agent miss its goal
        #  - emergency: at each time, was the agent in an emergency situation
        # level_0 -> agent_id, level_1 -> time_step
        # Ordered by level_0 and level_1
        criticality_dataframe.reset_index(inplace = True)

        module_logger.info("Computing danger criticality metric...")
        with TASK_TIMER.measure("evaluation.danger"):
            criticality_dataframe = augment_with_danger(simulation, criticality_dataframe)

        sync_progress_bar()

        module_logger.info("Computing interactions...")
        with TASK_TIMER.measure("evaluation.interactions"):
            criticality_dataframe = augment_with_interactions(simulation, trajectories_df, criticality_dataframe, trajectory_difference_threshold)

        sync_progress_bar()

        module_logger.info("Computing collisions...")
        with TASK_TIMER.measure("evaluation.collisions"):
            criticality_dataframe = augment_with_collisions(simulation, criticality_dataframe)

        sync_progress_bar()

        # collect data about missed goals/timeouts
        module_logger.info("Computing missed goals and timeouts...")
        with TASK_TIMER.measure("evaluation.missed_goals"):
            criticality_dataframe = augment_with_missed_goals(simulation, criticality_dataframe)

        sync_progress_bar()

        # collect data about emergency behaviors and disengagement
        module_logger.info("Computing emergencies and disengagements...")
        with TASK_TIMER.measure("evaluation.emergencies"):
            criticality_dataframe = augment_with_emergencies_and_disengagements(simulation, criticality_dataframe)

        sync_progress_bar()
        
        # Replace NaN with None otherwise SQLITE Complains
        criticality_dataframe = criticality_dataframe.replace({float('nan'): None})

        criticality_dataframe.rename(columns = {'level_0': 'agent', 'level_1': 'time_step'}, inplace = True)
        criticality_dataframe.set_index(["agent", "time_step"], inplace = True)

        criticality_dataframe = sanitize_dataframe(criticality_dataframe)

        # Add missing columns to the database if they don't exist yet
        with TASK_TIMER.measure("evaluation.database"):
            con = simulation.sim_logger.con
            existing = {row[1] for row in con.execute("PRAGMA table_info(scenario_evaluation)")}

            for col in criticality_dataframe.columns:
                if col not in existing:
                    con.execute(f'ALTER TABLE scenario_evaluation ADD COLUMN "{col}" ANY')

            con.execute("DELETE FROM scenario_evaluation WHERE scenario=?", (simulation.config.simulation.name_scenario,))
            con.commit()

            # Make sure we log everything inside the dataframe
            simulation.sim_logger.log_evaluation(criticality_dataframe)
        sync_progress_bar()

        return criticality_dataframe