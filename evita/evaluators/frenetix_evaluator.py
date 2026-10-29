from ..interfaces import AbstractScenarioEvaluator
from ..generators.utils import augment_with_danger, augment_with_interactions, augment_with_collisions, augment_with_missed_goals, augment_with_emergencies_and_disengagements
from ..data_analysis import _rebuild_trajectory
from ..utils import sanitize_dataframe, EVITA_MAIN_LOGGER_NAME

import os
import sys
import logging
import pandas as pd

from cr_scenario_handler.evaluation.simulation_evaluation import evaluate_simulation
from cr_scenario_handler.simulation.simulation import Simulation

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME + "." + __name__)

class FrenetixScenarioEvaluator(AbstractScenarioEvaluator):
    """
    This class is responsible for evaluating the simulation using the already implemented metrics, then augment the result with some additional metrics
    """
    def __init__(self, output_folder:str, **kwargs) -> None:           
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
        self.output_folder = output_folder
    
    def evaluate(self, simulation: Simulation, trajectory_difference_threshold = 1.0) -> pd.DataFrame:
        """
        Evaluate the simulation using the already implemented metrics, then augment the result with
        our additional metrics
        """

        # Retrieve scenario agents
        agent_ids = simulation.agent_id_list

        # Retrieve the path to the logs folder
        log_files_folder = simulation.config.simulation.log_path

        # Initialize the DataFrame to store the trajectories
        trajectories_df = pd.DataFrame(columns = ["vehicle_id", "timestep", "trajectory"])

        for agent_id in agent_ids:
            # Open agent's log file
            agent_log_file_path = os.path.join(log_files_folder, str(agent_id), "logs.csv")
            agent_log = pd.read_csv(agent_log_file_path, sep = ';')

            # There's a dictionary for each line, and a line for each time_step
            data = agent_log.to_dict(orient = "records")

            # Sort by time_step - Reconstruct all the trajectories
            for timestep, xs, ys, orientations, velocities in sorted([(int(d["trajectory_number"]), d["x_positions_m"], d["y_positions_m"], d["theta_orientations_rad"], d["velocities_mps"]) for d in data], key = lambda tup: tup[0]):
                trajectories_df.loc[len(trajectories_df)] = [agent_id, timestep, _rebuild_trajectory(timestep, xs, ys, orientations, velocities)]

        for obstacle in simulation.scenario.obstacles:
            module_logger.debug(f"Obstacle {obstacle.obstacle_id} has initial state at timestep {obstacle.initial_state.time_step} and final state at timestep {obstacle.prediction.trajectory.state_list[-1].time_step}")

        # Invoke the original code - this also stores the DB
        module_logger.info("Evaluating simulation...")
        criticality_dataframe = evaluate_simulation(simulation).criticality_dataframe

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
        criticality_dataframe = augment_with_danger(simulation, criticality_dataframe)

        module_logger.info("Computing interactions...")
        criticality_dataframe = augment_with_interactions(simulation, trajectories_df, criticality_dataframe, trajectory_difference_threshold)

        module_logger.info("Computing collisions...")
        criticality_dataframe = augment_with_collisions(simulation, criticality_dataframe)

        # collect data about missed goals/timeouts
        module_logger.info("Computing missed goals and timeouts...")
        criticality_dataframe = augment_with_missed_goals(simulation, criticality_dataframe)

        # collect data about emergency behaviors and disengagement
        module_logger.info("Computing emergencies and disengagements...")
        criticality_dataframe = augment_with_emergencies_and_disengagements(simulation, criticality_dataframe)
        
        # Replace NaN with None otherwise SQLITE Complains
        criticality_dataframe = criticality_dataframe.replace({float('nan'): None})

        criticality_dataframe.rename(columns = {'level_0': 'agent', 'level_1': 'time_step'}, inplace = True)
        criticality_dataframe.set_index(["agent", "time_step"], inplace = True)

        criticality_dataframe = sanitize_dataframe(criticality_dataframe)

        # Add missing columns to the database if they don't exist yet
        con = simulation.sim_logger.con

        existing = {row[1] for row in con.execute("PRAGMA table_info(scenario_evaluation)")}

        for col in criticality_dataframe.columns:
            if col not in existing:
                con.execute(f'ALTER TABLE scenario_evaluation ADD COLUMN "{col}" ANY')

        simulation.sim_logger.con.execute("DELETE FROM scenario_evaluation WHERE scenario=?", (simulation.config.simulation.name_scenario,))

        con.commit()

        # Make sure we log everything inside the dataframe
        simulation.sim_logger.log_evaluation(criticality_dataframe)

        return criticality_dataframe