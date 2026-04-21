from ..interfaces import AbstractScenarioEvaluator
from ..generators.utils import augment_with_danger, augment_with_interactions, augment_with_collisions, augment_with_missed_goals, augment_with_emergencies_and_disengagements
from typing import Tuple

# import csv
import os
import sys

# import shutil
import pandas as pd

import traceback
from datetime import datetime

from cr_scenario_handler.evaluation.simulation_evaluation import evaluate_simulation, Evaluator
from cr_scenario_handler.simulation.simulation import Simulation

class FrenetixScenarioEvaluator(AbstractScenarioEvaluator):
    
    def __init__(self, output_folder:str, **kwargs) -> None:           
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
        self.output_folder = output_folder

    def evaluate(self, simulation: Simulation) -> pd.DataFrame:
        """
        Evaluate the simulation using the already implemented metrics, then augment the result with
        our additional metrics
        """
        
        # Invoke the original code - this also store the DB
        # - DCE is broken for colliding vehicles?
        criticality_dataframe = evaluate_simulation(simulation).criticality_dataframe
        # Extend evaluation with additional metrics for each agent
        #  - max_danger: at each time, what's the max_danger for each agent
        #  - interaction: at each time, was the agent under some interaction
        # level_0 -> agent_id, level_1 -> time_step
        # Ordered by level_0 and level_1
        criticality_dataframe.reset_index(inplace=True)

        criticality_dataframe = augment_with_danger(simulation, criticality_dataframe)
        
        criticality_dataframe = augment_with_interactions(simulation, criticality_dataframe)

        # Check the data from the collision report
        criticality_dataframe = augment_with_collisions(simulation, criticality_dataframe)
        
        # collect data about missed goals/timeouts
        criticality_dataframe = augment_with_missed_goals(simulation, criticality_dataframe)

        # collect data about emergency behaviors and disengagement
        criticality_dataframe = augment_with_emergencies_and_disengagements(simulation, criticality_dataframe)

        # Replace NaN with None otherwise SQLITE Complains
        criticality_dataframe = criticality_dataframe.replace({float('nan'): None})

        criticality_dataframe.rename(columns={'level_0': 'agent', 'level_1': 'time_step'}, inplace=True)
        criticality_dataframe.set_index(["agent", "time_step"], inplace=True)

        # Make sure we log everything inside the dataframe
        simulation.sim_logger.log_evaluation(criticality_dataframe)

        return criticality_dataframe

