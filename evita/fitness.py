import pandas as pd
from typing import List

from .interfaces import AbstractFitnessFunction

class MaxDangerFitnessFunction(AbstractFitnessFunction):

    def __init__(self, **kwargs) -> None:
        """
        This basic function has no state
        """
        pass

    def compute(self, agents_data, evaluation_data: pd.DataFrame) -> float|List[float]:
        """
        Assume a danger value for each agent in each time_step has been already computed
        by the orchestrator. We need the overall maximum, so we return the max value of the "max_danger"
        column.
        """
        return evaluation_data["max_danger"].max()
    

class MaxInteractionsFitnessFunction(AbstractFitnessFunction):

    def __init__(self, **kwargs) -> None:
        """
        This basic function has no state
        """
        pass

    def compute(self, agents_data, evaluation_data: pd.DataFrame) -> float|List[float]:
        """
        Compute the overall number of times AV changed their trajectory because of a "likely" interaction.
        This is likely to increase with the number of AV, however, increasing the number of AV also leads to an increase of the complexity of generating the scenarios
        """

        return evaluation_data["is_interacting"].sum()