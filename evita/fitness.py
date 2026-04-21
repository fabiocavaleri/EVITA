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
        #  agent_interactions_count_df = pd.DataFrame(None, index=list(range(t_start, t_end+1)), columns=["is_interacting", "is_interacting_with"])
        # agent_interactions_count_df["is_interacting"] = is_interacting
        # agent_interactions_count_df["is_interacting_with"] = is_interacting_with
        return evaluation_data["is_interacting"].sum()