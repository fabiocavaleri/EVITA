from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict
from enum import Enum

import pandas as pd

from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.evaluation.simulation_evaluation import Evaluator

class AbstractFitnessFunction(ABC):
    """
    This is the reference (abstract) class defining the interface of the Fitness Function. We use a class to allow for archive and other state management
    """

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """
        Use whatever arguments are passed
        """
        pass

    @abstractmethod
    def compute(self, agents_data, evaluation_data) -> float|List[float]:
        """
        Return the fitness evaluation, which can be a float or a list of float in the case of Multi-Objective evaluation
        """
        pass

class AbstractScenarioGenerator(ABC):
    """
    This is the reference (abstract) class defining the interface of the Scenario Generator
    """

    @abstractmethod
    def __init__(self, map_file: Optional[str]=None, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, **kwargs) -> List:
        """
        Return new scenarios (in CommonRoad) format
        """
        pass


    # original_scenario, simulated_scenario, av_data
    @abstractmethod
    def evaluate(self, simulation_data: List[Dict]) -> None:
        """
        Evaluate the simulated scenarios (in CommonRoad format) and additional metrics about agent behavior.
        """
        pass

class AbstractScenarioExecutor(ABC):
    """
    This is the reference (abstract) class defining the interface of the Scenario Executor
    """

    @abstractmethod
    def __init__(self, output_folder:str, **kwargs) -> None:
        pass

    @abstractmethod
    def execute(self, scenario, planning_problem_set) -> Simulation:
        """
        Return a Simulation Object
        """
        pass

class AbstractScenarioEvaluator(ABC):
    """
    This is the reference (abstract) class defining the interface of the Scenario Evaluator
    """

    @abstractmethod
    def __init__(self, output_folder:str, **kwargs) -> None:
        pass

    @abstractmethod
    def evaluate(self, simulation: Simulation) -> pd.DataFrame:
        """
        Return a Evaluator object
        """
        pass
