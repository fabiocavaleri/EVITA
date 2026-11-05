"""
Shared utilities: paths, logging, progress bars, result folders, the time budget, and helpers for scenarios and
data frames.
"""
import json
import os
import sys
import time
import logging
import enlighten
import numpy as np
import pandas as pd

from copy import deepcopy
from enum import Enum
from typing import Iterator, Optional, Tuple

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

from .generators.lanelet_extension import extend_the_outgoing_lanelets_with_a_straight

# ========== Global constants ==========

# Name of EVITA's main logger
EVITA_MAIN_LOGGER_NAME = "evita.main_logger"

# Folder of this package, root folder of the project, and folder where each run stores its results
EVITA_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EVITA_SCRIPT_DIR, os.pardir))
RESULTS_FOLDER = os.path.join(ROOT_DIR, "results")

# Checkout of the Frenetix-Motion-Planner submodule
FRENETIX_ROOT = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")

# Metres of straight road added after the end of the road network, so planners do not stop before the end of the
# road; can be overridden with the LANELET_EXTENSION environment variable
try:
    LANELET_EXTENSION = int(os.getenv("LANELET_EXTENSION", "150"))
except ValueError:
    LANELET_EXTENSION = 150

# ========== Logging and progress bars ==========

def initialize_main_logger(debug: bool = False) -> None:
    """
    Configure EVITA's main logger: level DEBUG or INFO, printing to stdout.
    The handler is added only once, so calling this function again does not duplicate the log lines.
    """
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    
    if debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logger.setLevel(loglevel)

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(levelname)-6s [%(asctime)s] %(message)s ", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(stream_handler)

    for handler in logger.handlers:
        handler.setLevel(loglevel)

def get_main_logger() -> logging.Logger:
    """
    EVITA's main logger.
    """
    return logging.getLogger(EVITA_MAIN_LOGGER_NAME)

def get_progress_bar_manager():
    """
    Manager of the terminal progress bars.
    """
    try:
        terminal = open("/dev/tty", "w")
        return enlighten.get_manager(stream = terminal)
    except OSError:
        return enlighten.get_manager()

# ========== Result folders ==========

def simulation_folders(run_folder: str) -> Iterator[Tuple[str, str]]:
    """
    (path, name) of each simulation (scenario) folder of a run, in name order, skipping checkpoint and debug folders.
    """
    names = sorted(name for name in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, name)))
    for name in names:
        if "checkpoints" in name or "debug" in name:
            continue
        yield os.path.join(run_folder, name), name

def collisions_folders(simulation_folder: str) -> Iterator[Tuple[str, str]]:
    """
    (path, name) of the collisions folder of a simulation, if it exists.
    """
    collisions_folder = os.path.join(simulation_folder, "logs", "collisions")
    if os.path.exists(collisions_folder):
        yield collisions_folder, "collisions"

def find_csv_files(path_to_dir: str, suffix: str = ".csv") -> Iterator[Tuple[str, str]]:
    """
    (path, name) of each file in the folder whose name ends with `suffix`.
    """
    for filename in os.listdir(path_to_dir):
        if filename.endswith(suffix):
            yield os.path.join(path_to_dir, filename), filename

def folders_of_completed_runs(result_folder: str, limit: Optional[int] = None) -> Iterator[Tuple[str, str]]:
    """
    (path, name) of each run folder that contains all the files a completed run writes, in name order;
    at most `limit` folders if given.
    """
    required_file_names = ["behavior.csv",
                           "collision_map.npy",
                           "collision_map_impact_location_and_riv.npy",
                           "collision_map_only_impact_location.npy",
                           "interaction_map.npy",
                           "metadata.csv",
                           "performance.csv",
                           "requirements_violations.csv"]

    print(f"Checking run folders in {result_folder}")

    count = 0
    names = sorted(name for name in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, name)))
    for name in names:
        if limit is not None and count >= limit:
            return

        full_path = os.path.join(result_folder, name)
        missing = [f for f in required_file_names if not os.path.exists(os.path.join(full_path, f))]
        if missing:
            print(f"Missing file {os.path.join(full_path, missing[0])} in run folder {full_path}")
            continue

        count += 1
        yield full_path, name

# ========== Time budget ==========

class Budget:
    """
    This class is responsible for keeping track of the time budget and the elapsed time during the execution of the generator
    """
    # Take from: https://github.com/sbft-cps-tool-competition/cps-tool-competition/

    def __init__(self, time_budget: Optional[float] = None) -> None:
        self.time_budget = time_budget
        self.start_real_time = None
        self.start_pause = None
        self.time_spend_in_pause = 0.0
        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

    def get_time_budget(self) -> Optional[float]:
        """
        Total budget in seconds.
        """
        return self.time_budget

    def get_start_time(self) -> Optional[float]:
        """
        perf_counter() value at start(), None if not started.
        """
        return self.start_real_time

    def start(self) -> None:
        """
        Start counting.
        """
        self.start_real_time = time.perf_counter()

    def _active_time(self) -> float:
        """
        Seconds since start(), excluding completed pauses and any pause in progress; 0 if not started.
        """
        if self.start_real_time is None:
            return 0.0
        
        now = time.perf_counter()
        paused = self.time_spend_in_pause + ((now - self.start_pause) if self.start_pause is not None else 0.0)
        return max(0.0, now - self.start_real_time - paused)

    def get_elapsed(self) -> int:
        """
        Return how much "execution" time has passed since the beginning
        """
        return int(self._active_time())

    def get_remaining_real_time(self) -> float:
        """
        Active seconds left before the budget is over.
        """
        return self.time_budget - self._active_time()

    def pause(self) -> None:
        """
        Stop counting until resume(); calling it again while paused has no effect.
        """
        if self.start_pause is None:
            self.start_pause = time.perf_counter()
    
    def resume(self) -> None:
        """
        Count again after pause(); has no effect if not paused.
        """
        if self.start_pause is not None:
            self.time_spend_in_pause += time.perf_counter() - self.start_pause
            self.start_pause = None

    def is_over(self) -> bool:
        """
        True when the budget is used up.
        """
        return self.get_remaining_real_time() <= 0

# ========== Scenarios ==========

def extend_scenario_after_goal_areas(scenario: Scenario, extend_by_meters: int = 50) -> Scenario:
    """
    Copy of the scenario with a straight road of `extend_by_meters` metres added after each lanelet without
    successors, so that planners do not stop before the end of the road.
    Code based on Stretch
    """

    extended_scenario = deepcopy(scenario)
    extended_scenario._lanelet_network = extend_the_outgoing_lanelets_with_a_straight(extended_scenario.lanelet_network, extension_distance = extend_by_meters)
    return extended_scenario

def time_shift(scenario: Scenario, planning_problem_set: PlanningProblemSet) -> PlanningProblemSet:
    """
    Copy of the planning problems with their initial time steps shifted so that the earliest one starts at 0.
    Not supported for scenarios with obstacles, whose trajectories would also need shifting.
    """
    if len(scenario.obstacles) > 0:
        raise ValueError("Cannot time shift a scenario with obstacles (not supported)")

    planning_problems = list(planning_problem_set.planning_problem_dict.values())
    time_shift_by = min(planning_problem.initial_state.time_step for planning_problem in planning_problems)

    shifted = []
    for planning_problem in planning_problems:
        shifted_planning_problem = deepcopy(planning_problem)
        shifted_planning_problem.initial_state.time_step -= time_shift_by
        shifted.append(shifted_planning_problem)

    return PlanningProblemSet(shifted)

# ========== Data frames ==========

def _to_sqlite_value(value):
    """
    `value` as a type SQLite can store: None, int, float, str or bytes. Containers become JSON strings.
    """
    # Arrays and containers first: pd.isna() does not apply to them
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value)

    # NumPy scalars become native Python values
    if isinstance(value, np.generic):
        return value.item()

    # Missing values
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, Enum):
        return value.value

    # Already compatible
    if isinstance(value, (int, float, str, bytes)):
        return value

    return str(value)

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy of `df` with only SQLite-compatible values (None, int, float, str, bytes).
    """
    df = df.copy()

    for column in df.columns:
        df[column] = df[column].apply(_to_sqlite_value)
    
    return df