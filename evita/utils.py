import logging
import sys
import os
import time

from typing import Tuple

### GLOBAL CONSTANTS

# Create the main logger
EVITA_MAIN_LOGGER_NAME = "evita.main_logger"

# This is where "this script" is stored
EVITA_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This is your Project Root
ROOT_DIR = os.path.abspath(os.path.join(EVITA_SCRIPT_DIR, os.pardir))
# This is where each execution of any generator is stored 
RESULTS_FOLDER = os.path.join(ROOT_DIR, "results")

# This is the location where the Frenetix-Motion-Planner submodule is checkout
FRENETIX_ROOT = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")

LANELET_EXTENSION = 150 # Default
try:
    LANELET_EXTENSION = int(os.getenv('LANELET_EXTENSION', 150))
except:
    pass

def initialize_main_logger(debug: bool=False) -> None:
    """
    Initialize the given logger in place
    """

    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

    # Set logging level
    if debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logger.setLevel(loglevel)

    # Create log formatter
    log_formatter = logging.Formatter("%(levelname)-6s [%(asctime)s] %(message)s ", "%Y-%m-%d %H:%M:%S")

    # Create stream handler (prints to stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(loglevel)

    stream_handler.setFormatter(log_formatter)

    # Add handlers
    logger.addHandler(stream_handler)

def get_main_logger() -> logging.Logger:
    return logging.getLogger(EVITA_MAIN_LOGGER_NAME)

def simulation_folders(run_folder: str) -> Tuple[str, str]:
    """
    Return the simulation folders that can be found in the runfolder
    """
    # Iterate over all the directories in the run folder, and return those that are not checkpoints or debug folders
    for simulation_folder_name in sorted([name for name in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, name))]):
        if "checkpoints" in simulation_folder_name:
            continue
        elif "debug" in simulation_folder_name:
            continue
        else:
            yield os.path.join(run_folder, simulation_folder_name), simulation_folder_name

def collisions_folders(simulation_folder: str) -> Tuple[str, str]:
    """
    Return the collision folders that can be found in the runfolder
    """
    if os.path.exists(os.path.join(simulation_folder, "logs", "collisions")):
        yield os.path.join(simulation_folder, "logs", "collisions"), "collisions"

def find_csv_files(path_to_dir, suffix=".csv" ) -> Tuple[str, str]:
    """
    Return the csv files that can be found in the given directory
    """
    # https://stackoverflow.com/questions/9234560/find-all-csv-files-in-a-directory-using-python
    for filename in os.listdir(path_to_dir):
        if filename.endswith( suffix ):
            yield os.path.join(path_to_dir, filename), filename

def run_folders(result_folder: str) -> Tuple[str, str]:
    """
    Return the folders of the runs that can be found in the result folder
    """
    # Iterate over all the directories
    for run_folder_name in sorted([name for name in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, name))]):
        yield os.path.join(result_folder, run_folder_name), run_folder_name

def folders_of_completed_runs(result_folder: str, limit: int = None) -> Tuple[str, str]:
    """
    Return only the folders of the runs that succesfully completed
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

    # Iterate over all the directories
    for run_folder_name in sorted([name for name in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, name))]):
        count += 1
        # Reconstruct the full path to the run folder
        full_path = os.path.join(result_folder, run_folder_name)

        # Check whether the folder contains the required files
        all_in = True
        for file_name in required_file_names:
            the_file = os.path.join(full_path, file_name)
            if not os.path.exists(the_file):
                print(f"Missing file {the_file} in run folder {full_path}")
                all_in = False
                break

        if all_in:
            # If all the files are there, return the folder
            yield os.path.join(result_folder, run_folder_name), run_folder_name

        if limit and count > limit:
            return

class Budget:
    """
    This class is responsible for keeping track of the time budget and the elapsed time during the execution of the generator
    """
    # Take from: https://github.com/sbft-cps-tool-competition/cps-tool-competition/

    def __init__(self, time_budget: float = None):
        self.time_budget = time_budget
        self.start_real_time = None
        self.start_pause = None
        self.time_spend_in_pause = 0
        self.last_check = None
        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

    def get_time_budget(self):
        return self.time_budget
    
    def get_elapsed(self) -> int:
        """
        Return how much "execution" time has passed since the beginning
        """
        now = time.perf_counter()    
        
        real_time_passed_since_beginning = now - self.start_real_time

        time_passed_since_the_beginning = max(0, real_time_passed_since_beginning - self.time_spend_in_pause)
        
        return int(time_passed_since_the_beginning)
    
    def get_progress(self) :
        """
        Return how much "execution" time has passed since the last time we invoked this method
        """
        if not self.last_check:
            # This is the first time we check progress, so we start from the beginning
            self.last_check = self.get_start_time()
        
        now = time.perf_counter()    
        real_time_passed_since_the_last_time_we_checked = now - self.last_check
        # Cannot return a negative number!
        time_passed_since_the_last_time_we_checked = max(0, real_time_passed_since_the_last_time_we_checked - self.time_spend_in_pause)
        
        self.last_check = now
        
        return int(time_passed_since_the_last_time_we_checked)

    def start(self):
        self.start_real_time = time.perf_counter()

    def get_start_time(self):
        return self.start_real_time

    def get_remaining_time(self):
        # We need to ensure that we always return the same type of object:
        return {"time-budget": self.get_remaining_real_time()}

    def get_remaining_real_time(self):
        """
        Return how much "real" time is left before the budget is over
        """
        now = time.perf_counter()
        passed_time_since_beginning = now - self.start_real_time
        actual_time_spent_in_doing = passed_time_since_beginning + self.time_spend_in_pause

        remaining_time = self.time_budget - actual_time_spent_in_doing
        
        return remaining_time

    def pause(self):
        # Start the counter only if not yet started
        if self.start_pause is None:
            self.start_pause = time.perf_counter()
    
    def resume(self):
        if self.start_pause:
            now = time.perf_counter()
            self.time_spend_in_pause = self.time_spend_in_pause + (now - self.start_pause)
            self.start_pause = None

    # General Check
    def is_over(self):
        return self.get_remaining_real_time() <= 0

from typing import Tuple
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from .generators.lanelet_extension import extend_the_outgoing_lanelets_with_a_straight
from copy import deepcopy

def extend_scenario_after_goal_areas(scenario: Scenario, extend_by_meters: int = 50) -> Scenario:
    """
    Make sure we add some road after the goal area to avoid the planner stopping in front of the end of the road.
    Code based on Stretch
    """

    extended_scenario = deepcopy(scenario)
    extended_scenario._lanelet_network = extend_the_outgoing_lanelets_with_a_straight(extended_scenario.lanelet_network, extension_distance = extend_by_meters)
    return extended_scenario

def time_shift(scenario: Scenario, planning_problem_set: PlanningProblemSet) -> PlanningProblemSet:
    """
    Shift the time of all the agents in the scenario so that the earliest one starts at time == 0
    Make sure that at least one agent starts at time == 0.
    NOTE: This works ONLY if there are not obstacles!
    """
    assert len(scenario.obstacles) == 0, "Cannot time shift if there are obstacles (Not yet supported)"

    time_shift_by = min([planning_problem.initial_state.time_step for planning_problem in planning_problem_set.planning_problem_dict.values()])

    planning_problem_list = []
    for planning_problem in planning_problem_set.planning_problem_dict.values():
        time_shifted_planning_problem = deepcopy(planning_problem)
        time_shifted_planning_problem.initial_state.time_step = time_shifted_planning_problem.initial_state.time_step - time_shift_by
        planning_problem_list.append(time_shifted_planning_problem)

    return PlanningProblemSet(planning_problem_list)

def augment_with_delay(planning_problem_set: PlanningProblemSet) -> PlanningProblemSet:
    """
    Introduce a new attribute, delay, and reset initial_state.time_step to 0
    """

    planning_problem_list = []
    for planning_problem in planning_problem_set.planning_problem_dict.values():
        time_shifted_planning_problem = deepcopy(planning_problem)
        setattr(time_shifted_planning_problem, "initial_delay", time_shifted_planning_problem.initial_state.time_step)
        time_shifted_planning_problem.initial_state.time_step = 0
        planning_problem_list.append(time_shifted_planning_problem)

    return PlanningProblemSet(planning_problem_list)