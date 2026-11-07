"""
Entry point of EVITA: NSGA-II search on a seed road network, with FrenetiX.

The initial population is random; each generation is produced by crossover and mutation and executed, until the time budget (minutes) is used up
The scenarios, the metadata and the performance counters are stored in the output folder, followed by the report.
"""
import sys

# The Apollo-related modules imported by the operators live in the sibling EVITA_DoppelTest project
sys.path.append("../EVITA_DoppelTest")

import os
import csv
import random
import shutil
import logging
import argparse
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import generate_report
from evita.generators import utils as scenario_generation_utils
from evita.generators.NSGAIIproblem import NSGAIIproblem
from evita.generators.nsgaII_operators import (CustomTimeTermination, ScenarioCrossover, ScenarioDuplicateElimination,
                                               ScenarioMutation, ScenarioSampling)
from evita.orchestrator import write_performance_file
from evita.timing import TASK_TIMER
from evita.utils import EVITA_MAIN_LOGGER_NAME, get_progress_bar_manager, initialize_main_logger

# FrenetiX is the driving agent of every vehicle
DRIVING_AGENT_NAME = "frenetix"

# Th_ED: a replanning (condition C1) is detected when a planned position moves by at least this many metres
TRAJECTORY_DIFFERENCE_THRESHOLD = 2.5

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Run EVITA's NSGA-II search.")

    parser.add_argument("--population-size", type = int, required = True, help = "Number of scenarios in each generation.")
    parser.add_argument("--time-budget", type = float, required = True, help = "Time budget in minutes.")
    parser.add_argument("--scenario-seed-file", type = str, required = True, help = "Path to the CommonRoad scenario seed file (the road network).")

    parser.add_argument("--output-folder-name", type = str, default = None, help = "Output folder name. Defaults to an automatically generated name.")
    parser.add_argument("--seed", type = int, default = None, help = "Random seed for reproducibility.")
    parser.add_argument("--threshold", type = float, default = TRAJECTORY_DIFFERENCE_THRESHOLD, help = "Trajectory difference threshold Th_ED in metres (default: 1.0).")
    parser.add_argument("--n-bins", type = int, default = None, help = "Number of bin edges per axis of the interaction map (default: configuration).")
    parser.add_argument("--checkpoint-every", type = int, default = 60, help = "Checkpoint every N minutes of budget time (default: 60; 0 disables).")
    parser.add_argument("--force-max-duration", type = int, default = None, help = "Force the goal time interval of every planning problem to [0, N] (as `evita run --force-max-duration`).")
    parser.add_argument("--debug", action = "store_true", help = "Enable debug logging and plots.")

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    # Set the random seed for reproducibility, if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Initialize the main logger
    initialize_main_logger(debug = args.debug)
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    logger.info(30 * "=" + " Start EVITA " + 30 * "=")

    # Load the scenario object from the seed file, and leave only the lanelet network
    base_scenario, base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(args.scenario_seed_file, remove_obstacles = True, remove_planning_problems = True)

    # Initialize the scenario generator configuration
    scenario_generator_configuration = ScenarioGeneratorConfig()

    progress_bar_manager = get_progress_bar_manager()
    termination_criterion = CustomTimeTermination(args.time_budget, checkpoint_every = args.checkpoint_every, progress_bar_manager = progress_bar_manager)

    # Initialize the NSGAII algorithm ti generate scenario and the mutation and crossover operators
    sampling = ScenarioSampling(base_scenario, base_planning_problem_set, scenario_generator_configuration, termination_criterion)
    mutation = ScenarioMutation(args.scenario_seed_file, scenario_generator_configuration, termination_criterion)
    crossover = ScenarioCrossover(scenario_generator_configuration, termination_criterion)
    algorithm = NSGA2(pop_size = args.population_size,
                      sampling = sampling,
                      mutation = mutation,
                      crossover = crossover,
                      eliminate_duplicates = ScenarioDuplicateElimination())
    
    problem = NSGAIIproblem(args.scenario_seed_file,
                            DRIVING_AGENT_NAME,
                            args.population_size,
                            termination_criterion,
                            args.threshold,
                            args.n_bins,
                            output_folder_name = args.output_folder_name,
                            debug = args.debug,
                            force_max_duration = args.force_max_duration
                        )

    # Copy the base scenario file in the output folder
    shutil.copyfile(args.scenario_seed_file, os.path.join(problem.output_folder, os.path.basename(args.scenario_seed_file)))
    
    # Generate metadafile
    metadata_file = os.path.join(problem.output_folder, "metadata.csv")
    with open(metadata_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["generator", "scenario_name", "time_budget", "timestamp_id"])
        csv_writer.writerow([f"{problem.scenario_generator_module}.{problem.scenario_generator_class}", str(base_scenario.scenario_id), str(args.time_budget * 60), str(problem.timestamp_id)])

    algorithm.setup(problem, termination = termination_criterion, save_history = False, verbose = True, seed = args.seed)

    # Start the time budget
    termination_criterion.start()
    TASK_TIMER.reset()

    try:
        algorithm.run()
    finally:
        # Close the progress bar of the last generation
        if problem.generation_bar is not None:
            problem.generation_bar.close(clear = True)
            problem.generation_bar = None
 
        # Final refresh of the main bar, then restore the terminal
        termination_criterion.update_main_progress_bar()
        progress_bar_manager.stop()

    logger.info(f"Execution of {problem.output_folder} is over ")
    
    # Log performance metrics
    logger.info("Generation Performance Metrics")
    
    performance_file_path = os.path.join(problem.output_folder, "performance.csv")
    write_performance_file(performance_file_path,
                           termination_criterion.all_scenarios,
                           termination_criterion.valid_scenarios,
                           termination_criterion.invalid_scenarios,
                           termination_criterion.total_generation_time,
                           termination_criterion.skipped_scenarios)
    
    logger.info("Saved Performance Metrics")

    # Budget time actually used, for the cross-check in task_times.csv
    budget = termination_criterion.budget
    budget_elapsed_seconds = budget.get_time_budget() - budget.get_remaining_real_time()

    # Now analysize the data collected so far and output to the scenario folder
    with TASK_TIMER.measure("report"):
        generate_report(problem.output_folder)

    # Time spent in each task (task_times.csv, task_times_per_scenario.csv)
    TASK_TIMER.write(problem.output_folder, budget_elapsed_seconds)
    logger.info("Saved Task Times")

if __name__ == '__main__':
    main()