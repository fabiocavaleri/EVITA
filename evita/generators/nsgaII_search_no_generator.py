import sys
sys.path.append("../EVITA_DoppelTest")  # Add the path to the EVITA_DoppelTest package to the system path

import logging
import os
import csv
import argparse
import shutil
import enlighten  # Progress bars
from pymoo.algorithms.moo.nsga2 import NSGA2

import random
import numpy as np

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import generate_report
from evita.generators.NSGAIIproblem import NSGAIIproblem
from evita.generators.nsgaII_operators import ScenarioSampling, ScenarioMutation, ScenarioCrossover, \
    ScenarioDuplicateElimination, CustomTimeTermination
from evita.generators import utils as scenario_generation_utils

from evita.utils import initialize_main_logger, EVITA_MAIN_LOGGER_NAME, get_progress_bar_manager

if __name__ == '__main__':
    """
    This is the main entry point for the NSGAII Search without a generator
    It will generate scenarios by applying mutation and crossover operators to the seed scenario, and evaluate them using the given evaluation pipeline
    The generated scenarios will be stored in the output folder, together with metadata and performance metrics
    """  
    parser = argparse.ArgumentParser(description = "Run NSGAII Search without a generator.")

    # Required arguments
    parser.add_argument(
        "--population-size",
        type = int,
        required = True,
        help = "Number of scenarios to generate in each generation."
    )
    parser.add_argument(
        "--time-budget",
        type = float,
        required = True,
        help = "Maximum generation time in minutes."
    )
    parser.add_argument(
        "--scenario-seed-file",
        type = str,
        required = True,
        help = "Path to the CommonRoad scenario seed file."
    )

    # Optional arguments
    parser.add_argument(
        "--output-folder-name",
        type = str,
        default = None,
        help = "Output folder name. Defaults to an automatically generated name."
    )
    parser.add_argument(
        "--seed",
        type = int,
        default = None,
        help = "Random seed for reproducibility."
    )
    parser.add_argument(
        "--threshold",
        type = float,
        default = 1.0,
        help = "Evaluation threshold (default: 1.0)."
    )
    parser.add_argument(
        "--n-bins",
        type = int,
        default = None,
        help = "Number of bins (default: None)."
    )
    parser.add_argument(
        "--checkpoint-every",
        type = int,
        default = 60,
        help = "Checkpoint every N minutes (default: 60)."
    )
    parser.add_argument(
        "--force-max-duration",
        type = int,
        default = None,
        help = "Force the goal time interval of every planning problem to [0, N] (as `evita run --force-max-duration`)."
    )
    parser.add_argument(
        "--debug",
        action = "store_true",
        help = "Enable debug logging."
    )

    args = parser.parse_args()

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

    driving_agent_name = "frenetix"

    progress_bar_manager = get_progress_bar_manager()

    # By default checkpoint is every 60 minutes after the population evaluation
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
                            driving_agent_name,
                            args.population_size,
                            termination_criterion,
                            args.threshold,
                            args.n_bins,
                            output_folder_name = args.output_folder_name,
                            debug = args.debug,
                            force_max_duration = args.force_max_duration
                        )
    termination_criterion.problem_ref = problem

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
    performance_file = os.path.join(problem.output_folder, "performance.csv")
    with open(performance_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(termination_criterion.all_scenarios), str(termination_criterion.valid_scenarios), str(termination_criterion.invalid_scenarios), str(termination_criterion.total_generation_time)])
    logger.info("Saved Performance Metrics")

    # Now analysize the data collected so far and output to the scenario folder
    generate_report(problem.output_folder)