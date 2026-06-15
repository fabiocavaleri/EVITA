import sys
import logging
import os
import csv
from pymoo.algorithms.moo.nsga2 import NSGA2

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import generate_report
from evita.generators.NSGAIIproblem import NSGAIIproblem
from evita.generators.nsgaII_operators import ScenarioSampling, ScenarioMutation, ScenarioCrossover, \
    ScenarioDuplicateElimination, CustomTimeTermination
from evita.generators import utils as scenario_generation_utils

from evita.utils import initialize_main_logger, EVITA_MAIN_LOGGER_NAME

if __name__ == '__main__':
    """
    This is the main entry point for the NSGAII Search without a generator
    It will generate scenarios by applying mutation and crossover operators to the seed scenario, and evaluate them using the given evaluation pipeline
    The generated scenarios will be stored in the output folder, together with metadata and performance metrics
    """
    # Get the debug flag from the environment variable, default to False if not set or if there is an error
    debug = False
    try:
        debug = bool(os.getenv("EVITA_DEBUG", False))
    except Exception:
        pass    

    # Initialize the main logger
    initialize_main_logger(debug = debug)
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    
    # Get the parameters from the command line arguments
    population_size = int(sys.argv[1]) # number of scenarios to generate in each generation
    max_time_minutes = float(sys.argv[2]) # maximum generation time in minutes
    commonroad_scenario_seed_file = sys.argv[3] # path to the commonroad scenario seed file, which will be used as the base scenario for generation
    # Optional output folder name, if not provided, it will be generated based on the timestamp and scenario generator configuration
    if len(sys.argv) > 4:
        output_folder_name = sys.argv[4]
    else:
        output_folder_name = None

    # Load the scenario object from the seed file, and leave only the lanelet network
    base_scenario, base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(commonroad_scenario_seed_file, remove_obstacles = True, remove_planning_problems = True)

    # Initialize the scenario generator configuration
    scenario_generator_configuration = ScenarioGeneratorConfig()

    driving_agent_name = "frenetix"

    # Initialize the NSGAII algorithm ti generate scenario and the mutation and crossover operators
    sampling = ScenarioSampling(base_scenario, base_planning_problem_set, scenario_generator_configuration)
    mutation = ScenarioMutation()
    crossover = ScenarioCrossover()
    algorithm = NSGA2(pop_size = population_size, sampling = sampling, mutation = mutation,
                      crossover = crossover,
                      eliminate_duplicates = ScenarioDuplicateElimination())

    
    # By default checkpoint is every 60 minutes after the population evaluation
    termination_criterion = CustomTimeTermination(max_time_minutes)
    
    problem = NSGAIIproblem(commonroad_scenario_seed_file, driving_agent_name, population_size, termination_criterion, output_folder_name = output_folder_name)
    termination_criterion.problem_ref = problem

    algorithm.setup(problem, termination = termination_criterion, save_history = True, verbose = True)

    algorithm.run()
    result = algorithm.result()

    termination_criterion.update_elapsed_time()
    total_generation_time = termination_criterion.elapsed_time_seconds

    logger.info(f"Execution of {problem.output_folder} is over ")
    
    # Generate metadafile
    metadata_file = os.path.join(problem.output_folder, "metadata.csv")
    with open(metadata_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["generator", "scenario_name", "time_budget", "timestamp_id"])
        csv_writer.writerow([f"{problem.scenario_generator_module}.{problem.scenario_generator_class}", str(base_scenario.scenario_id), str(max_time_minutes * 60), str(problem.timestamp_id)])
    
    # Log the number of valid and invalid scenarios generated, as well as the total generation time, to a performance file in the output folder
    all_scenarios = problem.valid_scenarios + problem.invalid_scenarios
    valid_scenarios = problem.valid_scenarios
    invalid_scenarios = problem.invalid_scenarios

    # Log performance metrics right away
    logger.info("Generation Performance Metrics")
    performance_file = os.path.join(problem.output_folder, "performance.csv")
    with open(performance_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow(
            [str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    # Now analysize the data collected so far and output to the scenario folder
    generate_report(problem.output_folder)
