import csv
import sys
sys.path.append("../")

import os
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2

from evita.config import ScenarioGeneratorConfig
from evita.generators.NSGAIIproblem import EVITADoppelTestNSGAIIproblem
from evita.generators.nsgaII_operators import (
    EVITAScenarioSampling,
    EVITAScenarioMutation,
    EVITAScenarioCrossover,
    ScenarioDuplicateElimination,
    CustomTimeTermination
)
from evita.generators import utils as scenario_generation_utils
from evita.utils import initialize_main_logger, EVITA_MAIN_LOGGER_NAME
from config import SCENARIO_UPPER_LIMIT, HD_MAP_PATH, MAX_ADC_COUNT, RUN_FOR_HOUR, RECORDS_DIR_BASE, RuntimeConfig


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    # Build experiment unique identifier
    start_time = datetime.now()
    experiment_time = start_time.strftime("%Y%m%d%H%M%S")
    map_name = Path(HD_MAP_PATH).parent.name
    experiment_name = os.path.join(f"MAX_ADC_COUNT_{MAX_ADC_COUNT}_RUN_FOR_HOUR_{RUN_FOR_HOUR}_SCENARIO_MAP_{map_name}_{experiment_time}")

    # Dynamically set record files directory for the current experiment
    RuntimeConfig.RECORDS_DIR = os.path.join(RECORDS_DIR_BASE, experiment_name)
    os.makedirs(RuntimeConfig.RECORDS_DIR)

    # Save experiments configurations in the output folder
    config_file_output_path = os.path.join(RuntimeConfig.RECORDS_DIR, "config.py")
    with open("config.py", "r") as f:
        config_src = f.read()

    config_src = config_src.replace(
        'RECORDS_DIR = None',
        f'RECORDS_DIR = "{RuntimeConfig.RECORDS_DIR}"'
    )

    with open(config_file_output_path, "w") as f:
        f.write(config_src)

    # ---------------- Initialize Logging ----------------
    initialize_main_logger(debug=True)
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

    # ---------------- Define Additional Parameters ----------------
    POP_SIZE = 10
    DRIVING_AGENT = "apollo_evita"

    # ---------------- Apollo Containers and Scenario Runner ----------------
    python_38 = "../.venv38/bin/python"
    srunner_script = f"./apollo_runner_startup.py"

    # Run script that initializes and launches containers
    subprocess.run([python_38, srunner_script])

    # ---------------- Import Base scenario ----------------
    scenario_map_file_path = os.path.join("conversion_scripts", "converted_maps", f"{map_name}.xml")
    base_scenario, base_planning_problem_set = scenario_generation_utils.get_base_scenario_from_file(
        scenario_map_file_path,
        remove_obstacles = True, 
        remove_planning_problems = True
    )

    # Initialize scenario generation configurations
    config = ScenarioGeneratorConfig(SCENARIO_UPPER_LIMIT * 10, MAX_ADC_COUNT)

    # ---------------- Initialize Pymoo Operators ----------------
    sampling = EVITAScenarioSampling(base_scenario, base_planning_problem_set, config, scenario_map_file_path)
    mutation = EVITAScenarioMutation(scenario_map_file_path)
    crossover = EVITAScenarioCrossover()

    # Initialize NSGA2 algorithm
    algorithm = NSGA2(
        pop_size = POP_SIZE,
        sampling = sampling,
        crossover = crossover,
        mutation = mutation,
        eliminate_duplicates = ScenarioDuplicateElimination()
    )

    # ---------------- Define Termination Criteria Object ----------------
    termination = CustomTimeTermination(RUN_FOR_HOUR * 60, apollo = True)

    # ---------------- Initialize Problem ----------------
    problem = EVITADoppelTestNSGAIIproblem(
        scenario_map_file_path,
        DRIVING_AGENT,
        POP_SIZE,
        termination,
        RuntimeConfig.RECORDS_DIR,
        output_folder_name = RuntimeConfig.RECORDS_DIR
    )

    termination.problem_ref = problem

    # ---------------- Setup and Run NSGA-II ----------------
    algorithm.setup(
        problem,
        termination = termination,
        save_history = True,
        verbose = True
    )

    problem.algorithm = algorithm
    
    algorithm.run()
    result = algorithm.result()

    # Extract and save performance metrics
    output_folder = result.problem.output_folder
    valid_scenarios = result.problem.valid_scenarios
    invalid_scenarios = result.problem.invalid_scenarios
    all_scenarios = valid_scenarios + invalid_scenarios
    total_generation_time = result.exec_time

    performance_file = os.path.join(output_folder, "performance.csv").replace("/records/", "/evita-records/")
    with open(performance_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    logger.info("NSGA-II run completed.")

if __name__ == '__main__':
    main()
