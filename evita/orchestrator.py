import click
import logging
import sys
import importlib
import time
import os
import enlighten # Progress bar
import csv
import shutil
import traceback
import tempfile

from typing import List, Tuple, Set


from .utils import EVITA_MAIN_LOGGER_NAME, ROOT_DIR, RESULTS_FOLDER, LANELET_EXTENSION, Budget, extend_scenario_after_goal_areas, time_shift, simulation_folders

from .interfaces import AbstractScenarioGenerator, AbstractScenarioExecutor, AbstractScenarioEvaluator
from .validator import ScenarioValidator
from .config import ValidationConfig
from .data_analysis import generate_report

from cr_scenario_handler.simulation.agent import AgentStatus

from commonroad.planning.goal import Interval
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader

TOTAL_GENERATED_SCENARIOS = 0


def backup_debug(_output_folder, _debug_folder, _scenario, _logger):
    """
    In case of a critical error during the execution or evaluation of a scenario, we want to move the folder of this scenario into a debug folder for later analysis
    """
    # We need to clean up this folder to avoid the data analysis to pick broken data up!
    _logger.critical(f"Moving problematic folder of invalid scenario {_scenario.generation_id} into debug folder")
    
    execution_folder_name = str(_scenario.generation_id).rjust(5, '0')
    execution_folder = os.path.join(_output_folder, execution_folder_name)
    if os.path.exists(execution_folder):
        os.makedirs(_debug_folder, exist_ok=True)
        shutil.move(execution_folder, _debug_folder)
    else:
        _logger.critical(f"Folder {execution_folder} does not exist")

def _get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt
    """
    Returns a function that maps each index in 0, 1, ..., n - 1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name
    """
    return plt.cm.get_cmap(name, n)

def debug_plot_planning_problem_to_file(scenario, planning_problem, output_file):
    """
    Visualize and plot the initial scenario for a single planning problem
    """
    import matplotlib.pyplot as plt
    from commonroad.visualization.mp_renderer import MPRenderer
    
    # Initialize the renderer
    rnd = MPRenderer(figsize = (20, 10))

    # Set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.dynamic_obstacle.draw_icon = True

    # Draw Road and Obstacles at time stamp
    scenario.draw(rnd)
    planning_problem.draw(rnd)

    # Render the plot
    rnd.render()

    # Add title
    plt.title(f"Agent {planning_problem.planning_problem_id}")

    plt.savefig(f"{output_file}", format = 'png', dpi = 500, bbox_inches = 'tight')

def debug_plot_to_file(scenario, planning_problem_set, output_file):
    """
    Visualize and plot the initial scenario
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from commonroad.visualization.mp_renderer import MPRenderer

    # Sort planning problems by ID
    planning_problems = sorted(planning_problem_set.planning_problem_dict.values(), key = lambda pp : pp.planning_problem_id)
    # Get a list of unique colors for each planning problem
    unique_colors = _get_cmap(len(planning_problems), 'rainbow')

    # Initialize the renderer
    rnd = MPRenderer(figsize = (20, 10))

    # Set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.dynamic_obstacle.draw_icon = True

    # Draw Road and Obstacles at time stamp
    scenario.draw(rnd)
    
    # Iterate over each planning problem
    for i, planning_problem in enumerate(planning_problems):
        # Set renderer draw params
        rnd.draw_params.planning_problem.initial_state.state.facecolor = colors.to_hex(unique_colors(i))
        rnd.draw_params.planning_problem.initial_state.state.draw_arrow = True
        rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
        rnd.draw_params.planning_problem.initial_state.label = f"{planning_problem.planning_problem_id}"

        rnd.draw_params.planning_problem.goal_region.shape.facecolor = colors.to_hex(unique_colors(i))
        rnd.draw_params.planning_problem.goal_region.shape.opacity = 0.5
            
        # Include the arrow here!
        planning_problem.draw(rnd)

    # Render the plot
    rnd.render()

    # Add title
    plt.title(f"Valid? {scenario.is_valid}" + "" if scenario.is_valid else scenario.validation_message)

    plt.savefig(f"{output_file}", format = 'png', dpi = 500, bbox_inches = 'tight')

def ensure_generation_id(scenario_tuples: List[Tuple[Scenario, PlanningProblemSet]]) -> None:
    """
    Ensure that all the scenarios have a generation_id attribute
    This is required for the debug plots and for the checkpointing system to work properly
    """
    global TOTAL_GENERATED_SCENARIOS
    # Debug Plot
    for scenario, planning_problem_set in scenario_tuples:
        TOTAL_GENERATED_SCENARIOS = TOTAL_GENERATED_SCENARIOS + 1
        # Tag the scenario with a generation ID
        setattr(scenario, "generation_id", TOTAL_GENERATED_SCENARIOS)

def checkpoint(checkpoint_id: int, output_folder: str, all_scenarios:int, valid_scenarios:int, invalid_scenarios:int, total_generation_time:int) -> None:
    """
    Link all the scenario folders generated so far + compute the results for the generation (aka csv files) to the checkpoint folder.
    """
    # Build the checkpoint folder path and create it
    checkpoint_folder = os.path.join(output_folder, "checkpoints", f"{str(checkpoint_id).rjust(4,'0')}")
    os.makedirs(checkpoint_folder, exist_ok=False)
    print(f"Creating checkpoint at {checkpoint_folder}")

    # Create a scenarios.csv file containining the id of all the scenarios generated so far
    scenario_file = os.path.join(checkpoint_folder, "scenarios.csv")
    with open(scenario_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["scenario_id", "scenario_file", "db_file"])
        # Iterate over each scenario folder and link it to the checkpoint
        for _, simulation_folder in simulation_folders(output_folder):
            scenario_id = simulation_folder
            simulation_db = os.path.join(output_folder, simulation_folder, "logs", "simulation.db")
            scenario_file = os.path.join(output_folder, simulation_folder, "scenario", f"scenario_{simulation_folder}.xml")
            csv_writer.writerow([str(scenario_id), str(scenario_file), str(simulation_db)])

    # Log performance metrics in the checkpoint folder
    performance_file = os.path.join(checkpoint_folder, "performance.csv")
    with open(performance_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    # Generate the usual report in the checkpoint folder using whatever data is available at this point
    generate_report(output_folder, checkpoint_folder)

@click.command()
@click.option("--scenario-file", type=click.Path(exists=True), required=True, multiple=False)
@click.option("--driving-agent-name", type=str, required=True, multiple=False)
@click.pass_context 
def replay(ctx, scenario_file, driving_agent_name):
    """
    This command allows to replay a single scenario and evaluate it
    It is useful for debugging purposes and to replay interesting scenarios that were generated during the evolution
    """
    debug = ctx.obj["debug"]

    # Get the main logger
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    
    logger.info(f"Replay Scenario {scenario_file}")

    # Get the output folder from the context, if not present create a temporary one
    if "output_folder" in ctx.obj:
        output_folder = ctx.obj["output_folder"]
    else:
        output_folder = tempfile.TemporaryDirectory().name
    os.makedirs(output_folder, exist_ok = True)

    # Load the Scenario Executor
    scenario_executor_path = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")
    scenario_executor_module = "evita.executors.frenetix_executor"
    scenario_executor_class = "FrenetixScenarioExecution"

    logger.debug(f"Add scenario execution to path {scenario_executor_path}")
    sys.path.append(scenario_executor_path)
        
    logger.debug(f"Try to import {scenario_executor_class} from {scenario_executor_module}")
    se_module = importlib.import_module(scenario_executor_module)
    se_class = getattr(se_module, scenario_executor_class)

    # Load the Scenario Evaluator
    scenario_evaluator_path = "."
    scenario_evaluator_module = "evita.evaluators.frenetix_evaluator"
    scenario_evaluator_class = "FrenetixScenarioEvaluator"

    logger.debug(f"Add scenario evaluator to path {scenario_evaluator_path}")
    sys.path.append(scenario_evaluator_path)
        
    logger.debug(f"Try to import {scenario_evaluator_class} from {scenario_evaluator_module}")
    sev_module = importlib.import_module(scenario_evaluator_module)
    sev_class = getattr(sev_module, scenario_evaluator_class)

    scenario_executor: AbstractScenarioExecutor
    scenario_executor = se_class(output_folder, driving_agent_name=driving_agent_name, debug=debug)

    scenario_evaluator: AbstractScenarioEvaluator
    scenario_evaluator = sev_class(output_folder)

    # Instantiate the scenario validator
    scenario_validator_configuration = ValidationConfig()
    scenario_validator = ScenarioValidator(scenario_validator_configuration)

    # Log the metadata of this execution in a csv file for future reference
    logger.info("*" * len(f"Outputting results to {output_folder}"))
    logger.info(f"Outputting results to {output_folder}")
    logger.info("*" * len(f"Outputting results to {output_folder}"))
    
    if debug:
        debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(debug_folder, exist_ok = True)
        logger.info(f"Outputting debug data to {debug_folder}")
    
    # Load the scenario from the file
    crfr = CommonRoadFileReader(scenario_file)
    scenario_tuples = [crfr.open()]
    
    # Ensure required attributes are there
    # This contains a static reference, so tests will fail
    # ensure_generation_id(scenario_tuples)
    scenario, planning_problem_set = scenario_tuples[0]
    setattr(scenario, "generation_id", 1)

    # Validate the generated scenarios. This will add the attribute: is_valid to the scenario object
    scenario_validator.validate_all(scenario_tuples)

    # Plot scenario if debug is enabled
    if debug:
        for scenario, planning_problem_set in scenario_tuples:
            debug_plot_file = os.path.join(debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}.png")
            debug_plot_to_file(scenario, planning_problem_set, debug_plot_file)
            # Plot each single planning problem
            for planning_problem in planning_problem_set.planning_problem_dict.values():
                debug_plot_planning_problem_file = os.path.join(debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}_{planning_problem.planning_problem_id}.png")
                debug_plot_planning_problem_to_file(scenario, planning_problem, debug_plot_planning_problem_file)

    # Iterate over the scenarios and try to execute them if they are valid, then evaluate them
    for scenario, planning_problem_set in scenario_tuples:
        if scenario.is_valid:
            # Log infos about the replayed scenario
            logger.info(f"Replay Scenario: {scenario.generation_id}")
            [logger.debug(f"- {pp.planning_problem_id} - {pp.initial_state.position} - {pp.initial_state.velocity} - planned start at {pp.initial_state.time_step}") for pp in planning_problem_set.planning_problem_dict.values()]

            # We empirically show that shorter extensions cause the agent to slow down considerably before reaching the goal area!
            
            # Make sure we enlong the lanelets nearby goal areas or the planner will fail to reach them
            # extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters=2 * LANELET_EXTENSION)
            # During Replay we should not extend again the scenario
            extended_scenario = scenario
            
            # Time Shift so at least one agent starts at time_step=0
            time_shifted_planning_problem_set = time_shift(scenario, planning_problem_set)

            # Make sure we execute the extended_scenario, not the original one
            # Both will be available afterwards, but the original shall be used for evolution
            simulation = scenario_executor.execute(extended_scenario, time_shifted_planning_problem_set)

            # Check if any agent has an ERROR state at time_step == 0
            # Run evaluation and postprocessing outside the budget
            for agent in simulation.agents:
                if agent.agent_state.status == AgentStatus.ERROR and agent.agent_state.last_timestep == 0:
                    logger.critical(f"The scenario is invalid. Agent {agent.id} did not even start")
                    scenario.is_valid = False
        else:
            logger.info(f"Invalid Scenario {scenario.validation_message}")

        # Save the simulation data and the evaluation results only for valid scenarios
        # For invalid ones we just want to keep track of the fact that they were invalid and the reason why
        if scenario.is_valid:
            logger.info(f"Evaluate the Scenario")
            scenario_evaluator.evaluate(simulation)
        
            logger.info(f"Generate the report")
            performance_file = os.path.join(output_folder, "performance.csv")
            with open(performance_file, "w", newline = "") as csvfile:
                csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
                csv_writer.writerow([str(1), str(1), str(0), str(100)])
                
            generate_report(output_folder)

            # This will be true only if forced in the yml or because of the debug flag
            simulation.make_gif()
        else:
            logger.critical("Skip Evaluation and Report Generation for invalid scenarios")
                 
    logger.info("*" * len(f"Outputting results to {output_folder}"))
    logger.info(f"Outputting results to {output_folder}")
    logger.info("*" * len(f"Outputting results to {output_folder}"))

@click.command()
@click.option("--scenario-generator-path", type = click.Path(exists = True), required = True, multiple = False)
@click.option("--scenario-generator-module", type = str, required = True, multiple = False)
@click.option("--scenario-generator-class", type = str, required = True, multiple = False)
@click.option("--map-file", type = click.Path(exists = True), required = False, multiple = False)
@click.option("--driving-agent-name", type = str, required = True, multiple = False)
@click.option("--time-budget", type = float, required = True, multiple = False) 
@click.option("--force-max-duration", type = int, required = False, multiple = False) 
@click.option("--checkpoint-every", type = int, required = False, multiple = False, default = 3600) 
@click.pass_context 
def run(ctx, scenario_generator_path: str, scenario_generator_module: str, scenario_generator_class: str,
        map_file: str,
        driving_agent_name: str,
        time_budget: float,
        force_max_duration: int,
        checkpoint_every: int):

    debug = ctx.obj["debug"]
    
    # Initialize the set of checkpoints
    checkpoints: Set[int] = set()
    
    # Get the main logger and log the start of the execution
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    logger.info("Start Execution")

    # Generate the Unique ID of this Execution
    # Create the unique folder that will host the results of this execution using the test generator data and a timestamp as ID
    timestamp_id = time.time() * 100000000 // 1000000
    output_folder_name = ctx.obj["output_folder_name"] if "output_folder_name" in ctx.obj else "_".join([str(scenario_generator_module), str(scenario_generator_class), str(timestamp_id)])
    output_folder = os.path.join(RESULTS_FOLDER, output_folder_name)
    debug_folder = os.path.join(output_folder, "debug")

    # Initialize the performance metrics
    all_scenarios = 0
    valid_scenarios = 0
    invalid_scenarios = 0
    total_generation_time = 0

    #
    # Implement the main loop
    #

    # Load the Scenario Generator by dynamically loading it
    scenario_generator: AbstractScenarioGenerator

    # Log the loading of the scenario generator
    logger.debug(f"Add scenario generator to path {scenario_generator_path}")
    sys.path.append(scenario_generator_path)
    logger.debug(f"Try to import {scenario_generator_class} from {scenario_generator_module}")
    sg_module = importlib.import_module(scenario_generator_module)
    sg_class = getattr(sg_module, scenario_generator_class)

    # Instantiate the Scenario Generator
    commoroad_scenario_seed_file = map_file

    kwargs = {
        "debug": debug,
        "debug_folder": debug_folder
    }
    scenario_generator = sg_class(commoroad_scenario_seed_file, **kwargs)

    # Load the Scenario Executor
    scenario_executor_path = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")
    scenario_executor_module = "evita.executors.frenetix_executor"
    scenario_executor_class = "FrenetixScenarioExecution"

    # Log the loading of the scenario executor
    logger.debug(f"Add scenario execution to path {scenario_executor_path}")
    sys.path.append(scenario_executor_path)
    logger.debug(f"Try to import {scenario_executor_class} from {scenario_executor_module}")
    se_module = importlib.import_module(scenario_executor_module)
    se_class = getattr(se_module, scenario_executor_class)

    # Load the Scenario Evaluator
    scenario_evaluator_path = "."
    scenario_evaluator_module = "evita.evaluators.frenetix_evaluator"
    scenario_evaluator_class = "FrenetixScenarioEvaluator"

    # Log the loading of the scenario evaluator
    logger.debug(f"Add scenario evaluator to path {scenario_evaluator_path}")
    sys.path.append(scenario_evaluator_path)
    logger.debug(f"Try to import {scenario_evaluator_class} from {scenario_evaluator_module}")
    sev_module = importlib.import_module(scenario_evaluator_module)
    sev_class = getattr(sev_module, scenario_evaluator_class)

    # Create the output folder if it does not exist. It might because we store the execution logs there are well from the bash scripts
    os.makedirs(output_folder, exist_ok = True)
    
    # Instantiate the scenario executor, evaluator and validator
    scenario_executor: AbstractScenarioExecutor
    scenario_executor = se_class(output_folder, driving_agent_name = driving_agent_name, debug = debug)

    scenario_evaluator: AbstractScenarioEvaluator
    scenario_evaluator = sev_class(output_folder)

    scenario_validator_configuration = ValidationConfig()
    scenario_validator = ScenarioValidator(scenario_validator_configuration)

    logger.info(f"Outputting results to [{output_folder}]")
    os.makedirs(debug_folder, exist_ok = True)
    
    if debug:
        logger.info(f"Outputting debug data to {debug_folder}")
    
    # Copy the base-scenario.xml in the output folder
    map_file_output_folder = os.path.join(output_folder, os.path.basename(map_file))
    shutil.copyfile(map_file, map_file_output_folder)
    scenario, _ = CommonRoadFileReader(map_file).open()

    # Log the metadata of this execution in a csv file for future reference
    metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(metadata_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["generator", "scenario_name", "time_budget", "timestamp_id"])
        csv_writer.writerow([f"{scenario_generator_module}.{scenario_generator_class}", str(scenario.scenario_id), str(time_budget), str(timestamp_id)])
            
    # Start the generation  + execution loop
    budget = Budget(time_budget)
    budget.start()

    # Define a progress bar with time
    progress_bar_manager = enlighten.get_manager()
    main_progress_bar = progress_bar_manager.counter(total = int(budget.get_time_budget()), desc = "Time Budget", unit = "seconds", color = "red", leave = False)
    main_progress_bar.update(budget.get_progress())
    
    # Loop until the time budget is over
    while not budget.is_over():
        # Log the remaining time at the beginning of each iteration
        logger.debug(f"Remaining time {budget.get_remaining_real_time()}")

        # Generate some scenarios
        generation_start = time.time()
        logger.info("")
        scenario_tuples = scenario_generator.generate(progress_bar_manager = progress_bar_manager)
        
        # If no scenarios were generated, we can skip the rest of the loop and continue to the next iteration
        if scenario_tuples is None:
            logger.info("No scenarios generated. Retrying...")
            continue

        total_generation_time = total_generation_time + (time.time() - generation_start)

        # Log generated scenarios
        all_scenarios = all_scenarios + len(scenario_tuples)

        # Setup the progress bar for the scenario validation
        scenario_validation_bar = progress_bar_manager.counter(total = len(scenario_tuples), desc = "Scenario Validation", unit = "scenario", color = "blue", leave = False)

        # Ensure required attributes are there
        ensure_generation_id(scenario_tuples)

        # Notify that some "real time passed since the last time"
        main_progress_bar.update(budget.get_progress())

        # Pause the budget while we validate and plot the scenarios
        budget.pause()

        # Validate the generated scenarios
        # This will add the attribute: is_valid to the scenario object
        for scenario, planning_problem_set in scenario_tuples:
            scenario_validator.validate(scenario, planning_problem_set)
            scenario_validation_bar.update()

        # Plot scenarios if debug is enabled
        if debug:
            for scenario, planning_problem_set in scenario_tuples:
                debug_plot_file = os.path.join(debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}.png")
                debug_plot_to_file(scenario, planning_problem_set, debug_plot_file)

        # Resume the budget for the execution and evaluation of the scenarios
        budget.resume()

        # Try to Execute them
        simulation_data = []

        # Iterate over each scenario and planning problem set tuple
        for scenario, planning_problem_set in scenario_tuples:
            # Make sure we keep track of the time passing
            main_progress_bar.update(budget.get_progress())

            # Force max planning problem duration if required
            if force_max_duration:
                logger.critical(f"Forcing max duration {force_max_duration} for scenario {scenario.generation_id}")
                for planning_problem in planning_problem_set.planning_problem_dict.values():
                    for goal_state in planning_problem.goal.state_list:
                        if "time_step" in goal_state.attributes:
                            goal_state.time_step = Interval(0, force_max_duration)

            # Check if the time budget is over before executing the scenario, if it is, mark it as invalid and skip to the next one
            if budget.is_over():
                logger.critical("Timeout is over. Force the current scenario to be invalid")
                scenario.is_valid = False
                scenario.validation_message = "Timeout is over"
                continue
            
            # Execution
            if scenario.is_valid:
                # In some cases, the scenario is valid, but the execution fails because of some critical error
                try:
                    # Log info of the scenario that is going to be executed
                    logger.info(f"Executing scenario: {scenario.generation_id} - {budget.get_remaining_real_time()}")
                    [logger.debug(f"- {pp.planning_problem_id} - {pp.initial_state.position} - {pp.initial_state.velocity} - planned start at {pp.initial_state.time_step}") for pp in planning_problem_set.planning_problem_dict.values()]

                    # Extend the scenario by adding some road AFTER the goal areas
                    # Make sure we enlong the lanelets nearby goal areas or the planner will fail to reach them
                    extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters = LANELET_EXTENSION)

                    # Time Shift so at least one agent starts at time_step=0
                    time_shifted_planning_problem_set = time_shift(scenario, planning_problem_set)

                    # Make sure we execute the extended_scenario, not the original one
                    # Both will be available afterwards, but the original shall be used for evolution
                    simulation = scenario_executor.execute(extended_scenario, time_shifted_planning_problem_set)

                    # Run evaluation and postprocessing outside the budget
                    for agent in simulation.agents:
                        if agent.agent_state.status == AgentStatus.ERROR and agent.agent_state.last_timestep == 0:
                            logger.critical(f"The scenario is invalid. Agent {agent.id} did not even start")
                            scenario.is_valid = False
                            scenario.validation_message = f"Agent {agent.id} could not compute a phyisically possible trajectory since the beginning."

                except Exception as e:
                    # It might fail, log the error and move on to the next scenario, but mark this one as invalid
                    msg = f"A critical error prevented {scenario.generation_id} to complete. Mark it as Error and set the validity flag to False"
                    logger.critical( "=" * ( len(msg) + 2 ) )
                    logger.critical( f" {msg} ")
                    logger.critical( "=" * ( len(msg) + 2) )
                    scenario.is_valid = False
                    scenario.validation_message = "Error during the simulation execution"
                    raise e

            # Executed scenario evaluation
            if scenario.is_valid:
                try:
                    # Log info of the scenario that is going to be evaluated
                    logger.info(f"Evaluating scenario: {scenario.generation_id} - {budget.get_remaining_real_time()}")

                    # Pause the budget while we evaluate the scenario
                    budget.pause()
                    
                    try:
                        # Compute the criticality metrics for this scenario and planning problem set
                        criticality_dataframe = scenario_evaluator.evaluate(simulation)
                    finally:
                        # Perform cleanup ONLY after evaluation is done
                        simulation.sim_logger.con.close()
                        simulation.close_processes()
    
                    # Resume the budget for the next scenario execution
                    budget.resume()

                    # Log validation completion and remaining time
                    logger.info(f"Done. - {budget.get_remaining_real_time()}")

                    simulation_data.append({
                        "individual": (scenario, planning_problem_set),
                        "agent-data": simulation.agents,
                        "evaluation-data": criticality_dataframe,
                    })
                except Exception as e:
                    msg = f" {scenario.generation_id} to be evaluated. Mark it as invalid"
                    logger.critical( "=" * ( len(msg) + 2 ) )
                    logger.critical( f" {msg} ")
                    logger.critical( "=" * ( len(msg) + 2) )

                    # If an error is triggered during the evaluation, get rid of the scenario
                    # This should take care of those DSE error
                    logger.critical(traceback.format_exc())
                    # Try to move any folder away
                    backup_debug(output_folder, debug_folder, scenario, logger)
                    simulation_data.append({
                        "individual": (scenario, planning_problem_set),
                        "agent-data": None,
                        "evaluation-data": None,
                    })
            elif scenario.validation_message == "Error during the simulation execution":
                # If the scenario is invalid because of an issue during the simulation, move the folder away
                backup_debug(output_folder, debug_folder, scenario, logger)
                simulation_data.append({
                    "individual": (scenario, planning_problem_set),
                    "agent-data": None,
                    "evaluation-data": None,
                })
            else:
                # Invalid scenarios have no folder, no need to move it anywhere
                simulation_data.append({
                    "individual": (scenario, planning_problem_set),
                    "agent-data": None,
                    "evaluation-data": None,
                })

        # Close the progress bars    
        scenario_validation_bar.close(clear = True)

        # Iterate over the scenarios and count how many of them are valid and invalid
        for scenario, planning_problem_set in scenario_tuples:
            # Scenarios that were not executed because of a timeout
            if scenario.is_valid or scenario.validation_message == "Timeout is over":
                valid_scenarios = valid_scenarios + 1
            else:
                invalid_scenarios = invalid_scenarios + 1

        # Inform the generator about newly executed scenario to Evaluate the scenarios we managed to simulate if there's still some time
        if not budget.is_over():
            # If the budget is over, there's no point to ask the algorithm to evaluate the population
            scenario_generator.evaluate(simulation_data)

        # At this point, we have completed a generation epoch, we can check whether we need to checkpoint it
        elapsed_time = budget.get_elapsed()
        checkpoint_id = int( elapsed_time / checkpoint_every)
        if checkpoint_id > 0 and checkpoint_id not in checkpoints:
            budget.pause()
            try:
                checkpoint(checkpoint_id, output_folder,
                           all_scenarios, valid_scenarios, invalid_scenarios, total_generation_time)
                checkpoints.add(checkpoint_id)
            except Exception as e:
                logger.critical(f"Failed to checkpoint execution at checkpoint {checkpoint_id}")
                raise e

            budget.resume()

    logger.info(f"Execution of {output_folder} is over ")

    # Log performance metrics right away - This is difficult/impossible to compute only looking at the simulation dbs
    logger.info("Generation Performance Metrics")
    performance_file = os.path.join(output_folder, "performance.csv")
    with open(performance_file, "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    progress_bar_manager.stop()

    # Now analysize the data collected so far and output to the scenario folder
    generate_report(output_folder)
