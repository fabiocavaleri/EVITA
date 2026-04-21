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

from typing import Optional, List, Tuple, Set


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

# Global - Bad Design... Better use a generator next time
TOTAL_GENERATED_SCENARIOS = 0


def backup_debug(_output_folder, _debug_folder, _scenario, _logger):
    # We need to clean up this folder to avoid the data analysis to pick broken data up!
    _logger.critical(f"Moving problematic folder of invalid scenario {_scenario.generation_id} into debug folder")
    exeuction_folder_name = str(_scenario.generation_id).rjust(5, '0')
    execution_folder = os.path.join(_output_folder, exeuction_folder_name)
    if os.path.exists(execution_folder):
        os.makedirs(_debug_folder, exist_ok=True)
        shutil.move(execution_folder, _debug_folder)
    else:
        _logger.critical(f"Folder {execution_folder} does not exist")


def _get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def debug_plot_planning_problem_to_file(scenario, planning_problem, output_file):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    from commonroad.visualization.mp_renderer import MPDrawParams, MPRenderer
    
    rnd = MPRenderer(figsize=(20, 10))

    # set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.dynamic_obstacle.draw_icon = True

    # Draw Road and Obstacles at time stamp
    scenario.draw(rnd)
    planning_problem.draw(rnd)

    # Render the plot
    rnd.render()

    # Add title
    plt.title(f"Agent {planning_problem.planning_problem_id}")

    plt.savefig(f"{output_file}", format='png', dpi=300, bbox_inches='tight')

def debug_plot_to_file(scenario, planning_problem_set, output_file):
    """
    Visualize and plot the initial scenario
    """

    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    from commonroad.visualization.mp_renderer import MPDrawParams, MPRenderer

    planning_problems = sorted(planning_problem_set.planning_problem_dict.values(), key= lambda pp : pp.planning_problem_id)

    unique_colors = _get_cmap(len(planning_problems), 'rainbow')

    rnd = MPRenderer(figsize=(20, 10))

    # set renderer draw params
    rnd.draw_params.time_begin = 0
    rnd.draw_params.dynamic_obstacle.draw_icon = True

    # Draw Road and Obstacles at time stamp
    scenario.draw(rnd)
    
    for i, planning_problem in enumerate(planning_problems):
        # set renderer draw params
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

    plt.savefig(f"{output_file}", format='png', dpi=300, bbox_inches='tight')

def ensure_generation_id(scenario_tuples: List[Tuple[Scenario, PlanningProblemSet]]) -> None:
    global TOTAL_GENERATED_SCENARIOS
    # Debug Plot
    for scenario, planning_problem_set in scenario_tuples:
        TOTAL_GENERATED_SCENARIOS = TOTAL_GENERATED_SCENARIOS + 1
        # Tag the scenario with a generation ID
        setattr(scenario, "generation_id", TOTAL_GENERATED_SCENARIOS)


def checkpoint(checkpoint_id: int, output_folder: str,
               all_scenarios:int, valid_scenarios:int, invalid_scenarios:int, total_generation_time:int) -> None:
    """
    Link all the scenario folders generated so far + compute the results for the generation (aka csv files) to the checkpoint folder.
    """
    checkpoint_folder = os.path.join(output_folder, "checkpoints", f"{str(checkpoint_id).rjust(4,'0')}")
    print(f"Creating checkpoint at {checkpoint_folder}")
    
    # Create a checkpoint folder
    os.makedirs(checkpoint_folder, exist_ok=False)

    # Create a scenarios.csv file containining the id of all the scenarios generated so far
    scenario_file = os.path.join(checkpoint_folder, "scenarios.csv")
    with open(scenario_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["scenario_id", "scenario_file", "db_file"])
        for _, simulation_folder in simulation_folders(output_folder):
            scenario_id = simulation_folder
            simulation_db = os.path.join(output_folder, simulation_folder, "logs", "simulation.db")
            scenario_file = os.path.join(output_folder, simulation_folder, "scenario", f"scenario_{simulation_folder}.xml")
            csv_writer.writerow([str(scenario_id), str(scenario_file), str(simulation_db)])

    performance_file = os.path.join(checkpoint_folder, "performance.csv")
    with open(performance_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    # Generate the usual report in the checkpoint folder using whatever data is available at this point
    generate_report(output_folder, checkpoint_folder)



@click.command()
@click.option("--scenario-file", type=click.Path(exists=True), required=True, multiple=False)
@click.option("--driving-agent-name", type=str, required=True, multiple=False)
# TODO Include a --force-max-duration for debugging purposes
@click.pass_context 
def replay(ctx, scenario_file, driving_agent_name):
    debug = ctx.obj["debug"]

    # Get the main logger
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    
    logger.info(f"Replay Scenario {scenario_file}")

    output_folder = ctx.obj["output_folder"] if "output_folder" in ctx.obj else tempfile.TemporaryDirectory().name

    # Load the Scenario Executor - Hardcoded for the moment
    scenario_executor_path = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")
    scenario_executor_module = "evita.executors.frenetix_executor"
    scenario_executor_class = "FrenetixScenarioExecution"

    logger.debug(f"Add scenario execution to path {scenario_executor_path}")
    sys.path.append(scenario_executor_path)
        
    logger.debug(f"Try to import {scenario_executor_class} from {scenario_executor_module}")
    se_module = importlib.import_module(scenario_executor_module)
    se_class = getattr(se_module, scenario_executor_class)

    # Load the Scenario Evaluator - Hardcoded for the moment
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

    logger.info("*" * len(f"Outputting results to {output_folder}"))
    logger.info(f"Outputting results to {output_folder}")
    logger.info("*" * len(f"Outputting results to {output_folder}"))
    
    os.makedirs(output_folder, exist_ok=True)
    
    if debug:
        debug_folder = os.path.join(output_folder, "debug")
        os.makedirs(debug_folder, exist_ok=True)
        logger.info(f"Outputting debug data to {debug_folder}")
    


    crfr = CommonRoadFileReader(scenario_file)
    scenario_tuples = [crfr.open()]
    
    # Ensure required attributes are there
    # This contains a static reference, so tests will fail
    # ensure_generation_id(scenario_tuples)
    scenario, planning_problem_set = scenario_tuples[0]
    setattr(scenario, "generation_id", 1)

    # Validate the generated scenarios. This will add the attribute: is_valid to the scenario object
    scenario_validator.validate_all(scenario_tuples)

    if debug:
        # Plot the scenario
        for scenario, planning_problem_set in scenario_tuples:
            debug_plot_file = os.path.join(debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}.png")
            debug_plot_to_file(scenario, planning_problem_set, debug_plot_file)
            # Plot each single planning problem
            for planning_problem in planning_problem_set.planning_problem_dict.values():
                debug_plot_planning_problem_file = os.path.join(debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}_{planning_problem.planning_problem_id}.png")
                debug_plot_planning_problem_to_file(scenario, planning_problem, debug_plot_planning_problem_file)


    for scenario, planning_problem_set in scenario_tuples:

        if scenario.is_valid:

            logger.info(f"Replay Scenario: {scenario.generation_id}")
            [ logger.debug(f"- {pp.planning_problem_id} - {pp.initial_state.position} - {pp.initial_state.velocity} - planned start at {pp.initial_state.time_step}") for pp in planning_problem_set.planning_problem_dict.values()]

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



        if scenario.is_valid:
            logger.info(f"Evaluate the Scenario")
            scenario_evaluator.evaluate(simulation)
        
            logger.info(f"Generate the report")
            performance_file = os.path.join(output_folder, "performance.csv")
            with open(performance_file, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
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
@click.option("--scenario-generator-path", type=click.Path(exists=True), required=True, multiple=False)
@click.option("--scenario-generator-module", type=str, required=True, multiple=False)
@click.option("--scenario-generator-class", type=str, required=True, multiple=False)
@click.option("--map-file", type=click.Path(exists=True), required=False, multiple=False)
@click.option("--driving-agent-name", type=str, required=True, multiple=False)
@click.option("--time-budget", type=float, required=True, multiple=False) 
@click.option("--force-max-duration", type=int, required=False, multiple=False) 
@click.option("--checkpoint-every", type=int, required=False, multiple=False, default=3600) 
@click.pass_context 
def run(ctx, scenario_generator_path,scenario_generator_module, scenario_generator_class,
        map_file,
        driving_agent_name,
        time_budget,
        force_max_duration,
        checkpoint_every):
    
    checkpoints: Set[int] = set()

    debug = ctx.obj["debug"]
    
    # Get the main logger
    logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
    
    # TODO Provide some info here, about log output and the like
    logger.info("Start Execution")

    # Generate the Unique ID of this Execution.
    
    # Create the unique folder that will host the results of this execution using the test generator data and a timestamp as id
    timestamp_id = time.time() * 100000000 // 1000000
    output_folder_name = ctx.obj["output_folder_name"] if "output_folder_name" in ctx.obj else "_".join([str(scenario_generator_module), str(scenario_generator_class), str(timestamp_id)])
    output_folder = os.path.join(RESULTS_FOLDER, output_folder_name)
    debug_folder = os.path.join(output_folder, "debug")

    # Collect some statistics:
    
    all_scenarios = 0
    valid_scenarios = 0
    invalid_scenarios = 0
    total_generation_time = 0

    # Implement the main loop

    # Load the Scenario Generator by dynamically loading it
    scenario_generator: AbstractScenarioGenerator

    logger.debug(f"Add scenario generator to path {scenario_generator_path}")
    sys.path.append(scenario_generator_path)
        
    logger.debug(f"Try to import {scenario_generator_class} from {scenario_generator_module}")
    sg_module = importlib.import_module(scenario_generator_module)
    sg_class = getattr(sg_module, scenario_generator_class)

    
    # Instantiate the Scenario Generator. For the moment, we assume only the (optional) map is provided.
    commoroad_scenario_seed_file=map_file
    # TODO Check if this is ok or we should pass kwargs = {} instead
    kwargs = {
        "debug": debug,
        "debug_folder": debug_folder
    }
    scenario_generator = sg_class(commoroad_scenario_seed_file, **kwargs)

    # Load the Scenario Executor - Hardcoded for the moment
    scenario_executor_path = os.path.join(ROOT_DIR, "Frenetix-Motion-Planner")
    scenario_executor_module = "evita.executors.frenetix_executor"
    scenario_executor_class = "FrenetixScenarioExecution"

    logger.debug(f"Add scenario execution to path {scenario_executor_path}")
    sys.path.append(scenario_executor_path)
        
    logger.debug(f"Try to import {scenario_executor_class} from {scenario_executor_module}")
    se_module = importlib.import_module(scenario_executor_module)
    se_class = getattr(se_module, scenario_executor_class)


    # Load the Scenario Evaluator - Hardcoded for the moment
    scenario_evaluator_path = "."
    scenario_evaluator_module = "evita.evaluators.frenetix_evaluator"
    scenario_evaluator_class = "FrenetixScenarioEvaluator"

    logger.debug(f"Add scenario evaluator to path {scenario_evaluator_path}")
    sys.path.append(scenario_evaluator_path)
        
    logger.debug(f"Try to import {scenario_evaluator_class} from {scenario_evaluator_module}")
    sev_module = importlib.import_module(scenario_evaluator_module)
    sev_class = getattr(sev_module, scenario_evaluator_class)

    # Create the output folder if it does not exist. It might because we store the execution logs there are well from the bash scripts
    os.makedirs(output_folder, exist_ok=True)
    

    scenario_executor: AbstractScenarioExecutor
    scenario_executor = se_class(output_folder, driving_agent_name=driving_agent_name, debug=debug)

    scenario_evaluator: AbstractScenarioEvaluator
    scenario_evaluator = sev_class(output_folder)

    # Instantiate the scenario validator
    scenario_validator_configuration = ValidationConfig()
    scenario_validator = ScenarioValidator(scenario_validator_configuration)

    logger.info(f"Outputting results to {output_folder}")    
    
    os.makedirs(debug_folder, exist_ok=True)
    
    if debug:
        logger.info(f"Outputting debug data to {debug_folder}")
    
    # Copy the base-scenario.xml in the output folder
    map_file_output_folder = os.path.join(output_folder, os.path.basename(map_file))
    shutil.copyfile(map_file, map_file_output_folder)
    scenario, _ = CommonRoadFileReader(map_file).open()

    metadata_file = os.path.join(output_folder, "metadata.csv")
    with open(metadata_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["generator", "scenario_name", "time_budget", "timestamp_id"])
        csv_writer.writerow([f"{scenario_generator_module}.{scenario_generator_class}", str(scenario.scenario_id), str(time_budget), str(timestamp_id)])
            
    # Start the generation  + execution loop
    budget = Budget(time_budget)
    budget.start()

    # Define a progress bar with time... 
    progress_bar_manager = enlighten.get_manager()

    main_progress_bar = progress_bar_manager.counter(total=int(budget.get_time_budget()), desc="Time Budget", unit="seconds", color="red", leave=False)
    main_progress_bar.update(budget.get_progress())

    while not budget.is_over():

        logger.debug(f"Remaining time {budget.get_remaining_real_time()}")

        # Generate some scenarios
        
        generation_start = time.time()
        scenario_tuples = scenario_generator.generate(progress_bar_manager=progress_bar_manager)
        total_generation_time = total_generation_time + (time.time() - generation_start)


        # Log generated scenarios
        all_scenarios = all_scenarios + len(scenario_tuples)

        # Setup the bars
        scenario_validation_bar = progress_bar_manager.counter(total=len(scenario_tuples), desc="Scenario Validation", unit="scenario", color="blue", leave=False)

        # Ensure required attributes are there
        ensure_generation_id(scenario_tuples)

        # Notify that some "real time passed since the last time"
        main_progress_bar.update(budget.get_progress())

        budget.pause()
        # Validate the generated scenarios. This will add the attribute: is_valid to the scenario object
        
        for scenario, planning_problem_set in scenario_tuples:
            scenario_validator.validate(scenario, planning_problem_set)
            scenario_validation_bar.update()

        if debug:
            # Plot the scenario
            for scenario, planning_problem_set in scenario_tuples:
                debug_plot_file = os.path.join(debug_folder, f"plot_{(str(scenario.generation_id).rjust(5, '0'))}.png")
                debug_plot_to_file(scenario, planning_problem_set, debug_plot_file)

        budget.resume()

        # Try to Execute them
        simulation_data = []

        for scenario, planning_problem_set in scenario_tuples:
            
            # Make sure we keep track of the time passing
            main_progress_bar.update(budget.get_progress())

            # NOTE: This is useful for debug, in theory is the scenario generator who decides how long are the simulations or the timeout of each AV
            # TODO Enfore max duration of the scenario, i.e., until when the planners can reach their goal area
            if force_max_duration:
                logger.critical(f"Forcing max duration {force_max_duration} for scenario {scenario.generation_id}")
                for planning_problem in planning_problem_set.planning_problem_dict.values():
                    for goal_state in planning_problem.goal.state_list:
                        if "time_step" in goal_state.attributes:
                            goal_state.time_step = Interval(0, force_max_duration)

            if budget.is_over():
                logger.critical("Timeout is over. Force the current scenario to be invalid")
                scenario.is_valid = False
                scenario.validation_message = "Timeout is over"
                continue
            
            # Execution
            if scenario.is_valid:
                # In some cases, the scenario is valid, but there are no
                try:
                    logger.info(f"Executing scenario: {scenario.generation_id} - {budget.get_remaining_real_time()}")
                    [ logger.debug(f"- {pp.planning_problem_id} - {pp.initial_state.position} - {pp.initial_state.velocity} - planned start at {pp.initial_state.time_step}") for pp in planning_problem_set.planning_problem_dict.values()]

                    # TODO: It seems that sometime the ID of the lanelet for the planning problem is not correcly set/found?
                    # TODO Extension can be also done by attaching a new lanelet if necessary, but that requires computing the right origentation and such..
                    # Extend the scenario by adding some road AFTER the goal areas

                    # Make sure we enlong the lanelets nearby goal areas or the planner will fail to reach them
                    extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters=LANELET_EXTENSION)

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
                            scenario.validation_message = "Agent {agent.id} could not compute a phyisically possible trajectory since the beginning."

                except Exception as e:
                    # It migh
                    msg = f"A critical error prevented {scenario.generation_id} to complete. Mark it as Error and set the validity flag to False"
                    logger.critical( "=" * ( len(msg) + 2 ) )
                    logger.critical( f" {msg} ")
                    logger.critical( "=" * ( len(msg) + 2) )
                    scenario.is_valid = False
                    scenario.validation_message = "Error during the simulation execution"

            # Evaluation
            if scenario.is_valid:
                try:
                    logger.info(f"Evaluating scenario: {scenario.generation_id} - {budget.get_remaining_real_time()}")

                    budget.pause()
                    criticality_dataframe = scenario_evaluator.evaluate(simulation)
                    # Somehow, this Kill the entire process...
                    # Let's resort to collecting the plot at every time_step
                    # simulation.make_gif()
                    budget.resume()
                    logger.info(f"Done. - {budget.get_remaining_real_time()}")

                    simulation_data.append({
                        "individual": (scenario, planning_problem_set),
                        "agent-data": simulation.agents,
                        "evaluation-data": criticality_dataframe,
                    })
                except Exception as e:
                    msg = f"A critical error prevented {scenario.generation_id} to be evaluated. Mark it as invalid"
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

        # Close the bars    
        scenario_validation_bar.close(clear=True)

        for scenario, planning_problem_set in scenario_tuples:
            # Scenarios that were not executed because of a timeout
            if scenario.is_valid or scenario.validation_message == "Timeout is over":
                valid_scenarios = valid_scenarios + 1
            else:
                invalid_scenarios = invalid_scenarios + 1

        # Inform the generator about newly executed scenario to Evaluate the scenarios we managed to simulate if there's still some time
        if not budget.is_over():
            # If the budget is over, there's no point to ask the algorithm to evaluate the population
            # TODO Some algos might need this to produce the "final" result...
            scenario_generator.evaluate(simulation_data)

        # At this point, we have completed a generation epoch, we can check whether we need to checkpoint it
        
        elapsed_time = budget.get_elapsed()
        checkpoint_id = int( elapsed_time / checkpoint_every)
        if  checkpoint_id > 0 and checkpoint_id not in checkpoints:
            budget.pause()
            try:
                checkpoint(checkpoint_id, output_folder,
                           all_scenarios, valid_scenarios, invalid_scenarios, total_generation_time)
                checkpoints.add(checkpoint_id)
            except Exception as e:
                # TODO Remove this
                logger.critical(f"Failed to checkpoint execution at checkpoint {checkpoint_id}")
                raise e

            budget.resume()

    logger.info(f"Execution of {output_folder} is over ")

    # Log performance metrics right away - This is difficult/impossible to compute only looking at the simulation dbs
    logger.info("Genearion Performace Metrics")
    performance_file = os.path.join(output_folder, "performance.csv")
    with open(performance_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"])
        csv_writer.writerow([str(all_scenarios), str(valid_scenarios), str(invalid_scenarios), str(total_generation_time)])

    progress_bar_manager.stop()

    # Now analysize the data collected so far and output to the scenario folder
    generate_report(output_folder)
