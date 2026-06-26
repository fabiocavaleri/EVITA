from ..interfaces import AbstractScenarioExecutor

import os
import sys

import traceback
from datetime import datetime

from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder

from commonroad.visualization.mp_renderer import MPRenderer

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_reader import CommonRoadFileReader

from ..utils import FRENETIX_ROOT

def _get_cmap(n, name = 'hsv'):
    """
    Get a function that maps each index in 0, 1, ..., n - 1 to a distinct RGB color
    the keyword argument name must be a standard mpl colormap name
    """
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(name, n)

class FrenetixScenarioExecution(AbstractScenarioExecutor):
    """
    This class is responsible for executing a scenario using the Frenetix motion planner
    """
    def __init__(self, output_folder:str, **kwargs) -> None:           
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
        if "debug" in kwargs and kwargs["debug"]:
            self.debug = kwargs["debug"]
        else:
            self.debug = False

        self.output_folder = output_folder
        self.execution_counter = 0
        self.driving_agent_name = kwargs["driving_agent_name"]
        
    def visualize_the_scenario(self, scenario, planning_problem_set, output_folder):
        """
        Visualize and plot the initial scenario
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        # Sort the planning problems by their ID to ensure consistent coloring
        planning_problems = sorted(planning_problem_set.planning_problem_dict.values(), key = lambda pp : pp.planning_problem_id)
        # Get a unique color for each planning problem
        unique_colors = _get_cmap(len(planning_problems), 'rainbow')

        # Initialize the renderer
        rnd = MPRenderer(figsize = (20, 10))

        # Set renderer draw params
        rnd.draw_params.time_begin = 0
        rnd.draw_params.dynamic_obstacle.draw_icon = True

        # Draw Road and Obstacles at time stamp
        scenario.draw(rnd)
        
        # Iterate over the planning problems and draw them with unique colors
        for i, planning_problem in enumerate(planning_problems):
            from commonroad.planning.planning_problem import PlanningProblem
            planning_problem: PlanningProblem
            # Set renderer draw params
            rnd.draw_params.planning_problem.initial_state.state.facecolor = colors.to_hex(unique_colors(i))
            rnd.draw_params.planning_problem.initial_state.state.draw_arrow = True
            rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
            
            # Add labels
            rnd.draw_params.planning_problem.initial_state.label= f"{planning_problem.planning_problem_id} - {planning_problem.initial_state.velocity}"

            rnd.draw_params.planning_problem.goal_region.shape.facecolor = colors.to_hex(unique_colors(i))
            rnd.draw_params.planning_problem.goal_region.shape.opacity = 0.5
                
            # Plot the planning problem
            planning_problem.draw(rnd)

        # Render the plot
        rnd.render()

        # Save the plot to a file
        plt.savefig(f"{output_folder}/scenario.png", format = 'png', dpi = 500, bbox_inches = 'tight')

    def execute(self, scenario, planning_problem_set) -> Simulation:
        """
        Return the original scenario, the simulated scenario, and additional information about the AVs
        """

        # Increment the execution counter for each execution
        self.execution_counter = self.execution_counter + 1 

        # Create the folders for the execution using the generation_id
        execution_folder_name = str(scenario.generation_id).rjust(5, '0')
        execution_folder = os.path.join(self.output_folder, execution_folder_name)
        os.makedirs(execution_folder, exist_ok=False)

        # Store here the input scenario, configurations, default configurations
        scenario_folder = os.path.join(execution_folder, "scenario")
        os.makedirs(scenario_folder, exist_ok=False)

        # Build the scenario file name and file path
        scenario_name = f"scenario_{execution_folder_name}"
        scenario_file_name = os.path.join(scenario_folder, f"{scenario_name}.xml")

        # Write the scenario to a CommonRoad file
        # CommonRoad truncate floating point numbers, so we need to ensure that we are indeed executing what we generate
        # Or at least be consistent to avoid reproducibility issues
        crfw = CommonRoadFileWriter(
            scenario,
            planning_problem_set,
            author = "EVITA",
            decimal_precision = 10)
        crfw.write_to_file(scenario_file_name)

        # To enforce consistency we re-read what we just wrote to file
        crfr = CommonRoadFileReader(scenario_file_name)
        scenario, planning_problem_set = crfr.open()

        # Plot the scenario for reference
        self.visualize_the_scenario(scenario, planning_problem_set, scenario_folder)
        # Collect here the execution logs
        logs_folder = os.path.join(execution_folder, "logs")
        os.makedirs(logs_folder, exist_ok=False)

        # Build the simulation configuration and the planner configuration
        config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, root_path = FRENETIX_ROOT, module = "simulation")
        #
        config_sim.simulation.use_multiagent = True
        # Make sure we write the logs in the right folder
        config_sim.simulation.log_path = logs_folder

        # Build the planner configuration
        config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name, module = "frenetix_motion_planner", root_path = FRENETIX_ROOT)
        config_planner.debug.use_cpp = True

        # Add debugging options if the debug flag is set
        if self.debug:
            config_sim.visualization.save_plots = True
            config_sim.visualization.save_gif = True

        # All the configurations used for this execution will be stored in simulation.db, table meta
        simulation = None

        try:
            # If any agent fails to start, this will trigger an exception, and stop the simulation
            simulation = Simulation(config_sim, config_planner)
            simulation.run_simulation()
        except Exception as e:
            error_traceback = traceback.format_exc()  # This gets the entire error traceback
            current_time = datetime.now().strftime('%H:%M:%S')
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            print("Error time: " + str(current_time) + " ; " +
                  "In Scenario Timestep: " + current_timestep + " ; " +
                  "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n")
            raise e
        finally:
            try:
                # close sim_logger
                simulation.sim_logger.con.close()
                # close child processes
                simulation.close_processes()
            except:
                pass
        
        return simulation