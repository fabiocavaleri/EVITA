from ..interfaces import AbstractScenarioExecutor

from typing import Tuple

# import csv
import os
import sys

# import shutil

import traceback
from datetime import datetime

from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder

from commonroad.visualization.mp_renderer import MPRenderer

from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.file_reader import CommonRoadFileReader

from ..utils import FRENETIX_ROOT

# get n unique colors
def _get_cmap(n, name='hsv'):
    import matplotlib.pyplot as plt
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

###########################
### TODO: Refactor me!!
###########################
class FrenetixScenarioExecution(AbstractScenarioExecutor):
    
    # TODO Add details about the planner !
    def __init__(self, output_folder:str, **kwargs) -> None:   
        # super(ScenarioExecutor, self).__init__(output_folder, **kwargs)
        
        if sys.platform == "darwin":
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
        self.debug = kwargs["debug"] if "debug" in kwargs and kwargs["debug"] else False

        self.output_folder = output_folder
        self.execution_counter = 0
        self.driving_agent_name = kwargs["driving_agent_name"]

        # We need to create here all the necessary YAML configurations
        
    def visualize_the_scenario(self, scenario, planning_problem_set, output_folder):
        """
        Visualize and plot the initial scenario
        """

        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        planning_problems = sorted(planning_problem_set.planning_problem_dict.values(), key= lambda pp : pp.planning_problem_id)

        unique_colors = _get_cmap(len(planning_problems), 'rainbow')

        rnd = MPRenderer(figsize=(20, 10))

        # set renderer draw params
        rnd.draw_params.time_begin = 0
        rnd.draw_params.dynamic_obstacle.draw_icon = True

        # Draw Road and Obstacles at time stamp
        scenario.draw(rnd)
        
        for i, planning_problem in enumerate(planning_problems):
            from commonroad.planning.planning_problem import PlanningProblem
            planning_problem: PlanningProblem
            # set renderer draw params
            rnd.draw_params.planning_problem.initial_state.state.facecolor = colors.to_hex(unique_colors(i))
            rnd.draw_params.planning_problem.initial_state.state.draw_arrow = True
            rnd.draw_params.planning_problem.initial_state.state.radius = 0.5
            
            # Add labels
            rnd.draw_params.planning_problem.initial_state.label= f"{planning_problem.planning_problem_id} - {planning_problem.initial_state.velocity}"

            rnd.draw_params.planning_problem.goal_region.shape.facecolor = colors.to_hex(unique_colors(i))
            rnd.draw_params.planning_problem.goal_region.shape.opacity = 0.5
                
            # Include the arrow here!
            planning_problem.draw(rnd)

        # Render the plot
        rnd.render()

        # # Get the x, y limts without vehicles
        # x_lim = rnd.ax.get_xlim()
        # y_lim = rnd.ax.get_ylim()

        # # Plot the reference path of the planners
        # for planner in planners:
        #     rnd.ax.plot(planner.reference_path[:, 0], planner.reference_path[:, 1], color='g', marker='.', markersize=1, zorder=19, linewidth=0.8,
        #                 label=f'reference path planner {planner.name}')
            
        # # Get the x, y limts without the annotations
        # rnd.ax.set_xlim(x_lim)
        # rnd.ax.set_ylim(y_lim)

        # # save as .png file
        # print(f"Plotting {output_folder}/scenario.pdf")
        # os.makedirs(output_folder, exist_ok=True)
        plt.savefig(f"{output_folder}/scenario.png", format='png', dpi=300, bbox_inches='tight')
        # plt.savefig(f"{output_folder}/scenario.png", bbox_inches='tight')


    def execute(self, scenario, planning_problem_set) -> Simulation:
        """
        Return the original scenario, the simulated scenario, and additional information about the AVs
        """

        # Setup this execution
        self.execution_counter = self.execution_counter + 1 

        # Create the folders for the execution - use the generation_id to
        # exeuction_folder_name = str(self.execution_counter).rjust(5, '0')
        exeuction_folder_name = str(scenario.generation_id).rjust(5, '0')
        execution_folder = os.path.join(self.output_folder, exeuction_folder_name)
        os.makedirs(execution_folder, exist_ok=False)

        # Store here the input scenario, configurations, default configurations
        scenario_folder = os.path.join(execution_folder, "scenario")
        os.makedirs(scenario_folder, exist_ok=False)

        scenario_name = f"scenario_{exeuction_folder_name}"
        scenario_file_name = os.path.join(scenario_folder, f"{scenario_name}.xml")

        # CommonRoad truncate floating point numbers, so we need to ensure that we are indeed executing what we generate
        # Or at least be consistent to avoid reproducibility issues
        crfw = CommonRoadFileWriter(
            scenario,
            planning_problem_set,
            author = "EVITA",
            decimal_precision=10)
        crfw.write_to_file(scenario_file_name)

        # Note this one:    
        # To enforce consistency we re-read what we just wrote to file!
        crfr = CommonRoadFileReader(scenario_file_name)
        scenario, planning_problem_set = crfr.open()


        # Plot the scenario for reference
        self.visualize_the_scenario(scenario, planning_problem_set, scenario_folder)
        # Collect here the execution logs
        logs_folder = os.path.join(execution_folder, "logs")
        os.makedirs(logs_folder, exist_ok=False)


        # TODO Score? Metrics? Maybe we can compute this afterwards?
        # with open(os.path.join(logs_path, "score_overview.csv"), 'a') as file:
        #         line = "scenario;agent;timestep;status;message\n"
        #         file.write(line)

        config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, root_path=FRENETIX_ROOT, module="simulation")
        #
        config_sim.simulation.use_multiagent = True
        # Make sure we write the logs in the right folder
        config_sim.simulation.log_path = logs_folder

        # This should be handled in a different way, probably from outside to configure the agent, and pass it here
        # TODO This has more configurations FRENETIX_ROOT
        config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name, module="frenetix_motion_planner", root_path=FRENETIX_ROOT)
        
        # Setting this to FALSE does not work...
        config_planner.debug.use_cpp = True
        

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
            # with open('logs/log_failures.csv', 'a', newline='') as f:
            #     writer = csv.writer(f)
            current_time = datetime.now().strftime('%H:%M:%S')
            #     # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            #     writer.writerow([str(count) +" ; " + "Scenario Name: " + log_path.split("/")[-1] + " ; " +
            #                     "Error time: " + str(current_time) + " ; " +
            #                     "In Scenario Timestep: " + current_timestep + " ; " +
            #                     "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n"])
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


   

