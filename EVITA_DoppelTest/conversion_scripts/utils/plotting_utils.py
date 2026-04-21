import os
import json
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from commonroad.scenario.state import CustomState
from commonroad.scenario.scenario import Scenario
from commonroad.prediction.prediction import Trajectory
from commonroad.visualization.mp_renderer import MPDrawParams, MPRenderer
from commonroad.planning.planning_problem import PlanningProblemSet, GoalRegion


def draw_scenario(scenario: Scenario, output_folder: str, planned_trajectories: dict, planning_problems_set : PlanningProblemSet = None):
    """
    Generate scenario plots and animate in .gif file
    
    :param scenario: CommonRoad scenario object
    :type scenario: Scenario
    :param output_folder: Scenario plot images output folder
    :type output_folder: str
    :param planned_trajectories: Scenario vehicles planned trajectories
    :type planned_trajectories: dict
    """

    #
    # ---------- Scenario Plots Generation ----------
    #

    parent_folder = os.path.dirname(output_folder)
    output_file = os.path.join(parent_folder, "simulation.gif")

    # If scenario has not been plotted yet
    if not os.path.exists(output_file):
        os.makedirs(output_folder, exist_ok = True)

        # Get scenario obstacles
        obstacles = sorted(scenario.dynamic_obstacles, key = lambda o : o.obstacle_id)

        # Get scenario traffic lights
        traffic_lights = scenario.lanelet_network.traffic_lights

        # Get color mapping for each scenario obstacle
        unique_colors = plt.cm.get_cmap('rainbow', len(obstacles))
        color_mapping = dict()
        
        # Get initial and final timesteps for each scenario obstacle
        initial_timesteps = []
        final_timesteps = []
        
        for i, obstacle in enumerate(obstacles):
            color_mapping[obstacle.obstacle_id] = colors.to_hex(unique_colors(i))
            
            initial_timesteps.append(obstacle.prediction.initial_time_step)
            final_timesteps.append(obstacle.prediction.final_time_step)

        # Compute scenario initial and final timesteps and its duration
        scenario_initial_timestep = min(initial_timesteps)
        scenario_final_timestep = max(final_timesteps)
        scenario_duration = scenario_final_timestep - scenario_initial_timestep

        # If the number of plotted frames is less than the scenario duration
        if len(os.listdir(output_folder)) < scenario_duration:
            # Remove incomplete frames folder and recreate it
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)

            # Initialize plotting objects
            _, ax = plt.subplots(figsize = (4, 8), dpi = 500)
            rnd = MPRenderer(ax = ax)

            # Set lanelets color
            rnd.draw_params.lanelet_network.lanelet.left_bound_color  = "#bbbbbb"
            rnd.draw_params.lanelet_network.lanelet.right_bound_color = "#bbbbbb"
            rnd.draw_params.lanelet_network.lanelet_center_bound_color = "#888888"

            # Avoid traffic lights to be drawn with lanelet network
            rnd.draw_params.lanelet_network.traffic_light.draw_traffic_lights = False

            # Draw traffic signs
            rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
            rnd.draw_params.lanelet_network.traffic_sign.scale_factor = 0.25

            # Draw stop lines
            rnd.draw_params.lanelet_network.lanelet.draw_stop_lines = True
            rnd.draw_params.lanelet_network.lanelet.stop_line_color = "black"

            # Set scenario initial and final timesteps parameters
            rnd.draw_params.time_begin = scenario_initial_timestep
            rnd.draw_params.time_end = scenario_final_timestep

            # Build a mapping between scenario obstacles ids and planning problems ids
            pp_id_map = {}
            if planning_problems_set is not None:
                obstacles_ids = [obstacle.obstacle_id for obstacle in obstacles]
                for o in obstacles_ids:
                    pp_id_map[o] = 10000 + (o - min(obstacles_ids)) + 1

            # Iterate over each scenario timestep
            for time_step in tqdm(range(scenario_initial_timestep, scenario_final_timestep + 1), desc = "Scenario Timesteps"):
                # Reset plotting objects and remove axis infos
                ax.cla()
                ax.axis('off')
                rnd.ax = ax

                # Draw the scenario roads, traffic signs and stop lines
                scenario.lanelet_network.draw(rnd)

                # Iterate over each scenario traffic light
                for traffic_light in traffic_lights:
                    # Get traffic light state at plotted timestep
                    traffic_light_state = traffic_light.get_state_at_time_step(time_step)
                    
                    # Plot the traffic light only if the state is valid
                    if traffic_light_state is not None:
                        tl_draw_params = MPDrawParams().traffic_light

                        # Define traffic light scale factor depending on the scenario map
                        if scenario.scenario_id.map_name.startswith('borregas'):
                            tl_draw_params.scale_factor = 0.3
                        else:
                            tl_draw_params.scale_factor = 0.15
                        
                        tl_draw_params.zorder = 600
                        tl_draw_params.color_mapping = {
                            "red": "#ff0000",
                            "yellow": "#ffff00",
                            "green": "#00ff00",
                        }
                        tl_draw_params.time_begin = time_step
                        tl_draw_params.time_end = scenario_final_timestep
                        
                        # Draw the traffic light
                        traffic_light.draw(rnd, draw_params = tl_draw_params)
                
                # Iterate over each scenario obstacle
                stop_points = {}
                for obstacle in obstacles:

                    # Get obstacle occupancy at plotting timestep
                    obstacle_occupancy = obstacle.occupancy_at_time(time_step)
                    # If the obstacle is present at this timestep, plot it
                    if obstacle_occupancy is not None:
                        # Set obstacle parameters
                        obs_params = MPDrawParams()
                        obs_params.dynamic_obstacle.draw_icon = True
                        obs_params.dynamic_obstacle.show_label = True

                        # We need both otherwise begin and end otherwise it will not show it
                        obs_params.dynamic_obstacle.time_begin = time_step
                        obs_params.dynamic_obstacle.time_end = obstacle.prediction.final_time_step

                        # Set vehicle icon colors
                        obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = color_mapping[obstacle.obstacle_id]
                        obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "black"

                        # Ensure this is draw above everything else
                        obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.zorder = 900
                        obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.opacity = 1

                        obs_params.dynamic_obstacle.draw_direction = True
                        obs_params.dynamic_obstacle.draw_signals = True
                        
                        # Draw a line instead of a set of points using the same color of the vehicle
                        # Highlight better the planned trajectories (bold lines)
                        obs_params.dynamic_obstacle.trajectory.draw_trajectory = True
                        obs_params.dynamic_obstacle.trajectory.draw_continuous = True
                        obs_params.dynamic_obstacle.trajectory.facecolor = color_mapping[obstacle.obstacle_id]
                        obs_params.dynamic_obstacle.trajectory.line_width = 1.5
                        obs_params.dynamic_obstacle.trajectory.zorder = 150

                        # Get obstacle planned trajectory (if exist)
                        # If the vehicle has planned a trajectory at time t, get it
                        # Otherwise get the closer planned trajectories from the past
                        planned_trajectory_at_t = None
                        vehicle_planned_trajectories = planned_trajectories.get(obstacle.obstacle_id, {})
                        if time_step in vehicle_planned_trajectories:
                            planned_trajectory_at_t = vehicle_planned_trajectories[time_step]["planned_trajectory"]
                            stop_point_at_t = vehicle_planned_trajectories[time_step]["stop_position"]
                        else:
                            t = time_step - 1
                            while t >= (time_step - 25):
                                if t in vehicle_planned_trajectories:
                                    planned_trajectory_at_t = vehicle_planned_trajectories[t]["planned_trajectory"]
                                    stop_point_at_t = vehicle_planned_trajectories[t]["stop_position"]
                                    break
                                t -= 1

                        # Draw planned trajectory if it exist
                        if planned_trajectory_at_t is not None:
                            # print(f"{type(planned_trajectory_at_t)} Agent: {obstacle.obstacle_id} - Planning Timestep: {planned_trajectory_at_t.state_list[0].time_step} - Trajectory length: {len(planned_trajectory_at_t.state_list)}")
                            traj_params = MPDrawParams().dynamic_obstacle.trajectory

                            traj_params.draw_trajectory = True
                            traj_params.draw_continuous = True
                            traj_params.facecolor = "black"
                            traj_params.line_width = 0.75
                            traj_params.zorder = 400

                            # Draw planned trajectory only if the difference between its initial state position and 
                            # obstacle position at time t is less than 10 meters in both x and y axis.
                            # This is to avoid plotting too old planned trajectories.
                            planned_trajectory_initial_position = planned_trajectory_at_t.state_list[0].position
                            vehicle_history = obstacle.prediction.trajectory.state_list
                            obstacle_position = [state.position for state in vehicle_history if state.time_step == time_step][0]

                            position_difference = abs(planned_trajectory_initial_position - obstacle_position)

                            if position_difference[0] <= 5 and position_difference[1] <= 5:
                                planned_trajectory_at_t.draw(rnd, draw_params = traj_params)
                                
                                if stop_point_at_t is not None:
                                    stop_points[obstacle.obstacle_id] = stop_point_at_t

                        # Finally render the obstacle
                        obstacle.draw(rnd, draw_params=obs_params)

                        # Check if there is a valid planning problem for the current obstacle
                        if planning_problems_set is not None and pp_id_map[obstacle.obstacle_id] in planning_problems_set.planning_problem_dict:
                            # Get planning problem and its goal region
                            planning_problem = planning_problems_set.planning_problem_dict[pp_id_map[obstacle.obstacle_id]]
                            goal_region = planning_problem.goal

                            # Iterate over each goal state in the goal region
                            for goal_state in goal_region.state_list:
                                # Build (sub)-goal region object
                                sub_goal = GoalRegion(state_list = [goal_state])

                                goal_region_params = MPDrawParams()
                                
                                goal_region_params.time_begin = scenario_initial_timestep
                                goal_region_params.time_end = scenario_final_timestep
                                # Match the color with the vehicle
                                goal_region_params.goal_region.shape.facecolor = color_mapping[obstacle.obstacle_id]
                                goal_region_params.goal_region.shape.opacity = 0.6
                                
                                # Draw the (sub)-goal region
                                sub_goal.draw(rnd, draw_params = goal_region_params)
                
                # Render the scenario onto the plot
                rnd.render()

                # for obstacle in obstacles:
                #     if obstacle.obstacle_id in stop_points:
                #         x, y = stop_points[obstacle.obstacle_id]
                #         rnd.ax.plot(
                #             x, y,
                #             marker = 'x',
                #             markersize = 5,
                #             color = color_mapping[obstacle.obstacle_id],
                #             zorder = 1500
                #         )

                
                # Show scenario timesteps
                middle_x = 0.5
                top_y = 0.98
                ax.text(middle_x, top_y, f"Time: {str(round(float(time_step * scenario.dt), 1)).rjust(5,' ')} sec", horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)

                # Save scenario frame image
                plot_file = os.path.join(output_folder, f"frame_{str(time_step).rjust(5, '0')}.png")
                plt.savefig(plot_file, bbox_inches = 'tight', pad_inches = 0.1)
                
            plt.close("all")

        #
        # ---------- Scenario Animation ----------
        #

        # Animate scenario frames images exist and if the .gif file doesn't exists
        if len(os.listdir(output_folder)) > 0 and not os.path.exists(output_file):

            # Build gifski CLI arguments
            gif_options = []
            gif_options += ["-o", f"{output_file}"]
            gif_options += ["--fps", str(int(1 / scenario.dt))]

            # All frames saved above follow pattern frame_XXXXX.png
            gif_options += [os.path.join(output_folder, "frame_*.png")]

            # Run gifski
            exit_code = subprocess.check_call(
                f"gifski {' '.join(gif_options)}",
                shell = True
            )
            assert exit_code == 0, "Something went wrong while generating the GIF"

def load_planned_trajectories(json_path: str) -> Dict:
    """
    Reads planned trajectories from the saved JSON file and reconstructs
    { vehicle_id : { initial_timestep : Trajectory(...) } }
    """

    # Open .json file containing scenario's planned trajectories
    with open(json_path, "r") as f:
        data = json.load(f)

    scenario_planned_trajectories = {}

    # Iterate over each vehicle planned trajectories list
    for vehicle, traj_dict in data.items():
        vehicle_id = int(vehicle)
        scenario_planned_trajectories[vehicle_id] = {}

        # Iterate over each trajectory state
        for planning_timestep, traj_data in traj_dict.items():
            initial_timestep = int(planning_timestep)

            trajectory = trajectory_from_dict(traj_data["planned_trajectory"])
            scenario_planned_trajectories[vehicle_id][initial_timestep] = {
                "planned_trajectory": trajectory,
                "stop_position": traj_data["stop_position"]
            }

    return scenario_planned_trajectories

def trajectory_from_dict(data: dict) -> Trajectory:
    """
    Convert from .json like representation to CommonRoad trajectory object
    
    :param data: Trajectory .json data
    :type data: dict
    """

    # Convert state list to CommonRoad state objects
    state_list = [state_from_dict(s) for s in data["state_list"]]

    return Trajectory(
        initial_time_step = data["initial_time_step"],
        state_list = state_list
    )

def state_from_dict(data: dict) -> CustomState:
    """
    Convert from .json like representation to CommonRoad state object
    
    :param data: CustomState .json data
    :type data: dict
    """

    position = np.array(data["position"], dtype = float)

    return CustomState(
        time_step = data["time_step"],
        position = position,
        orientation = data["orientation"],
        velocity = data["velocity"],
        acceleration = data["acceleration"],
        yaw_rate = data["yaw_rate"],
    )