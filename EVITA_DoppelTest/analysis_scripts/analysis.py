import os
import sys
import ast
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor, as_completed

from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_reader import XMLFileReader
from commonroad.scenario.traffic_sign import TrafficSign
from commonroad.scenario.traffic_light import TrafficLight


import traceback

def process_scenario(scenario_folder_path: str, impact_location_bins, riv_bins, ria_bins):
    """
    Process a single scenario and return all aggregated metrics.
    """

    # Initialize result data structure for this scenario
    result = {
        "vehicles_interactions": [],
        "traffic_lights_interactions": [],
        "stop_signs_interactions": [],
        "interactions": [],

        "interaction_positions": {
            "vehicles": [],
            "traffic_lights": [],
            "stop_signs": []
        },

        "collisions": 0,
        "unique_collisions": 0,
        "unique_collisions_POI": 0,
        "unique_collisions_POI_RIV": 0,

        "collision_map_only_impact_location": np.zeros(len(impact_location_bins)),
        "collision_map_impact_location_and_riv": np.zeros((len(impact_location_bins), len(riv_bins) + 1)),
        "collision_map": np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1)),

        "violations": [0, 0, 0],  # stop_sign, traffic_light, speeding
        "vehicles": 0,
        "start": [],
        "goal": [],
        "trajectories_points": [],
        "distances": [],
    }

    try:
        # Define relevant files paths
        scenario_file_path = os.path.join(scenario_folder_path, "scenario.xml")
        executed_scenario_file_path = os.path.join(scenario_folder_path, "executed_scenario.xml")
        interactions_file_path = os.path.join(scenario_folder_path, "interactions.csv")
        collisions_file_path = os.path.join(scenario_folder_path, "collisions.csv")

        # Check existance of the analysis files
        # If any is missing, skip the scenario
        if not (os.path.exists(executed_scenario_file_path) and os.path.exists(interactions_file_path) and os.path.exists(collisions_file_path)):
            return None

        # ================= COLLISIONS =================

        # Open scenario's collisions file
        collisions = pd.read_csv(collisions_file_path)

        # Initialize local collision maps
        local_only_impact_location = np.zeros(len(impact_location_bins))
        local_impact_location_and_riv = np.zeros((len(impact_location_bins), len(riv_bins) + 1))
        local_collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1))

        # Check if any collision happened in the scenario
        if len(collisions) > 0:
            # Iterate over each collision metric
            for impact_location, riv, ria in zip(collisions["impact_position"], collisions["riv"], collisions["ria"]):
                # Retrieve impact location index and RIV/RIA bins
                idx = impact_location_bins.index(impact_location)
                riv_bin = np.digitize(riv, riv_bins)
                ria_bin = np.digitize(ria, ria_bins)

                # Update local collision maps
                local_only_impact_location[idx] += 1
                local_impact_location_and_riv[idx][riv_bin] += 1
                local_collision_map[idx][riv_bin][ria_bin] += 1

            # Update scenario's result with collisions metrics
            result["collisions"] = len(collisions)
            result["unique_collisions"] = np.count_nonzero(local_collision_map)
            result["unique_collisions_POI"] = np.count_nonzero(local_only_impact_location)
            result["unique_collisions_POI_RIV"] = np.count_nonzero(local_impact_location_and_riv)

        # Update scenario's result with collisions maps
        result["collision_map_only_impact_location"] = local_only_impact_location
        result["collision_map_impact_location_and_riv"] = local_impact_location_and_riv
        result["collision_map"] = local_collision_map

        # ================= VIOLATIONS =================

        violations_file = os.path.join(scenario_folder_path, "scenario_violations.csv")

        # Check if violations file exists
        if os.path.exists(violations_file):
            # Open scenario's violations file
            violations = pd.read_csv(violations_file)

            # If any violation happened in the scenario
            if len(violations) > 0:
                # Count stop sign, traffic light and speeding violations
                stop = len(violations[violations["violation"].str.startswith("stopsign")])
                light = len(violations[violations["violation"].str.startswith("signal")])
                speed = len(violations) - stop - light

                # Save violations count in the scenario's result
                result["violations"] = [stop, light, speed]

        # ================= VEHICLES =================
        
        planned_trajectories_file = os.path.join(scenario_folder_path, "planned_trajectories.json")
        
        # Check if planned trajectories file exists
        if os.path.exists(planned_trajectories_file):
            with open(planned_trajectories_file, "r") as f:
                # Open planned trajectories file
                planned = json.load(f)
                # Count the number of vehicles in the scenario
                result["vehicles"] = len(planned.keys())

        # ================= DISTANCES =================

        # Open scenario's CommonRoad planning problems and executed scenario file
        _, planning_problems = XMLFileReader(scenario_file_path).open(lanelet_assignment = True)
        executed_scenario, _ = XMLFileReader(executed_scenario_file_path).open(lanelet_assignment = True)

        # Sort scenario's planning problems and agents by their IDs to ensure correct matching
        scenario_pp = sorted(planning_problems.planning_problem_dict.values(), key = lambda pp: pp.planning_problem_id)
        scenario_agents = sorted(executed_scenario.obstacles, key = lambda agent: agent.obstacle_id)

        # Iterate over each agent and its corresponding planning problem
        for agent, planning_problem in zip(scenario_agents, scenario_pp):
            # Extract agent's driven trajectory
            trajectory = agent.prediction.trajectory.state_list
            # Skip too short trajectories
            if len(trajectory) < 2:
                continue
            
            # Extract agent's goal area
            goal = planning_problem.goal.state_list[0].position.center

            # Extract agent's initial and final position
            start = trajectory[0].position
            end = trajectory[-1].position
            # Compute approximated driven distance
            result["distances"].append(np.linalg.norm(end - start))

            # Extract agent's positions along the driven trajectory
            traj_points = [state.position for state in trajectory]
            
            # Save agent's start and goal positions and trajectory points in the scenario's result
            result["start"].append(start)
            result["goal"].append(goal)
            result["trajectories_points"].extend(traj_points)

        # ================= INTERACTION MAP =================

        # Open scenario's interactions file
        interactions = pd.read_csv(interactions_file_path)

        # Iterate over each interaction detected in the scenario
        for _, row in interactions.iterrows():
            # Extract interacting vehicle ID, interaction timestep, interacting agents IDs and the effects of the interaction on the agent's trajectory
            vehicle_id = int(row["agent_id"])
            interaction_timestep = int(row["timestep"])
            is_interacting_with = ast.literal_eval(row["is_interacting_with"])
            diff_long = row["diff_long"]
            diff_lat = row["diff_lat"]

            # Retrieve agent's position at the interaction timestep
            agent = executed_scenario.obstacle_by_id(int(vehicle_id))
            agent_state = agent.state_at_time(interaction_timestep)

            # Extract agent's position if available and valid
            position = None
            if agent_state is not None:
                provisional_position = agent_state.position
                if provisional_position is not None and hasattr(provisional_position, 'x') and hasattr(provisional_position, 'y'):
                    position = [float(provisional_position.x), float(provisional_position.y)]

            # Iterate over each interacting agent
            for agent_id in is_interacting_with:
                # Try to retrieve interacting obstacle as CommonRoad traffic light object
                obstacle = executed_scenario.lanelet_network.find_traffic_light_by_id(int(agent_id))

                if obstacle is None:
                    # If interacting agent is not a traffic light
                    # Try to retrieve interacting obstacle as CommonRoad stop sign object
                    obstacle = executed_scenario.lanelet_network.find_traffic_sign_by_id(int(agent_id))

                if obstacle is None:
                    # If interacting agent is not a stop sign
                    # Try to retrieve interacting obstacle as CommonRoad dynamic obstacle
                    obstacle = executed_scenario.obstacle_by_id(int(agent_id))
                
                # Save the interaction's information depending on the type of the interacting obstacle (vehicle, traffic light or stop sign)
                if isinstance(obstacle, TrafficSign):
                    result["stop_signs_interactions"].append([diff_long, diff_lat])
                    if position is not None:
                        result["interaction_positions"]["stop_signs"].append(position)
                elif isinstance(obstacle, TrafficLight):
                    result["traffic_lights_interactions"].append([diff_long, diff_lat])
                    if position is not None:
                        result["interaction_positions"]["traffic_lights"].append(position)
                else:
                    result["vehicles_interactions"].append([diff_long, diff_lat])
                    if position is not None:
                        result["interaction_positions"]["vehicles"].append(position)

                result["interactions"].append([diff_long, diff_lat])

    except Exception:
        print(traceback.format_exc())
        return None

    return result

def collect_scenario_paths(experiments_folder_path: str):
    """
    Efficiently collect all scenario folder paths
    """
    scenario_paths = []

    # Iterate over each experiment
    for experiment in os.scandir(experiments_folder_path):
        # Skip non-directories and the analysis output folder
        if not experiment.is_dir() or experiment.name == "experiments_analysis":
            continue

        # Iterate over each generation in the experiments
        for generation in os.scandir(experiment.path):
            # Skip non-directories
            if not generation.is_dir():
                continue
            # Iterate over each scenario in the generations
            for scenario in os.scandir(generation.path):
                # Add only directories (scenarios) to the list
                if scenario.is_dir():
                    scenario_paths.append(scenario.path)

    return scenario_paths

def apollo_experiments_analysis(experiments_folder_path: str, max_workers = 5):
    """
    Parallel analysis of the experiments
    """

    # ================= INIT GLOBAL STRUCTURES =================
    n_bins = 10
    riv_bins = np.linspace(0, 100, n_bins)
    ria_bins = np.linspace(0, 180, n_bins)

    impact_location_bins = [
        f"{p}{s}"
        for p in ["F", "M", "B", "-"]
        for s in ["L", "C", "R", "+"]
    ]

    # Initialize aggregated collision maps
    collision_map_only_impact_location = np.zeros(len(impact_location_bins))
    collision_map_impact_location_and_riv = np.zeros((len(impact_location_bins), len(riv_bins) + 1))
    collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1))

    # List of experiments IDs (folder names)
    experiments_ids = [experiment.name for experiment in os.scandir(experiments_folder_path) if experiment.is_dir() and experiment.name != "experiments_analysis"]
    
    # Initialize aggregated lists for interactions, collisions, violations, vehicles number and trajectories metrics
    total_vehicles_interactions = {exp_id: [] for exp_id in experiments_ids}
    total_traffic_lights_interactions = {exp_id: [] for exp_id in experiments_ids}
    total_stop_signs_interactions = {exp_id: [] for exp_id in experiments_ids}
    total_interactions = {exp_id: [] for exp_id in experiments_ids}
    total_vehicle_interactions_positions = []
    total_stop_sign_interaction_positions = []
    total_traffic_lights_interactions_positions = []
    total_collisions = {exp_id: [] for exp_id in experiments_ids}
    total_unique_collisions = {exp_id: [] for exp_id in experiments_ids}
    total_unique_collisions_POI = {exp_id: [] for exp_id in experiments_ids}
    total_unique_collisions_POI_RIV = {exp_id: [] for exp_id in experiments_ids}
    scenario_violations = {exp_id: [] for exp_id in experiments_ids}
    vehicles_number = {exp_id: [] for exp_id in experiments_ids}
    start_positions = []
    goal_positions = []
    trajectories_points = []
    distances = {exp_id: [] for exp_id in experiments_ids}

    # Initialize aggregated lists for generation metrics
    total_generated_scenarios = {exp_id: [] for exp_id in experiments_ids}
    total_valid_scenarios = {exp_id: [] for exp_id in experiments_ids}
    total_invalid_scenarios = {exp_id: [] for exp_id in experiments_ids}
    total_generation_time = {exp_id: [] for exp_id in experiments_ids}

    # ================= COLLECT SCENARIOS PATHS =================
    scenario_paths = collect_scenario_paths(experiments_folder_path)
    print(f"Found {len(scenario_paths)} scenarios")

    # ================= READ EXPERIMENTS GENERATION METRICS =================

    # Iterate over each experiment
    for experiment in os.scandir(experiments_folder_path):
        # Skip non-directories and the analysis output folder
        if not experiment.is_dir() or experiment.name == "experiments_analysis":
            continue

        # Check if is an EVITA experiment (has performance.csv)
        performance_file = os.path.join(experiment.path, "performance.csv")

        # Check if generation performance file exists
        if os.path.exists(performance_file):
            # Read EVITA performance metrics
            perf = pd.read_csv(performance_file)

            # Save generation metrics
            total_generated_scenarios[experiment.name].append(int(perf["all_scenarios"].iloc[0]))
            total_valid_scenarios[experiment.name].append(int(perf["valid_scenarios"].iloc[0]))
            total_invalid_scenarios[experiment.name].append(int(perf["invalid_scenarios"].iloc[0]))
            total_generation_time[experiment.name].append(float(perf["total_generation_time"].iloc[0]))

    # ================= PARALLEL ANALYSIS PROCESSING =================
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        # Submit all scenarios for processing
        futures = {
            executor.submit(
                process_scenario,
                path,
                impact_location_bins,
                riv_bins,
                ria_bins,
            ): path
            for path in scenario_paths
        }

        # Iterate over completed scenarios analysis tasks
        for future in tqdm(as_completed(futures), total = len(futures), desc = "Processing Scenarios"):
            # Extract experiment ID from the scenario path
            path = futures[future]
            future_experiment_id = path.split(os.sep)[-3]

            # Extract scenario's analysis results
            res = future.result()

            # Skipped failed analysis
            if res is None:
                print("A scenario failed to process. Skipping.")
                continue

            # ===== SAVE SCENARIO'S ANALYSIS RESULTS =====

            # Save interactions metrics
            total_vehicles_interactions[future_experiment_id].append(res["vehicles_interactions"])
            total_traffic_lights_interactions[future_experiment_id].append(res["traffic_lights_interactions"])
            total_stop_signs_interactions[future_experiment_id].append(res["stop_signs_interactions"])
            total_interactions[future_experiment_id].append(res["interactions"])

            # Save interactions positions
            total_vehicle_interactions_positions.extend(res["interaction_positions"]["vehicles"])
            total_stop_sign_interaction_positions.extend(res["interaction_positions"]["stop_signs"])
            total_traffic_lights_interactions_positions.extend(res["interaction_positions"]["traffic_lights"])

            # Save collisions metrics
            total_collisions[future_experiment_id].append(res["collisions"])
            total_unique_collisions[future_experiment_id].append(res["unique_collisions"])
            total_unique_collisions_POI[future_experiment_id].append(res["unique_collisions_POI"])
            total_unique_collisions_POI_RIV[future_experiment_id].append(res["unique_collisions_POI_RIV"])

            # Save collision maps
            collision_map_only_impact_location += res["collision_map_only_impact_location"]
            collision_map_impact_location_and_riv += res["collision_map_impact_location_and_riv"]
            collision_map += res["collision_map"]

            # Save violations data
            scenario_violations[future_experiment_id].append(res["violations"])

            # Save vehicles number data
            vehicles_number[future_experiment_id].append(res["vehicles"])

            # Save trajectories metrics
            start_positions.extend(res["start"])
            goal_positions.extend(res["goal"])
            trajectories_points.extend(res["trajectories_points"])

            # Save driven distances
            distances[future_experiment_id].extend(res["distances"])

    # Count the number of interactions for each scenario in the aggregated interactions lists
    total_interactions_count = count_interactions(total_interactions)
    total_vehicles_interactions_count = count_interactions(total_vehicles_interactions)
    total_traffic_lights_interactions_count = count_interactions(total_traffic_lights_interactions)
    total_stop_signs_interactions_count = count_interactions(total_stop_signs_interactions)

    # ================= EXPORT EXPERIMENTS ANALYSIS =================

    # Define and create output folder for experiments analysis results
    experiments_analysis_folder_path = os.path.join(experiments_folder_path, "experiments_analysis")
    os.makedirs(experiments_analysis_folder_path, exist_ok=True)

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    def save_list(data, filename):
        file_path = os.path.join(experiments_analysis_folder_path, filename)

        clean_data = make_json_serializable(data)

        with open(file_path, "w") as f:
            json.dump(clean_data, f, indent = 4)

    def save_positions(data, filename):
        arr = np.asarray(data)

        df = pd.DataFrame(arr, columns = ["x", "y"])
        df.to_csv(os.path.join(experiments_analysis_folder_path, filename), index = False)

    # ================= PLOT HEATMAPS =================
    # Load one reference scenario for map data
    example_scenario_path = scenario_paths[0]
    scenario_file = os.path.join(example_scenario_path, "scenario.xml")
    scenario, _ = XMLFileReader(scenario_file).open()

    # ---- Start positions heatmap ----
    fig = plot_position_heatmap(start_positions, scenario, "Start Positions Heatmap")
    plt.savefig(os.path.join(experiments_analysis_folder_path, "start_heatmap.png"),
                dpi = 500,
                bbox_inches = "tight",
                transparent = False)
    plt.close(fig)

    # ---- Goal positions heatmap ----
    fig = plot_position_heatmap(goal_positions, scenario, "Goal Positions Heatmap")
    plt.savefig(os.path.join(experiments_analysis_folder_path, "goal_heatmap.png"),
                dpi = 500,
                bbox_inches = "tight",
                transparent = False)
    plt.close(fig)

    # ---- Trajectories  heatmap ----
    fig = plot_position_heatmap(trajectories_points, scenario, "Trajectories Heatmap")
    plt.savefig(os.path.join(experiments_analysis_folder_path, "trajectories_heatmap.png"),
                dpi = 500,
                bbox_inches = "tight",
                transparent = False)
    plt.close(fig)

    # ---- Vehicle interactions heatmap ----
    fig = plot_position_heatmap(total_vehicle_interactions_positions, scenario, "Vehicle Interactions Heatmap")
    plt.savefig(os.path.join(experiments_analysis_folder_path, "vehicle_interactions_heatmap.png"),
                dpi = 500,
                bbox_inches = "tight",
                transparent = False)
    plt.close(fig)

    # ---- Stop sign interactions heatmap ----
    fig = plot_position_heatmap(total_stop_sign_interaction_positions, scenario, "Stop Sign Interactions Heatmap")
    plt.savefig(os.path.join(experiments_analysis_folder_path, "stop_sign_interactions_heatmap.png"),
                dpi = 500,
                bbox_inches = "tight",
                transparent = False)
    plt.close(fig)

    # ---- Traffic light interactions heatmap ----
    fig = plot_position_heatmap(total_traffic_lights_interactions_positions, scenario, "Traffic Light Interactions Heatmap")
    plt.savefig(os.path.join(experiments_analysis_folder_path, "traffic_light_interactions_heatmap.png"),
                dpi = 500,
                bbox_inches = "tight",
                transparent = False)
    plt.close(fig)

    # ================= VIOLATIONS =================
    converted_scenario_violations = {}

    # Iterate over each experiment ID and violations lists
    for experiment_key, values in scenario_violations.items():
        # Check if violations data is available for the experiment
        if not values:
            # If not available save empty violations data for the experiment and skip to the next experiment
            converted_scenario_violations[experiment_key] = {
                "stop_sign": [],
                "traffic_light": [],
                "speeding": []
            }
            continue

        # Extract stop sign, traffic light and speeding violations lists
        first  = [x[0] for x in values]
        second = [x[1] for x in values]
        third  = [x[2] for x in values]

        # Save converted violations data for the experiment
        converted_scenario_violations[experiment_key] = {
            "stop_sign": first,
            "traffic_light": second,
            "speeding": third
        }

    # Save converted violations data for all experiments in a JSON file
    with open(os.path.join(experiments_analysis_folder_path, "scenario_violations.json"), "w") as f:
        json.dump(converted_scenario_violations, f, indent = 4)

    # ================= SAVE AGGREGATED METRICS =================
    save_list(total_vehicles_interactions, "total_vehicles_interactions.json")
    save_list(total_interactions_count, "total_interactions_count.json")
    save_list(total_vehicles_interactions_count, "total_vehicles_interactions_count.json")
    save_list(total_traffic_lights_interactions, "total_traffic_lights_interactions.json")
    save_list(total_traffic_lights_interactions_count, "total_traffic_lights_interactions_count.json")
    save_list(total_stop_signs_interactions, "total_stop_signs_interactions.json")
    save_list(total_stop_signs_interactions_count, "total_stop_signs_interactions_count.json")
    save_list(total_interactions, "total_interactions.json")
    save_positions(total_vehicle_interactions_positions, "total_vehicle_interactions_positions.csv")
    save_positions(total_stop_sign_interaction_positions, "total_stop_sign_interactions_positions.csv")
    save_positions(total_traffic_lights_interactions_positions, "total_traffic_light_interactions_positions.csv")
    save_list(total_collisions, "total_collisions.json")
    save_list(total_unique_collisions, "total_unique_collisions.json")
    save_list(total_unique_collisions_POI, "total_unique_collisions_POI.json")
    save_list(total_unique_collisions_POI_RIV, "total_unique_collisions_POI_RIV.json")
    save_list(vehicles_number, "vehicles_number.json")
    save_positions(start_positions, "start_positions.csv")
    save_positions(goal_positions, "goal_positions.csv")
    save_positions(trajectories_points, "trajectories.csv")
    save_list(distances, "distances.json")

    # ================= SAVE AGGREGATED GENERATION METRICS =================
    if total_generated_scenarios:
        save_list(total_generated_scenarios, "total_generated_scenarios.json")
        save_list(total_valid_scenarios, "total_valid_scenarios.json")
        save_list(total_invalid_scenarios, "total_invalid_scenarios.json")
        save_list(total_generation_time, "total_generation_time.json")

    print("Analysis completed.")

def compute_density(points, xlim, ylim, bins = 200):
    """
    Compute the density of points in a 2D space using a 2D histogram
    """
    points = np.asarray(points)

    x = points[:, 0]
    y = points[:, 1]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins = bins, range = [xlim, ylim])

    return heatmap.T, xedges, yedges

def get_scenario_bounds(scenario):
    """
    Compute the bounding box of the drivable area in the scenario based on the lanelet vertices
    """
    
    xs = []
    ys = []

    for lanelet in scenario.lanelet_network.lanelets:
        xs.extend(lanelet.left_vertices[:, 0])
        xs.extend(lanelet.right_vertices[:, 0])
        ys.extend(lanelet.left_vertices[:, 1])
        ys.extend(lanelet.right_vertices[:, 1])

    return min(xs), max(xs), min(ys), max(ys)

def plot_position_heatmap(points, scenario, title):
    """
    Plot a heatmap of the density of points in the scenario's map
    """

    # Skip plotting if there are no points to plot
    if len(points) == 0:
        return None

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize = (8, 8))

    # ===== DRAW ROAD LAYOUT =====
    rnd = MPRenderer(ax = ax)

    rnd.draw_params.lanelet_network.traffic_sign.draw_traffic_signs = True
    rnd.draw_params.lanelet_network.traffic_sign.scale_factor = 0.7
    rnd.draw_params.lanelet_network.traffic_light.scale_factor = 0.7
    rnd.draw_params.lanelet_network.lanelet.stop_line_color = "black"

    scenario.lanelet_network.draw(rnd)
    rnd.render()

    # ===== COMPUTE SCENARIO'S MAP BOUNDS =====
    xmin, xmax, ymin, ymax = get_scenario_bounds(scenario)

    # ===== COMPUTE DENSITY =====
    heatmap, xedges, yedges = compute_density(
        points,
        xlim = (xmin, xmax),
        ylim = (ymin, ymax),
        bins = 200
    )

    # Skip heatmap overlay if density is zero (no points or all points in the same position)
    if heatmap is None:
        return fig

    # ===== SMOOTH HEATMAP DATA =====
    heatmap = gaussian_filter(heatmap, sigma = 1.5)

    # Log scaling (fix dynamic range)
    heatmap = np.log1p(heatmap)

    # Normalize safely
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # ===== Overlay heatmap =====
    im = ax.imshow(
        heatmap,
        extent = extent,
        origin = "lower",
        alpha = 0.6,
        cmap = "inferno",
        zorder = 10
    )

    plt.colorbar(im, ax = ax, label = "Normalized Density")

    ax.set_title(title)
    ax.set_aspect("equal")

    return fig

def count_interactions(interactions_dict: dict):
    """
    Count the number of interactions for each scenario in the given interactions dictionary
    """
    # Initialize interactions count dictionary
    interactions_count = {}

    # Iterate over each experiment and its scenarios interactions list
    for experiment_key, interactions_list in interactions_dict.items():
        # Initialize interactions count list for the experiment
        interactions_count[experiment_key] = []

        # Iterate over each scenario interactions list
        for interactions in interactions_list:
            # Count the number of interactions for the scenario
            count = sum(len(ints) for ints in interactions)
            interactions_count[experiment_key].append(count)

    return interactions_count

if __name__ == "__main__":
    # Check if the script is being run with the correct number of arguments
    if len(sys.argv) not in [2, 3]:
        print("Usage: python analysis.py <experiments_folder_path> [<max_workers> (default = 5)]")
        sys.exit(1)

    experiments_folder_path = sys.argv[1]
    if len(sys.argv) == 3:
        max_workers = int(sys.argv[2]) 
        apollo_experiments_analysis(experiments_folder_path, max_workers)
    else:
        apollo_experiments_analysis(experiments_folder_path)