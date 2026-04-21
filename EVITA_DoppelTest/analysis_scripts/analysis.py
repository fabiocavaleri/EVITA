import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from commonroad.visualization.icons import get_obstacle_icon_patch
from commonroad.scenario.obstacle import ObstacleType

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import get_interaction_map_bins

def apollo_experiments_analysis(experiments_folder_path: str):
    """
    
    """
    long_bins, lat_bins = get_interaction_map_bins()

    # Include 0 - B[0]:B[9] - 10 and len(bins)
    rows = len(long_bins) + 2 # 0 <, bins, len(bins) +1 or +2?
    cols = len(lat_bins) + 2 # 0 <, bins, len(bins)

    n_bins = 10

    riv_bins = np.linspace(0, 100, n_bins)
    ria_bins = np.linspace(0, 180, n_bins)

    # Generate all possible pairs of impact locations
    impact_location_bins = list()
    # 
    # "-+" denotes the road we need two symbols

    for p1 in ["F", "M", "B", "-"]:
        for s1 in ["L", "C", "R", "+"]:
            s = f"{p1}{s1}"
            impact_location_bins.append(s)

    collision_map_only_impact_location = np.zeros(len(impact_location_bins))
    collision_map_impact_location_and_riv = np.zeros((len(impact_location_bins), len(riv_bins) + 1))
    collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1))

    total_interaction_map = np.zeros((rows, cols))
    total_interactions = []
    total_unique_interactions = []
    total_collisions = []
    total_unique_collisions = []
    total_unique_collisions_POI = []
    total_unique_collisions_POI_RIV = []
    scenario_violations = []
    vehicles_number = []
    total_generated_scenarios = []
    total_valid_scenarios = []
    total_invalid_scenarios = []
    total_generation_time = []

    # List all the experiments in the input folder
    experiments_folders = [folder for folder in os.listdir(experiments_folder_path) if folder != "experiments_analysis"]

    # Iterate over each experiment
    for experiment_id in tqdm(experiments_folders, desc = "Processing Experiments", position = 0, leave = False):
        # Build experiment's folder path
        experiment_folder_path = os.path.join(experiments_folder_path, experiment_id)

        performance_file_path = os.path.join(experiment_folder_path, "performance.csv")
        if os.path.exists(performance_file_path):
            performance = pd.read_csv(performance_file_path)

            total_generated_scenarios.append(int(performance["all_scenarios"].iloc[0]))
            total_valid_scenarios.append(int(performance["valid_scenarios"].iloc[0]))
            total_invalid_scenarios.append(int(performance["invalid_scenarios"].iloc[0]))
            total_generation_time.append(float(performance["total_generation_time"].iloc[0]))

        # List all generations in the experiment folder
        generations_folders = [folder for folder in os.listdir(experiment_folder_path) if os.path.isdir(os.path.join(experiment_folder_path, folder))]

        # Iterate over each generation
        for generation_id in tqdm(generations_folders, desc = "Processing Generations", position = 1, leave = False):
            # Build generation's folder path
            generation_folder_path = os.path.join(experiment_folder_path, generation_id)

            # List all scenarios in the generation folder
            scenarios_folders = os.listdir(generation_folder_path)

            # Iterate over each generation
            for scenario_id in tqdm(scenarios_folders, desc = "Processing Scenarios", position = 2, leave = False):
                # Build scenario's folder path
                scenario_folder_path = os.path.join(generation_folder_path, scenario_id)

                interaction_map_file_path = os.path.join(scenario_folder_path, "interactions_map.csv")
                interaction_map = pd.read_csv(interaction_map_file_path, header = None).astype(int).values

                interactions_count = np.sum(interaction_map)
                total_interactions.append(interactions_count)

                unique_interactions_count = np.count_nonzero(interaction_map)
                total_unique_interactions.append(unique_interactions_count)

                total_interaction_map += interaction_map

                collisions_file_path = os.path.join(scenario_folder_path, "collisions.csv")
                collisions = pd.read_csv(collisions_file_path)

                local_only_impact_location = np.zeros(len(impact_location_bins))
                local_impact_location_and_riv = np.zeros((len(impact_location_bins), len(riv_bins) + 1))
                local_collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1))

                collisions_POI = collisions["impact_position"]
                collisions_RIA = collisions["ria"]
                collisions_RIV = collisions["riv"]

                if len(collisions_POI) != 0:
                    for impact_location, riv, ria in zip(collisions_POI, collisions_RIV, collisions_RIA):
                        impact_location_mapped = impact_location_bins.index(impact_location)
                        digitized_riv = np.digitize(riv, riv_bins)
                        digitized_ria = np.digitize(ria, ria_bins)

                        local_only_impact_location[impact_location_mapped] += 1.0
                        local_impact_location_and_riv[impact_location_mapped][digitized_riv] += 1.0
                        local_collision_map[impact_location_mapped][digitized_riv][digitized_ria] += 1.0
            
                    collision_map_only_impact_location += local_only_impact_location
                    collision_map_impact_location_and_riv += local_impact_location_and_riv
                    collision_map += local_collision_map

                    # Compute unique
                    unique_collisions_only_impact_location = np.count_nonzero(local_only_impact_location)
                    unique_collisions_impact_location_and_riv = np.count_nonzero(local_impact_location_and_riv)
                    unique_collisions = np.count_nonzero(local_collision_map)

                    total_collisions.append(len(collisions))
                    total_unique_collisions.append(unique_collisions)
                    total_unique_collisions_POI.append(unique_collisions_only_impact_location)
                    total_unique_collisions_POI_RIV.append(unique_collisions_impact_location_and_riv)
                else:
                    total_collisions.append(0)
                    total_unique_collisions.append(0)
                    total_unique_collisions_POI.append(0)
                    total_unique_collisions_POI_RIV.append(0)

                violations_file_path = os.path.join(scenario_folder_path, "scenario_violations.csv")
                violations = pd.read_csv(violations_file_path)
                if len(violations) != 0:
                    stop_sign_violations = len(violations[violations["violation"].str.startswith("stopsign")])
                    traffic_light_violations = len(violations[violations["violation"].str.startswith("signal")])
                    speeding_violations = len(violations) - stop_sign_violations - traffic_light_violations
                else:
                    stop_sign_violations = 0
                    traffic_light_violations = 0
                    speeding_violations = 0

                scenario_violations.append([stop_sign_violations, traffic_light_violations, speeding_violations])

                planned_trajectories_file_path = os.path.join(scenario_folder_path, "planned_trajectories.json")
                with open(planned_trajectories_file_path, "r") as f:
                    planned_trajectories = json.load(f)

                vehicles_count = len(planned_trajectories.keys())
                vehicles_number.append(vehicles_count)

    experiments_analysis_folder_path = os.path.join(experiments_folder_path, "experiments_analysis")
    os.makedirs(experiments_analysis_folder_path, exist_ok = True)
    fig = plot_interaction_map(total_interaction_map, long_bins, lat_bins)
    output_interaction_map_file = os.path.join(experiments_analysis_folder_path, f"interactions_map.png")
    plt.savefig(output_interaction_map_file, dpi = 500)
    plt.close(fig)

    # ================= EXPORT RESULTS =================
    def save_list(data, filename, column_name = "value"):
        df = pd.DataFrame(data, columns = [column_name])
        df.to_csv(os.path.join(experiments_analysis_folder_path, filename), index = False)

    # Interaction map (matrix)
    pd.DataFrame(total_interaction_map).to_csv(
        os.path.join(experiments_analysis_folder_path, "total_interaction_map.csv"),
        index = False,
        header = False
    )

    violations_df = pd.DataFrame(
        scenario_violations,
        columns = ["stop_sign", "traffic_light", "speeding"]
    )

    violations_df.to_csv(
        os.path.join(experiments_analysis_folder_path, "scenario_violations.csv"),
        index = False
    )

    # Per-scenario metrics
    save_list(total_interactions, "total_interactions.csv", "interactions")
    save_list(total_unique_interactions, "total_unique_interactions.csv", "unique_interactions")
    save_list(total_collisions, "total_collisions.csv", "collisions")
    save_list(total_unique_collisions, "total_unique_collisions.csv", "unique_collisions")
    save_list(total_unique_collisions_POI, "total_unique_collisions_POI.csv", "unique_collisions_POI")
    save_list(total_unique_collisions_POI_RIV, "total_unique_collisions_POI_RIV.csv", "unique_collisions_POI_RIV")
    save_list(vehicles_number, "vehicles_number.csv", "vehicles")

    # Experiment-level stats
    save_list(total_generated_scenarios, "total_generated_scenarios.csv", "generated")
    save_list(total_valid_scenarios, "total_valid_scenarios.csv", "valid")
    save_list(total_invalid_scenarios, "total_invalid_scenarios.csv", "invalid")
    save_list(total_generation_time, "total_generation_time.csv", "generation_time")

def plot_interaction_map(interaction_map, long_bins, lat_bins):
    config = ScenarioGeneratorConfig()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Get the colormap and set the under and bad colors
    colMap = cm.inferno_r
    colMap.set_bad(color = 'white')
    colMap.set_under(color = 'white')

    masked_data = np.copy(interaction_map).astype(float)
    masked_data[masked_data == 0 ] = np.nan

    try:
        im = ax.imshow(masked_data, cmap = colMap)
    except Exception as exc:
        print("STOP ME")
        raise exc

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(len(long_bins)), labels = [f"{v:.2f}" for v in long_bins])
    ax.set_yticks(np.arange(len(lat_bins)), labels = [f"{v:.2f}" for v in lat_bins])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False, labelsize = 6)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = -30, ha = "right", rotation_mode = "anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(len(long_bins)+1)-.5, minor = True)
    ax.set_yticks(np.arange(len(lat_bins)+1)-.5, minor = True)

    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which = "minor", bottom = False, left = False)

    ax.set_title(f"Interaction Effects' Map")

    center_x = sum(ax.get_xlim()) * 0.5 - 0.5
    center_y = sum(ax.get_ylim()) * 0.5 - 0.5

    icon_patch_list = get_obstacle_icon_patch(ObstacleType.CAR,
        center_x, # pos_x: Union[int, float],
        center_y, # pos_y: Union[int, float],
        0.0, # orientation: Union[int, float],
        config.length * 0.5,
        config.width * 0.5,
        # zorder=10,
        vehicle_color = (1, 1, 1, 0.6),
        edgecolor = "black",
        lw = 0.5)
    
    patch_col = mpl.collections.PatchCollection(icon_patch_list, match_original = True)
    ax.add_collection(patch_col)

    fig.tight_layout()

    return fig

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <experiments_folder_path>")
        sys.exit(1)

    experiments_folder_path = sys.argv[1]

    apollo_experiments_analysis(experiments_folder_path)