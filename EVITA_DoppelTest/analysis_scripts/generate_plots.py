import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from statistics import mean, median
from scipy.stats import mannwhitneyu

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import get_interaction_map_bins
from commonroad.visualization.icons import get_obstacle_icon_patch
from commonroad.scenario.obstacle import ObstacleType

from evita.data_analysis import compute_effect_size

N_BINS = 8

MAP_NAMES_MAPPING = {
    "borregas_ave": "Borregas Ave",
    "CHN_Cho-2_2_I-1-1": "Cho",
    "DEU_Cologne-79_2_I-1": "Cologne",
    "USA_US101-23_1_T-1": "US101-23",
    "USA_US101-15_2_T-1": "US101-15",
    "USA_US101-9_1_T-1": "US101-9"
}

def compute_p_value(data_1: list, data_2: list) -> float:
    """
    Computes the p-value of the two-sided Mann-Whitney U test between two samples.
    Returns None if the test cannot be computed (e.g., identical or degenerate samples).
    """
    try:
        _, p_value = mannwhitneyu(data_1, data_2, alternative = "two-sided")
        return round(float(p_value), 3)
    except ValueError:
        return None

def plot_interaction_map(interaction_map, long_bins, lat_bins, plot_title: str = "Interaction Effects Map"):
    """
    Plots the interaction map
    """
    # Initialize scenario configuration object
    config = ScenarioGeneratorConfig()

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize = (8, 4))

    # Get the colormap and set the under and bad colors
    colMap = cm.inferno_r
    colMap.set_bad(color = 'white')
    colMap.set_under(color = 'white')

    # Prepare data for plotting
    masked_data = np.copy(interaction_map).astype(float)
    masked_data[masked_data == 0 ] = np.nan

    # Plot the heatmap
    try:
        ax.imshow(masked_data, cmap = colMap)
    except Exception as exc:
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

    ax.set_xticks(np.arange(len(long_bins) + 1) -.5, minor = True)
    ax.set_yticks(np.arange(len(lat_bins) + 1) -.5, minor = True)

    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which = "minor", bottom = False, left = False)

    # Set plot title
    ax.set_title(plot_title)

    # Define the center of the plot for placing the vehicle icon
    center_x = sum(ax.get_xlim()) * 0.5 - 0.5
    center_y = sum(ax.get_ylim()) * 0.5 - 0.5

    icon_patch_list = get_obstacle_icon_patch(ObstacleType.CAR,
        center_x,
        center_y,
        0.0,
        config.length * 0.5,
        config.width * 0.5,
        vehicle_color = (1, 1, 1, 0.6),
        edgecolor = "black",
        lw = 0.5)
    
    patch_col = mpl.collections.PatchCollection(icon_patch_list, match_original = True)
    ax.add_collection(patch_col)

    fig.tight_layout()

    return fig

def compute_interaction_maps(experiment_folder: str, long_bins, lat_bins):
    """
    Computes the interaction maps for the different interaction types and saves the results as plots and csv files
    """

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
        file_path = os.path.join(experiment_folder, filename)

        clean_data = make_json_serializable(data)

        with open(file_path, "w") as f:
            json.dump(clean_data, f, indent = 4)
    
    os.makedirs(os.path.join(experiment_folder, "csv"), exist_ok = True)
    os.makedirs(os.path.join(experiment_folder, "plots"), exist_ok = True)
    
    # ========== STOP SIGNS INTERACTIONS ==========

    # Open stop signs interactions file
    stop_signs_interactions_file_path = os.path.join(experiment_folder, "total_stop_signs_interactions.json")

    if os.path.exists(stop_signs_interactions_file_path):
        with open(stop_signs_interactions_file_path, "r") as f:
            stop_signs_interactions = json.load(f)

        # Compute interactions map and unique interactions
        stop_signs_interactions_map, stop_signs_unique_interactions = compute_map(stop_signs_interactions, long_bins, lat_bins)

        # Save interactions map and unique interactions
        save_list(stop_signs_unique_interactions, "total_stop_signs_unique_interactions.json")
        pd.DataFrame(stop_signs_interactions_map).to_csv(
            os.path.join(experiment_folder, "csv", "total_interaction_map.csv"),
            index = False,
            header = False
        )
        
        # Plot interactions map
        fig = plot_interaction_map(stop_signs_interactions_map, long_bins, lat_bins, "Stop Signs Interactions Map")
        plt.savefig(os.path.join(experiment_folder, "plots", "stop_signs_interactions_map.png"), dpi = 500)
        plt.close(fig)

    # ========== TRAFFIC LIGHTS INTERACTIONS ==========

    # Open traffic lights interactions file
    traffic_lights_interactions_file_path = os.path.join(experiment_folder, "total_traffic_lights_interactions.json")

    if os.path.exists(traffic_lights_interactions_file_path):
        with open(traffic_lights_interactions_file_path, "r") as f:
            traffic_lights_interactions = json.load(f)

        # Compute interactions map and unique interactions
        traffic_lights_interactions_map, traffic_lights_unique_interactions = compute_map(traffic_lights_interactions, long_bins, lat_bins)

        # Save interactions map and unique interactions
        save_list(traffic_lights_unique_interactions, "total_traffic_lights_unique_interactions.json")
        pd.DataFrame(traffic_lights_interactions_map).to_csv(
            os.path.join(experiment_folder, "csv", "total_interaction_map.csv"),
            index = False,
            header = False
        )

        # Plot interactions map
        fig = plot_interaction_map(traffic_lights_interactions_map, long_bins, lat_bins, "Traffic Lights Interactions Map")
        plt.savefig(os.path.join(experiment_folder, "plots", "traffic_lights_interactions_map.png"), dpi = 500)
        plt.close(fig)

    # ========== VEHICLES INTERACTIONS ==========

    # Open vehicles interactions file
    vehicles_interactions_file_path = os.path.join(experiment_folder, "total_vehicles_interactions.json")

    if os.path.exists(traffic_lights_interactions_file_path):
        with open(vehicles_interactions_file_path, "r") as f:
            vehicles_interactions = json.load(f)

        # Compute interactions map and unique interactions
        vehicles_interactions_map, vehicles_unique_interactions = compute_map(vehicles_interactions, long_bins, lat_bins)

        # Save interactions map and unique interactions
        save_list(vehicles_unique_interactions, "total_vehicles_unique_interactions.json")
        pd.DataFrame(vehicles_interactions_map).to_csv(
            os.path.join(experiment_folder, "csv", "total_interaction_map.csv"),
            index = False,
            header = False
        )

        # Plot interactions map
        fig = plot_interaction_map(vehicles_interactions_map, long_bins, lat_bins, "Vehicles Interactions Map")
        plt.savefig(os.path.join(experiment_folder, "plots", "vehicles_interactions_map.png"), dpi = 500)
        plt.close(fig)

    # ========== TOTAL INTERACTIONS ==========

    # Open total interactions file
    total_interactions_file_path = os.path.join(experiment_folder, "total_interactions.json")
    with open(total_interactions_file_path, "r") as f:
        total_interactions = json.load(f)

    # Compute interactions map and unique interactions
    total_interactions_map, total_unique_interactions = compute_map(total_interactions, long_bins, lat_bins)

    # Save interactions map and unique interactions
    save_list(total_unique_interactions, "total_unique_interactions.json")
    pd.DataFrame(total_interactions_map).to_csv(
        os.path.join(experiment_folder, "csv", "total_interaction_map.csv"),
        index = False,
        header = False
    )

    # Plot interactions map
    fig = plot_interaction_map(total_interactions_map, long_bins, lat_bins, "Total Interactions Map")
    plt.savefig(os.path.join(experiment_folder, "plots", "total_interactions_map.png"), dpi = 500)
    plt.close(fig)

def compute_map(interactions: dict, long_bins, lat_bins):
    """
    Computes the interactions map and the unique interactions for the given interactions data
    """

    rows = len(long_bins) + 2
    cols = len(lat_bins) + 2

    # Initialize the interaction map and the unique interactions dictionary
    interaction_map = np.zeros((rows, cols))
    unique_interactions = {key: [] for key in interactions}

    # Iterate over the experiments IDs and their interactions lists
    for experiment_key, experiments_interactions in interactions.items():
        # Iterate over the interactions lists of the current experiment
        for interactions_lists in experiments_interactions:
            # Initialize local interaction map
            local_interaction_map = np.zeros((rows, cols))
            # Iterate over the interactions of the current interactions list
            for interaction in interactions_lists:
                # Extract longitudinal and latitudinal interaction's effect
                long = interaction[0]
                lat = interaction[1]

                long_bin_index = np.digitize(long, long_bins)
                lat_bin_index = np.digitize(lat, lat_bins)

                # Update local interaction map
                local_interaction_map[long_bin_index, lat_bin_index] += 1

            # Compute local unique interactions
            unique_interactions[experiment_key].append(np.count_nonzero(local_interaction_map))
            # Update interaction map
            interaction_map += local_interaction_map

    return interaction_map, unique_interactions

def interactions_collisions_plots(data_map: dict, output_path: str, aggregate: bool = True):
    """
    Generates the comparison plot between Evita and DoppelTest for the interactions or collisions metrics and saves the result in the output folder
    """

    all_data = []

    # Iterate over each map and aggregated metric data paths lists
    for map_name in data_map.keys():
        map_data = data_map[map_name]
        for framework, file_path in map_data.items():
            if file_path is None or not os.path.exists(file_path):
                print(f"Warning: File not found for framework '{framework}' in map '{map_name}'. Expected at: {file_path}")
                continue

            data = load_json_data(file_path, aggregate)

            for _, val in data.items():
                all_data.append({"Map": MAP_NAMES_MAPPING.get(map_name, map_name), "Metric": val, "Framework": framework})

    # Create DataFrame with the collected data
    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data available for the provided data map. Skipping plot generation.")
        return {}

    # Add a virtual map "All" aggregating all maps for each framework
    if len(df["Map"].unique()) > 1:
        all_df = df.copy()
        all_df["Map"] = "All"
        df = pd.concat([df, all_df], ignore_index = True)

    # Initialize the figure and axis for the boxplot
    fig, ax = plt.subplots(figsize = (7, 4))
    sns.set_style("ticks")

    # Define colors for each framework's data
    # Generate a color palette from a colormap based on the frameworks considered
    frameworks = df["Framework"].unique()
    cmap = plt.get_cmap("tab10")
    palette = {
        framework: cmap(i / max(len(frameworks) - 1, 1))
        for i, framework in enumerate(frameworks)
    }
    
    # Plot the boxplot
    sns.boxplot(
        data = df, 
        x = "Map", 
        y = "Metric", 
        hue = "Framework", 
        palette = palette,
        linewidth = 1,
        fliersize = 4,
        width = 0.8,
        gap = 0.2,
        flierprops = {"marker": "o", "markeredgecolor": "gray"},
        ax = ax
    )

    # Styling
    ax.legend(loc = 'best')
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(top = False, right = False)

    ax.margins(x = 0.05)
    
    plt.tight_layout()
    
    # Save boxplot to file
    fig.savefig(output_path, dpi = 800)
    plt.close(fig)

    # Compute effect size for every pair of frameworks
    effect_sizes = {}

    # Compute effect size for each map
    for map_name in data_map.keys():
        frameworks_data = {}
        effect_sizes[map_name] = {}
        for framework in frameworks:
            data = df[(df["Map"] == MAP_NAMES_MAPPING.get(map_name, map_name)) & (df["Framework"] == framework)]["Metric"].to_list()

            if len(data) != 0:
                frameworks_data[framework] = data

        for framework_1, framework_2 in combinations(frameworks_data.keys(), 2):
            data_framework_1 = frameworks_data[framework_1]
            data_framework_2 = frameworks_data[framework_2]

            if len(data_framework_1) != 0 and len(data_framework_2) != 0:
                effect_size = (*compute_effect_size(data_framework_1, data_framework_2), compute_p_value(data_framework_1, data_framework_2))
            else:
                effect_size = (-1, "n/a", None)

            effect_sizes[map_name][f"{framework_1}_{framework_2}"] = effect_size

    return effect_sizes

def load_json_data(file_path: str, aggregate: bool = True) -> dict:
    """
    Loads JSON data from a file and optionally aggregates data by experiment ID.
    """

    with open(file_path, "r") as f:
        data = json.load(f)

    if not aggregate:
        return data

    aggregated_data = {}

    for experiment, values in data.items():
        if isinstance(values, list):
            aggregated_data[experiment] = sum(
                value for value in values
                if isinstance(value, int) and not isinstance(value, bool)
            )
        elif isinstance(values, int) and not isinstance(values, bool):
            # Handle values that are already aggregated.
            aggregated_data[experiment] = values
        else:
            raise ValueError(f"Unexpected data type for experiment '{experiment}': {type(values).__name__}")

    return aggregated_data

def violations_plots(data_map: dict, output_folder_path: str, aggregate: bool = False):
    """
    Generates the comparison plots between Evita and DoppelTest for the violations metrics
    """

    # Initialize data structures
    all_data = []
    violations = ["stop_sign", "traffic_light", "speeding"]
    violations_effect_size_data = {}

    # Iterate over each violation type
    for i in range(3):
        violations_effect_size_data[violations[i]] = {}

        # Iterate over each map and aggregated metric data paths lists
        for map_name in data_map.keys():
            map_data = data_map[map_name]
            for framework, file_path in map_data.items():
                if file_path is None or not os.path.exists(file_path):
                    print(f"Warning: File not found for framework '{framework}' in map '{map_name}'. Expected at: {file_path}")
                    continue

                data = load_json_data(file_path, aggregate)

                for val in data.values():
                    # Count violation occurrences in the scenario
                    data_sum = sum(val[violations[i]])
                    all_data.append({"Map": MAP_NAMES_MAPPING.get(map_name, map_name), "Violations": data_sum, "Framework": framework})
            
        # Create DataFrame with the collected data for the current violation type
        df = pd.DataFrame(all_data)

        if df.empty:
            print("No data available for the provided data map. Skipping plot generation.")
            return {}

        # Add a virtual map "All" aggregating all maps for each framework
        if len(df["Map"].unique()) > 1:
            all_df = df.copy()
            all_df["Map"] = "All"
            df = pd.concat([df, all_df], ignore_index = True)

        # Initialize the figure and axis for the boxplot
        fig, ax = plt.subplots(figsize = (7, 4))
        sns.set_style("ticks")
        
        # Define colors for each framework's data
        # Generate a color palette from a colormap based on the frameworks considered
        frameworks = df["Framework"].unique()
        cmap = plt.get_cmap("tab10")
        palette = {
            framework: cmap(i / max(len(frameworks) - 1, 1))
            for i, framework in enumerate(frameworks)
        }
        
        # Plot data
        sns.boxplot(
            data = df, 
            x = "Map", 
            y = "Violations", 
            hue = "Framework", 
            palette = palette,
            linewidth = 1,
            fliersize = 4,
            width = 0.8,
            gap = 0.2,
            flierprops = {"marker": "o", "markeredgecolor": "gray"},
            ax = ax
        )

        # Styling
        ax.legend(loc = 'best')
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(top = False, right = False)
        
        ax.margins(x = 0.05)
        
        plt.tight_layout()

        # Save boxplot to file
        fig.savefig(os.path.join(output_folder_path, f"scenario_{violations[i]}_violations_comparison.png"), dpi = 800)
        plt.close(fig)


         # Compute effect size for each map
        for map_name in data_map.keys():
            frameworks_data = {}
            violations_effect_size_data[violations[i]][map_name] = {}
            for framework in frameworks:
                data = df[(df["Map"] == MAP_NAMES_MAPPING.get(map_name, map_name)) & (df["Framework"] == framework)]["Violations"].to_list()
    
                if len(data) != 0:
                    frameworks_data[framework] = data
    
            for framework_1, framework_2 in combinations(frameworks_data.keys(), 2):
                data_framework_1 = frameworks_data[framework_1]
                data_framework_2 = frameworks_data[framework_2]
    
                if len(data_framework_1) != 0 and len(data_framework_2) != 0:
                    effect_size = (*compute_effect_size(data_framework_1, data_framework_2), compute_p_value(data_framework_1, data_framework_2))
                else:
                    effect_size = (-1, "n/a", None)
    
                violations_effect_size_data[violations[i]][map_name][f"{framework_1}_{framework_2}"] = effect_size

    return violations_effect_size_data

def generate_statistics_plots(data_map: dict, first_label: str, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the generation statistics metrics
    """
    effect_sizes = {}

    # Iterate over each generation statistic and corresponding data paths list
    for statistic, statistic_data in data_map.items():
        all_data = []
        
        for map_name in statistic_data.keys():
            map_data = statistic_data[map_name]
            for framework, file_path in map_data.items():
                data = load_json_data(file_path)
    
                for _, val in data.items():
                    all_data.append({"Map": MAP_NAMES_MAPPING.get(map_name, map_name), f"{statistic}": val, "Framework": framework})
        
        # Create DataFrame with the collected data for the current generation statistic
        df = pd.DataFrame(all_data)

        # Initialize the figure and axis for the boxplot
        fig, ax = plt.subplots(figsize = (7, 4))
        sns.set_style("ticks")
        
        # Define colors for each framework's data
        # Generate a color palette from a colormap based on the frameworks considered
        frameworks = df["Framework"].unique()
        cmap = plt.get_cmap("tab10")
        palette = {
            framework: cmap(i / max(len(frameworks) - 1, 1))
            for i, framework in enumerate(frameworks)
        }
        
        # Plot data
        sns.boxplot(
            data = df, 
            x = "Map", 
            y = f"{statistic}", 
            hue = "Framework", 
            palette = palette,
            linewidth = 1,
            fliersize = 4,
            width = 0.8,
            gap = 0.2,
            flierprops = {"marker": "o", "markeredgecolor": "gray"},
            ax = ax
        )

        # Styling
        ax.legend(loc = 'best')
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(top = False, right = False) 

        ax.margins(x = 0.05)
        
        plt.tight_layout()
        
        # Save boxplot to file
        output_file_path = os.path.join(output_folder_path, f"{statistic}_comparison.png")
        fig.savefig(output_file_path, dpi = 800)
        
        plt.close(fig)

        # Initialize the figure and axis for the boxplot
        fig, ax = plt.subplots(figsize = (7, 4))
        sns.set_style("ticks")

        # Add a virtual map "All" aggregating all maps for each framework
        if len(df["Map"].unique()) > 1:
            all_df = df.copy()
            all_df["Map"] = "All"
            df = pd.concat([df, all_df], ignore_index = True)

        # Plot single framework data
        sns.boxplot(
            data = df[(df["Framework"] == first_label)],
            x = "Map", 
            y = f"{statistic}", 
            hue = "Framework", 
            palette = palette,
            linewidth = 1,
            fliersize = 4,
            width = 0.8,
            gap = 0.2,
            flierprops = {"marker": "o", "markeredgecolor": "gray"},
            ax = ax
        )

        # Styling
        ax.legend(loc = 'best')
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(top = False, right = False) 

        ax.margins(x = 0.05)
        
        plt.tight_layout()
        
        # Save boxplot to file
        output_file_path = os.path.join(output_folder_path, f"{first_label}_{statistic}_comparison.png")
        fig.savefig(output_file_path, dpi = 800)
        
        plt.close(fig)

        # Compute effect size for every pair of frameworks
        # Compute effect size for each map

        effect_sizes[statistic] = {}

        for map_name in statistic_data.keys():
            frameworks_data = {}
            effect_sizes[statistic][map_name] = {}
            for framework in frameworks:
                data = df[(df["Map"] == MAP_NAMES_MAPPING.get(map_name, map_name)) & (df["Framework"] == framework)][f"{statistic}"].to_list()

                if len(data) != 0:
                    frameworks_data[framework] = data

            for framework_1, framework_2 in combinations(frameworks_data.keys(), 2):
                data_framework_1 = frameworks_data[framework_1]
                data_framework_2 = frameworks_data[framework_2]

                if len(data_framework_1) != 0 and len(data_framework_2) != 0:
                    effect_size = (*compute_effect_size(data_framework_1, data_framework_2), compute_p_value(data_framework_1, data_framework_2))
                else:
                    effect_size = (-1, "n/a", None)

                effect_sizes[statistic][map_name][f"{framework_1}_{framework_2}"] = effect_size

    return effect_sizes

def generate_percentage_valid_scenarios_plot(valid_data_map: dict, generated_data_map: dict, output_folder_path: str):
    """
    Generates the comparison plot between Evita and DoppelTest for the percentage of valid scenarios generated
    """

    all_valid_data = []
    all_generated_data = []
    
    for map_name in valid_data_map.keys():
        valid_map_data = valid_data_map[map_name]
        generated_map_data = generated_data_map[map_name]

        for framework in valid_map_data.keys():
            valid_file_path = valid_map_data[framework]
            generated_file_path = generated_map_data[framework]

            valid_data = load_json_data(valid_file_path)

            for _, val in valid_data.items():
                all_valid_data.append({"Map": MAP_NAMES_MAPPING.get(map_name, map_name), "Valid": val, "Framework": framework})

            generated_data = load_json_data(generated_file_path)
            for _, val in generated_data.items():
                all_generated_data.append({"Map": MAP_NAMES_MAPPING.get(map_name, map_name), "Generated": val, "Framework": framework})

    # Create DataFrame with the collected data
    valid_df = pd.DataFrame(all_valid_data)
    generated_df = pd.DataFrame(all_generated_data)

    # Add a virtual map "All" aggregating all maps for each framework
    if len(valid_df["Map"].unique()) > 1:
        all_df = valid_df.copy()
        all_df["Map"] = "All"
        valid_df = pd.concat([valid_df, all_df], ignore_index = True)

        all_df = generated_df.copy()
        all_df["Map"] = "All"
        generated_df = pd.concat([generated_df, all_df], ignore_index = True)

    for framework in valid_df["Framework"].unique():
        percentage_data = []
        
        for map_name in valid_df["Map"].unique():
            valid_count = valid_df[(valid_df["Framework"] == framework) & (valid_df["Map"] == map_name)]["Valid"].sum()
            generated_count = generated_df[(generated_df["Framework"] == framework) & (generated_df["Map"] == map_name)]["Generated"].sum()

            if generated_count > 0:
                percentage_valid = (valid_count / generated_count) * 100
            else:
                percentage_valid = 0

            percentage_data.append({
                "Map": map_name,
                "Framework": framework,
                "Percentage": percentage_valid
                })
                
        percentage_df = pd.DataFrame(percentage_data)

        if percentage_df.empty:
            print(f"No data available for framework: {framework}. Skipping plot generation.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.set_style("ticks")

        sns.barplot(
            data = percentage_df,
            x = "Map",
            y = "Percentage",
            dodge = True,
            errorbar = None,
            ax = ax
        )

        ax.set_title(f"{framework}: % Valid Scenarios", fontsize = 14, pad = 15)
        ax.set_xlabel("Map")
        ax.set_ylabel("% Scenarios")
        ax.set_ylim(0, 100)

        for container in ax.containers:
            labels = [
                f"{bar.get_height():.1f}%".rstrip("0").rstrip(".")
                if bar.get_height() > 0
                else ""
                for bar in container
            ]

            ax.bar_label(
                container,
                labels = labels,
                padding = 3,
                fontsize = 9
            )

        sns.despine()

        plt.tight_layout()

        fig.savefig(os.path.join(output_folder_path, f"{framework}_valid_scenarios_comparison.png"), dpi = 500, bbox_inches = "tight")

        plt.close(fig)
        
def agents_number_plots(vehicles_data_map: dict, valid_data_map: dict, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the number of agents in the scenarios
    """

    all_valid_data = []
            
    for map_name, map_data in valid_data_map.items():
        for framework, file_path in map_data.items():
            data = load_json_data(file_path)

            for _, val in data.items():
                all_valid_data.append({"Map": MAP_NAMES_MAPPING.get(map_name, map_name), "Valid": val, "Framework": framework})

    # Create DataFrame with the collected data
    valid_df = pd.DataFrame(all_valid_data)
    frameworks = valid_df["Framework"].unique()
    summed_valid = {}
    for map_name in valid_data_map.keys():
        summed_valid[map_name] = {}
        for framework in frameworks:
            summed_valid[map_name][framework] = valid_df[(valid_df["Framework"] == framework) & (valid_df["Map"] == MAP_NAMES_MAPPING.get(map_name, map_name))]["Valid"].sum()

    all_data = {}
    statistics = {}
    
    # Iterate over each map and aggregated metric data paths lists
    for map_name, map_data in vehicles_data_map.items():
        statistics[map_name] = {}

        for framework, file_path in map_data.items():
            vehicle_count_list = []

            with open(file_path, "r") as f:
                data = json.load(f)

            for _, val in data.items():
                vehicle_count_list.extend(val)

            statistics[map_name][framework] = {
                "Mean": mean(vehicle_count_list),
                "Median": median(vehicle_count_list)
            }

            all_data[framework] = dict(Counter(vehicle_count_list))

        percentage_data = []

        for framework in frameworks:
            # Skip frameworks that don't have vehicle data
            if framework not in all_data:
                continue

            scenarios_count = all_data[framework]

            total_valid = summed_valid[map_name][framework]

            # Fallback if generated count is unavailable
            if total_valid == 0:
                total_valid = sum(scenarios_count.values())

            for n in sorted(scenarios_count.keys()):
                # Ignore scenarios with fewer than 2 agents
                if n < 2 or n > 8:
                    continue
                
                percentage = (scenarios_count[n] / total_valid) * 100

                percentage_data.append({
                    "Agents": n,
                    "Framework": framework,
                    "Percentage": percentage
                })

        percentage_df = pd.DataFrame(percentage_data)

        if percentage_df.empty:
            print(f"No data available for map: {map_name}. Skipping plot generation.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.set_style("ticks")

        sns.barplot(
            data = percentage_df,
            x = "Agents",
            y = "Percentage",
            hue = "Framework",
            dodge = True,
            errorbar = None,
            ax = ax
        )

        ax.set_title(map_name, fontsize = 14, pad = 15)
        ax.set_xlabel("# Agents")
        ax.set_ylabel("% Scenarios")

        # Make sure all agent counts are shown
        agent_counts = sorted(percentage_df["Agents"].unique())
        ax.set_xticks(range(len(agent_counts)))
        ax.set_xticklabels(agent_counts)

        for container in ax.containers:
            labels = [
                f"{bar.get_height():.1f}%".rstrip("0").rstrip(".")
                if bar.get_height() > 0
                else ""
                for bar in container
            ]

            ax.bar_label(
                container,
                labels = labels,
                padding = 3,
                fontsize = 9
            )

        sns.despine()

        ax.legend(title = "Framework", loc = "best")

        plt.tight_layout()

        fig.savefig(os.path.join(output_folder_path, f"{map_name}_scenario_vehicles_comparison.png"), dpi = 500, bbox_inches = "tight")

        plt.close(fig)

    vehicle_number_statistics_file_path = os.path.join(output_folder_path, "vehicle_number_statistics.json")
    with open(vehicle_number_statistics_file_path, "w") as f:
        json.dump(statistics, f, indent = 4)

def initialize_effect_size(experiments_folder_paths: list, experiments_labels: list, effect_size_columns: dict, additional_columns: dict) -> tuple:
    """
    Initializes the effect size data dictionary with the required structure
    """
    effect_size_results = {}
    effect_size_data = {}
    maps_folders = None

    for experiment_folder_path in experiments_folder_paths:
        maps = set([f for f in os.listdir(experiment_folder_path) if os.path.isdir(os.path.join(experiment_folder_path, f)) and f != "interactions_collisions_analysis"])

        if maps_folders is None:
            maps_folders = maps
        else:
            maps_folders &= maps

    for effect_size_column in effect_size_columns.keys():
        effect_size_results[effect_size_column] = {}
        effect_size_data[effect_size_column] = {}
        for map_folder in maps_folders:
            effect_size_results[effect_size_column][map_folder] = None
            effect_size_data[effect_size_column][map_folder] = {}
            for experiment_label in experiments_labels:
                effect_size_data[effect_size_column][map_folder][experiment_label] = None

    for additional_column in additional_columns.keys():
        effect_size_data[additional_column] = {}
        for map_folder in maps_folders:
            effect_size_data[additional_column][map_folder] = {}
            for experiment_label in experiments_labels:
                effect_size_data[additional_column][map_folder][experiment_label] = None

    return effect_size_results, effect_size_data, maps_folders

def generate_comparison_plots(experiments_folder_paths: list, experiments_labels: list, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the Borregas Ave map and saves the results in the output folder
    """
    
    # Initialize effect size dictionary
    effect_size_columns = {"interactions": "total_interactions_count.json", "unique_interactions": "total_unique_interactions.json", "collisions": "total_collisions.json", "unique_collisions": "total_unique_collisions.json", "violations": "scenario_violations.json"}
    additional_columns = {"generated": "total_generated_scenarios.json", "valid": "total_valid_scenarios.json", "invalid": "total_invalid_scenarios.json", "vehicles_number": "vehicles_number.json"}

    effect_size_results, effect_size_data, maps_folders = initialize_effect_size(experiments_folder_paths, experiments_labels, effect_size_columns,  additional_columns)

    # Compute interaction maps bins
    long_bins, lat_bins = get_interaction_map_bins(N_BINS)

    # Compute and plot interaction maps for both experiments and all maps
    msg = "Processing interaction maps..."
    print("=" * len(msg))
    print(msg)
    print("=" * len(msg))

    for experiment_folder_path, experiment_label in tqdm(zip(experiments_folder_paths, experiments_labels), desc = "Processing experiments"):
        tqdm.write(f"Processing experiment folder: {experiment_folder_path}")
        for map_folder in tqdm(maps_folders, desc = "Processing maps", leave = False):
            tqdm.write(f"Processing map folder: {map_folder}")
            experiment_analysis_folder_path = os.path.join(experiment_folder_path, map_folder, "experiments_analysis")
            compute_interaction_maps(experiment_analysis_folder_path, long_bins, lat_bins)

            for effect_size_column in (effect_size_columns | additional_columns):
                if effect_size_column in additional_columns.keys():
                    effect_size_file_path = os.path.join(experiment_folder_path, map_folder, "experiments_analysis", additional_columns[effect_size_column])
                else:
                    effect_size_file_path = os.path.join(experiment_folder_path, map_folder, "experiments_analysis", effect_size_columns[effect_size_column])
    
                if os.path.exists(effect_size_file_path):
                    effect_size_data[effect_size_column][map_folder][experiment_label] = (effect_size_file_path)
                else:
                    print(f"Warning: {effect_size_column} file not found for experiment '{experiment_label}' in map '{map_folder}'. Expected at: {effect_size_file_path}")
                
    print("Generated interactions maps plot")

    for effect_size_column in effect_size_columns.keys():
        files_list = effect_size_data[effect_size_column]

        if effect_size_column == "violations":
            effect_size_list = violations_plots(files_list, output_folder_path)
        else:
            output_path = os.path.join(output_folder_path, f"{effect_size_column}_comparison.png")
            effect_size_list = interactions_collisions_plots(files_list, output_path)

        print(f"Generated {effect_size_column} comparison plot")

        for map_folder in effect_size_list.keys():
            effect_size_results[effect_size_column][map_folder] = effect_size_list[map_folder]


    # ========== PLOT GENERATION STATISTICS =========
    data = {
        "generated": effect_size_data["generated"],
        "valid": effect_size_data["valid"],
        "invalid": effect_size_data["invalid"]
    }
    
    effect_size_list = generate_statistics_plots(data, experiments_labels[0], output_folder_path)
    effect_size_results["generation_statistics"] = effect_size_list
    generate_percentage_valid_scenarios_plot(effect_size_data["valid"], effect_size_data["generated"], output_folder_path)

    print("Generated generation statistics comparison plots")

    # ========== PLOT AGENTS NUMBER =========
    agents_number_plots(effect_size_data["vehicles_number"], effect_size_data["valid"], output_folder_path)

    print("Generated agents number comparison plots")

    effect_size_data_file_path = os.path.join(output_folder_path, "effect_size_data.json")
    with open(effect_size_data_file_path, "w") as f:
        json.dump(effect_size_results, f, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate comparison plots for multiple experiments.")

    parser.add_argument(
        "--experiments",
        nargs = "+",
        required = True,
        help = "Paths to the experiment folders."
    )

    parser.add_argument(
        "--labels",
        nargs = "+",
        required = True,
        help = "Labels corresponding to each experiment."
    )

    parser.add_argument(
        "--output",
        required = True,
        help = "Output folder path."
    )

    args = parser.parse_args()

    if len(args.experiments) != len(args.labels):
        parser.error("The number of experiment folder paths must match the number of experiment labels.")

    os.makedirs(args.output, exist_ok = True)

    generate_comparison_plots(args.experiments, args.labels, args.output)