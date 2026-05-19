import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import get_interaction_map_bins
from commonroad.visualization.icons import get_obstacle_icon_patch
from commonroad.scenario.obstacle import ObstacleType

from evita.data_analysis import compute_effect_size

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
    
    # ========== STOP SIGNS INTERACTIONS ==========

    # Open stop signs interactions file
    stop_signs_interactions_file_path = os.path.join(experiment_folder, "total_stop_signs_interactions.json")
    with open(stop_signs_interactions_file_path, "r") as f:
        stop_signs_interactions = json.load(f)

    # Compute interactions map and unique interactions
    stop_signs_interactions_map, stop_signs_unique_interactions = compute_map(stop_signs_interactions, long_bins, lat_bins)

    # Save interactions map and unique interactions
    save_list(stop_signs_unique_interactions, "total_stop_signs_unique_interactions.json")
    pd.DataFrame(stop_signs_interactions_map).to_csv(
        os.path.join(experiment_folder, "total_interaction_map.csv"),
        index = False,
        header = False
    )
    
    # Plot interactions map
    fig = plot_interaction_map(stop_signs_interactions_map, long_bins, lat_bins, "Stop Signs Interactions Map")
    plt.savefig(os.path.join(experiment_folder, "stop_signs_interactions_map.png"), dpi = 500)
    plt.close(fig)

    # ========== TRAFFIC LIGHTS INTERACTIONS ==========

    # Open traffic lights interactions file
    traffic_lights_interactions_file_path = os.path.join(experiment_folder, "total_traffic_lights_interactions.json")
    with open(traffic_lights_interactions_file_path, "r") as f:
        traffic_lights_interactions = json.load(f)

    # Compute interactions map and unique interactions
    traffic_lights_interactions_map, traffic_lights_unique_interactions = compute_map(traffic_lights_interactions, long_bins, lat_bins)

    # Save interactions map and unique interactions
    save_list(traffic_lights_unique_interactions, "total_traffic_lights_unique_interactions.json")
    pd.DataFrame(traffic_lights_interactions_map).to_csv(
        os.path.join(experiment_folder, "total_interaction_map.csv"),
        index = False,
        header = False
    )

    # Plot interactions map
    fig = plot_interaction_map(traffic_lights_interactions_map, long_bins, lat_bins, "Traffic Lights Interactions Map")
    plt.savefig(os.path.join(experiment_folder, "traffic_lights_interactions_map.png"), dpi = 500)
    plt.close(fig)

    # ========== VEHICLES INTERACTIONS ==========

    # Open vehicles interactions file
    vehicles_interactions_file_path = os.path.join(experiment_folder, "total_vehicles_interactions.json")
    with open(vehicles_interactions_file_path, "r") as f:
        vehicles_interactions = json.load(f)

    # Compute interactions map and unique interactions
    vehicles_interactions_map, vehicles_unique_interactions = compute_map(vehicles_interactions, long_bins, lat_bins)

    # Save interactions map and unique interactions
    save_list(vehicles_unique_interactions, "total_vehicles_unique_interactions.json")
    pd.DataFrame(vehicles_interactions_map).to_csv(
        os.path.join(experiment_folder, "total_interaction_map.csv"),
        index = False,
        header = False
    )

    # Plot interactions map
    fig = plot_interaction_map(vehicles_interactions_map, long_bins, lat_bins, "Vehicles Interactions Map")
    plt.savefig(os.path.join(experiment_folder, "vehicles_interactions_map.png"), dpi = 500)
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
        os.path.join(experiment_folder, "total_interaction_map.csv"),
        index = False,
        header = False
    )

    # Plot interactions map
    fig = plot_interaction_map(total_interactions_map, long_bins, lat_bins, "Total Interactions Map")
    plt.savefig(os.path.join(experiment_folder, "total_interactions_map.png"), dpi = 500)
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

def generate_comparison_plots(evita_borregas_folder_path: str, doppeltest_borregas_folder_path: str, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the Borregas Ave map and saves the results in the output folder
    """
    
    # Initialize effect size data dictionary
    effect_size_data = {}
    # Compute interaction maps bins
    long_bins, lat_bins = get_interaction_map_bins(8)

    # Compute and plot interaction maps for both Evita and DoppelTest
    compute_interaction_maps(os.path.join(evita_borregas_folder_path, "experiments_analysis"), long_bins, lat_bins)
    compute_interaction_maps(os.path.join(doppeltest_borregas_folder_path, "experiments_analysis"), long_bins, lat_bins)
    print("Generated interactions maps plot")

    # ========== COMPARE TOTAL INTERACTIONS =========
    borregas_effect_size = interactions_collisions_plots({
            "Borregas Ave": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_interactions_count.json"), 
                os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_interactions_count.json")
            ),
        },
        os.path.join(output_folder_path, "interactions_comparison.png")
    )

    effect_size_data["interactions"] = {
        "borregas_ave": borregas_effect_size,
    }

    print("Generated interactions comparison plot")

    # ========== COMPARE UNIQUE INTERACTIONS =========
    borregas_effect_size = interactions_collisions_plots({
        "Borregas Ave": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_unique_interactions.json"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_unique_interactions.json")
        )
    },
    os.path.join(output_folder_path, "unique_interactions_comparison.png")
    )

    effect_size_data["unique_interactions"] = {
        "borregas_ave": borregas_effect_size,
    }

    print("Generated unique interactions comparison plot")

    # ========== COMPARE TOTAL COLLISIONS =========
    borregas_effect_size = interactions_collisions_plots({
            "Borregas Ave": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_collisions.json"), 
                os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_collisions.json")
            )
        },
        os.path.join(output_folder_path, "collisions_comparison.png")
    )

    effect_size_data["collisions"] = {
        "borregas_ave": borregas_effect_size,
    }

    print("Generated collisions comparison plot")

    # ========== COMPARE UNIQUE COLLISIONS =========
    borregas_effect_size = interactions_collisions_plots({
        "Borregas Ave": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_unique_collisions.json"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_unique_collisions.json")
        ),
    },
    os.path.join(output_folder_path, "unique_collisions_comparison.png")
    )

    effect_size_data["unique_collisions"] = {
        "borregas_ave": borregas_effect_size,
    }

    print("Generated unique collisions comparison plot")

    # ========== COMPARE VIOLATIONS =========
    violations_effect_size_data = violations_plots({
        "Borregas Ave": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "scenario_violations.json"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "scenario_violations.json")
        ),
    },
    output_folder_path
    )

    effect_size_data["violations"] = violations_effect_size_data

    print("Generated violations comparison plots")

    # ========== PLOT GENERATION STATISTICS =========
    generatation_statistics_plots({
            "generated": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_generated_scenarios.json"), 
            ),
            "valid": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_valid_scenarios.json"), 
            ),
            "invalid": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_invalid_scenarios.json"), 
            )
        },
        output_folder_path
    )

    print("Generated generation statistics comparison plots")

    # ========== PLOT AGENTS NUMBER =========
    agents_number_plots({
        "borregas": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "vehicles_number.json"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "vehicles_number.json")
        ),
    },
    output_folder_path
    )

    print("Generated agents number comparison plots")

    effect_size_data_file_path = os.path.join(output_folder_path, "effect_size_data.json")
    with open(effect_size_data_file_path, "w") as f:
        json.dump(effect_size_data, f, indent = 4)

def interactions_collisions_plots(data_map: dict, output_path: str):
    """
    Generates the comparison plot between Evita and DoppelTest for the interactions or collisions metrics and saves the result in the output folder
    """

    all_data = []

    # Iterate over each map and aggregated metric data paths lists
    for map, paths in data_map.items():
        # Extract aggregated metrics data paths
        evita_path, dt_path = paths
        
        # Load DoppelTest data
        doppeltest_data = load_json_data(dt_path)

        # Iterate over the aggregated metrics values for DoppelTest
        for val in doppeltest_data:
            all_data.append({"Map": map, "Metric": val, "Framework": "DoppelTest"})
            
        # Load Evita data
        evita_data = load_json_data(evita_path)

        # Iterate over the aggregated metrics values for EVITA
        for val in evita_data:
            all_data.append({"Map": map, "Metric": val, "Framework": "Evita"})

    # Create DataFrame with the collected data
    df = pd.DataFrame(all_data)

    # Initialize the figure and axis for the boxplot
    fig, ax = plt.subplots(figsize = (10, 5))
    sns.set_style("ticks")
    
    # Define colors for each framework's data'
    palette = {"DoppelTest": "#9fcfff", "Evita": "#f19e6e"}
    
    # Plot the boxplot
    sns.boxplot(
        data = df, 
        x = "Map", 
        y = "Metric", 
        hue = "Framework", 
        palette = palette,
        linewidth = 1,
        fliersize = 4,
        gap = 0.25,
        flierprops = {"marker": "o", "markeredgecolor": "gray"},
        ax = ax
    )

    # Styling
    ax.legend(loc = 'upper left')
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(top = False, right = False) 
    
    plt.tight_layout()
    
    # Save boxplot to file
    fig.savefig(output_path, dpi = 500)
    plt.close(fig)

    # Compute effect size for Borregas Ave map
    borregas_doppeltest_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "DoppelTest")]["Metric"].to_list()
    borregas_evita_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "Evita")]["Metric"].to_list()
    if len(borregas_doppeltest_data) != 0 and len(borregas_evita_data) != 0:
        borregas_effect_size = compute_effect_size(borregas_doppeltest_data, borregas_evita_data)
    else:
        borregas_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

    return borregas_effect_size

def load_json_data(file_path: str):
    """
    Loads JSON data from a file and aggregates data by experiment ID
    """

    with open(file_path, "r") as f:
        data = json.load(f)
    
    try:
        aggregated_data = {
            experiment: sum(values for values in values if isinstance(values, int))
            for experiment, values in data.items()
        }
    except Exception as exc:
        raise exc
    
    return aggregated_data

def violations_plots(data_map: dict, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the violations metrics
    """

    # Initialize data structures
    all_data = []
    violations = ["stop_sign", "traffic_light", "speeding"]
    violations_effect_size_data = {}

    # Iterate over each violation type
    for i in range(3):
        # Iterate over each map and aggregated metric data paths lists
        for map, paths in data_map.items():
            # Extract aggregated metrics data paths
            evita_path, dt_path = paths
            
            # Load DoppelTest data
            with open(dt_path, "r") as f:
                doppeltest_data = json.load(f)
            
            # Iterate over the aggregated metrics values for DoppelTest
            for val in doppeltest_data.values():
                # Count violation occurrences in the scenario
                data = sum(val[violations[i]])
                all_data.append({"Map": map, "Violations": data, "Framework": "DoppelTest"})

            # Load Evita data
            with open(evita_path, "r") as f:
                evita_df = json.load(f)

            # Iterate over the aggregated metrics values for EVITA
            for val in evita_df.values():
                # Count violation occurrences in the scenario
                data = sum(val[violations[i]])
                all_data.append({"Map": map, "Violations": data, "Framework": "Evita"})

        # Create DataFrame with the collected data for the current violation type
        df = pd.DataFrame(all_data)

        # Initialize the figure and axis for the boxplot
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.set_style("ticks")
        
        # Define a plotting color for each framework
        palette = {"DoppelTest": "#9fcfff", "Evita": "#f19e6e"}
        
        # Plot data
        sns.boxplot(
            data = df, 
            x = "Map", 
            y = "Violations", 
            hue = "Framework", 
            palette = palette,
            linewidth = 1,
            fliersize = 4,
            gap = 0.25,
            flierprops = {"marker": "o", "markeredgecolor": "gray"},
            ax = ax
        )

        # Styling
        ax.legend(loc = 'upper left')
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(top = False, right = False) 
        
        plt.tight_layout()

        # Save boxplot to file
        fig.savefig(os.path.join(output_folder_path, f"scenario_{violations[i]}_violations_comparison.png"), dpi = 500)
        plt.close(fig)

        # Compute effect size for Borregas Ave map
        doppeltest_borregas_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "DoppelTest")]["Violations"].to_list()
        evita_borregas_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "Evita")]["Violations"].to_list()
        if len(doppeltest_borregas_data) != 0 and len(evita_borregas_data) != 0:
            borregas_effect_size = compute_effect_size(doppeltest_borregas_data, evita_borregas_data)
        else:
            borregas_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

        violations_effect_size_data[violations[i]] = {
            "borregas": borregas_effect_size,
        }

    return violations_effect_size_data

def generatation_statistics_plots(data_map: dict, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the generation statistics metrics
    """
    
    # Iterate over each generation statistic and corresponding data paths list
    for statistic, paths in data_map.items():
        # Initialize data structure
        all_data = []

        # Extract EVITA data path 
        evita_path = paths[0]
        
        # Load Evita data
        borregas_data = load_json_data(evita_path)

        # Iterate over the aggregated metrics values for EVITA
        for val in borregas_data:
            all_data.append({"Map": "Borregas Ave", f"{statistic}": val, "Framework": "Evita"})

        # Create DataFrame with the collected data for the current generation statistic
        df = pd.DataFrame(all_data)

        # Initialize the figure and axis for the boxplot
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.set_style("ticks")
        
        # Define a plotting color for Evita
        palette = {"Evita": "#f19e6e"}
        
        # Plot data
        sns.boxplot(
            data = df, 
            x = "Map", 
            y = f"{statistic}", 
            hue = "Framework", 
            palette = palette,
            linewidth = 1,
            fliersize = 4,
            gap = 0.25,
            flierprops = {"marker": "o", "markeredgecolor": "gray"},
            ax = ax
        )

        # Styling
        ax.legend(loc = 'upper left')
        ax.set_xlabel("")
        ax.set_ylabel("")
        sns.despine(top = False, right = False) 
        
        plt.tight_layout()
        
        # Save boxplot to file
        output_file_path = os.path.join(output_folder_path, f"{statistic}_comparison.png")
        fig.savefig(output_file_path, dpi = 500)
        
        plt.close(fig)

def agents_number_plots(data_map: dict, output_folder_path: str):
    """
    Generates the comparison plots between Evita and DoppelTest for the number of agents in the scenarios
    """

    # Iterate over each map and corresponding data paths list
    for map, paths in data_map.items():
        # Extract data paths
        evita_path, dt_path = paths
        
        # Load DoppelTest data
        with open(dt_path, "r") as f:
            data = json.load(f)

        # Aggregate data by experiment ID
        concatenated_list = []
        for list in data.values():
            concatenated_list.extend(list)
        
        # Aggregated data for DoppelTest
        doppeltest_data = pd.DataFrame(concatenated_list, columns = ["vehicles"])
        doppeltest_data["Framework"] = "DoppelTest"  
              
        # Load Evita data
        with open(evita_path, "r") as f:
            data = json.load(f)

        # Aggregate data by experiment ID
        concatenated_list = []
        for list in data.values():
            concatenated_list.extend(list)

        # Aggregated data for Evita
        evita_data = pd.DataFrame(concatenated_list, columns = ["vehicles"])
        evita_data["Framework"] = "Evita"

        # Combine dataframes
        combined_df = pd.concat([doppeltest_data, evita_data], ignore_index = True)
        combined_df = combined_df[combined_df["vehicles"] > 1]  # Filter out scenarios with only 1 vehicle for better visualization
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("ticks")
        
        # Define colors for each framework's data
        palette = {"DoppelTest": "#9fcfff", "Evita": "#f19e6e"}

        # Generate Stacked Histogram
        sns.histplot(
            data = combined_df,
            x = "vehicles",
            hue = "Framework",
            multiple = "stack",
            discrete = True,
            shrink = 0.7,      # Adjusts bar width for visual spacing
            palette = palette,
            alpha = 0.6,       # Transparency matching the visual style
            edgecolor = "white",
            ax = ax
        )
        
        low = 2
        high = int(combined_df["vehicles"].max())
        ax.set_xticks(range(low, high + 1))

        # Formatting
        ax.set_title(map, fontsize = 14, pad = 15)
        ax.set_xlabel("# Agents")
        ax.set_ylabel("# Scenarios")
        
        # Clean up legend and axes
        sns.despine()
        
        plt.tight_layout()

        # Save plot to file
        fig.savefig(os.path.join(output_folder_path, f"{map}_scenario_vehicles_comparison.png"), dpi = 500)
        plt.close(fig)

        #####

        # Initialize the figure and axis for the boxplot
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.set_style("ticks")

        # Define colors for each framework's data
        palette = {"DoppelTest": "#9fcfff", "Evita": "#f19e6e"}

        # Plot the boxplot
        sns.boxplot(
            data = combined_df,
            x = "Framework",
            y = "vehicles",
            hue = "Framework",
            palette = palette,
            linewidth = 1,
            fliersize = 4,
            gap = 0.25,
            flierprops = {"marker": "o", "markeredgecolor": "gray"},
            ax = ax
        )

        # Styling (matching your other function)
        ax.legend(loc='upper left')
        ax.set_xlabel("")
        ax.set_ylabel("# Agents")

        sns.despine(top=False, right=False)
        plt.tight_layout()

        # Save boxplot to file
        fig.savefig(os.path.join(output_folder_path, f"{map}_scenario_vehicles_boxplot.png"), dpi = 500)
        plt.close(fig)

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python generate_plots.py <evita_borregas_folder_path> <doppeltest_borregas_folder_path> <output_folder_path>")
        sys.exit(1)

    evita_borregas_folder_path = sys.argv[1]
    doppeltest_borregas_folder_path = sys.argv[2]
    output_folder_path = sys.argv[3]

    # Create the output folder
    os.makedirs(output_folder_path, exist_ok = True)

    generate_comparison_plots(evita_borregas_folder_path, doppeltest_borregas_folder_path, output_folder_path)