import json
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from evita.data_analysis import compute_effect_size


def generate_comparison_plots(evita_borregas_folder_path: str, evita_shalun_folder_path: str, doppeltest_borregas_folder_path: str, doppeltest_shalun_folder_path: str, output_folder_path: str):
    """
    
    """
    effect_size_data = {}

    # Total interactions
    borregas_effect_size, shalun_effect_size, all_effect_size = interactions_collisions_plots({
            "Borregas Ave": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_interactions.csv"), 
                os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_interactions.csv")
            ),
            "Shalun": (
                os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_interactions.csv"), 
                os.path.join(doppeltest_shalun_folder_path, "experiments_analysis", "total_interactions.csv")
            )
        },
        "interactions",
        os.path.join(output_folder_path, "interactions_comparison.png")
    )

    effect_size_data["interactions"] = {
        "borregas_ave": borregas_effect_size,
        "shalun": shalun_effect_size,
        "all": all_effect_size
    }

    print("Generated interactions comparison plot")

    # Unique interactions
    borregas_effect_size, shalun_effect_size, all_effect_size = interactions_collisions_plots({
        "Borregas Ave": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_unique_interactions.csv"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_unique_interactions.csv")
        ),
        "Shalun": (
            os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_unique_interactions.csv"), 
            os.path.join(doppeltest_shalun_folder_path, "experiments_analysis", "total_unique_interactions.csv")
        )
    },
    "unique_interactions",
    os.path.join(output_folder_path, "unique_interactions_comparison.png")
    )

    effect_size_data["unique_interactions"] = {
        "borregas_ave": borregas_effect_size,
        "shalun": shalun_effect_size,
        "all": all_effect_size
    }

    print("Generated unique interactions comparison plot")

    # Total collisions
    borregas_effect_size, shalun_effect_size, all_effect_size = interactions_collisions_plots({
            "Borregas Ave": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_collisions.csv"), 
                os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_collisions.csv")
            ),
            "Shalun": (
                os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_collisions.csv"), 
                os.path.join(doppeltest_shalun_folder_path, "experiments_analysis", "total_collisions.csv")
            )
        },
        "collisions",
        os.path.join(output_folder_path, "collisions_comparison.png")
    )

    effect_size_data["collisions"] = {
        "borregas_ave": borregas_effect_size,
        "shalun": shalun_effect_size,
        "all": all_effect_size
    }

    print("Generated collisions comparison plot")

    # Unique collisions
    borregas_effect_size, shalun_effect_size, all_effect_size = interactions_collisions_plots({
        "Borregas Ave": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_unique_collisions.csv"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "total_unique_collisions.csv")
        ),
        "Shalun": (
            os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_unique_collisions.csv"), 
            os.path.join(doppeltest_shalun_folder_path, "experiments_analysis", "total_unique_collisions.csv")
        )
    },
    "unique_collisions",
    os.path.join(output_folder_path, "unique_collisions_comparison.png")
    )

    effect_size_data["unique_collisions"] = {
        "borregas_ave": borregas_effect_size,
        "shalun": shalun_effect_size,
        "all": all_effect_size
    }

    print("Generated unique collisions comparison plot")

    # Total violations
    violations_effect_size_data = violations_plots({
        "Borregas Ave": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "scenario_violations.csv"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "scenario_violations.csv")
        ),
        "Shalun": (
            os.path.join(evita_shalun_folder_path, "experiments_analysis", "scenario_violations.csv"), 
            os.path.join(doppeltest_shalun_folder_path, "experiments_analysis", "scenario_violations.csv")
        )
    },
    output_folder_path
    )

    effect_size_data["violations"] = violations_effect_size_data

    print("Generated violations comparison plots")

    # Generation statistics
    generatation_statistics_plots({
            "generated": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_generated_scenarios.csv"), 
                os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_generated_scenarios.csv")
            ),
            "valid": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_valid_scenarios.csv"), 
                os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_valid_scenarios.csv")
            ),
            "invalid": (
                os.path.join(evita_borregas_folder_path, "experiments_analysis", "total_invalid_scenarios.csv"), 
                os.path.join(evita_shalun_folder_path, "experiments_analysis", "total_invalid_scenarios.csv")
            )
        },
        output_folder_path
    )

    print("Generated generation statistics comparison plots")

    # Total agents
    agents_number_plots({
        "borregas": (
            os.path.join(evita_borregas_folder_path, "experiments_analysis", "vehicles_number.csv"), 
            os.path.join(doppeltest_borregas_folder_path, "experiments_analysis", "vehicles_number.csv")
        ),
        "shalun": (
            os.path.join(evita_shalun_folder_path, "experiments_analysis", "vehicles_number.csv"), 
            os.path.join(doppeltest_shalun_folder_path, "experiments_analysis", "vehicles_number.csv")
        )
    },
    output_folder_path
    )

    print("Generated agents number comparison plots")

    effect_size_data_file_path = os.path.join(output_folder_path, "effect_size_data.json")
    with open(effect_size_data_file_path, "w") as f:
        json.dump(effect_size_data, f, indent = 4)

def interactions_collisions_plots(data_map: dict, column_name: str, output_path: str):
    all_data = []

    for map, paths in data_map.items():
        evita_path, dt_path = paths
        
        # Load DoppelTest data
        doppeltest_df = pd.read_csv(dt_path)
        for val in doppeltest_df[column_name]:
            all_data.append({"Map": map, "Metric": val, "Framework": "DoppelTest"})
            
        # Load Evita data
        evita_df = pd.read_csv(evita_path)
        for val in evita_df[column_name]:
            all_data.append({"Map": map, "Metric": val, "Framework": "Evita"})

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Calculate the 'all' category (combined data)
    all_category_df = df.copy()
    all_category_df["Map"] = "All"
    df = pd.concat([df, all_category_df], ignore_index = True)

    # Plotting
    fig, ax = plt.subplots(figsize = (10, 5))
    sns.set_style("ticks")
    
    palette = {"DoppelTest": "#9fcfff", "Evita": "#f19e6e"}
    
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
    
    fig.savefig(output_path, dpi = 500)
    plt.close(fig)

    borregas_doppeltest_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "DoppelTest")]["Metric"].to_list()
    borregas_evita_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "Evita")]["Metric"].to_list()
    if len(borregas_doppeltest_data) != 0 and len(borregas_evita_data) != 0:
        borregas_effect_size = compute_effect_size(borregas_doppeltest_data, borregas_evita_data)
    else:
        borregas_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

    shalun_doppeltest_data = df[(df["Map"] == "Shalun") & (df["Framework"] == "DoppelTest")]["Metric"].to_list()
    shalun_evita_data = df[(df["Map"] == "Shalun") & (df["Framework"] == "Evita")]["Metric"].to_list()
    if len(shalun_doppeltest_data) != 0 and len(shalun_evita_data) != 0:
        shalun_effect_size = compute_effect_size(shalun_doppeltest_data, shalun_evita_data)
    else:
        shalun_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

    all_doppeltest_data = df[(df["Map"] == "All") & (df["Framework"] == "DoppelTest")]["Metric"].to_list()
    all_evita_data = df[(df["Map"] == "All") & (df["Framework"] == "Evita")]["Metric"].to_list()
    if len(all_doppeltest_data) != 0 and len(all_evita_data) != 0:
        all_effect_size = compute_effect_size(all_doppeltest_data, all_evita_data)
    else:
        all_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

    return borregas_effect_size, shalun_effect_size, all_effect_size

def violations_plots(data_map: dict, output_folder_path: str):
    all_data = []
    violations = ["stop_sign", "traffic_light", "speeding"]
    violations_effect_size_data = {}

    for i in range(3):
        for map, paths in data_map.items():
            evita_path, dt_path = paths
            
            # Load DoppelTest data
            doppeltest_df = pd.read_csv(dt_path)
            for val in doppeltest_df[violations[i]]:
                all_data.append({"Map": map, "Violations": val, "Framework": "DoppelTest"})
                
            # Load Evita data
            evita_df = pd.read_csv(evita_path)
            for val in evita_df[violations[i]]:
                all_data.append({"Map": map, "Violations": val, "Framework": "Evita"})

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Calculate the 'all' category (combined data)
        all_category_df = df.copy()
        all_category_df["Map"] = "All"
        df = pd.concat([df, all_category_df], ignore_index = True)

        # Plotting
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.set_style("ticks")
        
        palette = {"DoppelTest": "#9fcfff", "Evita": "#f19e6e"}
        
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
        fig.savefig(os.path.join(output_folder_path, f"scenario_{violations[i]}_violations_comparison.png"), dpi = 500)
        plt.close(fig)

        doppeltest_borregas_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "DoppelTest")]["Violations"].to_list()
        evita_borregas_data = df[(df["Map"] == "Borregas Ave") & (df["Framework"] == "Evita")]["Violations"].to_list()
        if len(doppeltest_borregas_data) != 0 and len(evita_borregas_data) != 0:
            borregas_effect_size = compute_effect_size(doppeltest_borregas_data, evita_borregas_data)
        else:
            borregas_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

        doppeltest_shalun_data = df[(df["Map"] == "Shalun") & (df["Framework"] == "DoppelTest")]["Violations"].to_list()
        evita_shalun_data = df[(df["Map"] == "Shalun") & (df["Framework"] == "Evita")]["Violations"].to_list()
        if len(doppeltest_shalun_data) != 0 and len(evita_shalun_data) != 0:
            shalun_effect_size = compute_effect_size(doppeltest_shalun_data, evita_shalun_data)
        else:
            shalun_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

        doppeltest_all_data = df[(df["Map"] == "All") & (df["Framework"] == "DoppelTest")]["Violations"].to_list()
        evita_all_data = df[(df["Map"] == "All") & (df["Framework"] == "Evita")]["Violations"].to_list()
        if len(doppeltest_all_data) != 0 and len(evita_all_data) != 0:
            all_effect_size = compute_effect_size(doppeltest_all_data, evita_all_data)
        else:
            all_effect_size = -1  # Indicating that the effect size cannot be computed due to lack of data

        violations_effect_size_data[violations[i]] = {
            "borregas": borregas_effect_size,
            "shalun": shalun_effect_size,
            "all": all_effect_size
        }

        return violations_effect_size_data

def generatation_statistics_plots(data_map: dict, output_folder_path: str):

    for statistic, paths in data_map.items():
        all_data = []

        borregas_path, shalun_path = paths
        
        # Load DoppelTest data
        shalun_df = pd.read_csv(shalun_path)
        for val in shalun_df[statistic]:
            all_data.append({"Map": "Shalun", f"{statistic}": val, "Framework": "Evita"})
            
        # Load Evita data
        borregas_df = pd.read_csv(borregas_path)
        for val in borregas_df[statistic]:
            all_data.append({"Map": "Borregas Ave", f"{statistic}": val, "Framework": "Evita"})

        # Create DataFrame
        df = pd.DataFrame(all_data)

        # Plotting
        fig, ax = plt.subplots(figsize = (10, 5))
        sns.set_style("ticks")
        
        palette = {"Evita": "#f19e6e"}
        
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
        
        output_file_path = os.path.join(output_folder_path, f"{statistic}_comparison.png")
        fig.savefig(output_file_path, dpi = 500)
        plt.close(fig)

def agents_number_plots(data_map: dict, output_folder_path: str):
    for map, paths in data_map.items():
        evita_path, dt_path = paths
        
        # Load DoppelTest data
        doppeltest_df = pd.read_csv(dt_path)
        doppeltest_df["Framework"] = "DoppelTest"        
        # Load Evita data
        evita_df = pd.read_csv(evita_path)
        evita_df["Framework"] = "Evita"

        # Combine dataframes
        combined_df = pd.concat([doppeltest_df, evita_df], ignore_index = True)
        combined_df = combined_df[combined_df["vehicles"] > 1]  # Filter out scenarios with only 1 vehicle for better visualization
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style("ticks")
        
        # Custom palette matching the screenshot (muted/pastel blue and orange)
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

        fig.savefig(os.path.join(output_folder_path, f"{map}_scenario_vehicles_comparison.png"), dpi = 500)
        plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python generate_plots.py <evita_borregas_folder_path> <evita_shalun_folder_path> <doppeltest_borregas_folder_path> <doppeltest_shalun_folder_path> <output_folder_path>")
        sys.exit(1)

    evita_borregas_folder_path = sys.argv[1]
    evita_shalun_folder_path = sys.argv[2]
    doppeltest_borregas_folder_path = sys.argv[3]
    doppeltest_shalun_folder_path = sys.argv[4]
    output_folder_path = sys.argv[5]

    os.makedirs(output_folder_path, exist_ok = True)

    generate_comparison_plots(evita_borregas_folder_path, evita_shalun_folder_path, doppeltest_borregas_folder_path, doppeltest_shalun_folder_path, output_folder_path)