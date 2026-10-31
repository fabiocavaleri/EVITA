import logging
import os
import csv
import sqlite3
import ast
import click
import math
import json
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from venny4py.venny4py import *

from typing import Tuple, List 
from bisect import bisect_left
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import CustomState
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.scenario import Scenario
from typing import Dict
from .config import ScenarioGeneratorConfig
from .generators.utils import compute_difference_in_cc_space
from .utils import EVITA_MAIN_LOGGER_NAME, simulation_folders, collisions_folders, find_csv_files, folders_of_completed_runs, RESULTS_FOLDER

TIMEOUT = 200

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

# Mapping of the generator full name to a more readable name for the plots and tables
GENERATORS = dict()
GENERATORS["evita.generators.basic_generator.EAOnePlusOneGenerator_MaxInteractions"] = "EA1+1 (Interactions)"
GENERATORS["evita.generators.basic_generator.EAOnePlusOneGenerator_MaxDanger"] = "EA1+1 (Danger)"
GENERATORS["evita.generators.doppeltest_generator.DoppelTestScenarioGenerator"] = "DoppelTest"
GENERATORS["evita.generators.random_generator.RandomScenarioGenerator"] = "Random"
GENERATORS["evita.generators.nsgaii.NSGAIIproblem"] = "Evita"

# Mapping of the base scenario name to a more readable name for the plots and tables
BASE_SCENARIOS = dict()
BASE_SCENARIOS['CHN_Cho-2_2_I-1-1'] = "Cho"
BASE_SCENARIOS['DEU_Cologne-79_2_I-1'] = "Cologne"
BASE_SCENARIOS['USA_US101-15_2_T-1'] = "US101 15"
BASE_SCENARIOS['USA_US101-23_1_T-1'] = "US101 23"
BASE_SCENARIOS['USA_US101-9_1_T-1'] = "US101 09"

def remap_generator_name(fullname:str) -> str:
    return GENERATORS[fullname] if fullname in GENERATORS else fullname

# Calculate Pearson's correlation
def compute_correlation(d1, d2):
    corr, _ = ss.pearsonr(d1, d2)
    return corr

def compute_p_value(treatment, control) -> float:
    _, p_value = ss.mannwhitneyu(treatment, control)
    return p_value

def compute_effect_size(treatment, control) -> float:
    # https://gist.github.com/jacksonpradolima/f9b19d65b7f16603c837024d5f8c8a65
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude and power and number of runs to have power>0.8
    """
    m = len(treatment)
    n = len(control)

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude#, power, nruns

def compute_metrics_for(df: pd.DataFrame, metrics: List[str]):
    """
    Produces a DataFrame containing p-value and A12 for all scenarios and each scenario comparing the generators
    """
    column_names = ["base_scenario", "metric_name", "generator_1", "generator_2", "p_value", "a_12", "n_obs_generator_1", "n_obs_generator_2"]
    results_df = None

    # Extract the generators and the base scenarios to compare
    generators = sorted(df["generator"].unique().tolist())
    base_scenarios = sorted(df["base_scenario"].unique().tolist())

    # Iterate over all the metrics, scenarios and pairs of generators to compare them
    for metric in metrics:
        for generator_1 in generators:
            for generator_2 in generators:
                for base_scenario in base_scenarios:
                    # Retrieve the values for the metric for the two generators in the given scenario
                    bs_values_1 = df[(df["generator"] == generator_1) & (df["base_scenario"] == base_scenario) ][metric].tolist()
                    bs_values_2 = df[(df["generator"] == generator_2) & (df["base_scenario"] == base_scenario) ][metric].tolist()

                    # Extract the number of observations for each generator in the given scenario
                    n_obs_generator_1 = len(bs_values_1)
                    n_obs_generator_2 = len(bs_values_2)

                    # Compare them ONLY if there's anything to compare
                    if len(bs_values_1) == 0 or len(bs_values_2) == 0:
                        module_logger.info(f"Missing values for scenario {base_scenario}, cannot compare {generator_1} - {len(bs_values_1)} and {generator_2} - {len(bs_values_2)}")
                        continue

                    # Compute the p-value and A12 effect size for the comparison of the two generators in the given scenario
                    bs_p_value = compute_p_value(bs_values_1, bs_values_2)
                    bs_a_12 = compute_effect_size(bs_values_1, bs_values_2)

                    # Store the results in the DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame([[base_scenario, metric, generator_1, generator_2, bs_p_value, bs_a_12, n_obs_generator_1, n_obs_generator_2]], columns = column_names)], ignore_index = True)

    return results_df

def store_to_html(output_file_html, df, metrics_and_plots: Dict[str, List[str]]):
    """
    Store the results of the analysis in an HTML file containing a table for each metric and the corresponding plots
    """
    output_folder = os.path.dirname(output_file_html)

    with open(output_file_html, "w", newline = "") as htmlfile:
        htmlfile.write("<HTML><BODY>")
    
        # Generate on table for each metric
        for metric in metrics_and_plots.keys():
            htmlfile.write(f"<H1> Metric: {metric} </H1>")
            
            # Show any plot
            for plot_file in metrics_and_plots[metric]:
                # Make the link relative to the output_folder
                htmlfile.write(f'<IMG src="{os.path.relpath(plot_file, output_folder)}">')
            
            # Select only the rows for the metric
            metric_df = df[df["metric_name"] == metric]
            
            # Replace the generator names
            metric_df["generator_1"] = metric_df["generator_1"].replace(GENERATORS)
            metric_df["generator_2"] = metric_df["generator_2"].replace(GENERATORS)
            metric_df["base_scenario"] = metric_df["base_scenario"].replace(BASE_SCENARIOS)

            sorted_df = metric_df.sort_values(['base_scenario', 'generator_1', 'generator_2'], ascending = [True, True, True], inplace = False)

            # Remove rows for the same generator
            sorted_df = sorted_df[sorted_df['generator_1'] != sorted_df['generator_2']]
            sorted_df = sorted_df.drop(columns = ["metric_name"])

            # Format the output
            # Highlight 
            EXPECTED_VALUES = 10
            def apply_formatting(col):
                if "n_obs" in col.name:
                    return ['background-color: red' if c < EXPECTED_VALUES / 3 else 'background-color: yellow' if c < EXPECTED_VALUES / 2 else 'background-color: light-green' for c in col.values]
                else:
                    return ['' for _ in col.values]

            html_content = sorted_df.style.apply(apply_formatting).to_html()

            htmlfile.write(html_content)
        
        htmlfile.write("</BODY></HTML>")

def venn_diagram(file_name, sets: dict):
    """
    Plot a venn diagram for the given sets and store it in the given file name
    """
    import venn

    sorted_names = list()
    sorted_values = list()
    for key in sorted(sets.keys()):
        sorted_names.append(remap_generator_name(key))
        sorted_values.append(sets[key])    

    labels = venn.generate_petal_labels(sorted_values)

    fig = None

    if len(sorted_names) == 5:
        fig, _ = venn.venn5(labels, names = sorted_names)
    if len(sorted_names) == 4:
        fig, _ = venn.venn4(labels, names = sorted_names)
    if len(sorted_names) == 3:
        fig, _ = venn.venn3(labels, names = sorted_names)
    if len(sorted_names) == 2:
        fig, _ = venn.venn2(labels, names = sorted_names)
    if len(sorted_names) < 2:
        print("Cannot plot a single venn diagram!")
        return
    
    fig.savefig(file_name, bbox_inches = 'tight', facecolor = 'w')

    plt.close(fig)

def accumulate_results_in_set(the_generator, the_base_scenario, the_sets, the_map):
    """
    Given a map of collisions or interactions, accumulate the results in the corresponding set for the given generator and base scenario
    """
    
    if "all" not in the_sets:
        the_sets["all"] = dict()

    if the_base_scenario not in the_sets:
        the_sets[the_base_scenario] = dict()

    if the_generator not in the_sets["all"]:
        the_sets["all"][the_generator] = set()

    if the_generator not in the_sets[the_base_scenario]:
        the_sets[the_base_scenario][the_generator] = set()

    # Those are the result from this run
    per_run_set = set(list(zip(*[ a.tolist() for a in np.nonzero(the_map) ])))

    # Accumulate them into the all scenarios
    the_sets["all"][the_generator] = the_sets["all"][the_generator].union(per_run_set)    
    
    # Accumulate them into the base scenarios
    the_sets[the_base_scenario][the_generator] = the_sets[the_base_scenario][the_generator].union(per_run_set)

def plot_as_grouped_bloxplot(df, metric_header: str, metric_label: str):
    """
    Plot the given metric as a box plot grouped by the base scenario and colored by the generator
    """
    sns.set_theme(style = "ticks", palette = "pastel")
    
    fig, ax = plt.subplots(figsize = (8, 4))
    ax = sns.boxplot(
        x = "base_scenario", 
        y = metric_header,
        # Refine
        hue = "generator",
        data = df)
    
    ax.set_xlabel("Base Scenario")
    ax.set_ylabel(metric_label)
    plt.legend(loc = 'upper left')
    # Make sure we do not cut labels
    plt.tight_layout()

    return fig
    
def plot_metrics_for(results_folder: str, plotting_df: pd.DataFrame, metrics: List[str]) -> List[str]:
    """
    Plot the given metrics as box plots grouped by the base scenario and colored by the generator and store them in the given results folder
    """
    output_files = []

    for metric_name in metrics:
        # Plot the results achieved by each generator by grouping them on base_scenarios
        fig = plot_as_grouped_bloxplot(plotting_df, metric_name, metric_name)
        
        output_file = os.path.join(results_folder, f'{metric_name}.png') 
        plt.savefig(output_file)
        
        plt.close(fig)
        
        output_files.append(output_file)

    return output_files

def _load_data_from_results_folders(results_folders: List):
    """
    Go over all the experimental run folders contained in each of the provided results_folders
    """
    
    # Initialize the dataframes and sets to accumulate the results
    requirements_violations_df, performace_df, behavior_df = None, None, None
    interaction_sets = dict()

    collision_only_impact_location_sets = dict()
    collision_impact_location_and_riv_sets = dict()
    collision_impact_location_riv_and_ria_sets = dict()

    # Iterate over the results folders and collect the data from each of the runs contained in them
    for results_folder in results_folders:
        # Return only the folders that contain valid data
        for run_folder, _ in folders_of_completed_runs(results_folder):
            # Save the metadata to file
            metadata_file = os.path.join(run_folder, "metadata.csv")
            with open(metadata_file, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
                data = [row for row in csv_reader][0]
                base_scenario = data["scenario_name"]
                # Find out which generator - By name + module
                generator = data["generator"]
                # Ensure runs are distinguishable
                timestamp_id = data["timestamp_id"]

            # Open requirements violations file and save the data in the dataframe
            requirements_violations_file = os.path.join(run_folder, "requirements_violations.csv")
            r_df = pd.read_csv(requirements_violations_file)
            r_df['generator'] = generator
            r_df['base_scenario'] = base_scenario
            r_df['timestamp_id'] = timestamp_id

            if requirements_violations_df is None:
                requirements_violations_df = r_df
            else:
                requirements_violations_df = pd.concat([requirements_violations_df, r_df], ignore_index = True)

            # Open behavior file and save the data in the dataframe
            behavior_file = os.path.join(run_folder, "behavior.csv")
            b_df = pd.read_csv(behavior_file)
            b_df['generator'] = generator
            b_df['base_scenario'] = base_scenario
            b_df['timestamp_id'] = timestamp_id

            if behavior_df is None:
                behavior_df = b_df
            else:
                behavior_df = pd.concat([behavior_df, b_df], ignore_index = True)
            
            # Open performance file and save the data in the dataframe
            performance_file = os.path.join(run_folder, "performance.csv")
            p_df = pd.read_csv(performance_file)
            p_df['generator'] = generator
            p_df['base_scenario'] = base_scenario
            p_df['timestamp_id'] = timestamp_id

            if performace_df is None:
                performace_df = p_df
            else:
                performace_df = pd.concat([performace_df, p_df], ignore_index = True)

            # Load the collisions maps
            # Read the collision and interaction maps, transform them into sets for the venn diagram visualization
            collision_map_file = os.path.join(run_folder, "collision_map.npy")
            collision_map = np.load(collision_map_file)

            accumulate_results_in_set(generator, base_scenario, collision_impact_location_riv_and_ria_sets, collision_map)

            collision_map_only_impact_location_file = os.path.join(run_folder, "collision_map_only_impact_location.npy")
            collision_map_only_impact_location = np.load(collision_map_only_impact_location_file)
            accumulate_results_in_set(generator, base_scenario, collision_only_impact_location_sets, collision_map_only_impact_location)

            collision_map_impact_location_and_riv_file = os.path.join(run_folder, "collision_map_impact_location_and_riv.npy")
            collision_map_impact_location_and_riv = np.load(collision_map_impact_location_and_riv_file)
            accumulate_results_in_set(generator, base_scenario, collision_impact_location_and_riv_sets, collision_map_impact_location_and_riv)

            interaction_map_file = os.path.join(run_folder, "interaction_map.npy")
            interaction_map = np.load(interaction_map_file)
            accumulate_results_in_set(generator, base_scenario, interaction_sets, interaction_map)

    return requirements_violations_df, performace_df, behavior_df, interaction_sets, collision_only_impact_location_sets, collision_impact_location_and_riv_sets, collision_impact_location_riv_and_ria_sets

def plot_venn_diagrams(the_results_folder: str, the_sets: dict, the_name: str):
    """
     Plot the venn diagrams for the given sets and store them in the given results folder with the given name
    """
    for the_base_scenario, the_result_sets in the_sets.items():
        venn_diagram_file = os.path.join(the_results_folder, f"{the_name}-{the_base_scenario}.png")
        venn_diagram(venn_diagram_file, the_result_sets)

def get_collisions_details_from_db(simulation_db) -> pd.DataFrame:
    """
    Get the details of the collisions from the db file produced by the simulation
    For each colliding agent we get the time of the collision, its RIV, and the impact location"""
    # Open connection
    connection = sqlite3.connect(simulation_db)
    try:
        return pd.read_sql_query("SELECT MIN(timestep) as collision_time, agent_id, riv, impact_location FROM scenario_evaluation WHERE has_collided=1.0 GROUP BY agent_id;", connection)
    finally:
        try:
            connection.close()
        except:
            pass

def get_scenario_details_from_db(simulation_db) -> pd.DataFrame:
    """
    Get the details of the scenario from the db file produced by the simulation.
    For each scenario we get the number of agents in the scenario, the time of the first collision, and the time of the last collision
    """
    # Open connection
    connection = sqlite3.connect(simulation_db)
    try:
        return pd.read_sql_query("SELECT COUNT(DISTINCT agent_id) AS n_agents FROM scenario_evaluation;", connection)
    finally:
        try:
            connection.close()
        except:
            pass

def plot_collisions_on_map(scenario: Scenario, df: pd.DataFrame):
    """
    Given a scenario, plot the collisions of whatever generators and base_scenarios are listed in the dataframe
    """
    # Extract the generators and the base scenarios to compare
    generators = df["generator"].unique().tolist()
    base_scenarios = df["base_scenario"].unique().tolist()

    # Initialize plot image
    fig, ax = plt.subplots(figsize = (8, 4))

    rnd = MPRenderer(ax = ax)
    scenario.draw(rnd)
    rnd.render()

    the_mins = None
    the_maxs = None

    # Iterate over the base scenarios and generators
    for base_scenario in base_scenarios:
        # We'd like to keep the markers, colors connected to the generators
        for generator, color, marker in zip(generators, ["red", "blue", "green"], [".", "*", "s"]):
            # Get the filtered dataframe
            filtered_df = df[(df["base_scenario"] == base_scenario) & (df["generator"] == generator)]
            
            # Some combinations might not be there, skip them
            if filtered_df.empty:
                continue

            # Extract the x and y positions of the collisions for that generator and base scenario
            x_y_positions_series = filtered_df[["x_position", "y_position"]]
            mins = x_y_positions_series.min().tolist()
            maxs = x_y_positions_series.max().tolist()

            # Set the limits of the figure to include all the collision points
            if the_mins is None:
                the_mins = mins
            if the_maxs is None:
                the_maxs = maxs

            # Check that the mins and maxs are not NaN
            assert not np.isnan(the_mins[0]), f"the_mins[0] none for {base_scenario} and {generator}"
            assert not np.isnan(the_mins[1]), f"the_mins[1] none for {base_scenario} and {generator}"
            assert not np.isnan(the_maxs[0]), f"the_maxs[0] none for {base_scenario} and {generator}"
            assert not np.isnan(the_maxs[1]), f"the_maxs[1] none for {base_scenario} and {generator}"

            # Extract mins and maxs values
            the_mins = min(the_mins[0], mins[0]), min(the_mins[1], mins[1])
            the_maxs = max(the_maxs[0], maxs[0]), max(the_maxs[1], maxs[1])

            serie_x, serie_y = [], []
            for x, y in x_y_positions_series.itertuples(index = False):
                serie_x.append(x)
                serie_y.append(y)

            # Plot the collisions for the generator and the base scenario
            plt.scatter(serie_x, serie_y, marker = marker, color = color, zorder = 50, alpha = 0.1, label = generator)

        # Set the limits of the figure to include the collision points and not the entire map
        plt.gca().set_xlim(the_mins[0] - 20, the_maxs[0] + 20)
        plt.gca().set_ylim(the_mins[1] - 20, the_maxs[1] + 20)

        plt.legend().set_zorder(100)
        plt.axis('off')
        plt.gca().set_aspect("equal")

    return fig

def collisions_analysis(requirements_violations_df: pd.DataFrame, collision_sets: dict, output_folder:str, indepth_collisions_df: pd.DataFrame = None):
    """
    Analyze the collisions found by the generators and produce an HTML report with the results of the analysis and the corresponding plots
    """
    # Define the output folder for the collisions analysis results
    collisions_analysis_output_folder = os.path.join(output_folder, "collisions_analysis")
    os.makedirs(collisions_analysis_output_folder, exist_ok=True)

    # Duplicate the metrics for the scenarios and tag them as all scenario
    all_scenarios_df = requirements_violations_df.copy()
    # Replace the base scenarios with the generic "all" label
    all_scenarios_df = all_scenarios_df.assign(base_scenario = 'all')
    # Concatenate the two dataframes
    requirements_violations_df = pd.concat([requirements_violations_df, all_scenarios_df], ignore_index = True)

    # Replace the generator and base scenario names with the more readable ones for the plots and tables
    requirements_violations_df = requirements_violations_df.replace(GENERATORS).replace(BASE_SCENARIOS)

    # Iterate over collision metrics
    for metric in ["total_collisions", "unique_collisions"]:
        # Select the data and compute the p - A12 for each pair of generators in each base scenario and by aggregating the results across all of the scenarios
        df_metrics = compute_metrics_for(requirements_violations_df, [metric])
        # Select the data and plot a box plot for each scenario and by aggregating the metrics across all the scenarios
        plot_files = plot_metrics_for(collisions_analysis_output_folder, requirements_violations_df, [metric])
        
        metrics_and_plots = dict()
        metrics_and_plots[metric] = plot_files
        # Create an HTML report
        report_file_html = os.path.join(collisions_analysis_output_folder, f"{metric}.html")
        store_to_html(report_file_html, df_metrics, {metric: plot_files})

    # Which generator finds collision types that other generators did not find?
    plot_venn_diagrams(collisions_analysis_output_folder, collision_sets, "venn-diagram-collisions")
    
    # ========================================
    # Detailed collision analysis:
    # - How critical are the found collisions?
    #   - Harm (distribution) - How can we compare those if the size of the distribution is different?
    #   - Riv 
    #   - pdof
    # - What's the time to observe collisions?
    # - Which collisions are found?
    # - Which collisions each generator finds?
    # ========================================
    
    if indepth_collisions_df is not None:
        # Replace the generator and base scenario names with the more readable ones for the plots and tables
        collisions_df = indepth_collisions_df.copy()
        collisions_df["generator"] = collisions_df["generator"].replace(GENERATORS)
        collisions_df["base_scenario"] = collisions_df["base_scenario"].replace(BASE_SCENARIOS)

        # Duplicate the metrics for the scenarios and tag them as all scenario
        collisions_all_scenarios_df = collisions_df.copy()
        # Replace the base scenarios with the generic "all" label
        collisions_all_scenarios_df = collisions_all_scenarios_df.assign(base_scenario = 'all')
        # Concatenate the two dataframes
        collisions_df = pd.concat([collisions_df, collisions_all_scenarios_df], ignore_index = True)

        # Iterate over collision criticality metrics and plot them for each scenario
        for metric in ["ego_harm", "time_step", "pdof", "riv", "collision_time"]:
            plot_metrics_for(collisions_analysis_output_folder, collisions_df, [metric])
        
        ######## 

        # - Where did the collision happened?
        # Plot the position of the vehicles when they crashed for each generator (color) in each scenario (map)
        base_scenarios = [ s for s in collisions_df["base_scenario"].unique().tolist() if str(s) != "all"]
        generators = collisions_df["generator"].unique().tolist()

        # The scenario is always the same so we load the map only once (the first we get matching the conditions)
        # But we plot it separatedly for each generator and once for all together
        for base_scenario in base_scenarios:
            from commonroad.scenario.scenario import Scenario
            from commonroad.common.file_reader import CommonRoadFileReader

            # Extract the scenario file for the current base scenario
            scenario_file = str(collisions_df[collisions_df["base_scenario"] == base_scenario]["scenario_file"].values[0])

            # Open scenario file and load the scenario
            crfr = CommonRoadFileReader(scenario_file)
            _scenario: Scenario
            _scenario, _ = crfr.open()

            # Iterate over generators and plot the collisions for each of them
            for generator in generators:
                per_generator_df = collisions_df[ (collisions_df["base_scenario"] == base_scenario) & (collisions_df["generator"] == generator) ]
        
                collisions_positions_per_generator_file = os.path.join(collisions_analysis_output_folder, f"collisions_positions_{base_scenario}_{generator}.png")

                fig = plot_collisions_on_map(_scenario, per_generator_df)
                fig.tight_layout()
                plt.figure(fig)
                plt.savefig(collisions_positions_per_generator_file)
                plt.close(fig)
            
            # Plot all the collisions together for the current base scenario
            per_scenario_df = collisions_df[collisions_df["base_scenario"] == base_scenario]

            # Generate one figure with all the generators together
            collisions_positions_file = os.path.join(collisions_analysis_output_folder, f"collisions_positions_{base_scenario}_all.png")

            fig = plot_collisions_on_map(_scenario, per_scenario_df)
            fig.tight_layout()
            plt.figure(fig)
            plt.savefig(collisions_positions_file)
            plt.close(fig)

            ############################

            # What is the distribution of collisions (per scenario)?
            per_scenario_collision_df = collisions_df[ collisions_df["base_scenario"] == base_scenario ]

            distribution_of_impact_location_file = os.path.join(collisions_analysis_output_folder, f"impact_locations_{base_scenario}.png")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax = sns.histplot(data=per_scenario_collision_df, x="impact_location", hue="generator")
            ax.set_xlabel("Impact Location")
            ax.set_ylabel("Collision Count")
            # Make sure we do not cut labels
            plt.tight_layout()
            plt.savefig(distribution_of_impact_location_file)
            plt.close(fig)

        ########

        # What is the distribution of collisions (for all the scenarios)?
        all_collisions_df = collisions_df[ collisions_df["base_scenario"] == "all" ]
        distribution_of_impact_location_file = os.path.join(collisions_analysis_output_folder, f"impact_locations_all.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax = sns.histplot(data=all_collisions_df, x="impact_location", hue="generator")
        ax.set_xlabel("Impact Location")
        ax.set_ylabel("Collision Count")
        # Make sure we do not cut labels
        plt.tight_layout()
        plt.savefig(distribution_of_impact_location_file)
        plt.close(fig)
  
def extend_with_collision_criticality(collision_df_from_db, simulation_folder: str) -> pd.DataFrame:
    """
    For each colliding agent in a scenario collected from the DB we need to extend the DF with data from the corresponding CSV files
    """
    collisions_criticality_df = None

    # Iterate over collision folders
    for collisions_folder, _ in collisions_folders(simulation_folder):
        module_logger.debug(f"Collecting data from collisions folder {collisions_folder}")
        
        # Collects the data from each agent from its CSV collision file
        for csv_file, csv_file_name in find_csv_files(collisions_folder):
            agent_id = int(csv_file_name.replace(".csv", ""))
            collision_criticality_df = pd.read_csv(csv_file)
            # Extract collision criticality data and add the agent id to be able to merge it with the data from the DB
            collision_criticality_df['agent_id'] = agent_id
            if collisions_criticality_df is None:
                collisions_criticality_df = collision_criticality_df
            else:
                collisions_criticality_df = pd.concat([collisions_criticality_df, collision_criticality_df], ignore_index = True)

    if collisions_criticality_df is not None:
        return collision_df_from_db.merge(collisions_criticality_df, on = 'agent_id')
    else:
        return collision_df_from_db

def extend_with_agent_position_at_collision(collision_df_from_db, logs_folder) -> pd.DataFrame:
    """
    For each of the colliding agents find their position at the time they collided
    NOTE: will fail if not invoked on a DB with collisions data!
    """
    data = []
    # Iterate over colliding agents
    for agent_id in collision_df_from_db["agent_id"].unique().tolist():
        # Note the -1. The reason is that the agent crashes between time_step_of_collision and (time_step_of_collision -1)
        #  so at time time_step_of_collision it does not have any trajectory recorded (altough it should be the dummy one made of its position only...)
        time_step_of_collision = int(collision_df_from_db[ collision_df_from_db["agent_id"] == agent_id ]["collision_time"].values[0]) - 1

        # Load the csv file containing all the agent's trajectories
        agent_logs_csv_file = os.path.join(logs_folder, f"{agent_id}", "logs.csv")
        agent_trajectory_df = pd.read_csv(agent_logs_csv_file, sep = ";")
        # Extract vehicle position at the time of collision and add it to the data to be merged with the data from the DB
        if len(agent_trajectory_df) > 0:
            x = agent_trajectory_df[ agent_trajectory_df["trajectory_number"] == time_step_of_collision]["x_position_vehicle_m"].values[0]
            y = agent_trajectory_df[ agent_trajectory_df["trajectory_number"] == time_step_of_collision]["y_position_vehicle_m"].values[0]
        
            data.append( (int(agent_id), float(x), float(y)) )
        
    collision_position_df = pd.DataFrame(data, columns = ['agent_id', 'x_position', 'y_position'])

    return collision_df_from_db.merge(collision_position_df, on = 'agent_id')

def load_detailed_collisions_data(results_folders: List[str]):
    """
    For each collision found by the generators we extract as much information as possible about it, such as:
    - The position of the collision
    - The RIV of the collision
    - The impact location of the collision
    - The time of the collision
    - The criticality of the collision (harm, pdof, etc.)
    """
    collisions_df: pd.DataFrame = None

    # Iterate over the results folders and collect the data from each of the runs contained in them
    for results_folder in results_folders:
        # Return only the folders that contain valid data
        for run_folder, _ in folders_of_completed_runs(results_folder):
            module_logger.debug(f"Collecting collisions data from folder {run_folder}")

            # Extract scenario's metadata
            metadata_file = os.path.join(run_folder, "metadata.csv")
            with open(metadata_file, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                data = [row for row in csv_reader][0]
                base_scenario = data["scenario_name"]
                # Find out which generator - By name + module
                generator = data["generator"]
                # Ensure runs are distinguishable
                timestamp_id = data["timestamp_id"]

            # Collect detailed data about collisions from .csv file and DB (RIV)
            for simulation_folder, scenario_folder_name in simulation_folders(run_folder):
                # Build scenario's log folder, scenario file path, and DB file path
                logs_folder = os.path.join(simulation_folder, "logs")
                scenario_file = os.path.join(simulation_folder, "scenario", f"scenario_{scenario_folder_name}.xml")
                db_scenario_file = os.path.join(logs_folder, "simulation.db")

                # Extract the data about collisions from the DB (time of the collision, RIV, and impact location) for each colliding agent
                collision_df_from_db = get_collisions_details_from_db(db_scenario_file)

                # Further process collisions only if there are any
                if len(collision_df_from_db) > 0:
                    # Extract the position of each collided agent from their logs.csv
                    collision_df_from_db = extend_with_agent_position_at_collision(collision_df_from_db, logs_folder)

                    # Extract additional information about collision from the collisions csv files
                    collision_df_from_db = extend_with_collision_criticality(collision_df_from_db, simulation_folder)

                    # Add the necessary metadata to distinguish collisions generated by different generators in various scenarios
                    # Generator name
                    collision_df_from_db['generator'] = generator
                    # Base map
                    collision_df_from_db['base_scenario'] = base_scenario
                    # Timestamp id of the generation run
                    collision_df_from_db['timestamp_id'] = timestamp_id
                    # File of the current scenario
                    collision_df_from_db['scenario_file'] = os.path.abspath(scenario_file)
                    # Scenario ID - HARD CODED
                    collision_df_from_db['scenario_id'] = int(scenario_folder_name)
                    # Make sure to distinguish agents by their ID and scenario
                    collision_df_from_db['local_agent_id'] = collision_df_from_db['agent_id']
                    # HARDCODED
                    collision_df_from_db['agent_id'] = collision_df_from_db['scenario_id']* 10000 + collision_df_from_db['agent_id'] - 10000
                    
                    # Append the data about this collision to the overall df of collisiosn
                    if collisions_df is None:
                        collisions_df= collision_df_from_db
                    else:
                        collisions_df = pd.concat([collisions_df, collision_df_from_db], ignore_index=True)
            
    return collisions_df

def load_detailed_interactions_data(results_folders: List[str]):
    """
    For each interaction found by the generators we extract as much information as possible about it
    """
    interactions_df: pd.DataFrame = None

    # Iterate over the results folders and collect the data from each of the runs contained in them
    for results_folder in results_folders:
        # Return only the folders that contain valid data
        for run_folder, _ in folders_of_completed_runs(results_folder):
            module_logger.debug(f"Collecting interactions data from folder {run_folder}")

            # Extract scenario's metadata
            metadata_file = os.path.join(run_folder, "metadata.csv")
            with open(metadata_file, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                data = [row for row in csv_reader][0]
                base_scenario = data["scenario_name"]
                # Find out which generator - By name + module
                generator = data["generator"]
                # Ensure runs are distinguishable
                timestamp_id = data["timestamp_id"]

            # Load the existing interaction map
            interactions_map_file = os.path.join(run_folder, "interaction_map.npy")
            interaction_map = np.load(interactions_map_file)

            # Merge the interaction map with the metadata
            data = {"base_scenario": [base_scenario],
                    "generator": [generator],
                    "timestamp_id": [timestamp_id],
                    "interaction_map": [interaction_map]
            }
            interaction_map_df = pd.DataFrame(data)

            # Save the interaction map data with the metadata to the overall df of interactions
            if interactions_df is None:
                interactions_df = interaction_map_df
            else:
                interactions_df = pd.concat([interactions_df, interaction_map_df], ignore_index=True)

    return interactions_df            

def load_detailed_test_generation_data(list_of_results_folders: List[str]):
    """
    Count how many agents are present in each scenario ("n_agents")
    """
    detailed_test_generation_df: pd.DataFrame = None

    # Iterate over the results folders and collect the data from each of the runs contained in them
    for results_folder in list_of_results_folders:
        # Return only the folders that contain valid data
        for run_folder, _ in folders_of_completed_runs(results_folder):
            module_logger.debug(f"Collecting interactions data from folder {run_folder}")

            # Extract scenario's metadata
            metadata_file = os.path.join(run_folder, "metadata.csv")
            with open(metadata_file, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                data = [row for row in csv_reader][0]
                base_scenario = data["scenario_name"]
                # Find out which generator - By name + module
                generator = data["generator"]
                # Ensure runs are distinguishable
                timestamp_id = data["timestamp_id"]

            # Collect detailed data about collisions from csv file and DB (RIV)
            for simulation_folder, scenario_folder_name in simulation_folders(run_folder):
                # Build scenario's log folder, scenario file path, and DB file path
                logs_folder = os.path.join(simulation_folder, "logs")
                scenario_file = os.path.join(simulation_folder, "scenario", f"scenario_{scenario_folder_name}.xml")
                db_scenario_file = os.path.join(logs_folder, "simulation.db")

                # Extract the data about the scenario from the DB (number of agents, time of the first collision, and time of the last collision)
                scenario_df_from_db = get_scenario_details_from_db(db_scenario_file)

                # Further process collisions only if there are any
                if len(scenario_df_from_db) > 0:
                    # Add the necessary metadata to distinguish collisions generated by different generators in various scenarios
                    # Generator name
                    scenario_df_from_db['generator'] = generator
                    # Base map
                    scenario_df_from_db['base_scenario'] = base_scenario
                    # Timestamp id of the generation run
                    scenario_df_from_db['timestamp_id'] = timestamp_id
                    # File of the current scenario
                    scenario_df_from_db['scenario_file'] = os.path.abspath(scenario_file)
                    # Scenario ID - HARD CODED
                    scenario_df_from_db['scenario_id'] = int(scenario_folder_name)
                    
                    # Append the data about this collision to the overall df of collisiosn
                    if detailed_test_generation_df is None:
                        detailed_test_generation_df= scenario_df_from_db
                    else:
                        detailed_test_generation_df = pd.concat([detailed_test_generation_df, scenario_df_from_db], ignore_index=True)
                    
    return detailed_test_generation_df            

def interactions_analysis(behavior_df: pd.DataFrame, interaction_sets: list, output_folder: str, detailed_interaction_df: pd.DataFrame):
    """
    Analyze the interactions found by the generators and produce an HTML report with the results of the analysis and the corresponding plots
    """
    # Prepare the output folder
    interactions_analysis_output_folder = os.path.join(output_folder, "interactions_analysis")
    os.makedirs(interactions_analysis_output_folder, exist_ok=True)

    # Duplicate the metrics for the scenarios and tag them as all scenario
    all_scenarios_df = behavior_df.copy()
    # Replace the base scenarios with the generic "all" label
    all_scenarios_df = all_scenarios_df.assign(base_scenario='all')
    # Concatenate the two dataframes
    behavior_df = pd.concat([behavior_df, all_scenarios_df], ignore_index=True)

    # Replace the generator and base scenario names with the more readable ones for the plots and tables
    behavior_df = behavior_df.replace(GENERATORS).replace(BASE_SCENARIOS)

    # Iterate over interaction metrics
    for metric in ["total_interactions", "unique_interactions"]:
        # Select the data and compute the p - A12 for each pair of generators in each base scenario and by aggregating the results across all of the scenarios
        df_metrics= compute_metrics_for(behavior_df, [metric])
        # Select the data and plot a box plot for each scenario and by aggregating the metrics across all the scenarios
        plot_files = plot_metrics_for(interactions_analysis_output_folder, behavior_df, [metric])
        
        metrics_and_plots = dict()
        metrics_and_plots[metric] = plot_files
        # Create an HTML report
        report_file_html = os.path.join(interactions_analysis_output_folder, f"{metric}.html")
        store_to_html(report_file_html, df_metrics, {metric: plot_files})

    # Plot the venn diagrams for the interaction sets
    plot_venn_diagrams(interactions_analysis_output_folder, interaction_sets, "venn-diagram-interactions")

    if detailed_interaction_df is not None:
        # Duplicate the metrics for the scenarios and tag them as all scenario
        interactions_all_scenarios_df = detailed_interaction_df.copy()
        # Replace the base scenarios with the generic "all" label
        interactions_all_scenarios_df = interactions_all_scenarios_df.assign(base_scenario = 'all')
        # Concatenate the two dataframes
        detailed_interaction_df = pd.concat([detailed_interaction_df, interactions_all_scenarios_df], ignore_index = True)

        # Rename generators and scenarios - Replace does not work if the cell contains a numpy array
        detailed_interaction_df["generator"] = detailed_interaction_df["generator"].replace(GENERATORS)
        detailed_interaction_df["base_scenario"] = detailed_interaction_df["base_scenario"].replace(BASE_SCENARIOS)

        base_scenarios = [s for s in detailed_interaction_df["base_scenario"].unique().tolist() if str(s) != "all"]
        generators = detailed_interaction_df["generator"].unique().tolist()

        # Iterate over base scenarios and generators
        for base_scenario in base_scenarios:
            for generator in generators:
                # Extract the data for the current generator and base scenario
                interaction_caused_by_generator_in_scenario_df = detailed_interaction_df[(detailed_interaction_df["base_scenario"] == base_scenario) & (detailed_interaction_df["generator"] == generator)]
                
                # Skip empty dataframes (some combinations of generator and base scenario might not be there)
                if interaction_caused_by_generator_in_scenario_df.empty:
                    continue

                # Plot interaction map for the current generator and base scenario
                fig = plot_interaction_map(interaction_caused_by_generator_in_scenario_df, base_scenario, generator)
                output_interaction_map_file = os.path.join(interactions_analysis_output_folder, f"interactions_map_{base_scenario}_{generator}.png")
                plt.savefig(output_interaction_map_file)
                plt.close(fig)

        pass

def plot_interaction_map(interaction_map_df: pd.DataFrame, base_scenario: str, generator: str):
    """
    Plot the interaction map for a given generator and base scenario
    """
    import matplotlib.cm as cm

    # Map Axes Definition - relies on the common configuration.
    long_bins, lat_bins = get_interaction_map_bins()
    
    # Initialize plot image
    fig, ax = plt.subplots(figsize = (8, 4))

    # Sum the interaction maps of all the rows in the dataframe to get the overall interaction map for the current generator and base scenario
    interaction_map = np.sum(interaction_map_df["interaction_map"].tolist(), axis = 0)

    # Draw an icon of a vehicle at 0,0 to show direction
    from commonroad.visualization.icons import get_obstacle_icon_patch
    from commonroad.scenario.obstacle import ObstacleType
    import matplotlib as mpl
    config = ScenarioGeneratorConfig()

    # Get the colormap and set the under and bad colors
    colMap = cm.inferno_r
    colMap.set_bad(color = 'white')
    colMap.set_under(color = 'white')

    # Replace the 0 values with NaN to make them white in the plot
    masked_data = np.copy(interaction_map)
    masked_data[masked_data == 0 ] = np.nan

    # Plot data
    try:
        im = ax.imshow(masked_data, cmap = colMap)
    except Exception as exc:
        raise exc

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(len(long_bins)), labels = [f"{v:.2f}" for v in long_bins])
    ax.set_yticks(np.arange(len(lat_bins)), labels = [f"{v:.2f}" for v in lat_bins])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = -30, ha = "right", rotation_mode = "anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(len(long_bins)+1)-.5, minor = True)
    ax.set_yticks(np.arange(len(lat_bins)+1)-.5, minor = True)
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which = "minor", bottom = False, left = False)

    # Set title
    ax.set_title(f"Interaction Effects' Map for {generator} in {base_scenario} ")

    # Compute the center of the plot to place the vehicle icon there
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

def test_generation_analysis(performace_df: pd.DataFrame, output_folder, detailed_test_generation_df: pd.DataFrame):
    """
     Analyze the generated tests and produce an HTML report with the results of the analysis and the corresponding plots
    """

    # Prepare the output folder
    test_generation_analysis_output_folder = os.path.join(output_folder, "test_generation_analysis")
    os.makedirs(test_generation_analysis_output_folder, exist_ok=True)

    # Duplicate the metrics for the scenarios and tag them as all scenario
    all_scenarios_df = performace_df.copy()
    # Replace the base scenarios with the generic "all" label
    all_scenarios_df = all_scenarios_df.assign(base_scenario = 'all')
    # Concatenate the two dataframes
    performace_df = pd.concat([performace_df, all_scenarios_df], ignore_index=True)

    # Replace the generator and base scenario names with the more readable ones for the plots and tables
    performace_df = performace_df.replace(GENERATORS).replace(BASE_SCENARIOS)
    
    # Iterate over test generation metrics
    for metric in ["all_scenarios", "valid_scenarios", "invalid_scenarios", "total_generation_time"]:
        # Select the data and compute the p - A12 for each pair of generators in each base scenario and by aggregating the results across all of the scenarios
        df_metrics= compute_metrics_for(performace_df, [metric])
        # Selecte the data and plot a box plot for each scenario and by aggregating the metrics across all the scenarios
        plot_files = plot_metrics_for(test_generation_analysis_output_folder, performace_df, [metric])
        
        metrics_and_plots = dict()
        metrics_and_plots[metric] = plot_files
        # Create an HTML report
        report_file_html = os.path.join(test_generation_analysis_output_folder, f"{metric}.html")
        store_to_html(report_file_html, df_metrics, {metric: plot_files})

    if detailed_test_generation_df is not None:
        # Rename generators and scenarios
        test_generation_df = detailed_test_generation_df.copy()
        test_generation_df["generator"] = test_generation_df["generator"].replace(GENERATORS)
        test_generation_df["base_scenario"] = test_generation_df["base_scenario"].replace(BASE_SCENARIOS)

        # Duplicate the metrics for the scenarios and tag them as all scenario
        test_generation_all_df = test_generation_df.copy()
        # Replace the base scenarios with the generic "all" label
        test_generation_all_df = test_generation_all_df.assign(base_scenario = 'all')
        # Concatenate the two dataframes
        test_generation_df = pd.concat([test_generation_df, test_generation_all_df], ignore_index=True)

        # Plot the position of the vehicles when they crashed for each generator (color) in each scenario (map)
        base_scenarios = [ s for s in test_generation_df["base_scenario"].unique().tolist() if str(s) != "all"]
        generators = test_generation_df["generator"].unique().tolist()

        for base_scenario in base_scenarios:
            # What is the distribution of collisions (per scenario)?
            per_scenario_test_generation_df = test_generation_df[ test_generation_df["base_scenario"] == base_scenario ]

            distribution_of_scenario_complexity_file = os.path.join(test_generation_analysis_output_folder, f"scenario_complexity_{base_scenario}.png")

            fig, ax = plt.subplots(figsize = (8, 4))
            ax = sns.histplot(data = per_scenario_test_generation_df, x = "n_agents", hue = "generator")
            ax.set_xlabel("# Agents")
            ax.set_ylabel("# Scenarios")
            # Make sure we do not cut labels
            plt.tight_layout()
            plt.savefig(distribution_of_scenario_complexity_file)
            plt.close(fig)

        ########

        # What is the distribution of collisions (for all the scenarios)?
        all_scenario_test_generation_df = test_generation_df[ test_generation_df["base_scenario"] == "all" ]
        distribution_of_scenario_complexity_file = os.path.join(test_generation_analysis_output_folder, f"impact_locations_all.png")
        fig, ax = plt.subplots(figsize = (8, 4))
        ax = sns.histplot(data = all_scenario_test_generation_df, x = "n_agents", hue = "generator")
        ax.set_xlabel("# Agents")
        ax.set_ylabel("# Scenarios")
        # Make sure we do not cut labels
        plt.tight_layout()
        plt.savefig(distribution_of_scenario_complexity_file)
        plt.close(fig)

@click.command()
@click.option("--results-folder", 'results_folders', type = click.Path(exists = True), required = False, multiple = True,  default = [RESULTS_FOLDER])
@click.option("--full", 'perform_full_analysis', is_flag = True, required = False, default = False)
@click.pass_context
def analyse(ctx, results_folders: List, perform_full_analysis: bool):
    """
    Run all the data analyses and by loading data from the given results-folder
    """

    #
    # ========== Stage 0: Setup ==========
    #

    # Prepare the output folder and ensure it exists
    if "output_folder" in ctx.obj:
        output_folder = ctx.obj["output_folder"]
    else:
        output_folder = RESULTS_FOLDER

    os.makedirs(output_folder, exist_ok=True)

    # Initialize the data structures to store the loaded data
    requirements_violations_df: pd.DataFrame
    performace_df: pd.DataFrame
    behavior_df: pd.DataFrame

    # Detailed data about collisions
    detailed_collisions_df: pd.DataFrame = None
    detailed_interaction_df: pd.DataFrame = None
    detailed_test_generation_df: pd.DataFrame = None
    
    # Data structure to build the Venn diagrams
    interaction_sets: dict
    collision_location_riv_ria_sets: dict

    def plot_the_banner():
        max_l = max(len(f"Reading input data from {results_folders}"), len(f"Generate output data to {output_folder}"))
        module_logger.critical( "=" * (max_l + 2))
        module_logger.critical(f"Reading input data from {results_folders}")
        module_logger.critical(f"Generate output data to {output_folder}")
        module_logger.critical( "=" * (max_l + 2))

    # Print the banner with the input and output folders    
    plot_the_banner()

    #
    # ========== Stage 1: Load data ==========
    #

    requirements_violations_df, performace_df, behavior_df, interaction_sets, collision_location_sets, collision_location_riv_sets, collision_location_riv_ria_sets = _load_data_from_results_folders(results_folders)

    # Load detailed data for a detailed analysis of the collisions, interactions, and test generation only if requested
    if perform_full_analysis:
        detailed_collisions_df = load_detailed_collisions_data(results_folders)
        detailed_interaction_df = load_detailed_interactions_data(results_folders)
        detailed_test_generation_df = load_detailed_test_generation_data(results_folders)

    #
    # ========== Stage 2: Perform the various analyses ==========
    #
    
    # ========== Stage 2.1: Requirement Violation Analysis ==========
    # ========== Stage 2.1.1: Collision analysis ==========
    
    # For the collision analysis, we use the collision map containing the most detailed information: impact location, RIV, and RIA
    collisions_analysis(requirements_violations_df, collision_location_riv_ria_sets, output_folder, detailed_collisions_df)

    # ========== Stage 2.2: Interaction/Behavior Analysis ==========
    # For the interaction analysis, we use the interaction map containing the most detailed information: 
    # The interaction map for each scenario and generator
    interactions_analysis(behavior_df, interaction_sets, output_folder, detailed_interaction_df)

    # ========== Stage 2.3: Test Generation Analysis ==========
    # For the test generation analysis, we use the performance map containing the most detailed information:
    # number of agents, time of the first collision, and time of the last collision for each scenario and generator
    test_generation_analysis(performace_df, output_folder, detailed_test_generation_df)

    # Print the banner with the input and output folders at the end of the analysis
    plot_the_banner()

def count_metric(simulation_db, metric: str) -> int:
    """
    Count how many `metric` == 1.0 are stored in the DB
    """
    # Open connection with database
    connection = sqlite3.connect(simulation_db)
    try:
        # Count how many unique collisions are stored
        result = connection.execute(f"SELECT timestep from scenario_evaluation WHERE {metric} = 1.0;")

        return len(result.fetchall())
    finally:
        try:
            connection.close()
        except:
            pass

def count_collisions(simulation_db) -> int:
    """
    Count how many unique collisions are stored in the DB
    """
    # Open connection with database
    connection = sqlite3.connect(simulation_db)
    try:
        # Count how many unique collisions are stored
        result = connection.execute("SELECT agent_id, has_collided_with from scenario_evaluation WHERE has_collided = 1.0 GROUP BY agent_id, has_collided_with;")
        collisions = set()

        # Iterate over the results and count unique collisions
        for tup in result.fetchall():
            # Extract the collided vehicles
            collision_group = str(tup[1])
            collisions.add(collision_group)

        return len(collisions)
    finally:
        try:
            connection.close()
        except:
            pass

def _agent_ids_from_db(simulation_db) -> List:
    """
    Get the list of agent ids from the DB
    """
    # Open connection
    connection = sqlite3.connect(simulation_db)

    try:
        # Make the query
        result = connection.execute("SELECT agent_ids from meta;")
        result_as_string = result.fetchone()[0]

        return ast.literal_eval(result_as_string)
    finally:
        try:
            connection.close()
        except:
            pass

def _to_float(value) -> float:
    """
    Convert a value stored as text in the DB ('1.5', 'None', 'nan', NULL) to float or None.
    """
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    
    if math.isnan(result):
        return None
    else:
        return result

def _interactions_from_db_for_agent(simulation_db, agent_id) -> List[int]:
    """
    Get the list of time steps at which the given agent interacted with other agents from the DB
    """
    # Open connection
    connection = sqlite3.connect(simulation_db)

    try:
        # Make the query
        result = connection.execute(f"SELECT timestep, diff_long, diff_lat from scenario_evaluation where agent_id = {agent_id} and is_interacting == 1.0;")
        return [(int(t[0]), _to_float(t[1]), _to_float(t[2])) for t in result.fetchall()]
    finally:
        try:
            connection.close()
        except:
            pass

def _get_impacts_features_from_db(simulation_db, run_folder, simulation_folder) -> List[Tuple[str, float, float]]:
    """
    Get the features of the impacts from the DB, such as the impact location, RIV, and RIA
    """
    
    # Open connection with database
    connection = sqlite3.connect(simulation_db)

    try:
        # Retrieve collisions data from the DB
        # Group data by agent_id and has_collided_with to get unique collisions
        result = connection.execute('SELECT MIN(timestep), agent_id, most_critical_agent, has_collided_with, riv, ria, impact_location from scenario_evaluation where has_collided = 1.0 group by agent_id, has_collided_with order by MIN(timestep), has_collided_with;')
        metrics = result.fetchall()

        collisions = dict()
        impacts = list()

        # Iterate over extracted collision data
        for time_step, agent_id, most_critical_agent, has_collided_with_str, riv, ria, impact_location in metrics:
            agent_id = int(agent_id)
            time_step = int(time_step)
            
            # Keep this a string
            has_collided_with_str = str(has_collided_with_str)

            # Parse to array
            has_collided_with = ast.literal_eval(has_collided_with_str)
            if len(has_collided_with) == 1:
                module_logger.critical(f"Agent {agent_id} collided with the road - {simulation_db}")
                most_critical_agent = "ROAD"
            else:
                assert most_critical_agent != "None", f"Missing most critical agent. Check {simulation_db}"
                most_critical_agent = int(most_critical_agent)

            riv = float(riv)
            ria = float(ria)
            # If there's no impact location default to ROAD
            if impact_location != "None":
                impact_location = str(impact_location)
            else:
                impact_location = "-+"

            # Filter out spurious entries across accidents order matter
            if [agent_id, most_critical_agent] in impacts:
                module_logger.debug(f"We already have an entry for {agent_id} - {most_critical_agent} - {impact_location} - {simulation_db}")
                continue
            else:
                module_logger.debug(f"Adding entry {agent_id} - {most_critical_agent} - {impact_location} - {simulation_db}")
                impacts.append([agent_id, most_critical_agent])

            if has_collided_with_str not in collisions:
                # Register the data for the collisions
                collisions[has_collided_with_str] = {
                    "v1": None,
                    "v2": None,
                    "ria": abs(ria),
                    "riv": riv,
                    "impact_location": None,
                    "v1_impact_location": None,
                    "v2_impact_location": None,
                }
                collisions[has_collided_with_str]["v1_impact_location"] = impact_location # FB, -+, etc
                collisions[has_collided_with_str]["v1"] = agent_id

                if most_critical_agent == "ROAD":
                    collisions[has_collided_with_str]["v2_impact_location"] = "-+" # Default
                    collisions[has_collided_with_str]["v2"] = "ROAD" # Default
            else:
                # Register the data about the impact location of the second vehicle. This cannot be the road
                collisions[has_collided_with_str]["v2_impact_location"] = impact_location
                collisions[has_collided_with_str]["v2"] = agent_id

        # A this point we need to merge the impact locations together
        for collision in collisions.values():
            collision["impact_location"] = frozenset((collision["v1_impact_location"], collision["v2_impact_location"]))
            module_logger.debug(f"Register {collision}")

        return [(c["impact_location"], c["riv"], c["ria"]) for c in collisions.values()]
    finally:
        try:
            connection.close()
        except:
            pass            

def _rebuild_trajectory(initial_time_step: str, xs: str, ys: str, orientations:str, velocities:str) -> Trajectory:
    """
    Rebuild the trajectory of an agent from the logs.csv file.
    The trajectory is represented as a list of states, each containing the time step, position, orientation, and velocity of the agent at that time step
    """
    state_list = list()
    initial_time_step = int(initial_time_step)

    # Iterate over the state variables and build the trajectory
    for idx, state_vars, in enumerate(zip(xs.split(","), ys.split(","), orientations.split(","), velocities.split(","))):
        # Extract position, orientation, and velocity
        x, y = float(state_vars[0]), float(state_vars[1])
        orientation = float(state_vars[2])
        velocity = float(state_vars[3])
       
        # Create state object and add it to the list of states
        state = CustomState(time_step = initial_time_step + idx, position = np.array([x, y]), orientation = orientation, velocity = velocity)
        state_list.append(state)
        
    return Trajectory(initial_time_step, state_list)

def get_interaction_map_bins(n_bins: int = None):
    """
    Define the bins of the interaction map
    """
    from evita.config import ScenarioGeneratorConfig
    scenario_generator_config = ScenarioGeneratorConfig()

    if n_bins is None:
        n_bins = scenario_generator_config.n_bins

    # One car in front/one car behind - How can one retrieve difference in speed/acc?
    long_bins = np.linspace(-scenario_generator_config.length, +scenario_generator_config.length, n_bins)

    # In the United States, the Interstate Highway standards for the Interstate Highway System
    # use a 12 ft (3.7 m) standard lane width, while narrower lanes are used on lower classification roads.
    lane_width = 3.7
    lat_bins = np.linspace(-lane_width, lane_width, n_bins)
    
    return long_bins, lat_bins

# Columns written by EVITA's evaluator: a scenario_evaluation table without them was never evaluated
EVALUATION_COLUMNS = {"is_interacting", "diff_long", "diff_lat", "has_collided", "has_collided_with",
                      "ria", "riv", "impact_location", "most_critical_agent", "has_timeout", "has_missed_goal"}

def _is_evaluated(simulation_db: str) -> bool:
    """
    True if the scenario was evaluated: the evaluation table has EVITA's columns and at least one row.
    Executed but invalid scenarios (e.g. an agent did not start) keep the fork's bare table and are not evaluated.
    """
    connection = sqlite3.connect(simulation_db)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(scenario_evaluation);")}
        if not EVALUATION_COLUMNS.issubset(columns):
            return False
        return connection.execute("SELECT COUNT(*) FROM scenario_evaluation;").fetchone()[0] > 0
    except sqlite3.Error:
        return False
    finally:
        connection.close()

def generate_report(run_folder: str, _output_to: str = None) -> None:
    """
    Generate the "final" reports of the entire generation as csv files.
    """
    # Define output folder
    if _output_to:
        output_to = _output_to
    else:
        output_to = run_folder

    module_logger.critical(f"Generating report for {run_folder} at {output_to}")
    
    # Build requirement violation report file path
    requirements_violations = os.path.join(output_to, "requirements_violations.csv")

    module_logger.critical("Computing Requirement Violation Metrics")
    with open(requirements_violations, "w", newline = "") as csvfile:
        # Write the header of the csv file
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["total_timeouts", "total_missed_goals", "total_collisions", "unique_collisions_only_impact_location", "unique_collisions_impact_location_and_riv", "unique_collisions"])

        # Requirement Violations
        # - [x] Total Collision Count - counted on the side
        # - [x] Total Missed Goals (Timeout)
        # - [x] Total Missed Goals (Missed)
        
        # - Total Collision Features -> Unique Collision (using the map as filter)
        #       - Number of vehicles involved ?
        #       - Angle (RIA)
        #       - Velocity (RIV)
        #       - Impact Location
        
        # Generate all possible pairs of impact locations
        impact_location_bins = list()
        # 
        # "-+" denotes the road we need two symbols

        for p1 in ["F", "M", "B", "-"]:
            for s1 in ["L", "C", "R", "+"]:
                for p2 in ["F", "M", "B", "-"]:
                    for s2 in ["L", "C", "R", "+"]:
                        s = frozenset( (f"{p1}{s1}", f"{p2}{s2}") )
                        
                        impact_location_bins.append(s)
        
        # Initialize RIA and RIV bins       
        n_bins = 10
        riv_bins = np.linspace(0, 100, n_bins)
        ria_bins = np.linspace(0, 180, n_bins)
    
        # Initialize collision maps
        collision_map_only_impact_location = np.zeros(len(impact_location_bins))
        collision_map_impact_location_and_riv = np.zeros( (len(impact_location_bins), len(riv_bins) + 1))
        collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1))

        total_timeouts, total_missed_goals, total_collisions = 0, 0, 0
    
        # Iterate over the simulations
        for _, simulation_folder in simulation_folders(run_folder):
            module_logger.debug(f"Processing {simulation_folder}")
            
            # Build scenario's DB file path
            simulation_db = os.path.join(run_folder, simulation_folder, "logs", "simulation.db")

            if not _is_evaluated(simulation_db):
                module_logger.debug(f"Skipping {simulation_folder}: not evaluated (invalid scenario)")
                continue

            total_timeouts += count_metric(simulation_db, "has_timeout")
            total_missed_goals += count_metric(simulation_db, "has_missed_goal")

            # Initialize local collision maps
            local_only_impact_location = np.zeros(len(impact_location_bins))
            local_impact_location_and_riv = np.zeros((len(impact_location_bins), len(riv_bins) + 1))
            local_collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1))

            # Count collisions
            total_collisions += count_collisions(simulation_db)

            # Get the features of the collisions from the DB
            collision_features = _get_impacts_features_from_db(simulation_db, run_folder, simulation_folder)

            # It might happen that those features were not stored, in this case we need to recompute them!
            # Iterate over the features of the collisions and update the collision maps            
            for impact_location_as_frozenset, riv, ria in collision_features:
                # Check that the impact location is not None
                if None in impact_location_as_frozenset or riv is None or ria is None:
                    module_logger.warning(f"Skipping collision with missing features: impact_location = {impact_location_as_frozenset}, riv = {riv}, ria = {ria} in {simulation_db}")
                    continue

                # Map the impact location, RIV, and RIA to the corresponding bins
                impact_location_mapped = impact_location_bins.index(impact_location_as_frozenset)
                digitized_riv = np.digitize(riv, riv_bins)
                digitized_ria = np.digitize(ria, ria_bins)

                # Update local collision maps
                local_only_impact_location[impact_location_mapped] += 1.0
                local_impact_location_and_riv[impact_location_mapped][digitized_riv] += 1.0
                local_collision_map[impact_location_mapped][digitized_riv][digitized_ria] += 1.0
            
            # Update collision maps
            collision_map_only_impact_location += local_only_impact_location
            collision_map_impact_location_and_riv += local_impact_location_and_riv
            collision_map += local_collision_map

        # Compute total and unique collisions
        total_collisions_double_check = int(np.sum(collision_map_only_impact_location))
        unique_collisions_only_impact_location = np.count_nonzero(collision_map_only_impact_location)
        unique_collisions_impact_location_and_riv = np.count_nonzero(collision_map_impact_location_and_riv)
        unique_collisions = np.count_nonzero(collision_map)

        # Check that the total number of collisions we counted on the side is the same as the one we get from the collision map
        if total_collisions != total_collisions_double_check:
            module_logger.warning(f"Collision count mismatch in {run_folder}: {total_collisions} counted, {total_collisions_double_check} mapped")
        
        # Merge all the counts together and write them to the csv file
        values = [total_timeouts, total_missed_goals, total_collisions, unique_collisions_only_impact_location, unique_collisions_impact_location_and_riv, unique_collisions]

        csv_writer.writerow([str(v) for v in values])
    
    # Behavioral/Interaction Diversity
    #   - [x] Total time spent in emergency "", total number of  "has_disengagement"
    #   - Avg emergency/disengagement per vehicle?
    #     
    #   - Number of Vehicles/Complexity - NO
    #   - Initial Positioning of Vehicle  - NO
    #   - Initial Speed of Vehicles  - NO
    #   - Position of Goal  - NO
    #   - Route Complexity (Distance, Number of Lane Switches)  - NO
    #   - [x] Total Interaction Count
    #   - [x] Unique Interactions using the defined Interaction Map (10 bins 2 * lane widht, 10 bins 2 * vehicle lenght)
    
    module_logger.info("Computing Behavior and Interaction Metrics")

    # Build behavior report file path
    behavior = os.path.join(output_to, "behavior.csv")
    with open(behavior, "w", newline = "") as csvfile:
        # Write the header of the csv file
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["total_emergency", "total_disengagement", "total_agents", "total_interactions", "unique_interactions"])

        # Get the bins of the interaction map
        long_bins, lat_bins = get_interaction_map_bins()

        # Include 0 - B[0]:B[9] - 10 and len(bins)
        rows = len(long_bins) + 1
        cols = len(lat_bins) + 1

        # Initialize the interaction map and the counters for the requirement violations
        interaction_map = np.zeros((rows, cols))
        total_emergency, total_disengagement, total_agents = 0, 0, 0

        # Iterate over the simulations folders
        for _, simulation_folder in simulation_folders(run_folder):
            module_logger.debug(f"Processing {simulation_folder}")

            # Build scenario's DB and scenario's file paths
            simulation_db = os.path.join(run_folder, simulation_folder, "logs", "simulation.db")
            scenario_file = os.path.join(run_folder, simulation_folder, "scenario", f"scenario_{simulation_folder}.xml")

            if not _is_evaluated(simulation_db):
                module_logger.debug(f"Skipping {simulation_folder}: not evaluated (invalid scenario)")
                continue

            total_emergency += count_metric(simulation_db, "has_emergency")
            total_disengagement += count_metric(simulation_db, "has_disengagement")

            # Extract the agent ids from the DB and update the total count of agents
            agent_ids = _agent_ids_from_db(simulation_db)
            total_agents = total_agents + len(agent_ids)

            # Initialize local interaction map
            local_interaction_map = np.zeros((rows, cols))
            for agent_id in agent_ids:
                # Extract the timesteps at which the agent interacted with other agents
                interactions = _interactions_from_db_for_agent(simulation_db, agent_id)

                # Iterate over agent's interaction timesteos
                for t, diff_long, diff_lat in interactions:
                    if diff_long is None or diff_lat is None:
                        module_logger.warning(f"Interaction at timestep {t} for agent {agent_id} has None values for diff_long or diff_lat.")
                        continue

                    # If values in x are beyond the bounds of bins, 0 or len(bins) is returned as appropriate.
                    # Retrieve the trajectories for the interactions and fill up the local map
                    digitized_long = np.digitize(diff_long, long_bins)
                    digitized_lat = np.digitize(diff_lat, lat_bins)

                    local_interaction_map [digitized_long][digitized_lat] = local_interaction_map [digitized_long][digitized_lat] + 1.0

            # Update the interaction map with the local interaction map
            interaction_map = interaction_map + local_interaction_map

        # Update the interaction map with the local interaction map
        total_interactions = int(np.sum(interaction_map))
        unique_interactions = np.count_nonzero(interaction_map)
        merged_results = [total_emergency, total_disengagement, total_agents, total_interactions, unique_interactions]
        
        csv_writer.writerow([str(v) for v in merged_results])

    # Store all the results also as npy files
    collision_map_only_impact_location_file = os.path.join(output_to, "collision_map_only_impact_location.npy")
    np.save(collision_map_only_impact_location_file, collision_map_only_impact_location)

    collision_map_impact_location_and_riv_file = os.path.join(output_to, "collision_map_impact_location_and_riv.npy")
    np.save(collision_map_impact_location_and_riv_file, collision_map_impact_location_and_riv)

    collision_map_file = os.path.join(output_to, "collision_map.npy")
    np.save(collision_map_file, collision_map)

    interaction_map_file = os.path.join(output_to, "interaction_map.npy")
    np.save(interaction_map_file, interaction_map)