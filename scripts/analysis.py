import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from evita.config import ScenarioGeneratorConfig


def main(experiments_folder_path: str, max_workers: int = 5):
    # Initialize the scenario generator configuration and define the bins for interactions
    scenario_generator_config = ScenarioGeneratorConfig()
    interactions_n_bins = scenario_generator_config.n_bins
    long_bins = np.linspace(-scenario_generator_config.length, +scenario_generator_config.length, interactions_n_bins)
    lane_width = 3.7
    lat_bins = np.linspace(-lane_width, lane_width, interactions_n_bins)

    rows = len(long_bins) + 1
    cols = len(lat_bins) + 1

    # Initialize the bins for RIV and RIA, and the impact location bins
    collisions_n_bins = 10
    riv_bins = np.linspace(0, 100, collisions_n_bins + 1)
    ria_bins = np.linspace(0, 180, collisions_n_bins + 1)

    impact_location_bins = [
        f"{p}{s}"
        for p in ["F", "M", "B", "-"]
        for s in ["L", "C", "R", "+"]
    ]

    ##################################################

    experiments_ids = sorted([f for f in os.listdir(experiments_folder_path) if os.path.isdir(os.path.join(experiments_folder_path, f)) and f != "experiments_analysis"])

    # Initialize aggregated interaction maps
    interaction_maps = {exp_id: np.zeros((rows, cols), dtype = int) for exp_id in experiments_ids}

    # Initialize aggregated collision maps
    collision_maps = {exp_id: np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1), dtype = int) for exp_id in experiments_ids}
    collision_map_only_impact_location = {exp_id: np.zeros(len(impact_location_bins), dtype = int) for exp_id in experiments_ids}
    collision_map_impact_location_and_riv = {exp_id: np.zeros((len(impact_location_bins), len(riv_bins) + 1), dtype = int) for exp_id in experiments_ids}
    
    # Initialize aggregated lists for interactions, collisions and vehicles number metrics
    total_interactions = {exp_id: [] for exp_id in experiments_ids}
    total_interactions_effect = {exp_id: [] for exp_id in experiments_ids}
    total_unique_interactions = {exp_id: [] for exp_id in experiments_ids}
    aggregated_unique_interactions = {exp_id: -1 for exp_id in experiments_ids}
    total_collisions = {exp_id: [] for exp_id in experiments_ids}
    total_unique_collisions = {exp_id: [] for exp_id in experiments_ids}
    aggregated_unique_collisions = {exp_id: -1 for exp_id in experiments_ids}
    total_unique_scenarios_collisions_POI = {exp_id: [] for exp_id in experiments_ids}
    total_unique_scenarios_collisions_POI_RIV = {exp_id: [] for exp_id in experiments_ids}
    vehicles_number = {exp_id: [] for exp_id in experiments_ids}

    # Initialize aggregated lists for generation metrics
    total_generated_scenarios = {exp_id: -1 for exp_id in experiments_ids}
    total_valid_scenarios = {exp_id: -1 for exp_id in experiments_ids}
    total_invalid_scenarios = {exp_id: -1 for exp_id in experiments_ids}
    total_generation_time = {exp_id: -1 for exp_id in experiments_ids}

    ##################################################

    def process_scenario_folder(scenario_folder_path):
        """
        Load and process one scenario.

        Each worker creates and owns its own SQLite connection.
        The connection is closed before returning the analysis result.
        """

        simulation_db_path = os.path.join(scenario_folder_path, "logs", "simulation.db")

        if not os.path.exists(simulation_db_path):
            return None

        columns = ['agent_id', 'timestep', 'max_danger', 'max_danger_caused_by_agent', 'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat', 'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location']
        # columns = ['agent_id', 'timestep', 'max_danger', 'max_danger_caused_by_agent', 'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat', 'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location', 'most_critical_agent', 'has_timeout', 'has_missed_goal', 'has_emergency', 'has_disengagement']

        conn = None

        try:
            # Each worker gets its own SQLite connection.
            conn = sqlite3.connect(simulation_db_path)

            criticality_df = pd.read_sql_query("SELECT * FROM scenario_evaluation", conn)

            if len(criticality_df) == 0:
                return None

            criticality_df = criticality_df[columns]
        finally:
            if conn is not None:
                conn.close()

        analysis_results = process_scenario(criticality_df, impact_location_bins, riv_bins, ria_bins)

        return analysis_results

    def process_scenario(criticality_df: pd.DataFrame, impact_location_bins: list, riv_bins: list, ria_bins: list):
        """
        Process a single scenario and return all aggregated metrics.
        """

        # Initialize result data structure for this scenario
        result = {
            "interactions": [],
            "interactions_effect": [],
            "unique_interactions": 0,
            "interactions_map": np.zeros((rows, cols), dtype = int),

            "collisions": 0,
            "unique_collisions": 0,
            "unique_collisions_POI": 0,
            "unique_collisions_POI_RIV": 0,

            "collision_map": np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1), dtype = int),
            "collision_map_only_impact_location": np.zeros(len(impact_location_bins), dtype = int),
            "collision_map_impact_location_and_riv": np.zeros((len(impact_location_bins), len(riv_bins) + 1), dtype = int),

            "vehicles": 0,
        }

        # ================================================================
        # VEHICLES
        # ================================================================

        vehicles_ids = criticality_df['agent_id'].unique()

        # Count the number of vehicles in the scenario
        result["vehicles"] = len(vehicles_ids)

        # ================================================================
        # COLLISIONS
        # ================================================================

        collisions_columns = ['agent_id', 'timestep', 'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location']

        collisions = criticality_df[criticality_df['has_collided'] == 1.0][collisions_columns].sort_values(["timestep", "agent_id"])

        # Initialize local collision maps
        local_only_impact_location = np.zeros(len(impact_location_bins), dtype = int)
        local_impact_location_and_riv = np.zeros((len(impact_location_bins), len(riv_bins) + 1), dtype = int)
        local_collision_map = np.zeros((len(impact_location_bins), len(riv_bins) + 1, len(ria_bins) + 1), dtype = int)

        # Check if any collision happened in the scenario
        if len(collisions) > 0:
            collisions = collisions.drop_duplicates(subset = ["agent_id"])
            crashes = collisions.groupby("has_collided_with", sort = False)

            # Iterate over each collision metric
            for impact_location, riv, ria in zip(collisions["impact_location"], collisions["riv"], collisions["ria"]):
                # Skip impacts with missing features ('None' / NULL), instead of failing the whole scenario
                riv = pd.to_numeric(riv, errors = "coerce")
                ria = pd.to_numeric(ria, errors = "coerce")
                if impact_location not in impact_location_bins or pd.isna(riv) or pd.isna(ria):
                    continue

                # Retrieve impact location index and RIV/RIA bins
                idx = impact_location_bins.index(impact_location)
                riv_bin = np.digitize(riv * 3.6, riv_bins) # m/s -> km/h
                ria_bin = np.digitize(np.degrees(abs(ria)), ria_bins) # rad -> deg

                # Update local collision maps
                local_only_impact_location[idx] += 1
                local_impact_location_and_riv[idx][riv_bin] += 1
                local_collision_map[idx][riv_bin][ria_bin] += 1

            # Update scenario's result with collisions metrics
            result["collisions"] = crashes.ngroups
            result["unique_collisions"] = np.count_nonzero(local_collision_map)
            result["unique_collisions_POI"] = np.count_nonzero(local_only_impact_location)
            result["unique_collisions_POI_RIV"] = np.count_nonzero(local_impact_location_and_riv)

        # Update scenario's result with collisions maps
        result["collision_map"] = local_collision_map
        result["collision_map_only_impact_location"] = local_only_impact_location
        result["collision_map_impact_location_and_riv"] = local_impact_location_and_riv

        # ================================================================
        # INTERACTION MAP
        # ================================================================

        interactions_columns = ['agent_id', 'timestep', 'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat']

        interactions = criticality_df[criticality_df['is_interacting'] == 1.0][interactions_columns]

        local_interactions_map = np.zeros((rows, cols), dtype = int)

        # Iterate over each interaction detected in the scenario
        for _, row in interactions.iterrows():
            # Extract interacting vehicle ID, interaction timestep, interacting agents IDs and the effects of the interaction on the agent's trajectory
            diff_long = pd.to_numeric(row["diff_long"], errors = "coerce")
            diff_lat = pd.to_numeric(row["diff_lat"], errors = "coerce")

            if pd.isna(diff_long) or pd.isna(diff_lat):
                continue

            result["interactions_effect"].append((diff_long, diff_lat))

            # Determine the bin indices for the longitudinal and lateral differences
            long_bin_index = np.digitize(diff_long, long_bins)
            lat_bin_index = np.digitize(diff_lat, lat_bins)

            # Increment the corresponding cell in the interaction map
            local_interactions_map[long_bin_index][lat_bin_index] += 1

        result["interactions"] = int(np.sum(local_interactions_map, axis = None))
        result["unique_interactions"] = np.count_nonzero(local_interactions_map)
        result["interactions_map"] = local_interactions_map

        return result
    
    ##################################################

    for experiment_id in tqdm(experiments_ids, desc = "Processing experiments"):
        tqdm.write(f"Processing experiment: {experiment_id}")

        experiment_folder_path = os.path.join(experiments_folder_path, experiment_id)

        # ============================================================
        # Check if the performance.csv file exists and read generation metrics
        # ============================================================

        performance_file = os.path.join(experiment_folder_path, "performance.csv")

        if os.path.exists(performance_file):
            # Read performance.csv metrics file
            performances = pd.read_csv(performance_file)

            # Save generation metrics
            total_generated_scenarios[experiment_id] = int(performances["all_scenarios"].iloc[0])
            total_valid_scenarios[experiment_id] = int(performances["valid_scenarios"].iloc[0])
            total_invalid_scenarios[experiment_id] = int(performances["invalid_scenarios"].iloc[0])
            total_generation_time[experiment_id] = round(float(performances["total_generation_time"].iloc[0]), 2)

        # ============================================================
        # Find scenarios
        # ============================================================

        scenarios_folders = sorted([f for f in os.listdir(experiment_folder_path) if os.path.isdir(os.path.join(experiment_folder_path, f)) and f not in ["checkpoints", "debug"]])
        scenario_paths = [os.path.join(experiment_folder_path, scenario_folder) for scenario_folder in scenarios_folders]

        # ============================================================
        # Process scenarios in parallel
        # ============================================================

        with ThreadPoolExecutor(max_workers = max_workers) as executor:
            futures = {
                executor.submit(
                    process_scenario_folder,
                    scenario_path
                ): scenario_path
                for scenario_path in scenario_paths
            }

            results = {}
            for future in tqdm(as_completed(futures), total = len(futures), desc = f"Processing scenarios [{experiment_id}]"):
                scenario_path = futures[future]

                try:
                    results[scenario_path] = future.result()
                except Exception as e:
                    tqdm.write(f"Error processing scenario " f"{scenario_path}: {e}")
                    results[scenario_path] = None
                    continue
                
            for scenario_path in scenario_paths:
                analysis_results = results[scenario_path]

                if analysis_results is None:
                    tqdm.write(f"Skipping scenario: {scenario_path}")
                    continue

                # ====================================================
                # Save interactions metrics
                # ====================================================

                total_interactions[experiment_id].append(analysis_results["interactions"])
                total_interactions_effect[experiment_id].append(analysis_results["interactions_effect"])
                total_unique_interactions[experiment_id].append(analysis_results["unique_interactions"])

                # ====================================================
                # Save interactions maps
                # ====================================================
                
                interaction_maps[experiment_id] += (analysis_results["interactions_map"])

                # ====================================================
                # Save collision metrics
                # ====================================================

                total_collisions[experiment_id].append(analysis_results["collisions"])
                total_unique_collisions[experiment_id].append(analysis_results["unique_collisions"])
                total_unique_scenarios_collisions_POI[experiment_id].append(analysis_results["unique_collisions_POI"])
                total_unique_scenarios_collisions_POI_RIV[experiment_id].append(analysis_results["unique_collisions_POI_RIV"])

                # ====================================================
                # Save collision maps
                # ====================================================

                collision_maps[experiment_id] += (analysis_results["collision_map"])
                collision_map_only_impact_location[experiment_id] += (analysis_results["collision_map_only_impact_location"])
                collision_map_impact_location_and_riv[experiment_id] += (analysis_results["collision_map_impact_location_and_riv"])

                # ====================================================
                # Save vehicles number data
                # ====================================================

                vehicles_number[experiment_id].append(analysis_results["vehicles"])

    # ================================================================
    # AGGREGATED UNIQUE INTERACTIONS AND COLLISIONS
    # ================================================================
    for experiment_id in experiments_ids:
        aggregated_unique_interactions[experiment_id] = np.count_nonzero(interaction_maps[experiment_id])
        aggregated_unique_collisions[experiment_id] = np.count_nonzero(collision_maps[experiment_id])

    # ================================================================
    # EXPORT EXPERIMENTS ANALYSIS
    # ================================================================

    # Define and create output folder for experiments analysis results
    experiments_analysis_folder_path = os.path.join(experiments_folder_path, "experiments_analysis")
    os.makedirs(experiments_analysis_folder_path, exist_ok = True)

    # ================================================================
    # SAVE AGGREGATED METRICS
    # ================================================================

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_list(data, filename):
        file_path = os.path.join(experiments_analysis_folder_path, filename)

        clean_data = make_json_serializable(data)

        with open(file_path, "w") as f:
            json.dump(clean_data, f, indent = 4)

    save_list(total_interactions, "total_interactions.json")
    save_list(total_interactions_effect, "total_interactions_effect.json")
    save_list(total_unique_interactions, "total_unique_interactions.json")
    save_list(aggregated_unique_interactions, "aggregated_unique_interactions.json")
    save_list(total_collisions, "total_collisions.json")
    save_list(interaction_maps, "interaction_maps.json")
    #
    save_list(total_unique_collisions, "total_unique_collisions.json")
    save_list(aggregated_unique_collisions, "aggregated_unique_collisions.json")
    save_list(total_unique_scenarios_collisions_POI, "total_unique_scenarios_collisions_POI.json")
    save_list(total_unique_scenarios_collisions_POI_RIV, "total_unique_scenarios_collisions_POI_RIV.json")
    save_list(collision_maps, "collision_map.json")
    save_list(collision_map_only_impact_location, "collision_map_only_impact_location.json")
    save_list(collision_map_impact_location_and_riv, "collision_map_impact_location_and_riv.json")
    #
    save_list(vehicles_number, "vehicles_number.json")

    # ================================================================
    # SAVE AGGREGATED GENERATION METRICS
    # ================================================================

    if total_generated_scenarios:
        save_list(total_generated_scenarios, "total_generated_scenarios.json")
        save_list(total_valid_scenarios, "total_valid_scenarios.json")
        save_list(total_invalid_scenarios, "total_invalid_scenarios.json")
        save_list(total_generation_time, "total_generation_time.json")

    print("Analysis completed.")

##################################################

if __name__ == "__main__":
    experiments_folder_path = sys.argv[1]

    if len(sys.argv) > 2:
        max_workers = int(sys.argv[2])
    else:
        max_workers = 5

    main(experiments_folder_path, max_workers)