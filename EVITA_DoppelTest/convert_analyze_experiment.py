import json
import os
import re
import sys
import subprocess
import numpy as np
import pandas as pd

sys.path.append("../")
sys.path.append("../Frenetix-Motion-Planner/")

from commonroad.geometry.shape import Rectangle
from commonroad.common.file_reader import XMLFileReader

#####

# Set standard vehicle shape
# Also defined in config.py
vehicle_width = 2.11
vehicle_length = 4.933
vehicle_heigth = 1.48
shape = Rectangle(vehicle_length, vehicle_width)

def analyze_generation(record_files_folder: str, experiment: str, output_files_folder: str):
    print(f"Processing Experiment {experiment}")

    # Detect all the generation folders in the experiment
    generation_folders = sorted([folder for folder in os.listdir(os.path.join(record_files_folder, experiment)) if os.path.isdir(os.path.join(record_files_folder, experiment, folder)) and folder.startswith("Generation_")])
    
    for generation in generation_folders:
        print(f"\n-- Generation {generation}")

        # Detect all the scenarios folders generated
        scenarios_folders = sorted([folder for folder in os.listdir(os.path.join(record_files_folder, experiment, generation)) if os.path.isdir(os.path.join(record_files_folder, experiment, generation, folder))])
        for scenario in scenarios_folders:
            print(f"\n---- Scenario {scenario}")
            
            # Extract scenario's map name
            pattern = r"SCENARIO_MAP_(.+?)_(\d{14})"
            match = re.search(pattern, experiment)
            map_name = match.group(1)

            # Set output file path for converted scenario
            executed_scenario_folder = os.path.join(output_files_folder, experiment, generation, scenario)
            executed_scenario_file_path = os.path.join(executed_scenario_folder, "executed_scenario.xml")
            interactions_file_path = os.path.join(executed_scenario_folder, "interactions.csv")
            interactions_map_file_path = os.path.join(executed_scenario_folder, "interactions_map.csv")
            os.makedirs(executed_scenario_folder, exist_ok = True)

            # Skip the conversion process if already done
            if os.path.exists(executed_scenario_file_path) and os.path.exists(interactions_file_path):
                continue

            # Set folder path containing the scenario .record files
            apollo_scenario_folder_path = os.path.join(record_files_folder, experiment, generation, scenario)

            # Define python 3.8 environment and script to run scenario using Apollo
            python_38 = "../.venv38/bin/python"
            scenario_conversion_script = "conversion_scripts.scenario_converter"

            ids_map_file_path = os.path.join(executed_scenario_folder, "ids_map.json")
            if os.path.exists(ids_map_file_path):
                with open(ids_map_file_path, 'r') as f:
                    ids_map = json.load(f)
            else:
                ids_map = None

            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.abspath("../") + ":" + os.path.abspath("../../")

                # Launch script to run convert Apollo scenario to CommmonRoad format and extract planned trajectories, collisions, and perceptions data
                if ids_map is not None:
                    _ = subprocess.run(
                        [python_38, "-m", scenario_conversion_script, apollo_scenario_folder_path, executed_scenario_folder, map_name, json.dumps(ids_map)],
                        check = True,
                        env = env
                    )
                else:
                    _ = subprocess.run(
                        [python_38, "-m", scenario_conversion_script, apollo_scenario_folder_path, executed_scenario_folder, map_name],
                        check = True,
                        env = env
                    )
            except subprocess.CalledProcessError as e:
                print("Subprocess failed!")
                print("Return code:", e.returncode)
                print("stdout:", e.stdout)
                print("stderr:", e.stderr)
                raise
            
            from conversion_scripts.utils.interactions_utils import augment_with_collisions, augment_with_interactions, planned_trajectories_to_df

            # Open executed scenario
            executed_scenario_file_path = os.path.join(executed_scenario_folder, "executed_scenario.xml")
            executed_scenario, _ = XMLFileReader(executed_scenario_file_path).open(lanelet_assignment = True)

            # Open scenario's collisions
            scenario_collisions_file_path = os.path.join(executed_scenario_folder, "apollo_scenario_collisions.csv")
            scenario_collisions = pd.read_csv(scenario_collisions_file_path)
            
            # Open planned trajectories
            scenario_planned_trajectories_file_path = os.path.join(executed_scenario_folder, "planned_trajectories.json")
            with open(scenario_planned_trajectories_file_path, "r") as f:
                scenario_planned_trajectories = json.load(f)

            # Convert planned trajectories Dict to a DataFrame
            scenario_planned_trajectories_df = planned_trajectories_to_df(scenario_planned_trajectories)

            # Compute scenario's interactions and the interaction map --- TO CHECK
            collisions_df, issue_rows = augment_with_collisions(executed_scenario, scenario_collisions)

            # Save scenario's collisions
            scenario_collisions_file_path = os.path.join(executed_scenario_folder, "collisions.csv")
            collisions_df.to_csv(scenario_collisions_file_path, index = False)
            # Save issue rows
            if len(issue_rows) > 0:
                issue_rows_df = pd.DataFrame(issue_rows, columns = scenario_collisions.columns.tolist() + ["detected_obstacles"])
                issue_rows_df.to_csv(os.path.join(executed_scenario_folder, "invalid_collisions.csv"), index = False)
            # Open scenario's vehicles perceptions 
            scenario_vehicles_perceptions_file_path = os.path.join(executed_scenario_folder, "vehicles_perception.csv")
            scenario_vehicles_perceptions = pd.read_csv(scenario_vehicles_perceptions_file_path)

            # Open scenario's traffic lights perceptions
            scenario_traffic_lights_perceptions_file_path = os.path.join(executed_scenario_folder, "traffic_lights_perceptions.csv")
            scenario_traffic_lights_perceptions = pd.read_csv(scenario_traffic_lights_perceptions_file_path)
            interactions_df, interactions_map = augment_with_interactions(executed_scenario,
                                                                          scenario_planned_trajectories_df, 
                                                                          scenario_vehicles_perceptions, 
                                                                          scenario_traffic_lights_perceptions,
                                                                          scenario_collisions
                                                                        )

            interactions_df.to_csv(interactions_file_path, index = False)
            np.savetxt(
                interactions_map_file_path,
                interactions_map,
                delimiter = ",",
                fmt = "%.6f"
            )

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python convert_experiments.py <record_files_folder> <experiment_name> <output_files_folder>")
        sys.exit(1)

    record_files_folder = sys.argv[1]
    experiment_name = sys.argv[2]
    output_files_folder = sys.argv[3]

    analyze_generation(record_files_folder, experiment_name, output_files_folder)