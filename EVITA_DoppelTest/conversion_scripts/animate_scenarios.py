import os
import sys
import matplotlib as plt
plt.use("Agg")

from commonroad.common.file_reader import XMLFileReader

from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.plotting_utils import draw_scenario, load_planned_trajectories


def process_single_scenario(experiment: str, generation: str, scenario: str, output_files_folder: str) -> str:
    """
    Plot and animate converted scenario
    
    :param experiment: Experiment folder name
    :type experiment: str
    :param generation: Generation folder name
    :type generation: str
    :param scenario: Scenario folder name
    :type scenario: str
    :param output_files_folder: Output folder name
    :type output_files_folder: str
    """
    try:
        # Reconstruct scenario's folder
        scenario_folder = os.path.join(output_files_folder, experiment, generation, scenario)
        plot_output_folder = os.path.join(scenario_folder, "plots")

        # Open CommonRoad executed scenario file
        executed_scenario_file = os.path.join(scenario_folder, "executed_scenario.xml")
        executed_scenario, _ = XMLFileReader(executed_scenario_file).open(lanelet_assignment=True)

        # Load scenario vehicles planned trajectories file
        scenario_planned_trajectories_file = os.path.join(scenario_folder, "planned_trajectories.json")
        scenario_planned_trajectories = load_planned_trajectories(scenario_planned_trajectories_file)

        # Check existance of CommonRoad scenario file and draw scenario
        scenario_file = os.path.join(scenario_folder, "scenario.xml")
        if os.path.exists(scenario_file):
            _, planning_problems_set = XMLFileReader(scenario_file).open(lanelet_assignment = True)
            # Plot and animate CommonRoad scenario
            draw_scenario(executed_scenario, plot_output_folder, scenario_planned_trajectories, planning_problems_set)
        else:
            # Plot and animate CommonRoad scenario
            draw_scenario(executed_scenario, plot_output_folder, scenario_planned_trajectories)

        return f"\nDone: {experiment}/{generation}/{scenario}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

#
# ------------------------------
#

# Folder where the experiments folder are
record_files_folder = "../data/records/"
output_files_folder = "../data/evita-records/"

def main(experiment: str):
    tasks = []

    with ProcessPoolExecutor(max_workers = 2) as executor: 
        # Detect all the generation folders in the experiment
        generation_folders = sorted([folder for folder in os.listdir(os.path.join(output_files_folder, experiment)) if os.path.isdir(os.path.join(output_files_folder, experiment, folder))])
        
        for generation in generation_folders:
            # Detect all the scenarios folders generated
            scenarios_folders = sorted([folder for folder in os.listdir(os.path.join(output_files_folder, experiment, generation)) if os.path.isdir(os.path.join(output_files_folder, experiment, generation, folder))])
            for scenario in scenarios_folders:
                future = executor.submit(
                                    process_single_scenario,
                                    experiment,
                                    generation,
                                    scenario,
                                    output_files_folder,
                                )
                tasks.append(future)

        # Show progress
        for f in as_completed(tasks):
            print(f.result())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python animate_scenarios.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    main(experiment_name)