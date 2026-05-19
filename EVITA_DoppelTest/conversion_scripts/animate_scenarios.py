import os
import sys
import matplotlib as plt
plt.use("Agg")

from commonroad.common.file_reader import XMLFileReader

from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.plotting_utils import draw_scenario, load_planned_trajectories


def process_single_scenario(experiment_folder_path: str, generation: str, scenario: str) -> str:
    """
    Plot and animate converted scenario
    
    :param experiment_folder_path: Path to the experiment folder
    :type experiment_folder_path: str
    :param generation: Generation folder name
    :type generation: str
    :param scenario: Scenario folder name
    :type scenario: str
    """
    try:
        # Reconstruct scenario's folder
        scenario_folder = os.path.join(experiment_folder_path, generation, scenario)
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

        return f"\nDone: {experiment_folder_path}/{generation}/{scenario}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

def main(experiment_folder_path: str, max_workers: int = 3):
    """
    Main function to plot and animate converted scenarios in parallel
    """
    tasks = []

    with ProcessPoolExecutor(max_workers = max_workers) as executor: 
        # Detect all the generation folders in the experiment
        generation_folders = sorted([folder for folder in os.listdir(experiment_folder_path) if os.path.isdir(os.path.join(experiment_folder_path, folder))])
        # Iterate over each generation folder    
        for generation in generation_folders:
            # Detect all the scenarios folders generated
            scenarios_folders = sorted([folder for folder in os.listdir(os.path.join(experiment_folder_path, generation)) if os.path.isdir(os.path.join(experiment_folder_path, generation, folder))])
            for scenario in scenarios_folders:
                # Submit a task to animate the scenario in parallel
                future = executor.submit(
                                    process_single_scenario,
                                    experiment_folder_path,
                                    generation,
                                    scenario,
                                )
                tasks.append(future)

        # Show progress
        for f in as_completed(tasks):
            print(f.result())

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) not in [2, 3]:
        print("Usage: python animate_scenarios.py <experiment_folder_path> [<max_workers> (default = 3)]")
        sys.exit(1)

    experiment_folder_path = sys.argv[1]
    if len(sys.argv) == 3:
        max_workers = int(sys.argv[2])
        main(experiment_folder_path, max_workers)
    else:
        main(experiment_folder_path)