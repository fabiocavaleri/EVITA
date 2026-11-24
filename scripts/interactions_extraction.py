"""
Interactions and collisions of executed scenarios, for two studies:
  - interaction diversity vs criticality: interaction maps of scenarios with and without a collision, per number of vehicles, on samples of equal size;
  - evasive maneuvers (qualitative): for each cell of the interaction map, the plans before and after the replanning of sampled interactions, overlaid in the vehicle's frame.

Three steps, run in order; each reads the outputs of the previous one:
    python interactions_extraction.py extract     --experiments results/EVITA-Experiments [--workers 8]
    python interactions_extraction.py criticality --experiments results/EVITA-Experiments [--sample-size 100 --seed 42]
    python interactions_extraction.py maneuvers  --experiments results/EVITA-Experiments [--frameworks EVITA]

Expected layout: <experiments>/<framework>/<map>/<run>/<scenario>/logs/simulation.db (and logs/<vehicle>/logs.csv).
Outputs, in <experiments>:
    aggregated_interactions.csv, aggregated_collisions.csv, scenarios.csv, interaction_map.txt   (extract)
    collisions_interaction_maps/                                                                  (criticality)
    interaction_plots/                                                                            (maneuvers)
"""
import os
import sys
import shutil
import sqlite3
import argparse
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from commonroad.scenario.obstacle import ObstacleType
from commonroad.visualization.icons import get_obstacle_icon_patch

from evita.config import ScenarioGeneratorConfig
from evita.data_analysis import _rebuild_trajectory, get_interaction_map_bins
from evita.generators.utils import crop_trajectories

from scripts.generate_plots import plot_interaction_map

# ========== Constants ==========

# Columns read from each scenario's simulation database
DB_COLUMNS = ['scenario', 'agent_id', 'timestep', 'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat',
              'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location']

# Columns of the aggregated tables
INTERACTIONS_COLUMNS = ['framework', 'map', 'run', 'scenario_folder_path', 'scenario', 'agent_id', 'timestep',
                        'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat',
                        'interaction_map_index_long', 'interaction_map_index_lat']
COLLISIONS_COLUMNS = ['framework', 'map', 'run', 'scenario_folder_path', 'scenario', 'agent_id', 'timestep',
                      'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location', 'num_vehicles']
SCENARIOS_COLUMNS = ['framework', 'map', 'run', 'scenario_folder_path', 'num_vehicles', 'collided']

# Columns of a vehicle's logs.csv holding its planned trajectory
TRAJECTORY_COLUMNS = ['x_positions_m', 'y_positions_m', 'theta_orientations_rad', 'velocities_mps']

# Folders of <experiments>, <framework> and <map> that are not data
SKIPPED_FOLDERS = {"interaction_plots", "collisions_interaction_maps", "comparison-plots", "experiments_analysis", "interactions_collisions_analysis", "debug", "checkpoints"}

# Output folder of the aggregated tables (extract)
EXTRACT_FOLDER = "interactions_collisions_analysis"

# File names
INTERACTIONS_CSV = "aggregated_interactions.csv"
COLLISIONS_CSV = "aggregated_collisions.csv"
SCENARIOS_CSV = "scenarios.csv"

def interaction_map_bins(n_bins: int):
    """
    Bin edges of the interaction map (as EVITA computes them) and the map's shape: (long_bins, lat_bins, rows, cols).
    """
    long_bins, lat_bins = get_interaction_map_bins(n_bins)

    return long_bins, lat_bins, len(long_bins) + 1, len(lat_bins) + 1

def subfolders(path: str):
    """
    Data subfolders of `path`, sorted (folders in SKIPPED_FOLDERS excluded).
    """
    return sorted(f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f not in SKIPPED_FOLDERS)

def accumulate_map(diff_long_index, diff_lat_index, rows: int, cols: int) -> np.ndarray:
    """
    Interaction map (counts per cell) from the cell indices of interactions.
    """
    interaction_map = np.zeros((rows, cols), dtype = np.int64)
    np.add.at(interaction_map, (np.asarray(diff_long_index, dtype = int), np.asarray(diff_lat_index, dtype = int)), 1)

    return interaction_map

def write_map_table(interaction_map: np.ndarray, path: str) -> None:
    """
    Interaction map as a text table, rows LONG_<i> and columns LAT_<j>.
    """
    table = pd.DataFrame(interaction_map).astype(int)
    table.index = [f"LONG_{i}" for i in range(len(table))]
    table.columns = [f"LAT_{j}" for j in range(len(table.columns))]

    with open(path, "w") as handle:
        handle.write(table.to_string())

# ========== Step 1: extract ==========

def extract_scenario(task):
    """
    Interactions and collisions of one executed scenario, from its simulation database.
    Returns (interactions, collisions, scenario row) or None if the database is missing or incomplete.
    """
    scenario_path, framework, map_name, run, long_bins, lat_bins = task

    simulation_db_path = os.path.join(scenario_path, "logs", "simulation.db")
    if not os.path.exists(simulation_db_path):
        return None

    connection = sqlite3.connect(simulation_db_path)
    try:
        criticality_df = pd.read_sql_query("SELECT * FROM scenario_evaluation", connection)
    except Exception:
        return None
    finally:
        connection.close()

    # Check that the database has the expected columns, otherwise skip this scenario
    if len(criticality_df) == 0 or not all(column in criticality_df.columns for column in DB_COLUMNS):
        return None

    # Extract scenario information
    criticality_df = criticality_df[DB_COLUMNS].copy()
    criticality_df['framework'] = framework
    criticality_df['map'] = map_name
    criticality_df['run'] = run
    criticality_df['scenario_folder_path'] = scenario_path
    num_vehicles = int(criticality_df['agent_id'].nunique())
    criticality_df['num_vehicles'] = num_vehicles

    # Extract scenario interactions and collisions, and compute the interaction map indices of the interactions
    interactions = criticality_df[criticality_df['is_interacting'] == 1.0].copy()
    interactions['interaction_map_index_long'] = np.digitize(interactions['diff_long'].to_numpy(dtype = float), long_bins)
    interactions['interaction_map_index_lat'] = np.digitize(interactions['diff_lat'].to_numpy(dtype = float), lat_bins)

    collisions = criticality_df[criticality_df['has_collided'] == 1.0][COLLISIONS_COLUMNS]

    # A scenario is a collision scenario if any vehicle collided, and a no-collision scenario otherwise (never both)
    scenario_row = {'framework': framework, 'map': map_name, 'run': run, 'scenario_folder_path': scenario_path, 'num_vehicles': num_vehicles, 'collided': bool(len(collisions) > 0)}

    return interactions[INTERACTIONS_COLUMNS], collisions, scenario_row

def extract(args) -> int:
    """
    Read every scenario's simulation database and write the aggregated interactions, collisions and scenarios, and the interaction map of all interactions.
    """

    experiments = args.experiments
    output_folder = os.path.join(experiments, EXTRACT_FOLDER)
    os.makedirs(output_folder, exist_ok = True)
    long_bins, lat_bins, rows, cols = interaction_map_bins(args.n_bins)

    # Skip if the aggregated tables already exist, unless --force
    if os.path.exists(os.path.join(output_folder, INTERACTIONS_CSV)) and not args.force:
        print(f"{INTERACTIONS_CSV} exists in {output_folder}: use --force to extract again")
        return 1

    # Initialize parallel analysis tasks: one per scenario folder, with its framework, map and run names, and the interaction map bins
    tasks = []
    for framework in subfolders(experiments):
        for map_name in subfolders(os.path.join(experiments, framework)):
            for run in subfolders(os.path.join(experiments, framework, map_name)):
                run_path = os.path.join(experiments, framework, map_name, run)
                for scenario in subfolders(run_path):
                    tasks.append((os.path.join(run_path, scenario), framework, map_name, run, long_bins, lat_bins))
    if not tasks:
        print(f"No scenario folders found in {experiments}")
        return 1

    # Initialize the aggregated tables
    interactions, collisions, scenarios, missing = [], [], [], 0

    # Run the extraction in parallel, with a progress bar
    with ProcessPoolExecutor(max_workers = args.workers) as executor:
        for result in tqdm(executor.map(extract_scenario, tasks, chunksize = 16), total = len(tasks), desc = "Scenarios"):
            if result is None:
                missing += 1
                continue

            scenario_interactions, scenario_collisions, scenario_row = result
            interactions.append(scenario_interactions)
            collisions.append(scenario_collisions)
            scenarios.append(scenario_row)

    # Concatenate the results into DataFrames, or create empty DataFrames if no data was found
    if interactions:
        interactions_df = pd.concat(interactions, ignore_index = True)
    else:
        interactions_df = pd.DataFrame(columns = INTERACTIONS_COLUMNS)

    if collisions:
        collisions_df = pd.concat(collisions, ignore_index = True)
    else:
        collisions_df = pd.DataFrame(columns = COLLISIONS_COLUMNS)
    
    scenarios_df = pd.DataFrame(scenarios, columns = SCENARIOS_COLUMNS)

    # Write the aggregated tables and the interaction map of all interactions
    interactions_df.to_csv(os.path.join(output_folder, INTERACTIONS_CSV), index = False)
    collisions_df.to_csv(os.path.join(output_folder, COLLISIONS_CSV), index = False)
    scenarios_df.to_csv(os.path.join(output_folder, SCENARIOS_CSV), index = False)
    write_map_table(accumulate_map(interactions_df['interaction_map_index_long'], interactions_df['interaction_map_index_lat'], rows, cols), os.path.join(output_folder, "interaction_map.txt"))

    print(f"{len(scenarios_df)} scenarios ({int(scenarios_df['collided'].sum())} with a collision), {len(interactions_df)} interactions, {len(collisions_df)} collision rows; {missing} scenarios without an evaluated database")

    return 0

# ========== Step 2: criticality ==========

def criticality(args) -> int:
    """
    For each number of vehicles: draw the same number of collision and no-collision scenarios, and write the interaction map of each group (table and figure).
    """
    experiments = args.experiments
    output_folder = os.path.join(experiments, EXTRACT_FOLDER)
    long_bins, lat_bins, rows, cols = interaction_map_bins(args.n_bins)

    # Check if the aggregated tables exist, otherwise skip this step
    for name in (INTERACTIONS_CSV, SCENARIOS_CSV):
        if not os.path.exists(os.path.join(output_folder, name)):
            print(f"No {name} in {output_folder}: run `extract` first")
            return 1

    # Read the aggregated tables
    interactions_df = pd.read_csv(os.path.join(output_folder, INTERACTIONS_CSV))
    scenarios_df = pd.read_csv(os.path.join(output_folder, SCENARIOS_CSV))

    # Filter by framework if requested
    if args.frameworks:
        interactions_df = interactions_df[interactions_df['framework'].isin(args.frameworks)]
        scenarios_df = scenarios_df[scenarios_df['framework'].isin(args.frameworks)]

    # Create the output folder for the interaction maps of collision and no-collision scenarios, removing any previous content
    output_folder = os.path.join(experiments, EXTRACT_FOLDER, "collisions_interaction_maps")
    shutil.rmtree(output_folder, ignore_errors = True)
    os.makedirs(output_folder, exist_ok = True)

    # Group the interactions by scenario folder path
    interactions_by_scenario = {path: group for path, group in interactions_df.groupby('scenario_folder_path')}

    summary = []

    # Iterate over the number of vehicles, skipping 0 and 1 (no interaction possible)
    for num_vehicles in sorted(scenarios_df['num_vehicles'].unique()):
        if num_vehicles < 2:
            continue

        filtered_scenarios = scenarios_df[scenarios_df['num_vehicles'] == num_vehicles]
        collision_scenarios = filtered_scenarios[filtered_scenarios['collided']]
        no_collision_scenarios = filtered_scenarios[~filtered_scenarios['collided']]

        # Sample the same number of collision and no-collision scenarios, up to --sample-size
        sample_size = min(args.sample_size, len(collision_scenarios), len(no_collision_scenarios))
        
        summary.append({'num_vehicles': int(num_vehicles), 'collision_scenarios': len(collision_scenarios), 'no_collision_scenarios': len(no_collision_scenarios), 'sample_size': sample_size})
        
        # Skip if there are no scenarios to sample
        if sample_size == 0:
            print(f"{num_vehicles} vehicles: {len(collision_scenarios)} collision and {len(no_collision_scenarios)} no-collision scenarios; skipped")
            continue

        # Sample and write the interaction map for each group (collision and no-collision)
        for group, group_scenarios in (("collisions", collision_scenarios), ("no_collisions", no_collision_scenarios)):
            sample = group_scenarios.sample(n = sample_size, random_state = args.seed)

            # Get the interactions of the sampled scenarios, if they exist
            sampled = [interactions_by_scenario[p] for p in sample['scenario_folder_path'] if p in interactions_by_scenario]
            if sampled:
                group_interactions = pd.concat(sampled)
            else:
                group_interactions = interactions_df.iloc[0:0]

            # Compute the interaction map of the sampled interactions and write it to a text file and a figure
            group_map = accumulate_map(group_interactions['interaction_map_index_long'], group_interactions['interaction_map_index_lat'], rows, cols)

            output_path = os.path.join(output_folder, f"{group}_interaction_map_{num_vehicles}_vehicles.txt")
            write_map_table(group_map, output_path)

            figure = plot_interaction_map(group_map, long_bins, lat_bins)
            figure.savefig(output_path.replace(".txt", ".png"), dpi = args.dpi)
            plt.close(figure)

    pd.DataFrame(summary).to_csv(os.path.join(output_folder, "samples.csv"), index = False)
    print(f"Interaction maps written to {output_folder} (sample sizes in samples.csv)")

    return 0

# ========== Step 3: maneuvers ==========

def trajectory_to_array(trajectory) -> np.ndarray:
    """
    CommonRoad Trajectory as an (N, 4) array [x, y, theta, v], one row per state.
    """
    return np.array([[s.position[0], s.position[1], s.orientation, s.velocity] for s in trajectory.state_list])

def to_ego_frame(trajectory_array: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
    """
    States in the frame of `reference_pose` ([x, y, theta]): origin at its position, +x along its heading.
    Columns become [longitudinal, lateral, relative heading wrapped to [-pi, pi], speed].
    """
    cos_heading, sin_heading = np.cos(reference_pose[2]), np.sin(reference_pose[2])
    offset = trajectory_array[:, :2] - reference_pose[:2]

    states_ego = trajectory_array.copy()
    states_ego[:, 0] = cos_heading * offset[:, 0] + sin_heading * offset[:, 1]     # ahead of the reference
    states_ego[:, 1] = -sin_heading * offset[:, 0] + cos_heading * offset[:, 1]    # left of the reference
    heading_difference = trajectory_array[:, 2] - reference_pose[2]
    states_ego[:, 2] = np.arctan2(np.sin(heading_difference), np.cos(heading_difference))

    return states_ego

def normalize_pair(cropped_plan_t, cropped_plan_t_1, common_reference: bool = True):
    """
    The plan computed at t (old) and at t + 1 (new), aligned by crop_trajectories, in the vehicle's frame.
    common_reference = True: both in the frame of the new plan's first state (the new plan starts at (0, 0));
    False: each in the frame of its own first state.
    """

    old_array, new_array = trajectory_to_array(cropped_plan_t), trajectory_to_array(cropped_plan_t_1)
    reference_new = new_array[0, :3]
    reference_old = reference_new if common_reference else old_array[0, :3]

    return to_ego_frame(old_array, reference_old), to_ego_frame(new_array, reference_new)

def draw_ego_icon(ax, vehicle_length: float, vehicle_width: float) -> None:
    """
    Vehicle icon at the origin of the ego frame, heading +x.
    """
    icon_patch_list = get_obstacle_icon_patch(ObstacleType.CAR, 0.0, 0.0, 0.0, vehicle_length * 0.5, vehicle_width * 0.5,
                                              vehicle_color = (1, 1, 1, 0.6), edgecolor = "black", lw = 0.5, zorder = 200)
    ax.add_collection(mpl.collections.PatchCollection(icon_patch_list, match_original = True))

def plot_cell_trajectories(pairs, i, j, save_dir, x_limits, y_limits, dpi, vehicle_length = 4.5, vehicle_width = 1.8) -> None:
    """
    One figure per cell: plan at t (grey), plan at t + 1 (blue), end of the plan at t + 1 (red), vehicle at the
    origin. Lines are semi-transparent, so consistent reactions pile up darker.
    """
    if not pairs:
        return

    fig, ax = plt.subplots(figsize = (8, 5))
    for old, new in pairs:
        ax.plot(old[:, 0], old[:, 1], color = "gray", alpha = 0.3, linewidth = 1)
        ax.plot(new[:, 0], new[:, 1], color = "tab:blue", alpha = 0.3, linewidth = 1)
        ax.scatter(new[-1, 0], new[-1, 1], color = "tab:red", s = 8, zorder = 3)
    draw_ego_icon(ax, vehicle_length, vehicle_width)

    # One legend entry per line type
    ax.plot([], [], color = "gray", label = "Plan at t")
    ax.plot([], [], color = "tab:blue", label = "Plan at t + 1")
    ax.scatter([], [], color = "tab:red", s = 8, label = "End of plan at t + 1")
    ax.legend(loc = "upper right")

    ax.set_xlabel("Longitudinal [m]")
    ax.set_ylabel("Lateral [m]")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable = "box")
    ax.set_title(f"Interaction map cell LONG_{i} / LAT_{j}  (n = {len(pairs)})")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"LONG{i}_LAT{j}.png"), dpi = dpi)
    plt.close(fig)

def plot_cell(task):
    """
    Plan pairs of the sampled interactions of one cell, and their figure.
    Returns (i, j, sampled, plotted, without logs, initial offsets).
    """
    sample_df, i, j, plots_folder, agent_id_offset, x_limits, y_limits, dpi = task

    # Initialize counters
    pairs, initial_offsets, no_logs = [], [], 0

    # Each vehicle's logs.csv is read once for all its samples
    for (scenario_folder_path, agent_id), agent_samples in sample_df.groupby(['scenario_folder_path', 'agent_id']):
        logs_file_path = os.path.join(scenario_folder_path, "logs", str(agent_id), "logs.csv")
        try:
            logs = pd.read_csv(logs_file_path, delimiter = ";", quotechar = '"')
        except FileNotFoundError:
            no_logs += len(agent_samples)   # no logs folder for this id: see --agent-id-offset
            continue

        # Extract agent's timesteps
        logs["trajectory_number"] = logs["trajectory_number"].astype(int)
        plans = logs.drop_duplicates("trajectory_number", keep = "last").set_index("trajectory_number")[TRAJECTORY_COLUMNS]

        # For each timestep, rebuild the planned trajectory at t and t + 1
        for timestep in agent_samples['timestep'].astype(int):
            # Skip if the plan at t or t + 1 is missing: nothing to compare
            if timestep not in plans.index or (timestep + 1) not in plans.index:
                continue
            
            plan_t = _rebuild_trajectory(timestep, *plans.loc[timestep, TRAJECTORY_COLUMNS])
            plan_t_1 = _rebuild_trajectory(timestep + 1, *plans.loc[timestep + 1, TRAJECTORY_COLUMNS])

            # Crop planned trajectories
            cropped_plan_t_1, cropped_plan_t = crop_trajectories(plan_t, plan_t_1)
            
            if cropped_plan_t is None:
                continue
            
            # Normalize planned trajectories (so they have the same point as origin)
            old, new = normalize_pair(cropped_plan_t, cropped_plan_t_1, common_reference = True)
            pairs.append((old, new))
            initial_offsets.append(float(np.linalg.norm(new[0, :2] - old[0, :2])))

    # Plot the plans of all sampled interactions of this cell, in the vehicle's frame
    plot_cell_trajectories(pairs, i, j, plots_folder, x_limits, y_limits, dpi)

    return i, j, len(sample_df), len(pairs), no_logs, initial_offsets

def maneuvers(args) -> int:
    """
    For each cell of the interaction map: at most one interaction per (scenario, agent), its first in the cell,
    then up to --sample-size at random (--seed); plot their plans at t and t + 1.
    """

    experiments = args.experiments
    output_folder = os.path.join(experiments, EXTRACT_FOLDER)
    long_bins, lat_bins, rows, cols = interaction_map_bins(args.n_bins)

    # Skip if the aggregated interactions table does not exist, otherwise read it
    if not os.path.exists(os.path.join(output_folder, INTERACTIONS_CSV)):
        print(f"No {INTERACTIONS_CSV} in {output_folder}: run `extract` first")
        return 1

    interactions_df = pd.read_csv(os.path.join(output_folder, INTERACTIONS_CSV))
    
    if args.frameworks:
        interactions_df = interactions_df[interactions_df['framework'].isin(args.frameworks)]
        print(f"Interactions of {args.frameworks} only: {len(interactions_df)}")

    # Create the output folder for the interaction plots, removing any previous content
    plots_folder = os.path.join(output_folder, "interaction_plots")
    shutil.rmtree(plots_folder, ignore_errors = True)
    os.makedirs(plots_folder, exist_ok = True)

    # One maneuver is detected over several consecutive time steps: keep each (scenario, agent)'s first interaction per cell
    candidates = (interactions_df.sort_values('timestep').drop_duplicates(subset = ['scenario_folder_path', 'agent_id', 'interaction_map_index_long', 'interaction_map_index_lat'], keep = 'first'))
    cells = {cell: group for cell, group in candidates.groupby(['interaction_map_index_long', 'interaction_map_index_lat'])}

    # Initialize parallel plotting tasks: one per cell, with its sampled interactions and the output folder
    tasks = []
    for i in range(rows):
        for j in range(cols):
            cell = cells.get((i, j))

            # Skip empty cells (no interactions)
            if cell is None:
                continue
            
            # Sample up to --sample-size interactions per cell, at random without replacement
            sample_df = cell.sample(n = min(args.sample_size, len(cell)), random_state = args.seed)
            tasks.append((sample_df, i, j, plots_folder, args.agent_id_offset, tuple(args.x_limits), tuple(args.y_limits), args.dpi))

    # Initialize the counters of the total number of samples, the number of samples without logs, and the number of figures written
    initial_offsets, total_no_logs, figures = [], 0, 0

    with ProcessPoolExecutor(max_workers = args.workers) as executor:
        for i, j, sampled, plotted, no_logs, offsets in tqdm(executor.map(plot_cell, tasks), total = len(tasks), desc = "Cells"):
            initial_offsets.extend(offsets)
            total_no_logs += no_logs
            figures += plotted > 0

            if no_logs:
                tqdm.write(f"Cell LONG_{i} / LAT_{j}: {plotted} of {sampled} samples plotted, {no_logs} without logs")

    print(f"{len(tasks)} cells with interactions, {figures} figures written to {plots_folder}")
    if total_no_logs:
        print(f"WARNING: {total_no_logs} samples had no logs folder: check --agent-id-offset (issue 2)")

    # If this is ~0, the vehicle tracks its plan closely and both reference options give practically the same plots
    if initial_offsets:
        print(f"Initial offset between the plans [m]: median = {np.median(initial_offsets):.3f}, max = {np.max(initial_offsets):.3f}")
    
    return 0

# ========== Command line ==========

def build_parser() -> argparse.ArgumentParser:
    # Common arguments
    common = argparse.ArgumentParser(add_help = False)
    common.add_argument("--experiments", required = True, help = "Experiments folder: <framework>/<map>/<run>/<scenario>")
    common.add_argument("--n-bins", type = int, default = ScenarioGeneratorConfig().n_bins, help = "Number of bin edges per axis of the interaction map (default: EVITA's configuration)")
    common.add_argument("--workers", type = int, default = 5, help = "Number of parallel processes (default: 5)")

    # Sampling arguments
    sampling = argparse.ArgumentParser(add_help = False)
    sampling.add_argument("--sample-size", type = int, default = 100, help = "Number of scenarios per group (criticality) or interactions per cell (maneuvers)")
    sampling.add_argument("--seed", type = int, default = 42, help = "Seed of the random samples")
    sampling.add_argument("--frameworks", nargs = "+", default = None, help = "Analyze only these frameworks (default: all)")
    sampling.add_argument("--dpi", type = int, default = 500, help = "Resolution of the figures")

    parser = argparse.ArgumentParser(description = "Interactions and collisions of executed scenarios: extraction, diversity vs criticality, evasive maneuvers.")
    commands = parser.add_subparsers(dest = "command", required = True)

    extract_parser = commands.add_parser("extract", parents = [common], help = "Aggregate interactions, collisions and scenarios from the simulation databases")
    extract_parser.add_argument("--force", action = "store_true", help = "Extract again, replacing the aggregated tables")

    commands.add_parser("criticality", parents = [common, sampling], help = "Generate interaction maps of collision and no-collision scenarios, per number of vehicles")

    maneuvers_parser = commands.add_parser("maneuvers", parents = [common, sampling], help = "Analyze plans before and after the replanning, per cell of the interaction map")
    maneuvers_parser.add_argument("--agent-id-offset", type = int, default = 10, help = "Subtracted from the database's agent_id to find logs/<id>/logs.csv")
    maneuvers_parser.add_argument("--x-limits", type = float, nargs = 2, default = [-5, 65], metavar = ("MIN", "MAX"), help = "Longitudinal axis limits [m]")
    maneuvers_parser.add_argument("--y-limits", type = float, nargs = 2, default = [-22, 22], metavar = ("MIN", "MAX"), help = "Lateral axis limits [m]")

    return parser

def main(argv = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.experiments):
        print(f"Experiments folder not found: {args.experiments}")
        return 1
    
    if args.command == "extract":
        return extract(args)
    if args.command == "criticality":
        return criticality(args)
    return maneuvers(args)

if __name__ == "__main__":
    sys.exit(main())