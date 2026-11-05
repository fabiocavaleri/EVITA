import os
import shutil
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib as mpl 
mpl.use("Agg")
import matplotlib.pyplot as plt
 
# Vehicle icon (same helpers you already use for your other plots).
from commonroad.scenario.obstacle import ObstacleType
from commonroad.visualization.icons import get_obstacle_icon_patch

from concurrent.futures import ProcessPoolExecutor

from evita.data_analysis import get_interaction_map_bins, _rebuild_trajectory
from evita.generators.utils import crop_trajectories

# from EVITA_DoppelTest.analysis_scripts.generate_plots import plot_interaction_map

N_BINS = 10
SAMPLE_SIZE = 100
PLOT_X_LIMITS = (-5, 65)     # longitudinal
PLOT_Y_LIMITS = (-22, 22)    # lateral

######################################################################

# ----------------------------------------------------------------------------
# 1. Normalize an (old plan, new plan) pair
# ----------------------------------------------------------------------------
def normalize_pair(cropped_trajectory_t_1, cropped_trajectory_t, common_reference = True):
    """
    Bring the old and new plan of ONE interaction into the same ego frame.
 
    Parameters
    ----------
    cropped_trajectory_t_1, cropped_trajectory_t : Trajectory
        Plans already aligned in time by `crop_trajectories` (same first time step `t` and same length).
    common_reference : bool
        True  -> both plans use the FIRST STATE OF THE NEW PLAN as reference.
                 The new plan starts exactly at (0, 0); the old plan starts
                 near it.
        False -> each plan uses its own first state. Both start at exactly (0, 0).
 
    Returns
    -------
    (old, new): two (N, 4) arrays in the ego frame.
    """

    t_1_trajectory_array = trajectory_to_array(cropped_trajectory_t_1)
    t_trajectory_array = trajectory_to_array(cropped_trajectory_t)
 
    ref_new = t_trajectory_array[0, :3] # Extract position_x, position_y and orientation of vehicle's state at timestep t
    if common_reference:
        ref_old = ref_new
    else:
        ref_old = t_1_trajectory_array[0, :3] # Extract planned position_x, position_y and orientation for vehicle at timestep t
 
    return to_ego_frame(t_1_trajectory_array, ref_old), to_ego_frame(t_trajectory_array, ref_new)

# ----------------------------------------------------------------------------
# 2. Trajectory -> numpy
# ----------------------------------------------------------------------------
def trajectory_to_array(trajectory):
    """
    Convert a CommonRoad `Trajectory` into an (N, 4) array with columns
    [x, y, theta, v], one row per state.
 
    Working with plain arrays makes the geometry below (rotations, differences)
    simple and fast.
    """

    return np.array([[s.position[0], s.position[1], s.orientation, s.velocity] for s in trajectory.state_list])

# ----------------------------------------------------------------------------
# 3. World frame -> ego frame
# ----------------------------------------------------------------------------
def to_ego_frame(trajectory_array, reference_pose):
    """
    Express `trajectory_array` in the ego frame defined by `reference_pose`.
 
    Parameters
    ----------
    states_world : (N, 4) array
        One row per state, columns [x, y, theta, v], in world coordinates.
    reference_pose : (3,) array
        [x, y, theta] of the vehicle at the reference moment (world coordinates).
        It defines the new frame:
            * the origin of the new frame is the position of `reference_pose`
            * the +x axis of the new frame is the heading of `reference_pose`
 
    Returns
    -------
    states_ego : (N, 4) array
        Same states as columns [longitudinal, lateral, relative_theta, v].
 
    Step by step:
        (a) translate: subtract the reference position -> the reference becomes (0, 0)
        (b) rotate by -reference_heading so the reference heading lines up with +x
 
    After this, longitudinal = offset ahead of / behind the reference and lateral = offset left / right of the reference heading.
    Velocity is a scalar speed, so it does not change with the frame.
    """
    reference_position = reference_pose[:2]     # (x, y) of the reference
    reference_heading = reference_pose[2]       # theta of the reference [rad]
 
    # Components of the vehicle's forward unit vector (cos, sin) in world axes.
    cos_heading = np.cos(reference_heading)
    sin_heading = np.sin(reference_heading)
 
    # (a) translation: vector from the reference position to every state
    offset = trajectory_array[:, :2] - reference_position
    offset_x = offset[:, 0]
    offset_y = offset[:, 1]
 
    states_ego = trajectory_array.copy()
 
    # (b) rotation: project each offset on the vehicle's forward and left axes.
    #     forward unit vector = ( cos, sin)  -> distance ahead of the reference
    #     left    unit vector = (-sin, cos)  -> distance to the left of the reference
    states_ego[:, 0] = cos_heading * offset_x + sin_heading * offset_y     # longitudinal
    states_ego[:, 1] = -sin_heading * offset_x + cos_heading * offset_y    # lateral
 
    # Orientations become relative to the reference heading; wrap to [-pi, pi]
    # so that e.g. 359 deg vs 1 deg is treated as a 2 deg difference.
    heading_difference = trajectory_array[:, 2] - reference_heading
    states_ego[:, 2] = np.arctan2(np.sin(heading_difference), np.cos(heading_difference))
 
    return states_ego

# ----------------------------------------------------------------------------
# 4. Plot all the samples of one interaction-map cell
# ----------------------------------------------------------------------------
def plot_cell_trajectories(pairs, i, j, save_dir, vehicle_length = 4.5, vehicle_width = 1.8):
    """
    One figure per interaction-map cell: the ego vehicle icon at the origin and
    the old (gray) and new (blue) plans of every sampled interaction overlaid.
 
    Reading the plot
    ----------------
    * x axis = longitudinal (the vehicle faces right), y axis = lateral (left is up).
    * gray lines = plan made at t-1 (before the interaction was detected).
    * blue lines = plan made at t   (after the replanning).
    * red dots   = last state of the new plan: a braking reaction ends earlier
                   than a free-flow one, so the dots show the longitudinal effect
                   that the path shape alone can hide.
    * Lines are semi-transparent: where many samples agree they pile up and look
      darker, so the density of the bundle shows how consistent the reaction is.
    """
    if not pairs:
        return
 
    fig, ax = plt.subplots(figsize=(8, 5))
 
    for old, new in pairs:
        # Columns of old / new: 0 = longitudinal, 1 = lateral.
        ax.plot(old[:, 0], old[:, 1], color="gray", alpha=0.3, linewidth=1)
        ax.plot(new[:, 0], new[:, 1], color="tab:blue", alpha=0.3, linewidth=1)
        ax.scatter(new[-1, 0], new[-1, 1], color="tab:red", s=8, zorder=3)
 
    # Vehicle icon at the origin, the common starting point of every plan
    # (reuses draw_ego_icon, defined in the grid section below).
    draw_ego_icon(ax, vehicle_length, vehicle_width)
 
    # Dummy handles so the legend has one entry per line type, not one per sample.
    ax.plot([], [], color="gray", label="Old plan (t-1)")
    ax.plot([], [], color="tab:blue", label="New plan (t)")
    ax.scatter([], [], color="tab:red", s=8, label="End of new plan")
    ax.legend(loc="upper right")
 
    ax.set_xlabel("Longitudinal [m]")
    ax.set_ylabel("Lateral [m]")

    ax.set_xlim(*PLOT_X_LIMITS)
    ax.set_ylim(*PLOT_Y_LIMITS)

    # Equal scaling so 1 m looks the same on both axes (swerves are not distorted
    # and the vehicle icon keeps its proportions).
    ax.set_aspect("equal", adjustable = "box")

    ax.autoscale_view()  # include the icon, which was added with add_collection
    ax.set_title(f"Interaction map cell LONG_{i} / LAT_{j}  (n={len(pairs)})")
 
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"LONG{i}_LAT{j}.png"), dpi = 400)
    plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Grid figure laid out like the interaction map
# ----------------------------------------------------------------------------
def draw_ego_icon(ax, vehicle_length, vehicle_width):
    """
    Draw the ego vehicle icon at the origin of the ego frame, heading +x
    (the plans are normalized so the vehicle always starts at (0, 0) facing +x).
    """
    icon_patch_list = get_obstacle_icon_patch(
        ObstacleType.CAR,
        0.0,                       # center_x: origin of the ego frame
        0.0,                       # center_y
        0.0,                       # orientation: +x, by construction of the ego frame
        vehicle_length * 0.5,
        vehicle_width * 0.5,
        vehicle_color = (1, 1, 1, 0.6),
        edgecolor = "black",
        lw = 0.5,
        zorder = 200,                 # keep the icon above the trajectories
    )

    ax.add_collection(mpl.collections.PatchCollection(icon_patch_list, match_original = True))

def plot_interaction_map_grid(args):
    sample_df, i, j, plots_folder = args

    initial_offsets = []
    cell_pairs = []
    
    for _, row in sample_df.iterrows():
        scenario_folder_path = row['scenario_folder_path']
        agent_id = int(row['agent_id']) - 10
        timestep = int(row['timestep'])

        logs_file_path = os.path.join(scenario_folder_path, "logs", str(agent_id), "logs.csv")

        try:
            logs = pd.read_csv(logs_file_path, delimiter = ";", quotechar = '"')
            logs["trajectory_number"] = logs["trajectory_number"].astype(int)

            log_at_time_t = logs[logs["trajectory_number"] == timestep][trajectory_columns].iloc[0]
            log_at_time_next_t = logs[logs["trajectory_number"] == (timestep + 1)][trajectory_columns].iloc[0]
        except (FileNotFoundError, IndexError):
            continue  # no log, or no plan at t - 1: nothing to compare

        trajectory_at_time_t = _rebuild_trajectory(timestep, log_at_time_t['x_positions_m'], log_at_time_t['y_positions_m'], log_at_time_t['theta_orientations_rad'], log_at_time_t['velocities_mps'])
        trajectory_at_time_next_t = _rebuild_trajectory((timestep + 1), log_at_time_next_t['x_positions_m'], log_at_time_next_t['y_positions_m'], log_at_time_next_t['theta_orientations_rad'], log_at_time_next_t['velocities_mps'])

        cropped_trajectory_at_time_next_t, cropped_trajectory_at_time_t = crop_trajectories(trajectory_at_time_t, trajectory_at_time_next_t)

        old, new = normalize_pair(cropped_trajectory_at_time_t, cropped_trajectory_at_time_next_t, common_reference = True)
        cell_pairs.append((old, new))
        initial_offsets.append(np.linalg.norm(new[0, :2] - old[0, :2]))

    plot_cell_trajectories(cell_pairs, i, j, plots_folder)

    return initial_offsets

######################################################################

def retrieve_interaction_map(scenarios_sample, interactions_df, long_bins, lat_bins, rows, cols, output_path):
    
    local_interaction_map = np.zeros((rows, cols), dtype = np.int64)
    
    for _, row in scenarios_sample.iterrows():
        scenario_folder_path = row['scenario_folder_path']
        scenario_interactions_df = interactions_df[interactions_df['scenario_folder_path'] == scenario_folder_path].copy()

        scenario_interactions_df['interaction_map_index_long'] = np.digitize(scenario_interactions_df['diff_long'].to_numpy(), long_bins)
        scenario_interactions_df['interaction_map_index_lat'] = np.digitize(scenario_interactions_df['diff_lat'].to_numpy(), lat_bins)


        np.add.at(
            local_interaction_map,
            (
                scenario_interactions_df['interaction_map_index_long'].to_numpy(),
                scenario_interactions_df['interaction_map_index_lat'].to_numpy()
            ),
            1
        )

    local_interaction_map_df = pd.DataFrame(local_interaction_map).astype(int)

    interaction_map_fig = plot_interaction_map(local_interaction_map, long_bins, lat_bins)
    interaction_map_fig.savefig(output_path.replace(".txt", ".png"), dpi = 400)
    plt.close(interaction_map_fig)

    local_interaction_map_df.index = [f"LONG_{i}" for i in range(len(local_interaction_map_df))]
    local_interaction_map_df.columns = [f"LAT_{i}" for i in range(len(local_interaction_map_df.columns))]

    with open(output_path, "w") as f:
        f.write(local_interaction_map_df.to_string())

import matplotlib.cm as cm
from evita.config import ScenarioGeneratorConfig

def plot_interaction_map(interaction_map, long_bins, lat_bins, plot_title: str = "Interaction Effects Map"):
    """
    Plots the interaction map
    """
    # Initialize scenario configuration object
    config = ScenarioGeneratorConfig()

    n_rows, n_cols = interaction_map.shape      # = len(long_bins) + 1, len(lat_bins) + 1

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize = (8, 4))

    # Get the colormap and set the under and bad colors.
    colMap = cm.inferno_r.copy()
    colMap.set_bad(color = 'white')
    colMap.set_under(color = 'white')

    # Prepare data for plotting: empty cells are shown in white
    masked_data = np.copy(interaction_map).astype(float)
    masked_data[masked_data == 0] = np.nan

    # Plot the heatmap: rows = longitudinal (vertical axis), columns = lateral (horizontal axis)
    ax.imshow(masked_data, cmap = colMap)

    # --- Tick labels -------------------------------------------------------------------------
    # The horizontal axis shows the COLUMNS, which are indexed with lat_bins and the vertical axis the ROWS, indexed with long_bins
    # bin edge k is the boundary between cell k and cell k + 1 -> position k + 0.5.
    ax.set_xticks(np.arange(len(lat_bins)) + 0.5, labels = [f"{v:.2f}" for v in lat_bins])
    ax.set_yticks(np.arange(len(long_bins)) + 0.5, labels = [f"{v:.2f}" for v in long_bins])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False, labelsize = 6)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation = -30, ha = "right", rotation_mode = "anchor")

    # Say what each axis is (it was easy to mix them up).
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Lateral difference [m]", fontsize = 7, labelpad = 18)
    ax.set_ylabel("Longitudinal difference [m]", fontsize = 7)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(n_cols + 1) - .5, minor = True)
    ax.set_yticks(np.arange(n_rows + 1) - .5, minor = True)

    ax.grid(which = "both", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which = "minor", bottom = False, left = False)

    # Set plot title
    ax.set_title(plot_title)

    # --- Vehicle icon -----------------------------------------------------------------------
    center_x = np.digitize(0.0, lat_bins)       # column (horizontal)
    center_y = np.digitize(0.0, long_bins)      # row    (vertical)

    # FIX 5: orientation. With orientation 0.0 the icon points along +x of the plot, but +x is the
    #        LATERAL axis. Forward is the longitudinal axis (vertical, increasing downwards because
    #        imshow puts row 0 at the top), so the icon is rotated by +90 degrees.
    #        (If it points the wrong way with your version of get_obstacle_icon_patch, use -np.pi / 2.)
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

######################################################################

def process_scenario_interactions_collisions(args):
    scenario_path, framework, map, run = args

    simulation_db_path = os.path.join(scenario_path, "logs", "simulation.db")
                    
    if not os.path.exists(simulation_db_path):
        print(f"Simulation database not found for scenario: {scenario_path}")
        return None, None, None, None, None
    
    conn = None

    try:
        # Each worker gets its own SQLite connection.
        conn = sqlite3.connect(simulation_db_path)

        criticality_df = pd.read_sql_query("SELECT * FROM scenario_evaluation", conn)

        if len(criticality_df) == 0:
            print(f"No data found in scenario_evaluation for scenario: {scenario_path}")
            return None, None, None, None, None

        if not all(col in criticality_df.columns for col in columns):
            print(f"Missing columns in scenario_evaluation for scenario: {scenario_path}")
            return None, None, None, None, None

        criticality_df = criticality_df[columns]
        criticality_df['framework'] = framework
        criticality_df['map'] = map
        criticality_df['run'] = run
        criticality_df['scenario_folder_path'] = scenario_path
        criticality_df['num_vehicles'] = len(criticality_df['agent_id'].unique())
    finally:
        if conn is not None:
            conn.close()

    local_interactions_df = criticality_df[interactions_columns]
    local_interactions_df = local_interactions_df[local_interactions_df['is_interacting'] == 1.0]

    local_interactions_df['interaction_map_index_long'] = np.digitize(local_interactions_df['diff_long'].to_numpy(), long_bins)
    local_interactions_df['interaction_map_index_lat'] = np.digitize(local_interactions_df['diff_lat'].to_numpy(), lat_bins)

    local_interaction_map = np.zeros((rows, cols), dtype = np.int64)

    np.add.at(
        local_interaction_map,
        (
            local_interactions_df['interaction_map_index_long'].to_numpy(),
            local_interactions_df['interaction_map_index_lat'].to_numpy()
        ),
        1
    )

    local_collisions_df = criticality_df[collisions_columns]
    local_no_collision_scenarios = local_collisions_df[local_collisions_df['has_collided'] == 0.0][['scenario_folder_path', 'num_vehicles']].drop_duplicates()
    local_collisions_df = local_collisions_df[local_collisions_df['has_collided'] == 1.0]
    local_collision_scenarios = local_collisions_df[['scenario_folder_path', 'num_vehicles']].drop_duplicates()

    return local_interactions_df, local_interaction_map, local_collisions_df, local_no_collision_scenarios, local_collision_scenarios

######################################################################

if __name__ == "__main__":
    columns = ['scenario', 'agent_id', 'timestep', 'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat', 'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location']
    interactions_columns = ['scenario_folder_path', 'scenario', 'agent_id', 'timestep', 'is_interacting', 'is_interacting_with', 'diff_long', 'diff_lat']
    collisions_columns = ['scenario_folder_path', 'scenario', 'agent_id', 'timestep', 'has_collided', 'has_collided_with', 'ria', 'riv', 'impact_location', 'num_vehicles']

    interactions_df = pd.DataFrame(columns = interactions_columns)
    collisions_df = pd.DataFrame(columns = collisions_columns)

    long_bins, lat_bins = get_interaction_map_bins(N_BINS)

    # Include 0 - B[0]:B[9] - 10 and len(bins)
    rows = len(long_bins) + 1
    cols = len(lat_bins) + 1

    # Initialize the interaction map and the counters for the requirement violations
    interaction_map = np.zeros((rows, cols))

    experiments_folder_path = "./results/TIAV-Experiments"

    scenario_tasks = []

    if os.path.exists(os.path.join(experiments_folder_path, "aggregated_interactions.csv")):
        interactions_df = pd.read_csv(os.path.join(experiments_folder_path, "aggregated_interactions.csv"))
    else:
        frameworks = [f for f in os.listdir(experiments_folder_path) if os.path.isdir(os.path.join(experiments_folder_path, f)) if f not in ["interaction_plots", "collisions_interaction_maps", "comparison-plots"]]

        for framework in tqdm(frameworks, desc = "Frameworks"):
            framework_path = os.path.join(experiments_folder_path, framework)
            map_folders = [f for f in os.listdir(framework_path) if os.path.isdir(os.path.join(framework_path, f))]
            for map in tqdm(map_folders, desc = f"{framework} Maps", leave = False):
                map_path = os.path.join(framework_path, map)
                
                experiment_runs_folders = [f for f in os.listdir(map_path) if os.path.isdir(os.path.join(map_path, f)) and f not in ["experiments_analysis", "interactions_collisions_analysis"]]
                for run in tqdm(experiment_runs_folders, desc = f"{framework}/{map} Runs", leave = False):
                    run_path = os.path.join(map_path, run)
                    
                    scenarios_folders = [f for f in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, f)) and f != "debug"]
                    for scenario in tqdm(scenarios_folders, desc = f"{framework}/{map}/{run} Scenarios", leave = False):
                        scenario_path = os.path.join(run_path, scenario)

                        scenario_tasks.append((scenario_path, framework, map, run))

        interactions = []
        collisions = []
        no_collision_scenarios = pd.DataFrame(columns = ['scenario_folder_path', 'num_vehicles'])
        collision_scenarios = pd.DataFrame(columns = ['scenario_folder_path', 'num_vehicles'])

        with ProcessPoolExecutor() as executor:
            results = executor.map(process_scenario_interactions_collisions, scenario_tasks)

            for local_interactions_df, local_interaction_map, local_collisions_df, local_no_collision_scenarios, local_collision_scenarios in tqdm(results, total = len(scenario_tasks), desc = "Scenarios"):
                if local_interactions_df is not None and local_collisions_df is not None:
                    interactions.append(local_interactions_df)
                    collisions.append(local_collisions_df)
                    interaction_map += local_interaction_map
                    no_collision_scenarios = pd.concat([no_collision_scenarios, local_no_collision_scenarios], ignore_index = True)
                    collision_scenarios = pd.concat([collision_scenarios, local_collision_scenarios], ignore_index = True)
                else:
                    continue

        interactions_df = pd.concat(interactions, ignore_index = True)
        collisions_df = pd.concat(collisions, ignore_index = True)
        interaction_map_df = pd.DataFrame(interaction_map).astype(int)

        interaction_map_df.index = [f"LONG_{i}" for i in range(len(interaction_map_df))]
        interaction_map_df.columns = [f"LAT_{i}" for i in range(len(interaction_map_df.columns))]

        with open(os.path.join(experiments_folder_path, "interaction_map.txt"), "w") as f:
            f.write(interaction_map_df.to_string())

        interactions_df.to_csv(os.path.join(experiments_folder_path, "aggregated_interactions.csv"), index = False)
        collisions_df.to_csv(os.path.join(experiments_folder_path, "aggregated_collisions.csv"), index = False)

        num_vehicles_range = max(no_collision_scenarios['num_vehicles'].max(), collision_scenarios['num_vehicles'].max())

        output_folder_path = os.path.join(experiments_folder_path, "collisions_interaction_maps")
        shutil.rmtree(output_folder_path, ignore_errors = True)
        os.makedirs(output_folder_path, exist_ok = True)

        for num_vehicles in tqdm(range(2, num_vehicles_range + 1), desc = "Num Vehicles"):
            no_collision_scenarios_filtered = no_collision_scenarios[no_collision_scenarios['num_vehicles'] == num_vehicles]
            collision_scenarios_filtered = collision_scenarios[collision_scenarios['num_vehicles'] == num_vehicles]

            sample_size = min(SAMPLE_SIZE, len(no_collision_scenarios_filtered), len(collision_scenarios_filtered))

            no_collision_scenarios_sample = no_collision_scenarios_filtered.sample(n = sample_size)
            collision_scenarios_sample = collision_scenarios_filtered.sample(n = sample_size)

            no_collisions_interaction_map = retrieve_interaction_map(no_collision_scenarios_sample, interactions_df, long_bins, lat_bins, rows, cols, os.path.join(output_folder_path, f"no_collisions_interaction_map_{num_vehicles}_vehicles.txt"))
            collisions_interaction_map = retrieve_interaction_map(collision_scenarios_sample, interactions_df, long_bins, lat_bins, rows, cols, os.path.join(output_folder_path, f"collisions_interaction_map_{num_vehicles}_vehicles.txt"))

    trajectory_columns = ['x_positions_m', 'y_positions_m', 'theta_orientations_rad', 'velocities_mps']

    scenario_tasks = []
    initial_offsets = []
    plots_folder = os.path.join(experiments_folder_path, "interaction_plots")
    shutil.rmtree(plots_folder, ignore_errors = True)
    os.makedirs(plots_folder, exist_ok = True)

    for i in tqdm(range(rows), desc = "Rows"):
        for j in tqdm(range(cols), desc = "Columns", leave = False):
            interaction_type_df = interactions_df[(interactions_df['interaction_map_index_long'] == i) & (interactions_df['interaction_map_index_lat'] == j)]

            sample_df = interaction_type_df.sample(n = min(SAMPLE_SIZE, len(interaction_type_df)))

            scenario_tasks.append((sample_df, i, j, plots_folder))

    with ProcessPoolExecutor() as executor:
        results = executor.map(plot_interaction_map_grid, scenario_tasks)

        for initial_offset in tqdm(results, total = len(scenario_tasks), desc = "Interaction Map Cells"):
            if initial_offset is not None:
                initial_offsets.extend(initial_offset)

    # If this is ~0, the vehicle tracks its plan closely and both reference options (common / per-trajectory) give practically the same plots.
    if initial_offsets:
        print(f"Initial old-vs-new offset [m]: median = {np.median(initial_offsets):.3f}, max = {np.max(initial_offsets):.3f}")