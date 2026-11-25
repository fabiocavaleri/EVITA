"""
Analysis of the results.

- generate_report(): per-run metrics, written at the end of every run (and at checkpoints) into the run folder:
  requirements_violations.csv (timeouts, missed goals, collisions, unique collisions), behavior.csv (emergencies,
  disengagements, vehicles, interactions, unique interactions), and the collision and interaction maps (.npy).
- compute_effect_size(): Vargha-Delaney effect size, used by scripts/generate_plots.py.

Unique collisions (paper, Section 5.2.2) combine the point of impact of both vehicles, the relative impact velocity
(RIV, 10 partitions of 0-100 km/h plus one for faster impacts) and the relative impact angle (RIA, 10 partitions
of 0-180 degrees plus one). The database stores RIV in m/s and RIA in radians; they are converted when binned.
"""
import os
import ast
import csv
import math
import logging
import sqlite3
from bisect import bisect_left
from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as ss

from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory

from .config import ScenarioGeneratorConfig
from .utils import EVITA_MAIN_LOGGER_NAME, simulation_folders

module_logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)

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

# ========== Per-run report ==========

# Bins of the collision features (paper, Section 5.2.2): 11 edges, i.e., 10 partitions plus one for larger values
COLLISION_FEATURE_EDGES = 11
RIV_BINS_KMH = np.linspace(0, 100, COLLISION_FEATURE_EDGES)
RIA_BINS_DEG = np.linspace(0, 180, COLLISION_FEATURE_EDGES)

# Columns written by EVITA's evaluator: a scenario_evaluation table without them was never evaluated
EVALUATION_COLUMNS = {"is_interacting", "diff_long", "diff_lat", "has_collided", "has_collided_with",
                      "ria", "riv", "impact_location", "most_critical_agent", "has_timeout", "has_missed_goal"}


def _query(simulation_db: str, sql: str, params: tuple = ()) -> list:
    """
    All rows of a query on a simulation database.
    """
    connection = sqlite3.connect(simulation_db)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def count_metric(simulation_db: str, metric: str) -> int:
    """
    Number of (agent, time step) rows where `metric` is 1.0.
    """
    return len(_query(simulation_db, f"SELECT timestep FROM scenario_evaluation WHERE {metric} = 1.0;"))


def count_agents_with(simulation_db: str, metric: str) -> int:
    """
    Number of agents for which `metric` is 1.0 at some time step (for per-agent outcomes, which the evaluator
    repeats on every time step).
    """
    return _query(simulation_db, f"SELECT COUNT(DISTINCT agent_id) FROM scenario_evaluation WHERE {metric} = 1.0;")[0][0]


def count_collisions(simulation_db: str) -> int:
    """
    Number of collisions: distinct groups of vehicles involved in a collision.
    """
    rows = _query(simulation_db, "SELECT agent_id, has_collided_with FROM scenario_evaluation WHERE has_collided = 1.0 GROUP BY agent_id, has_collided_with;")
    return len({str(has_collided_with) for _, has_collided_with in rows})


def _agent_ids_from_db(simulation_db: str) -> List:
    """
    Ids of the simulated agents.
    """
    return ast.literal_eval(_query(simulation_db, "SELECT agent_ids FROM meta;")[0][0])


def _to_float(value) -> Optional[float]:
    """
    A value stored in the database ('1.5', 'None', 'nan', NULL, 1.5) as float, or None.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _interactions_from_db(simulation_db: str) -> List[Tuple[int, int, Optional[float], Optional[float]]]:
    """
    (agent id, time step, diff_long, diff_lat) of every interaction of the scenario.
    """
    rows = _query(simulation_db, "SELECT agent_id, timestep, diff_long, diff_lat FROM scenario_evaluation WHERE is_interacting = 1.0;")
    return [(int(agent_id), int(t), _to_float(diff_long), _to_float(diff_lat)) for agent_id, t, diff_long, diff_lat in rows]


def _get_impacts_features_from_db(simulation_db: str, run_folder: str = None, simulation_folder: str = None) -> List[Tuple[frozenset, float, float]]:
    """
    (points of impact of the two vehicles, RIV in m/s, |RIA| in radians) of each collision of the scenario.
    A collision is identified by the group of vehicles involved; the first entry of each (agent, other vehicle)
    pair is used. Collisions with the road have the point of impact "-+" for the road.
    """
    rows = _query(simulation_db, 'SELECT MIN(timestep), agent_id, most_critical_agent, has_collided_with, riv, ria, impact_location FROM scenario_evaluation WHERE has_collided = 1.0 GROUP BY agent_id, has_collided_with ORDER BY MIN(timestep), has_collided_with;')

    collisions = {}
    impacts = set()

    for _, agent_id, most_critical_agent, has_collided_with_str, riv, ria, impact_location in rows:
        agent_id = int(agent_id)
        has_collided_with_str = str(has_collided_with_str)
        has_collided_with = ast.literal_eval(has_collided_with_str)

        if len(has_collided_with) == 1:
            module_logger.critical(f"Agent {agent_id} collided with the road - {simulation_db}")
            most_critical_agent = "ROAD"
        else:
            if most_critical_agent in (None, "None"):
                module_logger.warning(f"Missing most critical agent for agent {agent_id} in {simulation_db}")
                continue
            most_critical_agent = int(float(most_critical_agent))

        riv, ria = _to_float(riv), _to_float(ria)
        impact_location = str(impact_location) if impact_location not in (None, "None") else "-+"

        # One entry per (agent, other vehicle), the earliest
        if (agent_id, most_critical_agent) in impacts:
            module_logger.debug(f"We already have an entry for {agent_id} - {most_critical_agent} - {impact_location} - {simulation_db}")
            continue
        impacts.add((agent_id, most_critical_agent))

        if has_collided_with_str not in collisions:
            # First vehicle of this collision; for a collision with the road, the road is the second
            collisions[has_collided_with_str] = {"ria": abs(ria) if ria is not None else None,
                                                 "riv": riv,
                                                 "v1_impact_location": impact_location,
                                                 "v2_impact_location": "-+" if most_critical_agent == "ROAD" else None}
        else:
            # Second vehicle of this collision (never the road)
            collisions[has_collided_with_str]["v2_impact_location"] = impact_location

    # A collision whose second vehicle has no entry keeps None, and is skipped by the report
    return [(frozenset((c["v1_impact_location"], c["v2_impact_location"])), c["riv"], c["ria"]) for c in collisions.values()]


def _rebuild_trajectory(initial_time_step, xs: str, ys: str, orientations: str, velocities: str) -> Trajectory:
    """
    Planned trajectory from one row of an agent's logs.csv (comma-separated positions, orientations and velocities,
    the first one at `initial_time_step`).
    """
    initial_time_step = int(initial_time_step)
    state_list = [CustomState(time_step = initial_time_step + idx, position = np.array([float(x), float(y)]), orientation = float(orientation), velocity = float(velocity))
                  for idx, (x, y, orientation, velocity) in enumerate(zip(xs.split(","), ys.split(","), orientations.split(","), velocities.split(",")))]
    return Trajectory(initial_time_step, state_list)


def get_interaction_map_bins(n_bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin edges of the interaction map (as in generators/utils.py compute_interaction_map): n_bins edges per axis,
    one vehicle length each way longitudinally, one lane width (3.7 m) each way laterally.
    """
    scenario_generator_config = ScenarioGeneratorConfig()
    if n_bins is None:
        n_bins = scenario_generator_config.n_bins

    lane_width = 3.7
    long_bins = np.linspace(-scenario_generator_config.length, +scenario_generator_config.length, n_bins)
    lat_bins = np.linspace(-lane_width, lane_width, n_bins)
    return long_bins, lat_bins


def _is_evaluated(simulation_db: str) -> bool:
    """
    True if the scenario was evaluated: the evaluation table has EVITA's columns and at least one row.
    Executed but invalid scenarios (e.g. an agent did not start) keep FrenetiX's bare table and are not evaluated.
    """
    try:
        columns = {row[1] for row in _query(simulation_db, "PRAGMA table_info(scenario_evaluation);")}
        if not EVALUATION_COLUMNS.issubset(columns):
            return False
        return _query(simulation_db, "SELECT COUNT(*) FROM scenario_evaluation;")[0][0] > 0
    except sqlite3.Error:
        return False


def _impact_location_bins() -> List[frozenset]:
    """
    Every combination of the points of impact of two vehicles (unordered; "-+" is the road).
    """
    locations = [f"{p}{s}" for p in ["F", "M", "B", "-"] for s in ["L", "C", "R", "+"]]
    bins = []
    for first in locations:
        for second in locations:
            pair = frozenset((first, second))
            if pair not in bins:
                bins.append(pair)
    return bins


def generate_report(run_folder: str, _output_to: str = None, n_bins = None) -> None:
    """
    Write the run's metrics into `_output_to` (default: the run folder):
      - requirements_violations.csv: total_timeouts and total_missed_goals (vehicles), total_collisions, and the
        unique collisions by point of impact, by point of impact and RIV, and by point of impact, RIV and RIA;
      - behavior.csv: total_emergency and total_disengagement (time steps), total_agents, total_interactions,
        unique_interactions (occupied cells of the interaction map);
      - collision_map_only_impact_location.npy, collision_map_impact_location_and_riv.npy, collision_map.npy,
        interaction_map.npy.
    Only evaluated scenarios are considered.
    """
    output_to = _output_to or run_folder
    module_logger.critical(f"Generating report for {run_folder} at {output_to}")

    evaluated_dbs = []
    for _, simulation_folder in simulation_folders(run_folder):
        simulation_db = os.path.join(run_folder, simulation_folder, "logs", "simulation.db")
        if _is_evaluated(simulation_db):
            evaluated_dbs.append((simulation_folder, simulation_db))
        else:
            module_logger.debug(f"Skipping {simulation_folder}: not evaluated (invalid scenario)")

    # ===== Requirement violations =====
    module_logger.critical("Computing Requirement Violation Metrics")

    impact_location_bins = _impact_location_bins()
    collision_map_only_impact_location = np.zeros(len(impact_location_bins))
    collision_map_impact_location_and_riv = np.zeros((len(impact_location_bins), len(RIV_BINS_KMH) + 1))
    collision_map = np.zeros((len(impact_location_bins), len(RIV_BINS_KMH) + 1, len(RIA_BINS_DEG) + 1))

    total_timeouts, total_missed_goals, total_collisions = 0, 0, 0

    for simulation_folder, simulation_db in evaluated_dbs:
        module_logger.debug(f"Processing {simulation_folder}")

        # Per-vehicle outcomes, repeated on every time step by the evaluator: count vehicles
        total_timeouts += count_agents_with(simulation_db, "has_timeout")
        total_missed_goals += count_agents_with(simulation_db, "has_missed_goal")
        total_collisions += count_collisions(simulation_db)

        for impact_location, riv, ria in _get_impacts_features_from_db(simulation_db):
            if None in impact_location or riv is None or ria is None:
                module_logger.warning(f"Skipping collision with missing features: impact_location = {impact_location}, riv = {riv}, ria = {ria} in {simulation_db}")
                continue

            # RIV from m/s to km/h, |RIA| from radians to degrees (the paper's bins)
            location_index = impact_location_bins.index(impact_location)
            riv_index = np.digitize(riv * 3.6, RIV_BINS_KMH)
            ria_index = np.digitize(math.degrees(ria), RIA_BINS_DEG)

            collision_map_only_impact_location[location_index] += 1.0
            collision_map_impact_location_and_riv[location_index][riv_index] += 1.0
            collision_map[location_index][riv_index][ria_index] += 1.0

    mapped_collisions = int(np.sum(collision_map_only_impact_location))
    if total_collisions != mapped_collisions:
        module_logger.warning(f"Collision count mismatch in {run_folder}: {total_collisions} counted, {mapped_collisions} mapped")

    with open(os.path.join(output_to, "requirements_violations.csv"), "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["total_timeouts", "total_missed_goals", "total_collisions", "unique_collisions_only_impact_location", "unique_collisions_impact_location_and_riv", "unique_collisions"])
        csv_writer.writerow([total_timeouts, total_missed_goals, total_collisions,
                             np.count_nonzero(collision_map_only_impact_location), np.count_nonzero(collision_map_impact_location_and_riv), np.count_nonzero(collision_map)])

    # ===== Behaviour and interactions =====
    module_logger.info("Computing Behavior and Interaction Metrics")

    long_bins, lat_bins = get_interaction_map_bins(n_bins)
    interaction_map = np.zeros((len(long_bins) + 1, len(lat_bins) + 1))
    total_emergency, total_disengagement, total_agents = 0, 0, 0

    for simulation_folder, simulation_db in evaluated_dbs:
        total_emergency += count_metric(simulation_db, "has_emergency")
        total_disengagement += count_metric(simulation_db, "has_disengagement")
        total_agents += len(_agent_ids_from_db(simulation_db))

        for agent_id, t, diff_long, diff_lat in _interactions_from_db(simulation_db):
            if diff_long is None or diff_lat is None:
                module_logger.warning(f"Interaction at timestep {t} for agent {agent_id} has None values for diff_long or diff_lat.")
                continue
            interaction_map[np.digitize(diff_long, long_bins)][np.digitize(diff_lat, lat_bins)] += 1.0

    with open(os.path.join(output_to, "behavior.csv"), "w", newline = "") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(["total_emergency", "total_disengagement", "total_agents", "total_interactions", "unique_interactions"])
        csv_writer.writerow([total_emergency, total_disengagement, total_agents, int(np.sum(interaction_map)), np.count_nonzero(interaction_map)])

    # ===== Maps =====
    np.save(os.path.join(output_to, "collision_map_only_impact_location.npy"), collision_map_only_impact_location)
    np.save(os.path.join(output_to, "collision_map_impact_location_and_riv.npy"), collision_map_impact_location_and_riv)
    np.save(os.path.join(output_to, "collision_map.npy"), collision_map)
    np.save(os.path.join(output_to, "interaction_map.npy"), interaction_map)