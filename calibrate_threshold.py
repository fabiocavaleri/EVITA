"""
Calibration of Th_ED, the trajectory-deviation threshold of EVITA's interaction detection (condition C1).

EVITA detects an interaction when a vehicle replans (C1: a planned position moves by at least Th_ED metres between two consecutive plans) while another vehicle is nearby (C2: shape-to-shape distance below safety_distance).
FrenetiX also replans routinely (every 3 steps) when nothing happens around the vehicle, so Th_ED must sit above that routine replanning.
This script compares two distributions of the deviation between consecutive plans:
  - noise:   each vehicle simulated alone (single-vehicle scenarios), so every replanning is routine;
  - traffic: the same vehicles in their original multi-vehicle scenario, at steps where C2 holds.

Three tasks share a working folder; each reads only what the previous one wrote:

    python calibrate_threshold.py generate --maps <map files> --singles-per-map 100 --seed 1 --output calibration/
    python calibrate_threshold.py execute  --input calibration/ --workers 6
    python calibrate_threshold.py analyze  --input calibration/ [--percentiles 90 95 99]

Working folder:
    manifest.csv                       one row per simulation (the single record of progress)
    <map>/scenario_0001/multi.xml      the original multi-vehicle scenario
    <map>/scenario_0001/single_<id>.xml   each vehicle alone, with its original id
    <map>/scenario_0001/runs/<run id>/ executor output (scenario/, logs/<vehicle id>/logs.csv),
                                       simulated_scenario.xml (vehicles as obstacles) and run_info.json
    <map>/scenario_0001/runs/<run id>.log   console output of the simulation
    analysis/replans.csv, pairing.csv, summary.json, plots/

The script uses EVITA's own code for generation, validation, execution and trajectory comparison, so that the
calibration measures exactly what EVITA measures. Run it from the project root, in the same environment as
`python -m evita`.
"""
import os
import csv
import sys
import json
import math
import time
import pickle
import random
import shutil
import logging
import argparse
import traceback
import contextlib
import numpy as np
import multiprocessing as mp

from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

# ========== Import paths ==========

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

for _extra_path in (os.path.join(PROJECT_ROOT, "Frenetix-Motion-Planner"), os.path.join(PROJECT_ROOT, "EVITA_DoppelTest")):
    if os.path.isdir(_extra_path) and _extra_path not in sys.path:
        sys.path.append(_extra_path)

# ========== Constants ==========

# The five maps of the project (used when --maps is not given)
DEFAULT_MAPS = [os.path.join(PROJECT_ROOT, "base-scenarios", name) for name in (
    "CHN_Cho-2_2_I-1-1.cr.xml",
    "DEU_Cologne-79_2_I-1.cr.xml",
    "USA_US101-9_1_T-1.xml",
    "USA_US101-15_2_T-1.xml",
    "USA_US101-23_1_T-1.xml",
)]

MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = ["map", "scenario", "kind", "run_id", "vehicle_ids", "n_vehicles", "start_time_steps", "seed", "scenario_file", "status", "duration_s", "failure_reason"]

STATUS_GENERATED, STATUS_DONE, STATUS_FAILED = "generated", "done", "failed"
KIND_MULTI, KIND_SINGLE = "multi", "single"

SIMULATED_SCENARIO_XML = "simulated_scenario.xml"
SIMULATED_SCENARIO_PICKLE = "simulated_scenario.pkl"
RUN_INFO_JSON = "run_info.json"

# Th_ED currently used by EVITA, reported next to the candidates for reference
CURRENT_THRESHOLD = 1.0

# Two consecutive plans are the same plan (no replanning) if no shared state moved more than this (metres)
# The default is 1e-3, which is the same as EVITA's own comparison of planned positions (in meters) in the evaluator.
DEFAULT_IDENTICAL_TOLERANCE = 1e-3

# Measures of the deviation between two consecutive plans
MEASURES = ["max_dev", "mean_dev", "end_dev"]

# Percentiles reported for every distribution (the candidate percentiles are added to these)
REPORTED_PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# "Noise grows clearly with speed" if the Spearman correlation between max deviation and speed reaches this
SPEED_CORRELATION_ALERT = 0.3

logger = logging.getLogger("calibrate_threshold")

# ========== Small helpers ==========

class _NullCounter:
    """
    Progress bar that shows nothing
    """
    count = 0
    total = 0

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

class _NullProgressBarManager:
    """
    Stand-in for enlighten's manager: EVITA's sampling and executor create progress bars, which must stay silent
    in the parallel workers (and in the generation, which runs without a terminal bar).
    """
    def counter(self, *args, **kwargs):
        return _NullCounter()

    def stop(self):
        pass

def _setup_logging(debug: bool = False) -> None:
    """
    Console logging for this script, and EVITA's main logger at the same level
    """
    from evita.utils import initialize_main_logger

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)-7s [%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    
    logger.setLevel(level)
    logger.propagate = False

    initialize_main_logger(debug)
    logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)

def map_name_of(map_file: str) -> str:
    """
    Map name from its file name: DEU_Cologne-79_2_I-1.cr.xml -> DEU_Cologne-79_2_I-1
    """
    name = os.path.basename(map_file)
    
    for suffix in (".xml", ".cr", ".pb"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    return name

def _ids_to_text(ids) -> str:
    return ";".join(str(i) for i in ids)

def _ids_from_text(text: str) -> List[int]:
    return [int(i) for i in str(text).split(";") if str(i).strip()]

# ========== Manifest ==========

def manifest_path(folder: str) -> str:
    return os.path.join(folder, MANIFEST_NAME)

def read_manifest(folder: str) -> List[Dict[str, str]]:
    path = manifest_path(folder)

    if not os.path.exists(path):
        return []
    
    with open(path, newline = "") as handle:
        return [dict(row) for row in csv.DictReader(handle)]

def write_manifest(folder: str, rows: List[Dict[str, str]]) -> None:
    """
    Write the manifest atomically (temporary file, then rename), so an interruption never leaves it truncated
    """
    path = manifest_path(folder)
    temporary = path + ".tmp"
    
    with open(temporary, "w", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = MANIFEST_COLUMNS, extrasaction = "ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})

    os.replace(temporary, path)

def _row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return row["map"], row["scenario"], row["run_id"]

@contextlib.contextmanager
def manifest_lock(folder: str):
    """
    Exclusive lock on the manifest, so that several processes (e.g., one `generate` per map, run in parallel) can
    update it: each one re-reads it under the lock and changes only its own rows.
    """
    import fcntl

    os.makedirs(folder, exist_ok = True)
    with open(manifest_path(folder) + ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def replace_map_rows(folder: str, map_name: str, new_rows: List[Dict[str, str]]) -> None:
    """
    Replace the rows of one map in the manifest, keeping the other maps' rows as they are now on disk
    """
    with manifest_lock(folder):
        rows = [row for row in read_manifest(folder) if row["map"] != map_name]
        write_manifest(folder, rows + new_rows)

def update_manifest_row(folder: str, key: Tuple[str, str, str], fields: Dict[str, str]) -> None:
    """
    Update one row of the manifest, keeping every other row as it is now on disk
    """
    with manifest_lock(folder):
        rows = read_manifest(folder)
        for row in rows:
            if _row_key(row) == key:
                row.update(fields)
        write_manifest(folder, rows)

# ========== Task 1: generate ==========

def _write_scenario_file(scenario, planning_problems, path: str) -> None:
    """
    Write a scenario with the given planning problems, as the executor does
    """
    from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
    from commonroad.planning.planning_problem import PlanningProblemSet

    problem_set = PlanningProblemSet([deepcopy(pp) for pp in planning_problems])
    CommonRoadFileWriter(scenario, problem_set, author = "EVITA", decimal_precision = 10).write_to_file(path, overwrite_existing_file = OverwriteExistingFile.ALWAYS)

def generate_map(map_file: str, output: str, seed: int, singles_target: int, max_rejected: int) -> List[Dict[str, str]]:
    """
    Random valid multi-vehicle scenarios on one map, each split into single-vehicle scenarios, until the map has at
    least `singles_target` single-vehicle scenarios. Returns the manifest rows of the map.
    """
    from commonroad.common.file_reader import CommonRoadFileReader

    from evita.config import ScenarioGeneratorConfig
    from evita.generators.nsgaII_operators import CustomTimeTermination, ScenarioSampling
    from evita.generators.utils import get_base_scenario_from_file

    map_name = map_name_of(map_file)
    map_folder = os.path.join(output, map_name)
    os.makedirs(map_folder, exist_ok = True)

    # 1. Same seed for every map: each map's scenarios are reproducible on their own
    random.seed(seed)
    np.random.seed(seed)

    # 2. Road network only, as EVITA loads it
    base_scenario, base_planning_problem_set = get_base_scenario_from_file(map_file, remove_obstacles = True, remove_planning_problems = True)

    # 3. EVITA's initial-population sampling, with the default configurations
    #    The termination criterion is only used for its budget pauses: a large budget, never started, silent bars.
    termination = CustomTimeTermination(max_time_minutes = 10 ** 6, checkpoint_every = 0, progress_bar_manager = _NullProgressBarManager())
    sampling = ScenarioSampling(base_scenario, base_planning_problem_set, ScenarioGeneratorConfig(), termination)

    rows: List[Dict[str, str]] = []
    n_singles, n_multis, n_rejected = 0, 0, 0

    # 4. Stop when the map has enough single-vehicle scenarios; the last scenario is split completely
    while n_singles < singles_target:
        # Generate a random multi-vehicle
        scenario, planning_problem_set = sampling._generate_random_individual()  # raises if no scenario can be made
        planning_problems = sorted(planning_problem_set.planning_problem_dict.values(), key = lambda pp: pp.planning_problem_id)

        # 4a. Save to file the original multi-vehicle scenario.
        # Check that it is still valid as written (rounding in the file).
        scenario_name = f"scenario_{n_multis + 1:04d}"
        scenario_folder = os.path.join(map_folder, scenario_name)
        if os.path.exists(scenario_folder):
            shutil.rmtree(scenario_folder)
        os.makedirs(scenario_folder)

        multi_file = os.path.join(scenario_folder, "multi.xml")
        _write_scenario_file(scenario, planning_problems, multi_file)

        written_scenario, written_problems = CommonRoadFileReader(multi_file).open()
        valid, message = sampling.scenario_validator.is_scenario_valid(written_scenario, written_problems)

        # If the written scenario is invalid, discard it and try again (up to max_rejected times)
        if not valid:
            n_rejected += 1
            logger.warning(f"{map_name}: generated scenario not valid once written ({message}); discarded")
            shutil.rmtree(scenario_folder)
            if n_rejected > max_rejected:
                raise RuntimeError(f"{map_name}: more than {max_rejected} generated scenarios were invalid once written")
            continue

        # The written scenario is valid: keep it and split it into single-vehicle scenarios
        n_multis += 1
        ids = [pp.planning_problem_id for pp in planning_problems]

        start_times = _ids_to_text(f"{pp.planning_problem_id}:{pp.initial_state.time_step}" for pp in planning_problems)
        common = {"map": map_name, "scenario": scenario_name, "n_vehicles": str(len(ids)), "seed": str(seed),
                  "status": STATUS_GENERATED, "duration_s": "", "failure_reason": ""}

        rows.append({**common, "kind": KIND_MULTI, "run_id": KIND_MULTI, "vehicle_ids": _ids_to_text(ids), "start_time_steps": start_times, "scenario_file": os.path.relpath(multi_file, output)})

        # 4b. One single-vehicle scenario per vehicle: same road network, only that vehicle, original id.
        #     Not validated: removing vehicles cannot break validity
        for planning_problem in planning_problems:
            vehicle_id = planning_problem.planning_problem_id
            single_file = os.path.join(scenario_folder, f"single_{vehicle_id}.xml")
            _write_scenario_file(scenario, [planning_problem], single_file)

            rows.append({**common, "kind": KIND_SINGLE, "run_id": f"single_{vehicle_id}", "vehicle_ids": str(vehicle_id),
                         "start_time_steps": f"{vehicle_id}:{planning_problem.initial_state.time_step}",
                         "scenario_file": os.path.relpath(single_file, output)})
            n_singles += 1

        logger.info(f"{map_name}: {scenario_name} with {len(ids)} vehicles ({n_singles}/{singles_target} single-vehicle scenarios)")

    logger.info(f"{map_name}: {n_multis} multi-vehicle and {n_singles} single-vehicle scenarios ({n_rejected} discarded)")

    return rows

def check_generated(output: str, rows: List[Dict[str, str]], singles_target: int) -> List[str]:
    """
    Acceptance checks of `generate`:
    - At least `singles_target` singles per map;
    - Each single's vehicle is in its multi scenario;
    - Every multi scenario passes the validator.
    """
    from commonroad.common.file_reader import CommonRoadFileReader

    from evita.config import ScenarioGeneratorConfig, ValidationConfig
    from evita.validator import ScenarioValidator

    problems = []
    validator = ScenarioValidator(ValidationConfig(), ScenarioGeneratorConfig())

    # Iterate over the maps in the manifest
    for map_name in sorted({row["map"] for row in rows}):
        # Extract the map's rows and its single-vehicle scenarios
        map_rows = [row for row in rows if row["map"] == map_name]
        singles = [row for row in map_rows if row["kind"] == KIND_SINGLE]

        # Check that the map has enough single-vehicle scenarios
        if len(singles) < singles_target:
            problems.append(f"{map_name}: only {len(singles)} single-vehicle scenarios (target {singles_target})")

        # Extract the multi-vehicle scenarios and check that each single's vehicle is in its multi scenario
        multis = {row["scenario"]: row for row in map_rows if row["kind"] == KIND_MULTI}
        for row in singles:
            multi = multis.get(row["scenario"])
            if multi is None or int(row["vehicle_ids"]) not in _ids_from_text(multi["vehicle_ids"]):
                problems.append(f"{map_name}/{row['scenario']}: vehicle {row['vehicle_ids']} not in the multi-vehicle scenario")

        # Check that every multi-vehicle scenario passes the validator, and that its vehicles match the manifest
        for row in multis.values():
            scenario, problem_set = CommonRoadFileReader(os.path.join(output, row["scenario_file"])).open()
            valid, message = validator.is_scenario_valid(scenario, problem_set)

            if not valid:
                problems.append(f"{map_name}/{row['scenario']}: multi-vehicle scenario not valid ({message})")
            
            if sorted(problem_set.planning_problem_dict) != sorted(_ids_from_text(row["vehicle_ids"])):
                problems.append(f"{map_name}/{row['scenario']}: vehicles in the file differ from the manifest")

    return problems

def command_generate(args) -> int:
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok = True)

    # 1. Check that the map files exist, before starting generation
    for map_file in args.maps:
        if not os.path.isfile(map_file):
            logger.error(f"Map file not found: {map_file}")
            return 1

    # 2. Generate each map in order, skipping those already generated (unless --force is given)
    for map_file in args.maps:
        map_name = map_name_of(map_file)
        existing = [row for row in read_manifest(output) if row["map"] == map_name]

        # Skip if the map is already generated, unless --force is given
        if existing and not args.force:
            logger.info(f"{map_name}: already generated ({len(existing)} rows in the manifest); use --force to regenerate")
            continue

        # Regenerate: forget the map's rows and outputs
        replace_map_rows(output, map_name, [])
        if os.path.isdir(os.path.join(output, map_name)):
            shutil.rmtree(os.path.join(output, map_name))

        # Generate the map's scenarios and write them to the manifest
        logger.info(f"{map_name}: generating from {map_file} (seed {args.seed})")
        replace_map_rows(output, map_name, generate_map(map_file, output, args.seed, args.singles_per_map, args.max_rejected))

    # 3. Check that the generated scenarios are valid and complete, and report any problems
    generated_maps = {map_name_of(m) for m in args.maps}
    problems = check_generated(output, [row for row in read_manifest(output) if row["map"] in generated_maps], args.singles_per_map)

    for problem in problems:
        logger.error(problem)

    if problems:
        return 1

    logger.info(f"Generation complete; manifest: {manifest_path(output)}")

    return 0

# ========== Task 2: execute ==========

@contextlib.contextmanager
def _redirect_output_to(path: str):
    """
    Send this process's stdout and stderr (file descriptors 1 and 2, inherited by FrenetiX's own processes) to a file.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        for fd in (saved_stdout, saved_stderr, log_fd):
            os.close(fd)

class _CriticalMessages(logging.Handler):
    """Keep EVITA's critical messages, to report why a simulation failed (the executor logs the traceback)."""
    def __init__(self):
        super().__init__(level = logging.CRITICAL)
        self.messages: List[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())

def _short_reason(text: str, limit: int = 300) -> str:
    """
    One-line failure reason for the manifest
    """
    lines = [line.strip() for line in str(text).strip().splitlines() if line.strip()]
    
    if not lines:
        return "unknown error"
    
    reason = lines[0] if len(lines) == 1 else f"{lines[0]} | {lines[-1]}"
    
    return reason.replace(",", ";")[:limit]

def _worker_initializer(debug: bool) -> None:
    """
    Import EVITA and FrenetiX once per worker process, and apply the resource settings
    """
    from evita.executors import frenetix_executor
    from evita.utils import initialize_main_logger
    
    initialize_main_logger(debug)
    logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)

    # Run every simulation in this worker's process (FrenetiX's sequential loop).
    original = frenetix_executor.PatchedConfigurationBuilder.build_sim_configuration

    def build_sim_configuration_without_multiprocessing(*args, **kwargs):
        configuration = original(*args, **kwargs)
        configuration.simulation.multiprocessing = False

        return configuration

    frenetix_executor.PatchedConfigurationBuilder.build_sim_configuration = staticmethod(build_sim_configuration_without_multiprocessing)

def _did_not_start(agent) -> bool:
    """
    Check if an agent failed at time step 0, or never moved past its initial time step
    """
    status = getattr(agent.agent_state.status, "name", str(agent.agent_state.status))
    last_timestep = agent.agent_state.last_timestep

    return (status == "ERROR" and last_timestep == 0) or agent.planning_problem.initial_state.time_step == last_timestep

def _write_simulated_scenario(simulation, run_folder: str) -> str:
    """
    Save the simulated scenario (every vehicle as an obstacle with its driven trajectory) for condition C2.
    CommonRoad XML as planned; a pickle if the XML writer cannot handle the simulated states.
    """
    from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
    from commonroad.planning.planning_problem import PlanningProblemSet

    xml_path = os.path.join(run_folder, SIMULATED_SCENARIO_XML)

    try:
        CommonRoadFileWriter(simulation.scenario, PlanningProblemSet(), author = "EVITA", decimal_precision = 10).write_to_file(xml_path, overwrite_existing_file = OverwriteExistingFile.ALWAYS)

        return xml_path
    except Exception:
        logging.getLogger("calibrate_threshold").warning(f"Cannot write {xml_path}, saving a pickle instead:\n{traceback.format_exc()}")
        
        if os.path.exists(xml_path):
            os.remove(xml_path)
        
        pickle_path = os.path.join(run_folder, SIMULATED_SCENARIO_PICKLE)
        with open(pickle_path, "wb") as handle:
            pickle.dump(simulation.scenario, handle)
        
        return pickle_path

def execute_one(task: Dict[str, str]) -> Dict[str, str]:
    """
    Prepare and run one scenario in its own output folder.
    Runs in a worker process.
    Returns the outcome for the manifest; never raises.
    """
    started = time.perf_counter()
    outcome = {"key": task["key"], "status": STATUS_FAILED, "failure_reason": "", "duration_s": ""}

    runs_folder = task["runs_folder"]
    run_id = task["run_id"]
    run_folder = os.path.join(runs_folder, run_id)
    log_file = os.path.join(runs_folder, f"{run_id}.log")

    # The executor fails if its folder exists: start a rerun from scratch
    if os.path.exists(run_folder):
        shutil.rmtree(run_folder)
    os.makedirs(runs_folder, exist_ok = True)

    evita_logger = logging.getLogger("evita.main_logger")
    critical_messages = _CriticalMessages()
    evita_logger.addHandler(critical_messages)

    simulation = None
    try:
        with _redirect_output_to(log_file):
            try:
                from commonroad.common.file_reader import CommonRoadFileReader
                from evita.executors.frenetix_executor import FrenetixScenarioExecution
                from evita.utils import LANELET_EXTENSION, extend_scenario_after_goal_areas, time_shift

                scenario, planning_problem_set = CommonRoadFileReader(task["scenario_file"]).open()

                # The executor names its folder after generation_id: unique within the scenario's runs/ folder
                setattr(scenario, "generation_id", run_id)

                # Preparation: extended and time shift
                extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters = LANELET_EXTENSION)
                # shifted_planning_problem_set = time_shift(scenario, planning_problem_set)
                shifted_planning_problem_set = planning_problem_set

                # Simulate scenario
                executor = FrenetixScenarioExecution(runs_folder, driving_agent_name = "frenetix", debug = False)
                simulation = executor.execute(extended_scenario, shifted_planning_problem_set, _NullProgressBarManager())

                if simulation is None:
                    failure = next((m for m in reversed(critical_messages.messages) if "failed" in m), "simulation failed")
                    outcome["failure_reason"] = _short_reason(failure)
                else:
                    # Save simulation
                    simulated_file = _write_simulated_scenario(simulation, run_folder)

                    agents = [{"id": int(agent.id),
                               "status": getattr(agent.agent_state.status, "name", str(agent.agent_state.status)),
                               "last_timestep": int(agent.agent_state.last_timestep),
                               "initial_time_step": int(agent.planning_problem.initial_state.time_step)}
                              for agent in simulation.agents]
                    
                    run_info = {"run_id": run_id,
                                "scenario_file": task["scenario_file"],
                                "simulated_scenario": os.path.basename(simulated_file),
                                "log_path": simulation.config.simulation.log_path,
                                "frenetix_multiprocessing": bool(simulation.config.simulation.multiprocessing),
                                "frenetix_num_procs": getattr(simulation.config.simulation, "num_procs", None),
                                "lanelet_extension": LANELET_EXTENSION,
                                "agents": agents}

                    with open(os.path.join(run_folder, RUN_INFO_JSON), "w") as handle:
                        json.dump(run_info, handle, indent = 2)

                    # Mark a scenario as invalid if any agent failed at time step 0 or never moved past its initial time step
                    not_started = [agent.id for agent in simulation.agents if _did_not_start(agent)]
                    if not_started:
                        outcome["failure_reason"] = f"agents {_ids_to_text(not_started)} did not start"
                    else:
                        outcome["status"] = STATUS_DONE
            except Exception:
                traceback.print_exc()
                outcome["failure_reason"] = _short_reason(traceback.format_exc().strip().splitlines()[-1])
            finally:
                # Close the simulation's database connection and processes
                if simulation is not None:
                    with contextlib.suppress(Exception):
                        simulation.sim_logger.con.close()
                    with contextlib.suppress(Exception):
                        if simulation.config.simulation.multiprocessing:
                            simulation.close_processes()
    finally:
        evita_logger.removeHandler(critical_messages)

    outcome["duration_s"] = f"{time.perf_counter() - started:.1f}"

    return outcome

def command_execute(args) -> int:
    folder = os.path.abspath(args.input)
    rows = read_manifest(folder)

    # Check if the manifest exists and has rows; if not, the user must run `generate` first
    if not rows:
        logger.error(f"No manifest in {folder}: run `generate` first")
        return 1

    # 1. Rows not done yet; failed ones only on request
    selectable = {STATUS_GENERATED} | ({STATUS_FAILED} if args.retry_failed else set())
    selected = [row for row in rows if row["status"] in selectable and (not args.maps or row["map"] in args.maps)]

    # Limit the number of simulations to run, if requested
    if args.limit is not None:
        selected = selected[:args.limit]

    # Count the number of done and failed simulations
    n_done = sum(row["status"] == STATUS_DONE for row in rows)
    n_failed = sum(row["status"] == STATUS_FAILED for row in rows)
    logger.info(f"{len(rows)} simulations in the manifest: {n_done} done, {n_failed} failed; {len(selected)} to run with {args.workers} workers")
    
    # Check that there is something to run; if not, the user must run `generate` first or change the selection
    if not selected:
        return 0

    rows_by_key = {_row_key(row): row for row in rows}
    tasks = []
    # Iterate over scenarios to run
    for row in selected:
        scenario_folder = os.path.dirname(os.path.join(folder, row["scenario_file"]))
        tasks.append({"key": "|".join(_row_key(row)),
                      "run_id": row["run_id"],
                      "scenario_file": os.path.join(folder, row["scenario_file"]),
                      "runs_folder": os.path.join(scenario_folder, "runs")})

    # 2. One simulation per task, in independent processes
    context = mp.get_context("spawn")
    completed = 0
    with ProcessPoolExecutor(max_workers = args.workers, mp_context = context, initializer = _worker_initializer, initargs = (args.debug,)) as pool:
        futures = {pool.submit(execute_one, task): task for task in tasks}
        try:
            for future in as_completed(futures):
                task = futures[future]

                try:
                    outcome = future.result()
                except BrokenProcessPool:
                    # A worker failed to start or was killed: the remaining simulations did not run, so they stay as generated
                    logger.error("The worker pool broke (a worker failed to start or was killed); simulations not yet recorded stay as generated. Fix the cause and run `execute` again.")
                    for pending in futures:
                        pending.cancel()
                    break
                except Exception as error:  # the worker process died
                    outcome = {"key": task["key"], "status": STATUS_FAILED, "duration_s": "", "failure_reason": _short_reason(f"worker error: {error!r}")}

                # Update the manifest row with the outcome, and log the result
                row = rows_by_key[tuple(outcome["key"].split("|"))]
                row.update(status = outcome["status"], duration_s = outcome["duration_s"], failure_reason = outcome["failure_reason"])
                update_manifest_row(folder, _row_key(row), {"status": row["status"], "duration_s": row["duration_s"], "failure_reason": row["failure_reason"]})

                completed += 1
                message = f"[{completed}/{len(tasks)}] {row['map']}/{row['scenario']}/{row['run_id']}: {row['status']} in {row['duration_s']} s"
                if row["status"] == STATUS_DONE:
                    logger.info(message)
                else:
                    logger.warning(f"{message} ({row['failure_reason']})")
        except KeyboardInterrupt:
            logger.warning("Interrupted: completed simulations are recorded; run `execute` again to continue")
            for future in futures:
                future.cancel()
            raise

    # Log the final counts of done, failed, and not run simulations
    rows = read_manifest(folder)
    n_done = sum(row["status"] == STATUS_DONE for row in rows)
    n_failed = sum(row["status"] == STATUS_FAILED for row in rows)

    logger.info(f"Execution finished: {n_done} done, {n_failed} failed, {len(rows) - n_done - n_failed} not run")

    # Acceptance check: a simulated scenario for every completed multi-vehicle run
    missing = [row for row in rows if row["status"] == STATUS_DONE and row["kind"] == KIND_MULTI and not _simulated_scenario_file(folder, row)]
    for row in missing:
        logger.error(f"{row['map']}/{row['scenario']}: no simulated scenario for the completed multi-vehicle run")

    return 1 if missing else 0

# ========== Task 3: analyze ==========

def _run_folder(folder: str, row: Dict[str, str]) -> str:
    return os.path.join(os.path.dirname(os.path.join(folder, row["scenario_file"])), "runs", row["run_id"])

def _simulated_scenario_file(folder: str, row: Dict[str, str]) -> Optional[str]:
    for name in (SIMULATED_SCENARIO_XML, SIMULATED_SCENARIO_PICKLE):
        path = os.path.join(_run_folder(folder, row), name)
        if os.path.exists(path):
            return path

    return None

def _load_simulated_scenario(path: str):
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    from commonroad.common.file_reader import CommonRoadFileReader
    return CommonRoadFileReader(path).open()[0]

def _planned_trajectories(log_path: str, vehicle_id: int) -> Dict[int, object]:
    """
    Planned trajectory of the vehicle at each time step, rebuilt from its logs.csv
    As in the evaluator, a later row with the same time step replaces an earlier one.
    """
    from evita.evaluators.frenetix_evaluator import FrenetixScenarioEvaluator

    stand_in = SimpleNamespace(agent_id_list = [vehicle_id], config = SimpleNamespace(simulation = SimpleNamespace(log_path = log_path)))
    table = FrenetixScenarioEvaluator._planned_trajectories(stand_in)

    return {int(timestep): trajectory for timestep, trajectory in zip(table["timestep"], table["trajectory"])}

def _path_length(trajectory) -> float:
    positions = np.array([np.asarray(state.position, dtype = float) for state in trajectory.state_list])

    if len(positions) < 2:
        return 0.0
    
    return float(np.sum(np.linalg.norm(np.diff(positions, axis = 0), axis = 1)))

def extract_run(folder: str, row: Dict[str, str], tolerance: float) -> Tuple[List[Dict], Dict]:
    """
    Replanning steps of every vehicle of one completed simulation: one dict per step, and counters for the summary.
    """
    from evita.generators.utils import NEARBY_DISTANCE, crop_trajectories, list_anyone_nearby

    # Open the run's info, to get the agents and their logs folder
    run_folder = _run_folder(folder, row)
    with open(os.path.join(run_folder, RUN_INFO_JSON)) as handle:
        run_info = json.load(handle)

    agents = {agent["id"]: agent for agent in run_info["agents"]}
    log_path = os.path.join(run_folder, "logs")

    vehicle_ids = _ids_from_text(row["vehicle_ids"])
    is_multi = row["kind"] == KIND_MULTI

    simulated_scenario = None
    if is_multi:
        simulated_scenario = _load_simulated_scenario(_simulated_scenario_file(folder, row))

    stats = {"pairs": 0, "replans": 0, "identical": 0, "no_overlap": 0, "after_collision": 0, "missing_logs": 0}
    rows = []

    # For each vehicle, rebuild its planned trajectories from its logs, and compare every two consecutive planned trajectories
    for vehicle_id in vehicle_ids:
        if not os.path.exists(os.path.join(log_path, str(vehicle_id), "logs.csv")):
            stats["missing_logs"] += 1
            continue

        plans = _planned_trajectories(log_path, vehicle_id)
        agent = agents.get(vehicle_id, {})
        collided = agent.get("status") == "COLLISION"
        last_timestep = agent.get("last_timestep")

        # In a multi-vehicle scenario, get the obstacle of this vehicle and the other obstacles at every time step
        if is_multi:
            obstacle = simulated_scenario.obstacle_by_id(vehicle_id)
            other_obstacles = [simulated_scenario.obstacle_by_id(other) for other in vehicle_ids if other != vehicle_id]
            other_obstacles = [other for other in other_obstacles if other is not None]

        for t in sorted(plans):
            # Skip if there is no planned trajectory at the next time step
            if t + 1 not in plans:
                continue

            # After a collision (at t or t + 1) the plans no longer reflect interactions
            if collided and last_timestep is not None and t + 1 >= last_timestep:
                stats["after_collision"] += 1
                continue

            stats["pairs"] += 1
            cropped_new, cropped_old = crop_trajectories(plans[t], plans[t + 1])
            if cropped_new is None:
                stats["no_overlap"] += 1
                continue

            # Distance between the two plans at every shared time step (as is_trajectory_similar)
            deviations = [float(np.linalg.norm(np.asarray(a.position, dtype = float) - np.asarray(b.position, dtype = float))) for a, b in zip(cropped_new.state_list, cropped_old.state_list)]

            # If the maximum deviation is within the tolerance, the two plans are considered identical and no replanning is counted
            if max(deviations) <= tolerance:
                stats["identical"] += 1
                continue

            stats["replans"] += 1

            first_state = plans[t].state_list[0]
            record = {"map": row["map"], "scenario": row["scenario"], "kind": row["kind"], "run_id": row["run_id"],
                      "vehicle_id": vehicle_id, "n_vehicles": int(row["n_vehicles"]), "time_step": t,
                      "max_dev": max(deviations), "mean_dev": float(np.mean(deviations)), "end_dev": deviations[-1],
                      "n_shared_steps": len(deviations), "plan_length": _path_length(cropped_old),
                      "speed": float(getattr(first_state, "velocity", np.nan)),
                      "deviations": json.dumps([round(d, 4) for d in deviations]),
                      "c2": False, "nearby_ids": "[]", "distances_to_others": "{}", "nearest_distance": np.nan}

            if is_multi:
                # Condition C2 exactly as EVITA, at time t (the time of the older plan)
                nearby_ids = list_anyone_nearby(t, obstacle, other_obstacles) if obstacle is not None else []
                record["c2"] = bool(nearby_ids)
                record["nearby_ids"] = json.dumps(sorted(int(i) for i in nearby_ids))

                # Distance to every other vehicle active at t, computed the same way (shape to shape)
                distances = {}
                occupancy = obstacle.occupancy_at_time(t) if obstacle is not None else None
                if occupancy is not None:
                    shape = occupancy.shape.shapely_object
                    for other in other_obstacles:
                        other_occupancy = other.occupancy_at_time(t)
                        if other_occupancy is not None:
                            distances[int(other.obstacle_id)] = round(float(shape.distance(other_occupancy.shape.shapely_object)), 4)

                record["distances_to_others"] = json.dumps({str(k): v for k, v in sorted(distances.items())})
                record["nearest_distance"] = min(distances.values()) if distances else np.nan

            rows.append(record)

    stats["nearby_distance"] = NEARBY_DISTANCE

    return rows, stats

def _extract_task(arguments):
    """
    Pool entry point: (folder, row, tolerance) -> (key, rows, stats, error)
    """
    folder, row, tolerance = arguments
    try:
        rows, stats = extract_run(folder, row, tolerance)
        return _row_key(row), rows, stats, None
    except Exception:
        return _row_key(row), [], {}, traceback.format_exc()

# ---------- Statistics ----------

def _percentiles(values: np.ndarray, percentiles: List[float]) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {f"p{p:g}": None for p in percentiles}
    return {f"p{p:g}": float(np.percentile(values, p)) for p in percentiles}

def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman rank correlation (no SciPy needed); None if undefined."""
    import pandas as pd

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    rank_x = pd.Series(x[mask]).rank().to_numpy()
    rank_y = pd.Series(y[mask]).rank().to_numpy()
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])

def _binned_profile(x: np.ndarray, y: np.ndarray, edges: List[float]) -> List[Dict]:
    """Count, median and 95th percentile of y in each bin of x."""
    profile = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = y[(x >= low) & (x < high)]
        profile.append({"from": low, "to": high if math.isfinite(high) else None, "n": int(len(selected)),
                        "median": float(np.median(selected)) if len(selected) else None,
                        "p95": float(np.percentile(selected, 95)) if len(selected) else None})
    return profile

def summarize_scope(replans, scope: str, candidate_percentiles: List[float], current_threshold: float) -> Dict:
    """
    Distributions, candidate thresholds and noise structure for one map (or all maps)
    """
    noise = replans[replans["kind"] == KIND_SINGLE]
    multi = replans[replans["kind"] == KIND_MULTI]
    traffic = multi[multi["c2"]]
    multi_far = multi[~multi["c2"]]

    percentiles = sorted(set(REPORTED_PERCENTILES) | set(candidate_percentiles))
    summary = {"scope": scope,
               "counts": {"noise": int(len(noise)), "traffic": int(len(traffic)), "multi_without_c2": int(len(multi_far)),
                          "noise_vehicles": int(noise.groupby(["map", "scenario", "vehicle_id"]).ngroups),
                          "traffic_vehicles": int(traffic.groupby(["map", "scenario", "vehicle_id"]).ngroups)},
               "measures": {}}

    for measure in MEASURES:
        noise_values = noise[measure].to_numpy(dtype = float)
        traffic_values = traffic[measure].to_numpy(dtype = float)
        far_values = multi_far[measure].to_numpy(dtype = float)

        candidates = []
        for p in candidate_percentiles:
            if len(noise_values) == 0:
                candidates.append({"percentile": p, "threshold": None})
                continue

            threshold = float(np.percentile(noise_values, p))
            # C1 flags a step when a deviation reaches the threshold (>=), as is_trajectory_similar
            
            candidates.append({"percentile": p, "threshold": threshold,
                               "noise_flagged": float(np.mean(noise_values >= threshold)),
                               "traffic_flagged": float(np.mean(traffic_values >= threshold)) if len(traffic_values) else None})

        reference = {"threshold": current_threshold,
                     "noise_flagged": float(np.mean(noise_values >= current_threshold)) if len(noise_values) else None,
                     "traffic_flagged": float(np.mean(traffic_values >= current_threshold)) if len(traffic_values) else None}

        summary["measures"][measure] = {"noise": _percentiles(noise_values, percentiles),
                                        "traffic": _percentiles(traffic_values, percentiles),
                                        "multi_without_c2": _percentiles(far_values, percentiles),
                                        "candidates": candidates,
                                        "current_threshold": reference}

    # Structure of the noise (on the max deviation, which is what C1 tests)
    if len(noise):
        max_dev = noise["max_dev"].to_numpy(dtype = float)
        speed = noise["speed"].to_numpy(dtype = float)
        time_step = noise["time_step"].to_numpy(dtype = float)
        n_vehicles = noise["n_vehicles"].to_numpy(dtype = float)
        plan_length = noise["plan_length"].to_numpy(dtype = float)

        speed_correlation = _spearman(speed, max_dev)
        relative = max_dev[plan_length > 0] / plan_length[plan_length > 0]
        summary["noise_structure"] = {
            "spearman_max_dev_vs_speed": speed_correlation,
            "spearman_max_dev_vs_time_step": _spearman(time_step, max_dev),
            "spearman_max_dev_vs_n_vehicles": _spearman(n_vehicles, max_dev),
            "noise_grows_with_speed": bool(speed_correlation is not None and speed_correlation >= SPEED_CORRELATION_ALERT),
            "by_speed_mps": _binned_profile(speed, max_dev, [0, 2, 5, 10, 15, 20, 25, np.inf]),
            "by_time_step": _binned_profile(time_step, max_dev, [0, 10, 20, 30, 50, 100, 200, np.inf]),
            "by_n_vehicles": [{"n_vehicles": int(n), "n": int((n_vehicles == n).sum()),
                               "median": float(np.median(max_dev[n_vehicles == n])),
                               "p95": float(np.percentile(max_dev[n_vehicles == n], 95))} for n in sorted(set(n_vehicles))],
            # If the noise grows with speed: a threshold relative to the planned path length (max_dev / plan length)
            "relative_max_dev": _percentiles(relative, percentiles),
            "relative_candidates": [{"percentile": p, "threshold": float(np.percentile(relative, p)) if len(relative) else None}
                                    for p in candidate_percentiles]}

    return summary

def pair_vehicles(replans, pairing_percentile: float):
    """
    Each vehicle's noise (its single-vehicle run) against its own traffic deviations (its multi-vehicle run, C2).
    """
    import pandas as pd

    keys = ["map", "scenario", "vehicle_id"]
    noise = replans[replans["kind"] == KIND_SINGLE]
    traffic = replans[(replans["kind"] == KIND_MULTI) & replans["c2"]]

    pairs = []
    for key, vehicle_noise in noise.groupby(keys):
        vehicle_traffic = traffic[(traffic["map"] == key[0]) & (traffic["scenario"] == key[1]) & (traffic["vehicle_id"] == key[2])]
        own_threshold = float(np.percentile(vehicle_noise["max_dev"], pairing_percentile))
        pairs.append({"map": key[0], "scenario": key[1], "vehicle_id": key[2],
                      "n_vehicles": int(vehicle_noise["n_vehicles"].iloc[0]),
                      "noise_replans": len(vehicle_noise),
                      f"noise_p{pairing_percentile:g}_max_dev": own_threshold,
                      "noise_max_max_dev": float(vehicle_noise["max_dev"].max()),
                      "traffic_replans": len(vehicle_traffic),
                      "traffic_median_max_dev": float(vehicle_traffic["max_dev"].median()) if len(vehicle_traffic) else np.nan,
                      "traffic_max_max_dev": float(vehicle_traffic["max_dev"].max()) if len(vehicle_traffic) else np.nan,
                      "traffic_share_above_own_noise": float((vehicle_traffic["max_dev"] >= own_threshold).mean()) if len(vehicle_traffic) else np.nan})

    return pd.DataFrame(pairs)

# ---------- Plots ----------

def _ecdf(values: np.ndarray):
    values = np.sort(values)
    return values, np.arange(1, len(values) + 1) / len(values)

def make_plots(replans, summaries: Dict[str, Dict], plots_folder: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_folder, exist_ok = True)
    scopes = [("overall", replans)] + [(name, replans[replans["map"] == name]) for name in sorted(replans["map"].unique())]
    colors = ["tab:green", "tab:orange", "tab:red", "tab:purple"]

    for scope, data in scopes:
        noise = data[data["kind"] == KIND_SINGLE]
        traffic = data[(data["kind"] == KIND_MULTI) & data["c2"]]
        summary = summaries[scope]

        # Noise vs traffic ECDFs, with the candidate thresholds and the current Th_ED
        for measure in MEASURES:
            figure, axis = plt.subplots(figsize = (8, 5))
            for values, label, color in ((noise[measure], f"noise (single, n={len(noise)})", "tab:blue"),
                                         (traffic[measure], f"traffic (multi, C2, n={len(traffic)})", "black")):
                values = values.to_numpy(dtype = float)
                values = values[values > 0]
                if len(values):
                    x, y = _ecdf(values)
                    axis.step(x, y, where = "post", label = label, color = color)
            for candidate, color in zip(summary["measures"][measure]["candidates"], colors):
                if candidate.get("threshold"):
                    axis.axvline(candidate["threshold"], color = color, linestyle = ":", label = f"noise p{candidate['percentile']:g} = {candidate['threshold']:.3f} m")
            axis.axvline(CURRENT_THRESHOLD, color = "grey", linestyle = "--", label = f"current Th_ED = {CURRENT_THRESHOLD:g} m")
            axis.set_xscale("log")
            axis.set_xlabel(f"{measure} between consecutive plans (m, log scale)")
            axis.set_ylabel("cumulative share of replanning steps")
            axis.set_title(f"{scope}: {measure}")
            axis.grid(alpha = 0.3)
            axis.legend(fontsize = 8)
            figure.tight_layout()
            figure.savefig(os.path.join(plots_folder, f"ecdf_{measure}_{scope}.png"), dpi = 120)
            plt.close(figure)

        if not len(noise):
            continue

        # Noise structure: against speed, time step and number of vehicles of the original scenario
        figure, axes = plt.subplots(1, 3, figsize = (16, 4.5))
        axes[0].scatter(noise["speed"], noise["max_dev"], s = 3, alpha = 0.3)
        axes[0].set_xlabel("speed (m/s)")
        axes[1].scatter(noise["time_step"], noise["max_dev"], s = 3, alpha = 0.3)
        axes[1].set_xlabel("time step (0.1 s)")
        groups = sorted(noise["n_vehicles"].unique())
        axes[2].boxplot([noise.loc[noise["n_vehicles"] == n, "max_dev"] for n in groups], labels = [str(int(n)) for n in groups], showfliers = False)
        axes[2].set_xlabel("vehicles in the original scenario")
        for axis in axes:
            axis.set_ylabel("noise max_dev (m)")
            axis.grid(alpha = 0.3)
        for axis in axes[:2]:
            axis.set_yscale("log")
        figure.suptitle(f"{scope}: structure of the noise (single-vehicle runs)")
        figure.tight_layout()
        figure.savefig(os.path.join(plots_folder, f"noise_structure_{scope}.png"), dpi = 120)
        plt.close(figure)

def command_analyze(args) -> int:
    import pandas as pd

    # Filter the manifest for completed simulations
    folder = os.path.abspath(args.input)
    rows = [row for row in read_manifest(folder) if row["status"] == STATUS_DONE]
    if not rows:
        logger.error(f"No completed simulation in {manifest_path(folder)}: run `execute` first")
        return 1

    # Create the analysis folder
    analysis_folder = os.path.join(folder, "analysis")
    os.makedirs(analysis_folder, exist_ok = True)
    logger.info(f"Extracting replanning steps from {len(rows)} completed simulations")

    # Data extraction
    jobs = [(folder, row, args.tolerance) for row in rows]
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers = args.workers, mp_context = mp.get_context("spawn")) as pool:
            for index, result in enumerate(pool.map(_extract_task, jobs, chunksize = 4), start = 1):
                results.append(result)
                if index % 50 == 0:
                    logger.info(f"  {index}/{len(jobs)} simulations")
    else:
        for index, job in enumerate(jobs, start = 1):
            results.append(_extract_task(job))
            if index % 50 == 0:
                logger.info(f"  {index}/{len(jobs)} simulations")

    all_rows, totals, errors = [], {}, 0
    # Iterate over the results of the extraction, logging errors and accumulating the extracted rows and summary stats
    for key, extracted, stats, error in results:
        if error:
            errors += 1
            logger.error(f"Cannot analyze {'/'.join(key)}:\n{error}")
            continue

        all_rows.extend(extracted)
        # Iterate over the stats of this simulation and accumulate them into the totals, except for nearby_distance which is constant
        for name, value in stats.items():
            if name != "nearby_distance":
                totals[name] = totals.get(name, 0) + value

    if not all_rows:
        logger.error("No replanning step found")
        return 1

    # Write the extracted replanning steps to a CSV file, sorted by map, scenario, kind, vehicle_id and time_step
    replans = pd.DataFrame(all_rows)
    replans["c2"] = replans["c2"].astype(bool)
    replans.sort_values(["map", "scenario", "kind", "vehicle_id", "time_step"], inplace = True)
    replans.to_csv(os.path.join(analysis_folder, "replans.csv"), index = False)
    logger.info(f"{len(replans)} replanning steps written to analysis/replans.csv ({totals})")

    # Distributions and thresholds, per map and overall
    summaries = {"overall": summarize_scope(replans, "overall", args.percentiles, args.current_threshold)}
    for map_name in sorted(replans["map"].unique()):
        summaries[map_name] = summarize_scope(replans[replans["map"] == map_name], map_name, args.percentiles, args.current_threshold)

    # Pairing: each vehicle's noise against its own traffic deviations
    pairing = pair_vehicles(replans, args.pairing_percentile)
    pairing.to_csv(os.path.join(analysis_folder, "pairing.csv"), index = False)
    paired = pairing.dropna(subset = ["traffic_median_max_dev"])

    # Acceptance checks of the analysis
    noise = replans[replans["kind"] == KIND_SINGLE]
    gaps = noise.groupby(["map", "scenario", "vehicle_id"])["time_step"].diff().dropna()
    checks = {"single_rows_with_c2": int(noise["c2"].sum()),
              "identical_pairs_excluded": int(totals.get("identical", 0)),
              "noise_gap_between_replans": {str(int(k)): int(v) for k, v in gaps.value_counts().sort_index().items()},
              "simulations_not_analyzed": errors}

    summary = {"settings": {"percentiles": args.percentiles, "identical_tolerance_m": args.tolerance,
                            "current_threshold_m": args.current_threshold, "pairing_percentile": args.pairing_percentile,
                            "nearby_distance_m": next((s["nearby_distance"] for _, _, s, e in results if not e and s), None),
                            "simulations": len(rows)},
               "extraction": totals,
               "checks": checks,
               "pairing": {"vehicles": int(len(pairing)), "vehicles_with_traffic": int(len(paired)),
                           "share_traffic_median_above_own_noise": float((paired["traffic_median_max_dev"] >= paired[f"noise_p{args.pairing_percentile:g}_max_dev"]).mean()) if len(paired) else None},
               "scopes": summaries}

    with open(os.path.join(analysis_folder, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent = 2, default = lambda value: None if isinstance(value, float) and not math.isfinite(value) else str(value))

    if checks["single_rows_with_c2"]:
        logger.error(f"{checks['single_rows_with_c2']} single-vehicle rows have C2 true")

    if not args.no_plots:
        make_plots(replans, summaries, os.path.join(analysis_folder, "plots"))

    # Console summary: candidate thresholds on the max deviation (what C1 tests)
    def share(value):
        return "n/a" if value is None else f"{value:.1%}"

    logger.info("Candidate thresholds on max_dev (share of noise / traffic flagged):")
    for scope, scope_summary in summaries.items():
        parts = [f"p{c['percentile']:g}=" + ("n/a" if c.get("threshold") is None else
                 f"{c['threshold']:.3f} m ({share(c['noise_flagged'])} / {share(c['traffic_flagged'])})")
                 for c in scope_summary["measures"]["max_dev"]["candidates"]]
        current = scope_summary["measures"]["max_dev"]["current_threshold"]
        logger.info(f"  {scope:<24} {'; '.join(parts)}; current {current['threshold']:g} m: "
                    f"{share(current['noise_flagged'])} / {share(current['traffic_flagged'])}")
        if scope_summary.get("noise_structure", {}).get("noise_grows_with_speed"):
            logger.warning(f"  {scope}: the noise grows with speed (Spearman {scope_summary['noise_structure']['spearman_max_dev_vs_speed']:.2f}); "
                           "consider a threshold relative to the planned path length (see relative_candidates)")

    logger.info(f"Analysis written to {analysis_folder}")
    
    return 1 if errors else 0

# ========== Command line ==========

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = "Calibration of Th_ED, the trajectory-deviation threshold of EVITA's interaction detection.")
    parser.add_argument("--debug", action = "store_true", help = "verbose logging")
    commands = parser.add_subparsers(dest = "command", required = True)

    generate = commands.add_parser("generate", help = "random valid scenarios, split into single-vehicle scenarios")
    generate.add_argument("--maps", nargs = "+", default = DEFAULT_MAPS, help = "map files (default: the five maps of the project)")
    generate.add_argument("--singles-per-map", type = int, default = 100, help = "target number of single-vehicle scenarios per map")
    generate.add_argument("--seed", type = int, default = 1, help = "random seed, the same for every map")
    generate.add_argument("--output", default = "calibration", help = "working folder")
    generate.add_argument("--force", action = "store_true", help = "regenerate maps already in the manifest")
    generate.add_argument("--max-rejected", type = int, default = 50, help = "give up a map after this many scenarios invalid once written")

    execute = commands.add_parser("execute", help = "run the scenarios with FrenetiX, in parallel")
    execute.add_argument("--input", default = "calibration", help = "working folder")
    execute.add_argument("--workers", type = int, default = 6, help = "parallel simulations (resource setting)")
    execute.add_argument("--retry-failed", action = "store_true", help = "also rerun simulations marked failed")
    execute.add_argument("--maps", nargs = "+", help = "only these map names")
    execute.add_argument("--limit", type = int, help = "run at most this many simulations (for a quick test)")

    analyze = commands.add_parser("analyze", help = "noise and traffic distributions, candidate thresholds, plots")
    analyze.add_argument("--input", default = "calibration", help = "working folder")
    analyze.add_argument("--percentiles", nargs = "+", type = float, default = [90, 95, 99], help = "noise percentiles proposed as thresholds")
    analyze.add_argument("--tolerance", type = float, default = DEFAULT_IDENTICAL_TOLERANCE, help = "consecutive plans closer than this (m) everywhere are the same plan")
    analyze.add_argument("--current-threshold", type = float, default = CURRENT_THRESHOLD, help = "Th_ED in use, reported for reference")
    analyze.add_argument("--pairing-percentile", type = float, default = 95, help = "percentile of each vehicle's own noise in pairing.csv")
    analyze.add_argument("--workers", type = int, default = 1, help = "parallel extraction processes")
    analyze.add_argument("--no-plots", action = "store_true", help = "skip the plots")

    return parser

def main(argv = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.debug)

    if args.command == "generate":
        return command_generate(args)
    if args.command == "execute":
        return command_execute(args)
    return command_analyze(args)

if __name__ == "__main__":
    sys.exit(main())