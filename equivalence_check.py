"""
Equivalence check for refactorings of EVITA's execution and evaluation (e.g., step 1 of the Apollo integration):
the same scenarios are executed and evaluated with the code before and after the change, and the results compared.

    python equivalence_check.py generate --map base-scenarios/USA_US101-15_2_T-1.xml --n 5 --output equivalence
    python equivalence_check.py execute  --input equivalence --label before --workers 6
    python equivalence_check.py execute  --input equivalence --label before_2 --workers 6     # FrenetiX determinism
    ... apply the change ...
    python equivalence_check.py execute  --input equivalence --label after --workers 6
    python equivalence_check.py compare  --input equivalence --labels before before_2           # baseline variation
    python equivalence_check.py compare  --input equivalence --labels before after

Working folder:
    scenarios/<framework>/scenario_<k>.xml        generated scenarios (valid), the same for every label
    runs/<label>/<framework>/<scenario>/          executor output, evaluation.csv, fitness.json
    runs/<label>/<framework>/<scenario>.log       console output of the run

Works with both versions of the executor/evaluator interface: what `execute` returns (a Simulation, or an
ExecutionResult) is passed to the evaluator as it is.
"""
import argparse
import contextlib
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

FRAMEWORKS = ["EVITA", "MaxDanger", "MaxInteractions", "Random", "DoppelTest"]

# ========== Helpers ==========


class _NullCounter:
    count = 0
    total = 0

    def update(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass


class _NullProgressBarManager:
    """
    Stand-in for enlighten's manager: progress bars stay silent.
    """
    def counter(self, *args, **kwargs):
        return _NullCounter()

    def stop(self):
        pass


@contextlib.contextmanager
def _redirect_output_to(path: str):
    """
    Send this process's stdout and stderr (file descriptors 1 and 2) to a file.
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


def _write_scenario(scenario, planning_problem_set, path: str) -> None:
    from commonroad.common.file_writer import CommonRoadFileWriter
    from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
    CommonRoadFileWriter(scenario, planning_problem_set, author = "EVITA", decimal_precision = 10).write_to_file(path, OverwriteExistingFile.ALWAYS)

# ========== Task 1: generate ==========

def _valid_scenarios(framework: str, map_file: str, n: int, max_attempts: int):
    """
    Up to n valid scenarios from the framework's own generation code (its initial, random individuals).
    """
    from evita.config import ScenarioGeneratorConfig, ValidationConfig
    from evita.validator import ScenarioValidator

    validator = ScenarioValidator(ValidationConfig(), ScenarioGeneratorConfig())
    found = []

    if framework == "EVITA":
        from evita.generators.nsgaII_operators import CustomTimeTermination, ScenarioSampling
        from evita.generators.utils import get_base_scenario_from_file
        base_scenario, base_planning_problem_set = get_base_scenario_from_file(map_file, remove_obstacles = True, remove_planning_problems = True)
        termination = CustomTimeTermination(10 ** 6, checkpoint_every = 0, progress_bar_manager = _NullProgressBarManager())
        sampling = ScenarioSampling(base_scenario, base_planning_problem_set, ScenarioGeneratorConfig(), termination)
        while len(found) < n:
            found.append(sampling._generate_random_individual())   # valid by construction
        return found

    for _ in range(max_attempts):
        if len(found) >= n:
            break
        if framework in ("MaxDanger", "MaxInteractions"):
            # A new generator gives a new random individual at its first generate()
            from evita.generators import basic_generator
            cls = basic_generator.EAOnePlusOneGenerator_MaxDanger if framework == "MaxDanger" else basic_generator.EAOnePlusOneGenerator_MaxInteractions
            candidates = cls(map_file).generate()
        elif framework == "Random":
            from evita.generators.random_generator import RandomScenarioGenerator
            candidates = RandomScenarioGenerator(map_file).generate() or []
        else:   # DoppelTest: its initial population, converted to CommonRoad
            from evita.generators.doppeltest_generator import DoppelTestScenarioGenerator
            candidates = DoppelTestScenarioGenerator(map_file).generate(progress_bar_manager = _NullProgressBarManager())

        for scenario, planning_problem_set in candidates:
            if planning_problem_set is None or len(found) >= n:
                continue
            if not hasattr(scenario, "is_valid"):
                validator.validate(scenario, planning_problem_set)
            if scenario.is_valid:
                found.append((scenario, planning_problem_set))
    return found


def generate(args) -> int:
    for framework in args.frameworks:
        # Same seed for every framework
        random.seed(args.seed)
        np.random.seed(args.seed)

        folder = os.path.join(args.output, "scenarios", framework)
        os.makedirs(folder, exist_ok = True)
        scenarios = _valid_scenarios(framework, args.map, args.n, args.max_attempts)
        for k, (scenario, planning_problem_set) in enumerate(scenarios, start = 1):
            _write_scenario(scenario, planning_problem_set, os.path.join(folder, f"scenario_{k:03d}.xml"))
        print(f"{framework}: {len(scenarios)} of {args.n} scenarios written to {folder}", flush = True)
    return 0

# ========== Task 2: execute ==========


def _agents_of(result) -> list:
    """
    The agents of an execution, from either interface version: a list of objects with id, status, last time step and
    driven states.
    """
    agents = getattr(result, "agents", None)
    if isinstance(agents, dict):    # ExecutionResult: AgentOutcome objects
        return [{"id": int(a.id), "status": str(a.status), "last_timestep": int(a.last_timestep), "states": list(a.driven_states)} for a in agents.values()]
    return [{"id": int(a.id), "status": a.agent_state.status.name, "last_timestep": int(a.agent_state.last_timestep), "states": list(a.record_state_list)}
            for a in result.agents]  # FrenetiX Simulation


def _release(executor, result) -> None:
    if hasattr(executor, "release"):
        executor.release(result)
        return
    try:
        result.sim_logger.con.close()
        if result.config.simulation.multiprocessing:
            result.close_processes()
    except Exception:
        pass


def _execute_one(scenario_file: str, run_folder: str, framework: str, threshold: float) -> dict:
    """
    Execute and evaluate one scenario as the pipeline does (worker process). Writes evaluation.csv and fitness.json.
    """
    started = time.perf_counter()
    name = os.path.splitext(os.path.basename(scenario_file))[0]
    outcome = {"framework": framework, "scenario": name, "status": "failed", "reason": ""}
    os.makedirs(run_folder, exist_ok = True)

    with _redirect_output_to(os.path.join(run_folder, f"{name}.log")):
        executor, result = None, None
        try:
            from commonroad.common.file_reader import CommonRoadFileReader
            from evita.evaluators.frenetix_evaluator import FrenetixScenarioEvaluator
            from evita.executors.frenetix_executor import FrenetixScenarioExecution
            from evita.generators.utils import compute_interaction_map
            from evita.roads import identify_segments
            from evita.utils import LANELET_EXTENSION, extend_scenario_after_goal_areas

            scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()
            setattr(scenario, "is_valid", True)
            setattr(scenario, "generation_id", name)

            extended_scenario = extend_scenario_after_goal_areas(scenario, extend_by_meters = LANELET_EXTENSION)
            executor = FrenetixScenarioExecution(run_folder, driving_agent_name = "frenetix", debug = False)
            result = executor.execute(extended_scenario, planning_problem_set, _NullProgressBarManager())

            if result is None:
                outcome["reason"] = "simulation failed"
            else:
                agents = _agents_of(result)
                evaluation = FrenetixScenarioEvaluator(run_folder).evaluate(result, threshold)
                evaluation.reset_index().to_csv(os.path.join(run_folder, name, "evaluation.csv"), index = False)

                # Fitness of every framework, from this one execution
                try:
                    segments = [identify_segments(agent["states"])[0] for agent in agents]          # after step 1: states
                except AttributeError:
                    segments = [identify_segments(a)[0] for a in getattr(result, "agents", [])]     # before: agent objects
                collided = evaluation[evaluation["has_collided"] == 1.0]
                dce = evaluation["dce"].min()
                fitness = {"evita_interactions": float(evaluation["is_interacting"].sum()),
                           "evita_vehicles": len(agents),
                           "evita_interaction_map": compute_interaction_map(evaluation, None).tolist(),
                           "max_danger": float(evaluation["max_danger"].max()),
                           "max_interactions": float(evaluation["is_interacting"].sum()),
                           "doppeltest_min_distance": None if dce is None or np.isnan(dce) else float(dce),
                           "doppeltest_decisions": int(sum(segments)),
                           "doppeltest_violations": len(set(collided.index.get_level_values(0))),
                           "agents": {str(a["id"]): {"status": a["status"], "last_timestep": a["last_timestep"]} for a in agents}}
                with open(os.path.join(run_folder, name, "fitness.json"), "w") as handle:
                    json.dump(fitness, handle, indent = 2)
                outcome["status"] = "done"
        except Exception:
            traceback.print_exc()
            outcome["reason"] = traceback.format_exc().strip().splitlines()[-1]
        finally:
            if result is not None:
                _release(executor, result)

    outcome["duration_s"] = round(time.perf_counter() - started, 1)
    return outcome


def execute(args) -> int:
    try:
        from evita.orchestrator import TRAJECTORY_DIFFERENCE_THRESHOLD as threshold
    except ImportError:
        threshold = 1.0

    tasks = []
    for framework in sorted(os.listdir(os.path.join(args.input, "scenarios"))):
        scenario_folder = os.path.join(args.input, "scenarios", framework)
        run_folder = os.path.join(args.input, "runs", args.label, framework)
        for scenario_file in sorted(f for f in os.listdir(scenario_folder) if f.endswith(".xml")):
            if os.path.exists(os.path.join(run_folder, os.path.splitext(scenario_file)[0])):
                print(f"{args.label}/{framework}/{scenario_file}: already executed (delete its folder to rerun)")
                continue
            tasks.append((os.path.join(scenario_folder, scenario_file), run_folder, framework, threshold))

    print(f"{len(tasks)} scenarios to execute with {args.workers} workers (label {args.label}, Th_ED {threshold})", flush = True)
    failed = 0
    with ProcessPoolExecutor(max_workers = args.workers, mp_context = multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_execute_one, *task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start = 1):
            scenario_file, run_folder, framework, _ = futures[future]
            try:
                outcome = future.result()
            except Exception as error:
                outcome = {"framework": framework, "scenario": os.path.basename(scenario_file), "status": "failed", "reason": f"worker error: {error!r}", "duration_s": None}
            failed += outcome["status"] != "done"
            message = f"[{completed}/{len(tasks)}] {framework}/{outcome['scenario']}: {outcome['status']} in {outcome['duration_s']} s"
            if outcome["status"] != "done":
                message += f" ({outcome['reason']})"
            print(message, flush = True)

    print(f"Execution finished: {len(tasks) - failed} done, {failed} failed")
    return 1 if failed else 0

# ========== Task 3: compare ==========


def _compare_tables(a: pd.DataFrame, b: pd.DataFrame, tolerance: float) -> list:
    """
    Differences between two evaluation tables (same rows and columns expected); numbers within the tolerance and
    missing values in both count as equal.
    """
    if list(a.columns) != list(b.columns):
        return [f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"]
    if len(a) != len(b):
        return [f"{len(a)} rows vs {len(b)} rows"]
    keys = [c for c in ("agent", "time_step") if c in a.columns]
    a = a.sort_values(keys).reset_index(drop = True)
    b = b.sort_values(keys).reset_index(drop = True)

    differences = []
    for column in a.columns:
        x, y = a[column], b[column]
        both_missing = x.isna() & y.isna()
        numeric_x, numeric_y = pd.to_numeric(x, errors = "coerce"), pd.to_numeric(y, errors = "coerce")
        numeric = numeric_x.notna() & numeric_y.notna()
        equal = both_missing | (numeric & np.isclose(numeric_x, numeric_y, rtol = 0, atol = tolerance)) | (~numeric & (x.astype(str) == y.astype(str)))
        if not equal.all():
            row = int((~equal).idxmax())
            differences.append(f"{column}: {int((~equal).sum())} rows differ (e.g., agent {a.loc[row, 'agent'] if 'agent' in a else '?'}, "
                               f"time step {a.loc[row, 'time_step'] if 'time_step' in a else '?'}: {x[row]!r} vs {y[row]!r})")
    return differences


def compare(args) -> int:
    first, second = (os.path.join(args.input, "runs", label) for label in args.labels)
    identical, different, missing = 0, 0, 0

    for framework in sorted(os.listdir(first)):
        if not os.path.isdir(os.path.join(first, framework)):
            continue
        for name in sorted(n for n in os.listdir(os.path.join(first, framework)) if os.path.isdir(os.path.join(first, framework, n))):
            paths = [os.path.join(root, framework, name) for root in (first, second)]
            files = [(os.path.join(p, "evaluation.csv"), os.path.join(p, "fitness.json")) for p in paths]
            if not all(os.path.exists(f) for pair in files for f in pair):
                missing += 1
                print(f"{framework}/{name}: missing in one of the runs (failed execution?)")
                continue

            differences = _compare_tables(pd.read_csv(files[0][0]), pd.read_csv(files[1][0]), args.tolerance)
            fitness = [json.load(open(f[1])) for f in files]
            for key in fitness[0]:
                if fitness[0][key] != fitness[1].get(key):
                    differences.append(f"fitness {key}: {fitness[0][key]!r} vs {fitness[1].get(key)!r}"[:300])

            if differences:
                different += 1
                print(f"{framework}/{name}: DIFFERENT")
                for line in differences[:8]:
                    print(f"    {line}")
            else:
                identical += 1

    print(f"\n{args.labels[0]} vs {args.labels[1]}: {identical} identical, {different} different, {missing} missing")
    return 0 if different == 0 and missing == 0 else 1

# ========== Command line ==========


def main(argv = None) -> int:
    parser = argparse.ArgumentParser(description = "Execute the same scenarios before and after a change, and compare the results.")
    commands = parser.add_subparsers(dest = "command", required = True)

    g = commands.add_parser("generate", help = "valid scenarios from each framework's generation code")
    g.add_argument("--map", required = True, help = "CommonRoad map file")
    g.add_argument("--n", type = int, default = 5, help = "scenarios per framework")
    g.add_argument("--frameworks", nargs = "+", default = FRAMEWORKS, choices = FRAMEWORKS)
    g.add_argument("--seed", type = int, default = 1)
    g.add_argument("--max-attempts", type = int, default = 200, help = "generator calls per framework before giving up")
    g.add_argument("--output", default = "equivalence")

    e = commands.add_parser("execute", help = "execute and evaluate every generated scenario, in parallel")
    e.add_argument("--input", default = "equivalence")
    e.add_argument("--label", required = True, help = "name of this execution, e.g. before, after")
    e.add_argument("--workers", type = int, default = 4)

    c = commands.add_parser("compare", help = "compare two executions")
    c.add_argument("--input", default = "equivalence")
    c.add_argument("--labels", nargs = 2, required = True)
    c.add_argument("--tolerance", type = float, default = 1e-9, help = "absolute tolerance for numbers")

    args = parser.parse_args(argv)
    return {"generate": generate, "execute": execute, "compare": compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
