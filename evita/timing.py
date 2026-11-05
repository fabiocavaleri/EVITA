"""
Time spent in each task of the pipeline, written to the run folder at the end of a run.

Usage: `with TASK_TIMER.measure("simulation", key = scenario.generation_id): ...`

Each task records its own time only: when tasks are nested (e.g., validation inside generation), the inner task's time is subtracted from the outer one,
so the totals of all tasks add up to the measured time without double counting.

Output (write()):
  - task_times.csv: per task, whether it counts against the time budget, number of calls, total and mean seconds,
    share of the run; then the unmeasured remainder, the wall-clock time of the run, and the budget's elapsed time.
  - task_times_per_scenario.csv: per scenario (generation id), the seconds of each task measured with a key,
    nested tasks included (e.g., the whole evaluation of the scenario).
"""
import csv
import os
import time

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

# Tasks run while the budget is paused (as in orchestrator.py and nsgaII_operators.CustomTimeTermination);
# every other task counts against the budget
NOT_COUNTED_TASKS = {"validation", "debug_plots", "checkpoints"}

# Tasks run after the budget is over
AFTER_BUDGET_TASKS = {"report"}

# Order of the tasks in the output (others follow in alphabetical order)
TASK_ORDER = ["generation", "validation", "debug_plots", "execution_preparation", "simulation",
              "evaluation", "evaluation.planned_trajectories", "evaluation.frenetix_metrics", "evaluation.danger",
              "evaluation.interactions", "evaluation.collisions", "evaluation.missed_goals", "evaluation.emergencies",
              "evaluation.database", "fitness", "checkpoints", "report"]


class TaskTimer:
    """
    Accumulates the own (exclusive) time of each task, overall and per scenario.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Forget all measurements and start the run's wall clock.
        """
        self.totals: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)
        self.per_key: Dict[object, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._stack: List[list] = []
        self.started_at = time.perf_counter()

    @contextmanager
    def measure(self, task: str, key: Optional[object] = None):
        """
        Time the enclosed block as `task` (and for the scenario `key`, if given), excluding nested measured tasks.
        """
        frame = [task, key, time.perf_counter(), 0.0]   # task, key, start, time of nested tasks
        self._stack.append(frame)
        try:
            yield
        finally:
            self._stack.pop()
            elapsed = time.perf_counter() - frame[2]
            own = max(0.0, elapsed - frame[3])

            self.totals[task] += own
            self.calls[task] += 1

            # Per scenario: the full time of the task, nested tasks included (e.g., all of a scenario's evaluation)
            if key is not None:
                self.per_key[key][task] += elapsed

            # The enclosing task does not count this time as its own
            if self._stack:
                self._stack[-1][3] += elapsed

    def _ordered_tasks(self) -> List[str]:
        known = [task for task in TASK_ORDER if task in self.totals]
        return known + sorted(task for task in self.totals if task not in TASK_ORDER)

    @staticmethod
    def _budget_status(task: str) -> str:
        if task in AFTER_BUDGET_TASKS:
            return "after budget"
        if task in NOT_COUNTED_TASKS:
            return "no"
        return "yes"

    def write(self, output_folder: str, budget_elapsed_seconds: Optional[float] = None) -> None:
        """
        Write task_times.csv and task_times_per_scenario.csv into the output folder.
        `budget_elapsed_seconds`: the budget's own count of elapsed (non-paused) time, as a cross-check of the
        counted tasks plus the unmeasured remainder.
        """
        wall_clock = time.perf_counter() - self.started_at
        measured = sum(self.totals.values())
        unmeasured = max(0.0, wall_clock - measured)
        if wall_clock > 0:
            share = lambda seconds: f"{seconds / wall_clock:.4f}" 
        else:
            share = ""

        with open(os.path.join(output_folder, "task_times.csv"), "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["task", "counted_in_budget", "calls", "total_seconds", "mean_seconds", "share_of_run"])

            for task in self._ordered_tasks():
                total, calls = self.totals[task], self.calls[task]
                writer.writerow([task, self._budget_status(task), calls, f"{total:.3f}", f"{total / calls:.3f}" if calls else "", share(total)])

            # Time between measured tasks (progress bars, logging, the search algorithm's own bookkeeping): counted
            writer.writerow(["unmeasured", "yes", "", f"{unmeasured:.3f}", "", share(unmeasured)])
            writer.writerow(["total (wall clock)", "", "", f"{wall_clock:.3f}", "", share(wall_clock)])

            counted = sum(t for task, t in self.totals.items() if self._budget_status(task) == "yes") + unmeasured
            writer.writerow(["total counted in budget", "yes", "", f"{counted:.3f}", "", share(counted)])

            if budget_elapsed_seconds is not None:
                writer.writerow(["budget elapsed (from the budget)", "yes", "", f"{budget_elapsed_seconds:.3f}", "", share(budget_elapsed_seconds)])

        keyed_tasks = sorted({task for tasks in self.per_key.values() for task in tasks}, key = lambda t: (TASK_ORDER.index(t) if t in TASK_ORDER else len(TASK_ORDER), t))
        with open(os.path.join(output_folder, "task_times_per_scenario.csv"), "w", newline = "") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["scenario"] + [f"{task}_seconds" for task in keyed_tasks])
            for key in sorted(self.per_key, key = str):
                writer.writerow([key] + [f"{self.per_key[key].get(task, 0.0):.3f}" for task in keyed_tasks])

# One timer per process, shared by all modules
TASK_TIMER = TaskTimer()