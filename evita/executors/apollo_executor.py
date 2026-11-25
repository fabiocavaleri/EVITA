"""
Executes scenarios with Baidu Apollo, through DoppelTest's setup (EVITA_DoppelTest), with EVITA's own Python.

Start of a run: checks the setup (the runner and start-up script, the converted map is the HD map Apollo runs), then starts
Apollo's containers with EVITA_DoppelTest/apollo_runner_startup.py.

Each scenario, in <output folder>/<generation id, 5 digits>/ (as FrenetiX's executor):
1. traffic lights: a new random DoppelTest traffic control when `random_lights` is set.
    This is required for generators that do not evolve lights (Random, MaxDanger, MaxInteractions) or when the scenario has no DoppelTest lights
2. scenario.xml and c.json (a scenario that cannot be described for Apollo is invalid)
3. EVITA_DoppelTest/apollo_scenario_runner.py in its own process (EVITA's Python; EVITA_APOLLO_PYTHON overrides it), with a timeout;
4. The trace is converted into an ExecutionResult and its health check.

A rejected scenario is invalid.
A failed run (runner error, timeout, unreadable records) or an unhealthy one is retried RETRIES times.
The failed attempt kept in failed_attempt_<k>/; if every attempt fails, the scenario is invalid.
With `delete_records`, the .record files are deleted once the trace is written.
"""
import os
import re
import sys
import time
import shutil
import logging
import threading
import subprocess
from copy import deepcopy

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

from evita.timing import TASK_TIMER

from .. import simulators
from ..interfaces import AbstractScenarioExecutor
from ..execution import ExecutionResult, STATUS_COLLISION
from ..utils import EVITA_MAIN_LOGGER_NAME, ROOT_DIR
from ..apollo import trace as tr
from ..apollo import traffic_control as tc
from ..apollo import scenario_converter as converter
from ..apollo.health import health_problems
from ..apollo.map_data import ApolloMapData, MapDataError

DOPPELTEST_ROOT = os.path.join(ROOT_DIR, "EVITA_DoppelTest")
RUNNER = os.path.join(DOPPELTEST_ROOT, "apollo_scenario_runner.py")
STARTUP = os.path.join(DOPPELTEST_ROOT, "apollo_runner_startup.py")

# Seconds one run of the runner may take (DoppelTest's 30 s scenario, plus setting the instances up and writing records)
EXECUTION_TIMEOUT = 600
# Retries of a scenario whose run failed or was unhealthy
RETRIES = 1

# The runner's exit codes (apollo_scenario_runner.py)
EXIT_EXECUTED, EXIT_REJECTED, EXIT_UNEXPECTED_ERROR, EXIT_RECORD_ERROR = 0, 2, 3, 10
RUNNER_FAILURES = {EXIT_UNEXPECTED_ERROR: "Unexpected error (Docker, Apollo or DoppelTest; see runner.log)",
                   EXIT_RECORD_ERROR: "The record files are missing or cannot be read"}

# DoppelTest's scenario ticks in the runner's output, and the seconds without a new
# tick after which the scenario is over and the runner is saving the records and writing the trace
SCENARIO_TICK = re.compile(r"Scenario time: (\d+(?:\.\d+)?)")
SAVING_AFTER_SECONDS = 2.0
# How often the progress bars are refreshed while the runner works (seconds)
PROGRESS_REFRESH = 0.5

class ApolloSetupError(RuntimeError):
    pass

class RunnerOutput(threading.Thread):
    """
    Reads the runner's output as it arrives: writes it to runner.log (as before) and keeps the latest scenario time
    """
    def __init__(self, stream, log):
        super().__init__(daemon = True)
        self.stream, self.log = stream, log
        self.scenario_time = None     # seconds, from DoppelTest's latest tick
        self.last_tick = None         # when that tick arrived (time.perf_counter())
        self._tail = ""

    def run(self):
        for chunk in iter(lambda: self.stream.read1(4096), b""):
            text = chunk.decode(errors = "replace")
            self.log.write(text)
            self.log.flush()
            self._tail = (self._tail + text)[-200:]
            ticks = SCENARIO_TICK.findall(self._tail)
            if ticks:
                self.scenario_time = float(ticks[-1])
                self.last_tick = time.perf_counter()

class ApolloScenarioExecution(AbstractScenarioExecutor):
    """
    Executes scenarios with Baidu Apollo
    """
    def __init__(self, output_folder: str, **kwargs) -> None:
        self.output_folder = output_folder

        self.debug = bool(kwargs.get("debug", False))
        self.logger = logging.getLogger(EVITA_MAIN_LOGGER_NAME)
        
        if not kwargs.get("map_file"):
            raise ApolloSetupError("The Apollo executor needs the converted map (map_file)")
        self.map_file = os.path.abspath(kwargs["map_file"])
        self.map_data = ApolloMapData.load(self.map_file)
        
        self.delete_records = bool(kwargs.get("delete_records", False))
        self.random_lights = bool(kwargs.get("random_lights", False))
        self.python = kwargs.get("python") or os.environ.get("EVITA_APOLLO_PYTHON") or sys.executable
        self.timeout = float(kwargs.get("timeout") or os.environ.get("EVITA_APOLLO_TIMEOUT", EXECUTION_TIMEOUT))
        self.retries = int(kwargs.get("retries", RETRIES))

        self.max_time_step = simulators.generation_configuration(simulators.get_simulator(simulators.APOLLO)).max_time_step
        self.time_step = simulators.frenetix_time_step()
        
        self._check_setup()
        
        if kwargs.get("start_containers", True):
            self._start_containers()

    # ---------- Start of a run ----------

    def _environment(self):
        return dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python")

    def _check_setup(self) -> None:
        """
        Check the Python that runs the runner, the runner and the start-up script; the converted map is the HD map Apollo runs
        (DoppelTest's HD_MAP_PATH: <maps>/<map_name>/base_map.bin, converted map <map_name>.xml)
        """
        missing = [path for path in (self.python, RUNNER, STARTUP) if not os.path.exists(path)]
        if missing:
            raise ApolloSetupError(f"Missing for Apollo: {', '.join(missing)} (the runner's Python can be given with EVITA_APOLLO_PYTHON)")

        from EVITA_DoppelTest.config import HD_MAP_PATH
        hd_map = os.path.basename(os.path.dirname(os.path.normpath(HD_MAP_PATH)))
        converted = os.path.splitext(os.path.basename(self.map_file))[0]
        if hd_map != converted:
            raise ApolloSetupError(f"The converted map {converted} is not the HD map Apollo runs ({hd_map}, HD_MAP_PATH in "
                                   f"EVITA_DoppelTest/config.py): use the converted map of {hd_map}, or change HD_MAP_PATH "
                                   f"(and Apollo's global_flagfile)")

    def _start_containers(self) -> None:
        """
        Start Apollo's containers using `apollo_runner_startup.py`; its output goes to apollo_startup.log
        """
        os.makedirs(self.output_folder, exist_ok = True)
        log_path = os.path.join(self.output_folder, "apollo_startup.log")

        self.logger.info(f"Starting Apollo's containers (log: {log_path})")
        
        with open(log_path, "w") as log:
            result = subprocess.run([self.python, STARTUP], cwd = DOPPELTEST_ROOT, stdout = log, stderr = subprocess.STDOUT, env = self._environment())
        
        if result.returncode != 0:
            raise ApolloSetupError(f"Apollo's containers did not start (exit {result.returncode}, see {log_path})")

    # ---------- One scenario ----------

    def execute(self, scenario, planning_problem_set, progress_bar_manager = None, on_step = None) -> ExecutionResult:
        generation_id = scenario.generation_id

        # The scenario's progress bar while Apollo runs, and the main bar's refresh
        self._progress_bar_manager, self._on_step = progress_bar_manager, on_step

        folder = os.path.join(self.output_folder, str(generation_id).rjust(5, "0"))
        os.makedirs(folder, exist_ok = True)

        # Preparation: lights, DoppelTest's JSON, the scenario file (EVITA's task timer: simulation.apollo_preparation)
        with TASK_TIMER.measure("simulation.apollo_preparation", key = generation_id):
            # The search keeps its scenario: lights drawn here go to a copy
            scenario = deepcopy(scenario)
            self._set_lights(scenario)

            try:
                scenario_json = converter.build_scenario_json(scenario, planning_problem_set, self.map_data)
            except (converter.ScenarioSpecError, MapDataError) as exc:
                return self._failed(generation_id, folder, scenario, f"Apollo cannot run the scenario: {exc}")
        
            CommonRoadFileWriter(scenario, planning_problem_set).write_to_file(os.path.join(folder, "scenario.xml"), OverwriteExistingFile.ALWAYS)
            converter.write_scenario_json(scenario_json, os.path.join(folder, "c.json"))

        message, execution = None, None
        for attempt in range(1, self.retries + 2):
            code, message, runner_seconds = self._run(folder, generation_id)

            if on_step is not None:
                on_step()

            # A run without a trace: only its wall time (simulation.apollo_runner)
            if code != EXIT_EXECUTED:
                self._time_runner(generation_id, runner_seconds, {})

            if code == EXIT_REJECTED:
                return self._failed(generation_id, folder, scenario, message)

            if code == EXIT_EXECUTED:
                # The trace and its conversion (simulation.apollo_trace); then the runner's phases, from the trace
                with TASK_TIMER.measure("simulation.apollo_trace", key = generation_id):
                    try:
                        trace = tr.read_trace(os.path.join(folder, "trace.json.gz"))
                        execution = tr.to_execution_result(trace, scenario, self.map_data, folder, scenario_id = generation_id)
                    except Exception as exc:
                        trace, failure = None, f"Apollo's trace cannot be converted: {type(exc).__name__}: {exc}"
                
                self._time_runner(generation_id, runner_seconds, trace.scenario.get("timings", {}) if trace is not None else {})
                
                if trace is None:
                    return self._failed(generation_id, folder, scenario, failure)

                # Health check and records cleanup (simulation.apollo_health)
                with TASK_TIMER.measure("simulation.apollo_health", key = generation_id):
                    collisions = {a.id: a.last_timestep for a in execution.agents.values() if a.status == STATUS_COLLISION}
                    problems = health_problems(execution.native, collisions)
                    self._records_done(folder)

                if not problems:
                    return execution
                
                message = "Apollo's run is unhealthy: " + "; ".join(problems)

            if attempt <= self.retries:
                self.logger.warning(f"Scenario {generation_id}, attempt {attempt}: {message}. Retrying")
                self._keep_failed_attempt(folder, attempt)

        self.logger.critical(f"Scenario {generation_id}: {message} (after {self.retries + 1} attempts)")
        if execution is not None:
            execution.failure = message
            return execution
        
        return self._failed(generation_id, folder, scenario, message)

    def _set_lights(self, scenario) -> None:
        """
        A new random DoppelTest traffic control when random_lights is set or the scenario has no DoppelTest lights
        """
        try:
            tc.TrafficControl.from_commonroad(scenario.lanelet_network, scenario.dt)
            has_lights = True
        except tc.TrafficControlError:
            has_lights = False

        if self.random_lights or not has_lights:
            lights = [light.traffic_light_id for light in scenario.lanelet_network.traffic_lights]
            control = tc.random_traffic_control(self.map_data, lights, tc.max_duration_green(self.max_time_step, scenario.dt))
            control.to_commonroad(scenario.lanelet_network, scenario.dt)


    def _run(self, folder: str, generation_id):
        """
        One run of the runner: (exit code, message); a timeout counts as an unexpected error
        """
        trace_path = os.path.join(folder, "trace.json.gz")
        if os.path.exists(trace_path):
            os.remove(trace_path)

        command = [self.python, RUNNER, "--scenario-json", os.path.join(folder, "c.json"), "--records-dir", os.path.join(folder, "records"),
                   "--scenario", str(generation_id), "--trace", trace_path]

        started = time.perf_counter()
        bar = self._execution_bar()
        try:
            with open(os.path.join(folder, "runner.log"), "w") as log:
                process = subprocess.Popen(command, cwd = DOPPELTEST_ROOT, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, env = self._environment())
                output = RunnerOutput(process.stdout, log)
                output.start()

                # Wait for the runner, refreshing the progress bars; past the timeout it is stopped
                while True:
                    try:
                        code = process.wait(timeout = PROGRESS_REFRESH)
                        break
                    except subprocess.TimeoutExpired:
                        self._show_progress(bar, output)
                        if time.perf_counter() - started > self.timeout:
                            process.kill()
                            process.wait()
                            output.join(timeout = 5)
                            return EXIT_UNEXPECTED_ERROR, f"Apollo's runner timed out after {self.timeout:.0f} s", time.perf_counter() - started
                output.join(timeout = 5)
        finally:
            if bar is not None:
                bar.close(clear = True)

        if code == EXIT_EXECUTED:
            return code, "", time.perf_counter() - started
        
        if code == EXIT_REJECTED:
            reason = ""
            try:
                reason = tr.read_trace(trace_path).reason
            except Exception:
                pass
            return code, f"Apollo rejected the scenario: {reason or 'see runner.log'}", time.perf_counter() - started
        
        return code, f"Apollo's runner failed: {RUNNER_FAILURES.get(code, f'exit {code}')}", time.perf_counter() - started
    
    def _execution_bar(self):
        """
        The scenario's progress bar (as FrenetiX's "Scenario Execution"): Apollo's time steps, from DoppelTest's ticks
        """
        manager = getattr(self, "_progress_bar_manager", None)

        if manager is None:
            return None
        
        return manager.counter(total = self.max_time_step, desc = "Scenario Execution (Apollo: setting up)", unit = "timesteps", color = "orange", leave = False)

    def _show_progress(self, bar, output: RunnerOutput) -> None:
        """
        Update the scenario's bar from the latest tick, its description from the phase, and the main progress bar
        """
        if bar is not None:
            if output.scenario_time is None:
                phase = "setting up"
            elif time.perf_counter() - output.last_tick > SAVING_AFTER_SECONDS:
                phase = "saving records and trace"
            else:
                phase = "running"

            bar.count = min(self.max_time_step, int(round(output.scenario_time / self.time_step))) if output.scenario_time is not None else 0
            bar.desc = f"Scenario Execution (Apollo: {phase})"
            bar.refresh()
        
        on_step = getattr(self, "_on_step", None)
        
        if on_step is not None:
            on_step()

    def _time_runner(self, generation_id, wall_seconds: float, phases: dict) -> None:
        """
        EVITA's task timer: the runner's phases and the rest of its wall time
        """
        known = {phase: float(seconds) for phase, seconds in phases.items()}

        for phase, seconds in known.items():
            TASK_TIMER.add(f"simulation.apollo_runner.{phase}", seconds, key = generation_id)
        
        TASK_TIMER.add("simulation.apollo_runner", wall_seconds - sum(known.values()), key = generation_id)

    def _records_done(self, folder: str) -> None:
        """
        The trace is written: delete the .record files if asked
        """
        if self.delete_records:
            shutil.rmtree(os.path.join(folder, "records"), ignore_errors = True)

    def _keep_failed_attempt(self, folder: str, attempt: int) -> None:
        """
        Move a failed attempt's outputs (trace, runner log, records) to failed_attempt_<k>/
        """
        target = os.path.join(folder, f"failed_attempt_{attempt}")
        os.makedirs(target, exist_ok = True)

        for name in ("trace.json.gz", "runner.log", "records"):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                shutil.move(path, os.path.join(target, name))

        if self.delete_records:
            shutil.rmtree(os.path.join(target, "records"), ignore_errors = True)

    def _failed(self, generation_id, folder: str, scenario, message: str) -> ExecutionResult:
        self.logger.critical(f"Scenario {generation_id}: {message}")

        return ExecutionResult(scenario_id = generation_id, output_folder = folder, executed_scenario = scenario, agents = {}, simulator = "apollo", failure = message)
