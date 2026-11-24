"""
Executes scenarios with Baidu Apollo, through DoppelTest's setup (EVITA_DoppelTest, Python 3.8).

Start of a run: checks the setup (.venv38's Python, the runner, the converted map is the HD map Apollo runs), then starts
Apollo's containers with EVITA_DoppelTest/apollo_runner_startup.py.

Each scenario, in <output folder>/<generation id, 5 digits>/ (as FrenetiX's executor):
1. traffic lights: a new random DoppelTest traffic control when `random_lights` is set.
    This is required for generators that do not evolve lights (Random, MaxDanger, MaxInteractions) or when the scenario has no DoppelTest lights
2. scenario.xml and c.json (a scenario that cannot be described for Apollo is invalid)
3. EVITA_DoppelTest/apollo_scenario_runner.py with .venv38's Python and a timeout;
4. The trace is converted into an ExecutionResult and its health check.

A rejected scenario is invalid.
A failed run (runner error, timeout, unreadable records) or an unhealthy one is retried RETRIES times.
The failed attempt kept in failed_attempt_<k>/; if every attempt fails, the scenario is invalid.
With `delete_records`, the .record files are deleted once the trace is written.
"""
import os
import shutil
import logging
import subprocess
from copy import deepcopy

from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

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
DEFAULT_PYTHON38 = os.path.join(ROOT_DIR, ".venv38", "bin", "python")

# Seconds one run of the runner may take (DoppelTest's 30 s scenario, plus setting the instances up and writing records)
EXECUTION_TIMEOUT = 600
# Retries of a scenario whose run failed or was unhealthy
RETRIES = 1

# The runner's exit codes (apollo_scenario_runner.py)
EXIT_EXECUTED, EXIT_REJECTED, EXIT_UNEXPECTED_ERROR, EXIT_RECORD_ERROR = 0, 2, 3, 10
RUNNER_FAILURES = {EXIT_UNEXPECTED_ERROR: "Unexpected error (Docker, Apollo or DoppelTest; see runner.log)",
                   EXIT_RECORD_ERROR: "The record files are missing or cannot be read"}

class ApolloSetupError(RuntimeError):
    pass

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
        self.python38 = kwargs.get("python38") or os.environ.get("EVITA_PYTHON38", DEFAULT_PYTHON38)
        self.timeout = float(kwargs.get("timeout") or os.environ.get("EVITA_APOLLO_TIMEOUT", EXECUTION_TIMEOUT))
        self.retries = int(kwargs.get("retries", RETRIES))

        self.max_time_step = simulators.generation_configuration(simulators.get_simulator(simulators.APOLLO)).max_time_step
        
        self._check_setup()
        
        if kwargs.get("start_containers", True):
            self._start_containers()

    # ---------- Start of a run ----------

    def _environment(self):
        return dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python")

    def _check_setup(self) -> None:
        """
        Check .venv38's Python virtual environment, the runner and the start-up script; the converted map is the HD map Apollo runs
        (DoppelTest's HD_MAP_PATH: <maps>/<map_name>/base_map.bin, converted map <map_name>.xml)
        """
        missing = [path for path in (self.python38, RUNNER, STARTUP) if not os.path.exists(path)]
        if missing:
            raise ApolloSetupError(f"Missing for Apollo: {', '.join(missing)} (.venv38's Python can be given with EVITA_PYTHON38)")

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
            result = subprocess.run([self.python38, STARTUP], cwd = DOPPELTEST_ROOT, stdout = log, stderr = subprocess.STDOUT, env = self._environment())
        
        if result.returncode != 0:
            raise ApolloSetupError(f"Apollo's containers did not start (exit {result.returncode}, see {log_path})")

    # ---------- One scenario ----------

    def execute(self, scenario, planning_problem_set, progress_bar_manager = None, on_step = None) -> ExecutionResult:
        generation_id = scenario.generation_id
        folder = os.path.join(self.output_folder, str(generation_id).rjust(5, "0"))
        os.makedirs(folder, exist_ok = True)

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
            code, message = self._run(folder, generation_id)

            if on_step is not None:
                on_step()

            if code == EXIT_REJECTED:
                return self._failed(generation_id, folder, scenario, message)

            if code == EXIT_EXECUTED:
                try:
                    execution = tr.to_execution_result(tr.read_trace(os.path.join(folder, "trace.json.gz")), scenario, self.map_data, folder, scenario_id = generation_id)
                except Exception as exc:
                    return self._failed(generation_id, folder, scenario, f"Apollo's trace cannot be converted: {type(exc).__name__}: {exc}")
                
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

        command = [self.python38, RUNNER, "--scenario-json", os.path.join(folder, "c.json"), "--records-dir", os.path.join(folder, "records"),
                   "--scenario", str(generation_id), "--trace", trace_path]
        
        with open(os.path.join(folder, "runner.log"), "w") as log:
            try:
                code = subprocess.run(command, cwd = DOPPELTEST_ROOT, stdout = log, stderr = subprocess.STDOUT, env = self._environment(), timeout = self.timeout).returncode
            except subprocess.TimeoutExpired:
                return EXIT_UNEXPECTED_ERROR, f"Apollo's runner timed out after {self.timeout:.0f} s"

        if code == EXIT_EXECUTED:
            return code, ""
        
        if code == EXIT_REJECTED:
            reason = ""
            try:
                reason = tr.read_trace(trace_path).reason
            except Exception:
                pass
            return code, f"Apollo rejected the scenario: {reason or 'see runner.log'}"
        
        return code, f"Apollo's runner failed: {RUNNER_FAILURES.get(code, f'exit {code}')}"

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
