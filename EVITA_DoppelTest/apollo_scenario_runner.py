"""
Run one scenario with Apollo and write its trace (Python 3.8 side).

    python apollo_scenario_runner.py --scenario-json <scenario.json> --records-dir <dir> \
                                     --scenario <i> --trace <trace.json.gz>

The Apollo containers must already be running (apollo_runner_startup.py).
The scenario JSON is DoppelTest's format; if it has an "evita_vehicle_ids" list (one id per vehicle, in order), the trace keeps EVITA's id of each vehicle.

Before running, the scenario is checked against what DoppelTest would silently change: DoppelTest drops a vehicle that
starts within 5 m of an earlier one, and runs at most MAX_ADC_COUNT vehicles.
Either change would shift the vehicles to the wrong containers, so such a scenario is not run.

Exit codes:
    0   the scenario ran and its trace was written
    2   the scenario was rejected before running (the trace explains why; no record files)
    10  the record files are missing or cannot be read
    3   unexpected error (e.g. Docker or Apollo failure): the message is on stderr
    1   wrong arguments
"""
# The process's start, before the imports (its start-up time is part of the runner's timings)
import time
PROCESS_START = time.time()

import os
import sys
import json
import logging
import argparse
import traceback
from typing import Any, Dict, List

from hdmap.MapParser import MapParser
from apollo.ApolloContainer import ApolloContainer
from framework.scenario import Scenario as ApolloScenario
from framework.scenario.ScenarioRunner import ScenarioRunner
from config import (APOLLO_ROOT, APOLLO_VEHICLE_LENGTH, APOLLO_VEHICLE_WIDTH, APOLLO_VEHICLE_back_edge_to_center,
                    HD_MAP_PATH, MAX_ADC_COUNT, SCENARIO_UPPER_LIMIT, RuntimeConfig)

from utils import get_scenario_logger
from record_export import RecordExportError, build_trace, write_trace

EXIT_OK = 0
EXIT_WRONG_ARGUMENTS = 1
EXIT_REJECTED = 2
EXIT_RECORD_ERROR = 10
EXIT_UNEXPECTED_ERROR = 3

def scenario_section(args, status: str, json_scenario: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    """
    Description of the run kept in the trace: names, outcome, map, the scenario as given, DoppelTest's settings and
    the vehicle's dimensions (Apollo's position is a point at back_edge_to_center from the rear edge, not the center)
    """
    return {
        # "generation": f"Generation_{args.generation}",
        "scenario": f"Scenario_{args.scenario}",
        "status": status,
        "reason": reason,
        "hd_map": HD_MAP_PATH,
        "scenario_upper_limit_s": SCENARIO_UPPER_LIMIT,
        "max_adc_count": MAX_ADC_COUNT,
        "vehicle": {"length": APOLLO_VEHICLE_LENGTH, "width": APOLLO_VEHICLE_WIDTH, "back_edge_to_center": APOLLO_VEHICLE_back_edge_to_center},
        "json": json_scenario
    }

def check_doppeltest_changes(json_scenario: Dict[str, Any], apollo_scenario) -> str:
    """
    Reason why DoppelTest would not run the scenario as given, or an empty string
    """
    requested = len(json_scenario["ad_section"]["adcs"])
    kept = len(apollo_scenario.ad_section.adcs)

    if requested > MAX_ADC_COUNT:
        return f"{requested} vehicles, but at most {MAX_ADC_COUNT} Apollo instances"

    if kept != requested:
        return f"DoppelTest dropped {requested - kept} of {requested} vehicles (start within 5 m of another vehicle)"
    
    return ""

def vehicles_that_ran(runners, json_scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Container, Apollo id and EVITA id of each vehicle that ran.
    Check that DoppelTest kept every vehicle in order, so the i-th runner drives the i-th vehicle of the JSON; the routing is compared to make sure.
    """
    adcs = json_scenario["ad_section"]["adcs"]
    evita_ids = json_scenario.get("evita_vehicle_ids")

    vehicles = []
    for json_index, (runner, agent) in enumerate(runners):
        if list(agent.routing) != list(adcs[json_index]["routing"]):
            raise RuntimeError(f"Vehicle {json_index} ran a different routing than the scenario JSON")

        vehicles.append({
            "container": runner.container.container_name,
            "apollo_id": runner.nid,
            "json_index": json_index,
            "evita_id": evita_ids[json_index] if evita_ids else None,
            "routing": list(agent.routing),
            "start_s": agent.start_s,
            "dest_s": agent.dest_s,
            "start_t": agent.start_t
        })
        
    return vehicles

class ScenarioClock(logging.Handler):
    """
    Times of DoppelTest's first and last scenario ticks ("Scenario time: ...", logged every 0.1 s by ScenarioRunner):
    the scenario's own duration, without changing DoppelTest
    """
    def __init__(self):
        super().__init__(level = logging.INFO)
        self.first = None
        self.last = None

    def emit(self, record):
        if str(record.getMessage()).startswith("Scenario time:"):
            if self.first is not None:
                self.first = self.first
            else:
                self.first = record.created

            self.last = record.created

def phase_timings(marks: Dict[str, float], clock: ScenarioClock) -> Dict[str, float]:
    """
    Seconds of each phase of the run, from the marks taken in main and the scenario's ticks:
    - Startup (imports, arguments),
    - Scenario_loading (map, JSON, DoppelTest's checks),
    - Instances (ScenarioRunner, set_scenario, init_scenario),
    - Setup (run_scenario until the first tick),
    - Scenario (first to last tick),
    - Recorders (last tick until run_scenario),
    - Recorders (last tick until run_scenario returns: stopping the recorders, saving the records),
    - Export (the trace built from the records)
    """
    timings = {"startup": marks["main"] - PROCESS_START,
               "scenario_loading": marks["loaded"] - marks["main"],
               "instances": marks["run"] - marks["loaded"]}
    
    if clock.first is not None:
        timings.update(setup = clock.first - marks["run"], scenario = clock.last - clock.first, recorders = marks["ran"] - clock.last)
    else:
        timings["run_scenario"] = marks["ran"] - marks["run"]
    
    if "exported" in marks:
        timings["export"] = marks["exported"] - marks["ran"]
    
    return {phase: round(max(0.0, seconds), 3) for phase, seconds in timings.items()}

def main(args) -> int:
    """
    Run one scenario with Apollo and write its trace.
    The scenario is rejected if DoppelTest would silently change it.
    """
    marks = {"main": time.time()}

    # DoppelTest's scenario classes and oracles use the map parser
    MapParser(HD_MAP_PATH)

    # Open the scenario JSON specifications and parse it into Apollo's classes
    with open(args.scenario_json, "r") as f:
        json_scenario = json.load(f)

    apollo_scenario = ApolloScenario.from_json(args.scenario_json)

    # Check if DoppelTest would silently change the scenario; if so, reject it and write a trace explaining why
    reason = check_doppeltest_changes(json_scenario, apollo_scenario)
    if reason:
        write_trace({"format_version": 1, "scenario": scenario_section(args, "rejected", json_scenario, reason), "vehicles": []}, args.trace)
        print(f"Scenario rejected: {reason}", file = sys.stderr)
        return EXIT_REJECTED

    marks["loaded"] = time.time()

    # Set the names of the records folder
    # Names of the records folder: DoppelTest saves the record files to <records dir>/<generation>/<scenario>;
    # EVITA gives every scenario its own records folder, so the generation name is empty (os.path.join skips it) and the
    # records go to <records dir>/<scenario>. The scenario name also names the record files (<container>.<scenario>)
    # generation_name = f"Generation_{args.generation}"
    generation_name = ""
    scenario_name = f"Scenario_{args.scenario}"

    # Run the scenario; DoppelTest saves the record files to <records dir>/<generation>/<scenario>
    containers = [ApolloContainer(APOLLO_ROOT, f"ROUTE_{x}") for x in range(MAX_ADC_COUNT)]
    runner = ScenarioRunner(containers)
    runner.set_scenario(apollo_scenario)
    runner.init_scenario()

    # DoppelTest's scenario logger, set up by DoppelTest's own function (run_scenario then finds it ready), plus the clock
    clock = ScenarioClock()
    get_scenario_logger().addHandler(clock)
    marks["run"] = time.time()
    runners = runner.run_scenario(generation_name, scenario_name, True)
    marks["ran"] = time.time()

    # Build the trace from the record files and the scenario description, and write it to the given file
    records_dir = os.path.join(RuntimeConfig.RECORDS_DIR, generation_name, scenario_name)
    try:
        trace = build_trace(records_dir, scenario_name, scenario_section(args, "executed", json_scenario), vehicles_that_ran(runners, json_scenario))
    except RecordExportError as exc:
        print(f"Cannot export the records: {exc}", file = sys.stderr)
        return EXIT_RECORD_ERROR
    
    marks["exported"] = time.time()

    # Seconds of each phase, for EVITA's task timer (the trace's scenario section)
    trace["scenario"]["timings"] = phase_timings(marks, clock)
    write_trace(trace, args.trace)
    return EXIT_OK

def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(description = "Run one scenario with Apollo and write its trace")
    parser.add_argument("--scenario-json", required = True, help = "Scenario in DoppelTest's JSON format")
    parser.add_argument("--records-dir", required = True, help = "Folder where DoppelTest saves the record files")
    # parser.add_argument("--generation", required = True, help = "Generation number (names the records folder)")
    parser.add_argument("--scenario", required = True, help = "Scenario number (names the records folder)")
    parser.add_argument("--trace", required = True, help = "Trace file to write (.json.gz)")

    return parser.parse_args(argv)

def run(argv: List[str]) -> int:
    """
    Parse the arguments, run the scenario and return the exit code
    """
    try:
        arguments = parse_arguments(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_WRONG_ARGUMENTS

    RuntimeConfig.RECORDS_DIR = arguments.records_dir
    
    try:
        return main(arguments)
    except Exception:
        # Any other failure (Docker, Apollo, DoppelTest) must not look like wrong arguments (Python's default exit code 1)
        traceback.print_exc()
        return EXIT_UNEXPECTED_ERROR

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))