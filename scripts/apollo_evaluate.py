"""
Evaluate Apollo traces with EVITA's evaluator and generate EVITA's report, without running EVITA's search (test of
step 2 on real data; step 3's executor will do the same for each scenario of a search).

    python scripts/apollo_evaluate.py --traces <folder> --map <converted map.xml> --out <run folder> [--no-report]

Every trace.json.gz under --traces is one scenario; its generated scenario is the scenario.xml next to it (as written by
scripts/apollo_live_run.py), else the converted map (old runs without the generated scenario). For each scenario:
<run folder>/<scenario name>/logs/simulation.db (the evaluation, read by the report), then the report of the run
folder (data_analysis.generate_report), as for a FrenetiX run. A summary per scenario is printed.

Run in .venv310.
"""

import os
import sys
import glob
import logging
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from commonroad.common.file_reader import CommonRoadFileReader

from evita.apollo.map_data import ApolloMapData
from evita.apollo import trace as tr
from evita.evaluators.scenario_evaluator import ScenarioEvaluator
from evita.data_analysis import generate_report


def scenario_name(trace_path: str, traces_root: str) -> str:
    folder = os.path.relpath(os.path.dirname(trace_path), traces_root)
    return "root" if folder == "." else folder.replace(os.sep, "_")


def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Evaluate Apollo traces and generate EVITA's report")
    parser.add_argument("--traces", required = True, help = "folder with trace.json.gz files (searched recursively)")
    parser.add_argument("--map", required = True, help = "converted CommonRoad map (its id files next to it)")
    parser.add_argument("--out", required = True, help = "run folder to write (one subfolder per scenario)")
    parser.add_argument("--no-report", action = "store_true", help = "only evaluate")
    args = parser.parse_args(argv)
    traces_root, out = os.path.abspath(args.traces), os.path.abspath(args.out)

    logging.basicConfig(level = logging.ERROR)
    map_data = ApolloMapData.load(args.map)
    traces = sorted(glob.glob(os.path.join(traces_root, "**", "trace.json.gz"), recursive = True))
    print(f"{len(traces)} traces in {traces_root}")

    evaluated = 0
    for path in traces:
        name = scenario_name(path, traces_root)
        trace = tr.read_trace(path)
        if trace.status != tr.EXECUTED:
            print(f"  {name}: not executed ({trace.status}: {trace.reason}), skipped")
            continue
        scenario_xml = os.path.join(os.path.dirname(path), "scenario.xml")
        generated, _ = CommonRoadFileReader(scenario_xml if os.path.exists(scenario_xml) else args.map).open()

        folder = os.path.join(out, name)
        execution = tr.to_execution_result(trace, generated, map_data, folder, scenario_id = name)
        table = ScenarioEvaluator(folder).evaluate(execution, None)
        evaluated += 1

        statuses = {}
        for agent in execution.agents.values():
            statuses[agent.status] = statuses.get(agent.status, 0) + 1
        interacting = int(table["is_interacting"].astype(float).sum()) if len(table) else 0
        collided = sorted({a for a, _ in table.index[table["has_collided"].astype(float) == 1.0]}) if len(table) else []
        print(f"  {name}: {len(execution.agents)} vehicles {statuses}; {interacting} interacting time steps; "
              f"max danger {float(table['max_danger'].astype(float).max()) if len(table) else 0:.2f}; collided {collided}")

    print(f"\n{evaluated} scenarios evaluated into {out}")
    if evaluated and not args.no_report:
        generate_report(out)
        print(f"Report generated in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
