"""
Export a trace from record files that already exist (e.g. from earlier Apollo runs), without running Apollo.

    python export_existing_records.py --records-dir <folder with ROUTE_k.<scenario> files> --trace <trace.json.gz>
                                      [--scenario-name Scenario_3] [--ids-map <ids_map.json>]

The folder is DoppelTest's records folder of one scenario (<records>/<generation>/<scenario>), which also holds the
scenario JSON (c.json). Every container with record files becomes a vehicle; the container ...ROUTE_k (e.g.
apollo_dev_ROUTE_k) is the k-th vehicle of the JSON. Apollo ids (under which the vehicles perceive each other) come from the old runner's ids_map.json
({apollo id: container}) when given; EVITA ids from the JSON's "evita_vehicle_ids" when it has them (older JSONs do not:
scripts/apollo_check.py can take them from the generated scenario instead).
"""

import os
import re
import sys
import json
import argparse

from hdmap.MapParser import MapParser
from config import (APOLLO_VEHICLE_LENGTH, APOLLO_VEHICLE_WIDTH, APOLLO_VEHICLE_back_edge_to_center, HD_MAP_PATH,
                    MAX_ADC_COUNT, SCENARIO_UPPER_LIMIT)
from record_export import RecordExportError, build_trace, write_trace


def containers_with_records(records_dir: str, scenario_name: str):
    """
    Containers (named ...ROUTE_k, e.g. apollo_dev_ROUTE_k) that have record files for the scenario, in order of k
    """
    pattern = re.compile(r"^(\w*ROUTE_(\d+))\." + re.escape(scenario_name) + r"(\.\d+)?$")
    found = {}
    for filename in os.listdir(records_dir):
        match = pattern.match(filename)
        if match:
            found[match.group(1)] = int(match.group(2))
    return sorted(found, key = lambda container: found[container])


def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Export a trace from existing record files")
    parser.add_argument("--records-dir", required = True, help = "records folder of one scenario")
    parser.add_argument("--trace", required = True, help = "trace file to write (.json.gz)")
    parser.add_argument("--scenario-name", help = "scenario name in the record file names (default: the folder's name)")
    parser.add_argument("--ids-map", help = "the old runner's ids_map.json ({apollo id: container})")
    args = parser.parse_args(argv)

    records_dir = os.path.abspath(args.records_dir)
    scenario_name = args.scenario_name or os.path.basename(records_dir.rstrip("/"))

    json_path = os.path.join(records_dir, "c.json")
    json_scenario = json.load(open(json_path)) if os.path.exists(json_path) else {}
    evita_ids = json_scenario.get("evita_vehicle_ids")

    apollo_ids = {}
    if args.ids_map:
        apollo_ids = {container: int(apollo_id) for apollo_id, container in json.load(open(args.ids_map)).items()}

    containers = containers_with_records(records_dir, scenario_name)
    if not containers:
        print(f"No record files named <container>ROUTE_k.{scenario_name} in {records_dir}", file = sys.stderr)
        return 10

    vehicles = []
    for container in containers:
        index = int(container.rsplit("_", 1)[-1])
        adcs = json_scenario.get("ad_section", {}).get("adcs", [])
        adc = adcs[index] if index < len(adcs) else {}
        vehicles.append({
            "container": container,
            "apollo_id": apollo_ids.get(container),
            "json_index": index,
            "evita_id": evita_ids[index] if evita_ids and index < len(evita_ids) else None,
            "routing": adc.get("routing", []), "start_s": adc.get("start_s"), "dest_s": adc.get("dest_s"),
            "start_t": adc.get("start_t")
        })

    scenario = {
        "generation": os.path.basename(os.path.dirname(records_dir)), "scenario": scenario_name,
        "status": "executed", "reason": "exported from existing record files",
        "hd_map": HD_MAP_PATH, "scenario_upper_limit_s": SCENARIO_UPPER_LIMIT, "max_adc_count": MAX_ADC_COUNT,
        "vehicle": {"length": APOLLO_VEHICLE_LENGTH, "width": APOLLO_VEHICLE_WIDTH,
                    "back_edge_to_center": APOLLO_VEHICLE_back_edge_to_center},
        "json": json_scenario
    }

    # DoppelTest's stop sign oracle reads the HD map
    MapParser(HD_MAP_PATH)
    try:
        trace = build_trace(records_dir, scenario_name, scenario, vehicles)
    except RecordExportError as exc:
        print(f"Cannot export the records: {exc}", file = sys.stderr)
        return 10

    write_trace(trace, args.trace)
    print(f"Trace of {len(vehicles)} vehicles ({', '.join(containers)}) written to {args.trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
