"""
Runs DoppelTest's own search (main_ga.py, unchanged) with one run's parameters; used by generate_DoppelTest_dataset.sh.

    python run_main_ga.py --output-folder <folder> [--hours <h>] [--map <map>]
    python run_main_ga.py --defaults          # prints config.py's run length and map: "<hours> <map>"

main_ga.py takes its settings from config.py when imported, and so do several DoppelTest modules (the Apollo runner
and containers, the message broker, the oracles, the scenario classes). This launcher sets the run's values on
config before importing anything else from DoppelTest, so DoppelTest's code runs as published:
    RUN_FOR_HOUR                --hours (default: config.py's)
    HD_MAP_PATH                 data/maps/<--map>/base_map.bin (default: config.py's map)
    RuntimeConfig.RECORDS_DIR   --output-folder: main_ga.py's records (Generation_*/Scenario_*), DoppelTest's
                                summary.csv (violations) go directly into the run's folder
MAX_ADC_COUNT and SCENARIO_UPPER_LIMIT stay as in config.py: EVITA takes them from there too, on Apollo.

The map must be the one Apollo loads (--map_dir in Apollo's modules/common/data/global_flagfile.txt): checked.
In the output folder: run_parameters.json (the run's values and main_ga.py's GA hyperparameters) and a copy of
config.py at the start; main_ga.py's logbook and hall of fame (data/log.bin, data/hof.bin, which every run
overwrites) and run_completed.json at the end. Must run from EVITA_DoppelTest, with
DoppelTest's Python 3.8 environment (.venv38).
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config


def apollo_map() -> str:
    """
    The map Apollo loads (--map_dir in its global_flagfile.txt), or "" if the file cannot be read
    """
    flagfile = os.path.join(config.APOLLO_ROOT, "modules", "common", "data", "global_flagfile.txt")
    try:
        with open(flagfile) as f:
            for line in f:
                line = line.strip()
                if line.startswith("--map_dir="):
                    return os.path.basename(line.split("=", 1)[1].rstrip("/"))
    except OSError:
        pass
    return ""


def main(argv) -> int:
    parser = argparse.ArgumentParser(description = "Run DoppelTest's search (main_ga.py) with one run's parameters")
    parser.add_argument("--output-folder", help = "the run's folder (main_ga.py's records go inside)")
    parser.add_argument("--hours", type = float, default = None, help = "run length in hours (default: config.py's RUN_FOR_HOUR)")
    parser.add_argument("--map", default = None, help = "DoppelTest map, a folder of data/maps (default: config.py's HD_MAP_PATH)")
    parser.add_argument("--defaults", action = "store_true", help = "print config.py's run length and map, and exit")
    args = parser.parse_args(argv)

    default_map = Path(config.HD_MAP_PATH).parent.name
    if args.defaults:
        print(f"{config.RUN_FOR_HOUR} {default_map}")
        return 0
    if not args.output_folder:
        parser.error("--output-folder is required")

    hours = config.RUN_FOR_HOUR if args.hours is None else (int(args.hours) if float(args.hours).is_integer() else args.hours)
    map_name = args.map or default_map
    map_path = os.path.join(str(config.DT_ROOT), "data", "maps", map_name, "base_map.bin")
    if not os.path.exists(map_path):
        print(f"Error: no DoppelTest map {map_name} ({map_path})", file = sys.stderr)
        return 2
    loaded = apollo_map()
    if loaded and loaded != map_name:
        print(f"Error: Apollo loads the map {loaded} (--map_dir in its global_flagfile.txt), not {map_name}: "
              f"use MAP={loaded}, or change Apollo's map", file = sys.stderr)
        return 2

    # The run's values, before main_ga.py and DoppelTest's modules read config
    config.RUN_FOR_HOUR = hours
    config.HD_MAP_PATH = map_path
    output_folder = os.path.abspath(args.output_folder)
    config.RECORDS_DIR = output_folder
    config.RuntimeConfig.RECORDS_DIR = output_folder
    os.makedirs(output_folder, exist_ok = True)

    # main_ga.py's GA hyperparameters (constants of its main()), read from its source for the record
    with open(os.path.join(HERE, "main_ga.py")) as f:
        source = f.read()
    hyperparameters = {name: float(value) if "." in value else int(value)
                       for name, value in re.findall(r"^\s*(POP_SIZE|OFF_SIZE|CXPB|MUTPB)\s*=\s*([0-9.]+)", source, re.M)}

    started = datetime.now()
    with open(os.path.join(output_folder, "run_parameters.json"), "w") as f:
        json.dump({"run_for_hour": hours, "map": map_name, "hd_map_path": map_path, "apollo_root": config.APOLLO_ROOT,
                   "max_adc_count": config.MAX_ADC_COUNT, "max_pd_count": config.MAX_PD_COUNT,
                   "scenario_upper_limit": config.SCENARIO_UPPER_LIMIT, "ga": hyperparameters,
                   "python": sys.version.split()[0], "started": started.isoformat(timespec = "seconds")}, f, indent = 2)
    shutil.copy2(os.path.join(HERE, "config.py"), os.path.join(output_folder, "config.py"))

    import main_ga
    main_ga.main()

    # main_ga.py writes its logbook and hall of fame to data/ (overwritten by every run): keep this run's
    for name in ("log.bin", "hof.bin"):
        source = os.path.join(HERE, "data", name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(output_folder, name))
    with open(os.path.join(output_folder, "run_completed.json"), "w") as f:
        json.dump({"started": started.isoformat(timespec = "seconds"), "finished": datetime.now().isoformat(timespec = "seconds")}, f, indent = 2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
