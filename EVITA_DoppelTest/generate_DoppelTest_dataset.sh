#!/usr/bin/env bash
#
# DoppelTest's own baseline on Apollo: repetitions of DoppelTest's search (main_ga.py, unchanged), one after the other.
#
# Usage (from anywhere; DoppelTest's Python 3.8 environment active):
#   source .venv38/bin/activate
#   [HOURS=<hours>] [MAP=<map>] bash EVITA_DoppelTest/generate_DoppelTest_dataset.sh <start_repetition> <number_of_repetitions>
#
# HOURS (optional) is the duration of each run in hours (default: RUN_FOR_HOUR in config.py; decimals allowed, e.g.
# HOURS=0.33 for 20 minutes).
# MAP (optional) is DoppelTest's map, a folder of EVITA_DoppelTest/data/maps (default: the map of HD_MAP_PATH in
# config.py). It must be the map Apollo loads (--map_dir in Apollo's global_flagfile.txt): checked before every run.
# MAX_ADC_COUNT and SCENARIO_UPPER_LIMIT stay as in config.py (EVITA takes them from there too, on Apollo).
#
# Every repetition gets its own folder, next to EVITA's Apollo results:
#   results/Main-Experiment/Apollo/DoppelTest/<map>/DoppelTest_RUN_FOR_<h>_HOURS_SCENARIO_MAP_<map>_REPETITION_<k>_<time>/
#   logs/Main-Experiment/Apollo/DoppelTest/<map>/<same name>.log
# with DoppelTest's records (Generation_*/Scenario_*) and summary.csv (violations), run_parameters.json (with
# main_ga.py's GA hyperparameters), a copy of config.py, main_ga.py's log.bin and hof.bin, and run_completed.json once
# the run is over. A completed repetition is skipped; an interrupted one is moved
# to _incomplete/ and repeated, so the script can be restarted safely.
#
# Example:
#   ./generate_DoppelTest_dataset.sh 1 10
#   HOURS=3 ./generate_DoppelTest_dataset.sh 11 5

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: [HOURS=<hours>] [MAP=<map>] $0 <start_repetition> <number_of_repetitions>" >&2
    exit 1
fi
start_repetition="$1"
number_of_repetitions="$2"
for value in "$start_repetition" "$number_of_repetitions"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: <start_repetition> and <number_of_repetitions> must be positive integers, not: $value" >&2
        exit 1
    fi
done

# Run from EVITA_DoppelTest (main_ga.py uses paths relative to it); results and logs go to the repository root
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(dirname "$script_dir")"
cd "$script_dir"

# DoppelTest's baseline runs as published, with its Python 3.8 environment
python_bin="${DOPPELTEST_PYTHON:-python}"
if ! "$python_bin" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 8) else 1)' 2>/dev/null; then
    echo "Error: DoppelTest's baseline needs its Python 3.8 environment: source .venv38/bin/activate" >&2
    echo "       (create it with: bash setup.sh --with-doppeltest-baseline)" >&2
    exit 1
fi

# Defaults from config.py
read -r default_hours default_map < <(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "$python_bin" run_main_ga.py --defaults)
hours="${HOURS:-$default_hours}"
map="${MAP:-$default_map}"
if ! [[ "$hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v h="$hours" 'BEGIN { exit !(h > 0) }'; then
    echo "Error: HOURS must be a positive number of hours, not: $hours" >&2
    exit 2
fi
if [[ ! -f "data/maps/$map/base_map.bin" ]]; then
    echo "Error: no DoppelTest map '$map' (expected data/maps/$map/base_map.bin)" >&2
    exit 2
fi

output_parent_dir="$repository_root/results/Main-Experiment/Apollo/DoppelTest/$map"
log_parent_dir="$repository_root/logs/Main-Experiment/Apollo/DoppelTest/$map"
mkdir -p "$output_parent_dir" "$log_parent_dir"

completed_runs=0
skipped_runs=0
failed_runs=0
end_repetition=$((start_repetition + number_of_repetitions - 1))

for ((repetition = start_repetition; repetition <= end_repetition; repetition++)); do
    prefix="DoppelTest_RUN_FOR_${hours}_HOURS_SCENARIO_MAP_${map}_REPETITION_${repetition}_"

    # A completed run of this repetition: skip it; an interrupted one: move it aside, then repeat it
    completed=""
    for folder in "$output_parent_dir/$prefix"*; do
        [[ -d "$folder" ]] || continue
        if [[ -f "$folder/run_completed.json" ]]; then
            completed="$folder"
        else
            mkdir -p "$output_parent_dir/_incomplete"
            mv "$folder" "$output_parent_dir/_incomplete/"
            echo "Moved the interrupted run $(basename "$folder") to _incomplete/"
        fi
    done
    if [[ -n "$completed" ]]; then
        skipped_runs=$((skipped_runs + 1))
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] SKIPPED (already completed): $(basename "$completed")"
        continue
    fi

    experiment_name="${prefix}$(date +"%Y%m%d%H%M%S")"
    output_folder="$output_parent_dir/$experiment_name"
    log_file="$log_parent_dir/$experiment_name.log"

    echo "=================================================================="
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Run $((repetition - start_repetition + 1))/${number_of_repetitions}"
    echo "  Map:         ${map}"
    echo "  Repetition:  ${repetition}"
    echo "  Time budget: ${hours}h"
    echo "  Output name: ${output_folder}"
    echo "  Log file:    ${log_file}"
    echo "=================================================================="

    set +e
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "$python_bin" run_main_ga.py --output-folder "$output_folder" --hours "$hours" --map "$map" \
        2>&1 | tee "$log_file"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e

    if (( pipeline_status[0] != 0 )); then
        failed_runs=$((failed_runs + 1))
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAILED (exit ${pipeline_status[0]}): ${experiment_name}" | tee -a "$log_file" || true
    elif (( pipeline_status[1] != 0 )); then
        failed_runs=$((failed_runs + 1))
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAILED: tee could not write the log correctly: ${experiment_name}" >&2
    else
        completed_runs=$((completed_runs + 1))
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE: ${experiment_name}"
    fi
    echo
done

echo "##################################################################"
echo "DoppelTest baseline summary (map ${map}, ${hours}h per run)"
echo "  Successful runs: ${completed_runs}"
echo "  Skipped runs:    ${skipped_runs}"
echo "  Failed runs:     ${failed_runs}"
echo "  Planned runs:    ${number_of_repetitions}"
echo "##################################################################"

if (( failed_runs > 0 )); then
    echo "Finished with errors. Logs saved under: ${log_parent_dir}/" >&2
    exit 1
fi
echo "All DoppelTest runs completed successfully."
echo "Logs saved under: ${log_parent_dir}/"
echo "Results saved under: ${output_parent_dir}/"
