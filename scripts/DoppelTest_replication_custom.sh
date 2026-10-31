#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./run_DoppelTest.sh <start_repetition> <number_of_repetitions> <map_path1> [map_path2 ...]
#
# Examples:
#   ./run_DoppelTest.sh 1 10 ../base-scenarios/CHN_Cho-2_2_I-1-1.cr.xml
#   ./run_DoppelTest.sh 3 5 ../base-scenarios/CHN_Cho-2_2_I-1-1.cr.xml ../base-scenarios/DEU_Cologne-79_2_I-1.cr.xml
#
# The second example executes repetitions 3, 4, 5, 6, and 7
# for each map.

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <start_repetition> <number_of_repetitions> <map_path1> [map_path2 ...]" >&2
    exit 2
fi

start=$1
number_of_repetitions=$2
shift 2

maps_paths=("$@")

# Validate numeric arguments before arithmetic expansion.
if [[ ! "$start" =~ ^[0-9]+$ ||
      ! "$number_of_repetitions" =~ ^[0-9]+$ ]]; then
    echo "Error: start_repetition and number_of_repetitions must be integers." >&2
    exit 2
fi

# Force base-10 interpretation, avoiding octal numbers such as 08.
start=$((10#$start))
number_of_repetitions=$((10#$number_of_repetitions))

if (( start < 1 || number_of_repetitions < 1 )); then
    echo "Error: both arguments must be greater than zero." >&2
    exit 2
fi

end=$((start + number_of_repetitions - 1))

time_budget_hours=3
total_runs=$(( ${#maps_paths[@]} * number_of_repetitions ))

# Validate all map files before starting experiments.
for map_path in "${maps_paths[@]}"; do
    if [[ ! -f "$map_path" ]]; then
        echo "Error: map file not found: $map_path" >&2
        exit 2
    fi
done

log_parent_dir="./logs/test/DoppelTest"
output_parent_dir="./results/test/DoppelTest"

mkdir -p "$log_parent_dir" "$output_parent_dir"

overall_completed_runs=0
overall_skipped_runs=0
overall_failed_runs=0
overall_analysis_failures=0
run_count=0

for map_path in "${maps_paths[@]}"; do
    # Derive a clean map name from the file.
    map_name=$(basename -- "$map_path")
    map_name=${map_name%.xml}
    map_name=${map_name%.cr}

    log_dir="${log_parent_dir}/${map_name}"
    output_base_dir="${output_parent_dir}/${map_name}"

    mkdir -p "$log_dir" "$output_base_dir"

    completed_runs=0
    skipped_runs=0
    failed_runs=0

    experiment_name_prefix="DoppelTest_RUN_FOR_${time_budget_hours}_HOURS_SCENARIO_MAP_${map_name}"

    echo
    echo "##################################################################"
    echo "Processing map: ${map_path}"
    echo "Repetitions: ${start} through ${end}"
    echo "##################################################################"

    for ((j = start; j <= end; j++)); do
        run_count=$((run_count + 1))

        # Skip only if the output directory for this specific repetition
        # already exists. This avoids incorrectly skipping repetitions
        # when existing runs are not consecutive.
        existing_run=$(find "$output_base_dir" \
            -mindepth 1 -maxdepth 1 -type d \
            -name "${experiment_name_prefix}_REPETITION_${j}_*" \
            -print -quit)

        if [[ -n "$existing_run" ]]; then
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] SKIPPED: repetition ${j} already exists for ${map_name}"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        experiment_time=$(date +"%Y%m%d%H%M%S")
        experiment_name="DoppelTest_RUN_FOR_${time_budget_hours}_HOURS_SCENARIO_MAP_${map_name}_REPETITION_${j}_${experiment_time}"

        output_folder="$(realpath -m "${output_base_dir}/${experiment_name}")"
        log_file="${log_dir}/${experiment_name}.log"

        echo "=================================================================="
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Run ${run_count}/${total_runs}"
        echo "  Map:         ${map_path}"
        echo "  Repetition:  ${j}"
        echo "  Time budget: ${time_budget_hours}h"
        echo "  Output name: ${output_folder}"
        echo "  Log file:    ${log_file}"
        echo "=================================================================="

        # Run the experiment from the repository root.
        # Temporarily disable errexit to handle failures explicitly.
        set +e
        evita --output-folder-name "$output_folder" \
            run --scenario-generator-path . \
            --scenario-generator-module evita.generators.doppeltest_generator \
            --scenario-generator-class DoppelTestScenarioGenerator \
            --driving-agent-name frenetix \
            --time-budget "$((time_budget_hours * 3600))" \
            --map-file "$map_path" \
            2>&1 | tee "$log_file"

        # Capture both statuses before executing another command.
        pipeline_status=("${PIPESTATUS[@]}")
        status=${pipeline_status[0]}
        tee_status=${pipeline_status[1]}
        set -e

        if (( status != 0 )); then
            failed_runs=$((failed_runs + 1))
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAILED (exit ${status}): ${experiment_name}" \
                | tee -a "$log_file" || true
        elif (( tee_status != 0 )); then
            failed_runs=$((failed_runs + 1))
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAILED: tee could not write the log correctly: ${experiment_name}" >&2
        else
            completed_runs=$((completed_runs + 1))
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE: ${experiment_name}"
        fi

        echo
    done

    overall_completed_runs=$((overall_completed_runs + completed_runs))
    overall_skipped_runs=$((overall_skipped_runs + skipped_runs))
    overall_failed_runs=$((overall_failed_runs + failed_runs))

    echo "=================================================================="
    echo "Experiment summary for map: ${map_path}"
    echo "  Successful: ${completed_runs}"
    echo "  Skipped:    ${skipped_runs}"
    echo "  Failed:     ${failed_runs}"
    echo "  Planned:    ${number_of_repetitions}"
    echo "=================================================================="
done

echo "##################################################################"
echo "Overall DoppelTest experiment summary"
echo "  Maps processed:    ${#maps_paths[@]}"
echo "  Successful runs:   ${overall_completed_runs}"
echo "  Skipped runs:      ${overall_skipped_runs}"
echo "  Failed runs:       ${overall_failed_runs}"
echo "  Analysis failures: ${overall_analysis_failures}"
echo "  Planned runs:      ${total_runs}"
echo "##################################################################"

if (( overall_failed_runs > 0 || overall_analysis_failures > 0 )); then
    echo "Finished with errors. Logs saved under: ${log_parent_dir}/" >&2
    exit 1
fi

echo "All DoppelTest experiments completed successfully."
echo "Logs saved under: ${log_parent_dir}/"
echo "Results saved under: ${output_parent_dir}/"