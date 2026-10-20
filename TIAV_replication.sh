#!/usr/bin/env bash
set -euo pipefail

maps_paths=(
    # "base-scenarios/USA_US101-23_1_T-1.xml"
    # "base-scenarios/USA_US101-15_2_T-1.xml"
    # "base-scenarios/CHN_Cho-2_2_I-1-1.cr.xml"
    # "base-scenarios/DEU_Cologne-79_2_I-1.cr.xml"
    "base-scenarios/USA_US101-9_1_T-1.xml"
)
repetitions=10
time_budget_hours=3

total_runs=$(( ${#maps_paths[@]} * repetitions ))
run_count=0

log_parent_dir="./logs"

for map_path in "${maps_paths[@]}"; do
    # Derive a clean map name from the file (strip directory + extension(s))
    map_name=$(basename "$map_path")
    map_name=${map_name%.xml}
    map_name=${map_name%.cr}

    log_dir="${log_parent_dir}/TIAV/${map_name}"
    mkdir -p "$log_dir"

    output_base_dir="./results/TIAV/${map_name}"
    mkdir -p "$output_base_dir"

    for ((j = 1; j <= repetitions; j++)); do
        run_count=$(( run_count + 1 ))

        # Prefix shared by every repetition of this map, e.g. TIAV_RUN_FOR_3_HOURS_SCENARIO_MAP_USA_US101-23_1_T-1
        experiment_name_prefix="TIAV_RUN_FOR_${time_budget_hours}_HOURS_SCENARIO_MAP_${map_name}"

        # Count how many folders for this map already exist (from a previous,
        # possibly interrupted, run of this script).
        existing_experiments_count=$(find "$output_base_dir" -maxdepth 1 -type d -name "${experiment_name_prefix}*" 2>/dev/null | wc -l)

        if (( existing_experiments_count > 0 )); then
            echo "Found ${existing_experiments_count} existing run(s) for map '${map_name}' (prefix: ${experiment_name_prefix}*)"
            if (( j <= existing_experiments_count )); then
                continue
            fi
        fi

        # Build a unique experiment identifier (bash equivalent of the Python snippet)
        experiment_time=$(date +"%Y%m%d%H%M%S")
        experiment_name="${map_name}/TIAV_RUN_FOR_${time_budget_hours}_HOURS_SCENARIO_MAP_${map_name}_REPETITION_${j}_${experiment_time}"

        log_file="${log_parent_dir}/TIAV/${experiment_name}.log"

        echo "=================================================================="
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Run ${run_count}/${total_runs}"
        echo "  Map:         ${map_path}"
        echo "  Repetition:  ${j}/${repetitions}"
        echo "  Time budget: ${time_budget_hours}h"
        echo "  Output name: ${experiment_name}"
        echo "  Log file:    ${log_file}"
        echo "=================================================================="

        # Run the experiment, mirroring terminal output to the log file
        # while still showing it live in the terminal.
        evita --output-folder-name "TIAV/$experiment_name" \
            run --scenario-generator-path . \
            --scenario-generator-module evita.generators.basic_generator \
            --scenario-generator-class EAOnePlusOneGenerator_MaxDanger \
            --driving-agent-name frenetix \
            --time-budget $(( time_budget_hours * 3600 )) \
            --map-file "$map_path" \
            2>&1 | tee "$log_file"

        status=${PIPESTATUS[0]}
        if [[ $status -ne 0 ]]; then
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] FAILED (exit ${status}): ${experiment_name}" | tee -a "$log_file"
        else
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] DONE: ${experiment_name}"
        fi
        echo
    done
    python TIAV_analysis.py ${output_base_dir}
done

echo "All ${total_runs} experiments completed. Logs saved under ${log_parent_dir}/"