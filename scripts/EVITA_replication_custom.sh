#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   [CPUS=<cpu-list>] [SIMULATOR=frenetix|apollo] [DELETE_RECORDS=1] [TH_ED=<m>] [N_BINS=<n>] ./run_EVITA.sh <start_repetition> <number_of_repetitions> <map_path1> [map_path2 ...]
#
# CPUS (optional) pins every run of this script to the given cores, e.g. CPUS=0-3 or CPUS=0,2,4,6,
# and limits the numerical libraries to that many threads. Give each terminal its own, disjoint list.
#
# SIMULATOR (optional) chooses the simulator: frenetix (default) or apollo. With apollo, every map must be the
# converted map of the HD map Apollo runs (EVITA_DoppelTest/data/converted_maps/<map>/<map>.xml; HD_MAP_PATH in
# EVITA_DoppelTest/config.py), the results go to the Apollo branch of the folders, and TH_ED plays no role (Apollo's
# interactions come from its decisions).
# DELETE_RECORDS=1 (optional, apollo only) deletes each scenario's .record files once its trace is written.
#
# TH_ED (optional) sets the trajectory difference threshold Th_ED in metres, e.g. TH_ED=1.25, for the
# sensitivity analysis. Without it, EVITA uses its default (2.5 m). With it, runs are stored apart, under
# EVITA_ThED_<value>, so they never mix with the main experiment.
#
# N_BINS (optional) sets the number of bin edges per axis of the interaction map, e.g. N_BINS=20, for the
# sensitivity analysis (use even values: an odd number of edges puts 0 on an edge and removes the centre cell).
# Without it, EVITA uses its default (10). With it, runs are stored under EVITA[_ThED_<value>]_NBINS_<value>.
#
# Examples:
#   ./run_EVITA.sh 1 5 ../base-scenarios/DEU_Cologne-79_2_I-1.cr.xml
#   ./run_EVITA.sh 3 5 ../base-scenarios/DEU_Cologne-79_2_I-1.cr.xml ../base-scenarios/USA_US101-23_1_T-1.xml
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

# Optional CPU pinning (see CPUS above)
cpus="${CPUS:-}"
pin=()
if [[ -n "$cpus" ]]; then
    if ! n_threads=$(taskset -c "$cpus" nproc 2>/dev/null); then
        echo "Error: invalid CPU list in CPUS: $cpus" >&2
        exit 2
    fi
    export OMP_NUM_THREADS=$n_threads MKL_NUM_THREADS=$n_threads OPENBLAS_NUM_THREADS=$n_threads NUMEXPR_NUM_THREADS=$n_threads
    pin=(taskset -c "$cpus")
fi

# Optional simulator (see SIMULATOR above)
simulator="${SIMULATOR:-frenetix}"
case "$simulator" in
    frenetix) simulator_folder="Frenetix" ;;
    apollo) simulator_folder="Apollo" ;;
    *) echo "Error: SIMULATOR must be frenetix or apollo, not: $simulator" >&2; exit 2 ;;
esac
simulator_args=(--simulator "$simulator")
if [[ -n "${DELETE_RECORDS:-}" ]]; then
    if [[ "$DELETE_RECORDS" != "1" || "$simulator" != "apollo" ]]; then
        echo "Error: DELETE_RECORDS=1 applies only with SIMULATOR=apollo" >&2
        exit 2
    fi
    simulator_args+=(--delete-records)
fi

time_budget_hours=3
total_runs=$(( ${#maps_paths[@]} * number_of_repetitions ))

# Resolve the script's expected working directory.
if [[ ! -d "evita" ]]; then
    echo "Error: directory 'evita' not found in the current directory." >&2
    exit 2
fi

# Validate all map files before starting experiments.
for map_path in "${maps_paths[@]}"; do
    if [[ ! -f "$map_path" ]]; then
        echo "Error: map file not found: $map_path" >&2
        exit 2
    fi
done

# Optional Th_ED (see TH_ED above)
th_ed="${TH_ED:-}"
threshold_args=()
generator_label="EVITA"
if [[ -n "$th_ed" ]]; then
    if [[ ! "$th_ed" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error: TH_ED must be a positive number in metres, e.g. 1.25" >&2
        exit 2
    fi
    threshold_args=(--threshold "$th_ed")
    generator_label="EVITA_ThED_${th_ed}"
fi

# Optional interaction map resolution (see N_BINS above)
n_bins="${N_BINS:-}"
if [[ -n "$n_bins" ]]; then
    if [[ ! "$n_bins" =~ ^[0-9]+$ ]] || (( 10#$n_bins < 2 )); then
        echo "Error: N_BINS must be an integer of at least 2 (bin edges per axis), e.g. 20" >&2
        exit 2
    fi
    n_bins=$((10#$n_bins))
    if (( n_bins % 2 == 1 )); then
        echo "Warning: N_BINS=${n_bins} is odd: 0 falls on a bin edge and the interaction map has no centre cell" >&2
    fi
    threshold_args+=(--n-bins "$n_bins")
    generator_label="${generator_label}_NBINS_${n_bins}"
fi

log_parent_dir="./logs/Ablation-Study/${simulator_folder}/TotalInts-UniqueInts/${generator_label}/"
output_parent_dir="./results/Ablation-Study/${simulator_folder}/TotalInts-UniqueInts/${generator_label}/"

mkdir -p "$log_parent_dir" "$output_parent_dir"

overall_completed_runs=0
overall_skipped_runs=0
overall_failed_runs=0
overall_analysis_failures=0
run_count=0

# All supplied map paths are interpreted relative to the directory
# from which this script is invoked.
for map_path in "${maps_paths[@]}"; do
    map_name=$(basename -- "$map_path")
    map_name=${map_name%.xml}
    map_name=${map_name%.cr}

    log_dir="${log_parent_dir}/${map_name}"
    output_base_dir="${output_parent_dir}/${map_name}"

    mkdir -p "$log_dir" "$output_base_dir"

    completed_runs=0
    skipped_runs=0
    failed_runs=0

    experiment_name_prefix="${generator_label}_RUN_FOR_${time_budget_hours}_HOURS_SCENARIO_MAP_${map_name}"

    echo
    echo "##################################################################"
    echo "Processing map: ${map_path}"
    echo "Repetitions: ${start} through ${end}"
    echo "##################################################################"

    for ((j = start; j <= end; j++)); do
        run_count=$((run_count + 1))

        # Skip only if this repetition already finished: a completed run writes
        # performance.csv at the top of its output folder. A folder without it
        # is an interrupted run: it is moved to _incomplete/ and repeated.
        completed_run=""
        while IFS= read -r -d '' run_dir; do
            if [[ -f "${run_dir}/performance.csv" ]]; then
                completed_run="$run_dir"
                break
            fi
            # Move it out of the results, so the analysis never reads it
            incomplete_dir="${output_parent_dir%/}/_incomplete/${map_name}"
            mkdir -p "$incomplete_dir"
            mv -- "$run_dir" "$incomplete_dir/"
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] WARNING: incomplete run moved to ${incomplete_dir}/, repeating it: $(basename -- "$run_dir")"
        done < <(find "$output_base_dir" \
            -mindepth 1 -maxdepth 1 -type d \
            -name "${experiment_name_prefix}_REPETITION_${j}_*" \
            -print0)

        if [[ -n "$completed_run" ]]; then
            echo "[$(date +"%Y-%m-%d %H:%M:%S")] SKIPPED: repetition ${j} already completed for ${map_name}"
            skipped_runs=$((skipped_runs + 1))
            continue
        fi

        experiment_time=$(date +"%Y%m%d%H%M%S")
        experiment_name="${generator_label}_RUN_FOR_${time_budget_hours}_HOURS_SCENARIO_MAP_${map_name}_REPETITION_${j}_${experiment_time}"

        # Output paths are relative to the repository root.
        output_folder="$(realpath -m "${output_base_dir}/${experiment_name}")"
        log_file="${log_dir}/${experiment_name}.log"

        echo "=================================================================="
        echo "[$(date +"%Y-%m-%d %H:%M:%S")] Run ${run_count}/${total_runs}"
        echo "  Map:         ${map_path}"
        echo "  Repetition:  ${j}"
        echo "  Time budget: ${time_budget_hours}h"
        echo "  Output name: ${output_folder}"
        echo "  Log file:    ${log_file}"
        echo "  CPUs:        ${cpus:-all}"
        echo "  Th_ED:       ${th_ed:-default (2.5 m)}"
        echo "  n_bins:      ${n_bins:-default (10)}"
        echo "=================================================================="

        # Run from the evita directory, as in the original script.
        # Temporarily disable errexit to handle failures explicitly.
        set +e
        (
            cd "evita"
            ${pin[@]+"${pin[@]}"} python generators/nsgaII_search_no_generator.py \
            --population-size 10 \
            --time-budget "$((time_budget_hours * 60))" \
            --scenario-seed-file "../${map_path#../}" \
            --output-folder-name "$output_folder" \
            "${simulator_args[@]}" \
            ${threshold_args[@]+"${threshold_args[@]}"}
        ) 2>&1 | tee "$log_file"

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
echo "Overall EVITA experiment summary"
echo "  Maps processed:    ${#maps_paths[@]}"
echo "  Successful runs:   ${overall_completed_runs}"
echo "  Skipped runs:      ${overall_skipped_runs}"
echo "  Failed runs:       ${overall_failed_runs}"
echo "  Analysis failures: ${overall_analysis_failures}"
echo "  Planned runs:      ${total_runs}"
echo "##################################################################"

if (( overall_failed_runs > 0 || overall_analysis_failures > 0 )); then
    echo "Finished with errors. Logs saved under: ${log_parent_dir}" >&2
    exit 1
fi

echo "All EVITA experiments completed successfully."
echo "Logs saved under: ${log_parent_dir}"
echo "Results saved under: ${output_parent_dir}"