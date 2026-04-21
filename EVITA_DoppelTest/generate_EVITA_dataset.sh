#!/usr/bin/env bash
set -e

TOTAL_RUNS=20
LOG_DIR=dataset_generation_logs/EVITA
START_TIME_GLOBAL=$(date +%s)

mkdir -p "$LOG_DIR"

draw_header() {
    clear
    echo "============================================="
    echo " GA Batch Execution"
    echo "============================================="
    echo "Container: $(hostname)"
    echo "Total runs: $TOTAL_RUNS"
    echo "Current run: $CURRENT_RUN / $TOTAL_RUNS"
    echo "Started at: $START_TIME_RUN"
    echo "Elapsed (this run): $ELAPSED_RUN"
    echo "Elapsed (total): $ELAPSED_TOTAL"
    echo "Log file: $LOG_FILE"
    echo "============================================="
    echo
}

for ((i=1; i<=TOTAL_RUNS; i++)); do
    CURRENT_RUN=$i
    START_TIME_RUN=$(date)
    RUN_START_TS=$(date +%s)
    LOG_FILE="$LOG_DIR/run_${i}.log"

    # run script in background
    python main_ga_evita.py > "$LOG_FILE" 2>&1 &
    PID=$!

    while kill -0 $PID 2>/dev/null; do
        NOW=$(date +%s)
        ELAPSED_RUN="$((NOW - RUN_START_TS)) sec"
        ELAPSED_TOTAL="$((NOW - START_TIME_GLOBAL)) sec"
        draw_header
        tail -n 20 "$LOG_FILE"
        sleep 1
    done

    wait $PID
done

echo "===== All runs completed ====="
