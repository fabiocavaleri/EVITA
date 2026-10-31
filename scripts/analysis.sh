
#!/usr/bin/env bash

set -euo pipefail

# Usage:
# ./run_experiments.sh <experiments_folder_path> <analysis|plot|both> [max_workers]

# -------------------------
# Validate arguments
# -------------------------

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <experiments_folder_path> <analysis|plot|both> [max_workers]"
    exit 1
fi

EXPERIMENTS_FOLDER="$1"
TASK="$2"
MAX_WORKERS="${3:-5}"

if [[ ! -d "$EXPERIMENTS_FOLDER" ]]; then
    echo "Error: Experiments folder does not exist: $EXPERIMENTS_FOLDER"
    exit 1
fi

if [[ ! "$TASK" =~ ^(analysis|plot|both)$ ]]; then
    echo "Error: Task must be one of: analysis, plot, both"
    exit 1
fi

if ! [[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: max_workers must be a positive integer"
    exit 1
fi

# Resolve absolute path
EXPERIMENTS_FOLDER="$(cd "$EXPERIMENTS_FOLDER" && pwd)"

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANALYSIS_SCRIPT="$SCRIPT_DIR/analysis.py"
PLOT_SCRIPT="$SCRIPT_DIR/generate_plots.py"

# -------------------------
# Discover framework folders
# -------------------------

FRAMEWORKS=()

for framework_path in "$EXPERIMENTS_FOLDER"/*/; do
    [[ -d "$framework_path" ]] || continue
    FRAMEWORKS+=("${framework_path%/}")
done

if [[ ${#FRAMEWORKS[@]} -eq 0 ]]; then
    echo "Error: No framework folders found in $EXPERIMENTS_FOLDER"
    exit 1
fi

# -------------------------
# Analysis task
# -------------------------

run_analysis() {
    if [[ ! -f "$ANALYSIS_SCRIPT" ]]; then
        echo "Error: analysis.py not found at $ANALYSIS_SCRIPT"
        exit 1
    fi

    echo "======================================"
    echo "Starting analysis"
    echo "Max workers: $MAX_WORKERS"
    echo "======================================"

    for framework_path in "${FRAMEWORKS[@]}"; do
        framework_name="$(basename "$framework_path")"

        echo ""
        echo "Framework: $framework_name"

        map_found=false

        for map_path in "$framework_path"/*/; do
            [[ -d "$map_path" ]] || continue

            map_found=true
            map_path="${map_path%/}"

            echo "Analyzing map: $map_path"

            python "$ANALYSIS_SCRIPT" \
                "$map_path" \
                "$MAX_WORKERS"
        done

        if [[ "$map_found" == false ]]; then
            echo "No map folders found in $framework_path"
        fi
    done

    echo ""
    echo "Analysis completed."
}

# -------------------------
# Plot task
# -------------------------

run_plot() {
    if [[ ! -f "$PLOT_SCRIPT" ]]; then
        echo "Error: generate_plots.py not found at $PLOT_SCRIPT"
        exit 1
    fi

    echo "======================================"
    echo "Generating comparison plots"
    echo "======================================"

    FRAMEWORK_LABELS=()

    for framework_path in "${FRAMEWORKS[@]}"; do
        FRAMEWORK_LABELS+=("$(basename "$framework_path")")
    done

    OUTPUT_FOLDER="$EXPERIMENTS_FOLDER/comparison plots"

    mkdir -p "$OUTPUT_FOLDER"

    python "$PLOT_SCRIPT" \
        --experiments "${FRAMEWORKS[@]}" \
        --labels "${FRAMEWORK_LABELS[@]}" \
        --output "$OUTPUT_FOLDER"

    echo ""
    echo "Plots generated in: $OUTPUT_FOLDER"
}

# -------------------------
# Execute requested task
# -------------------------

case "$TASK" in
    analysis)
        run_analysis
        ;;
    plot)
        run_plot
        ;;
    both)
        run_analysis
        run_plot
        ;;
esac

echo ""
echo "All requested tasks completed successfully."