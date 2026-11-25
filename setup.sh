#!/bin/bash
# Sets up EVITA's Python environments, each with one role:
#   .venv310  EVITA (every generator, FrenetiX and Apollo; the Apollo runner runs with EVITA's Python)
#   .venv38   only DoppelTest's own baseline (EVITA_DoppelTest/generate_DoppelTest_dataset.sh -> main_ga.py), with
#             DoppelTest's pinned requirements, so that the baseline runs as published.
#             Optional: created only with --with-doppeltest-baseline (the only part that needs Python 3.8); otherwise .venv38 is left untouched.
#
# Usage:
#   bash setup.sh [--with-doppeltest-baseline]
#
# System requirements: python3.10 (python3.8 with --with-doppeltest-baseline), git, and the system libraries OpenCV
# needs (libGL and GLib; OpenCV comes with FrenetiX's prediction, wale-net). Missing libraries are installed with
# apt-get (as root, or through sudo); elsewhere the script stops with the command to run.
set -e

WITH_DOPPELTEST=0
for argument in "$@"; do
    case "$argument" in
        --with-doppeltest-baseline) WITH_DOPPELTEST=1 ;;
        *) echo "Usage: bash setup.sh [--with-doppeltest-baseline]" >&2; exit 2 ;;
    esac
done

# ----- Required Python versions -----
echo "Checking required python versions availability..."
command -v python3.10 >/dev/null 2>&1 || { echo "python3.10 not found"; exit 1; }
if [[ $WITH_DOPPELTEST == 1 ]]; then
    command -v python3.8 >/dev/null 2>&1 || { echo "python3.8 not found (needed by --with-doppeltest-baseline)"; exit 1; }
fi

# ----- System libraries needed by OpenCV (FrenetiX's prediction): libGL and GLib -----
echo "Checking system libraries..."
LDCONFIG=$(command -v ldconfig || echo /sbin/ldconfig)
missing_libraries=()
for library in libGL.so.1 libgthread-2.0.so.0; do
    "$LDCONFIG" -p 2>/dev/null | grep -q "$library" || missing_libraries+=("$library")
done
if (( ${#missing_libraries[@]} > 0 )); then
    echo "Missing system libraries: ${missing_libraries[*]} (needed by OpenCV, used by FrenetiX's prediction)"
    if command -v apt-get >/dev/null 2>&1; then
        if [[ $EUID -eq 0 ]]; then
            apt-get update -q && apt-get install -y -q libgl1 libglib2.0-0
        elif command -v sudo >/dev/null 2>&1; then
            echo "Installing them with sudo (it may ask for your password)..."
            sudo apt-get update -q && sudo apt-get install -y -q libgl1 libglib2.0-0
        else
            echo "Error: install them as root, then rerun this script:  apt-get install -y libgl1 libglib2.0-0" >&2
            exit 1
        fi
    else
        echo "Error: install the system packages providing ${missing_libraries[*]} (OpenGL and GLib), then rerun this script" >&2
        exit 1
    fi
fi

# ----- DoppelTest's baseline environment (optional) -----
if [[ $WITH_DOPPELTEST == 1 ]]; then
    echo "Creating the Python 3.8 virtual environment for DoppelTest's baseline..."
    rm -rf .venv38
    python3.8 -m venv .venv38
    source .venv38/bin/activate

    echo "Updating pip in Python 3.8 virtual environment..."
    pip install -q --upgrade pip

    echo "Installing DoppelTest (baseline) in Python 3.8 virtual environment..."
    cd EVITA_DoppelTest
    pip install -qr requirements.txt
    pip install -q "requests<2.32" "urllib3<2"
    cd ..
    deactivate
else
    echo "Skipping DoppelTest's baseline environment (.venv38 left as it is; use --with-doppeltest-baseline to create it)"
fi

# ----- EVITA, its Apollo runner and Frenetix-Motion-Planner in the Python 3.10 virtual environment -----
echo "Removing the existing EVITA environment and Frenetix-Motion-Planner repository directory..."
rm -rf .venv310
rm -rf Frenetix-Motion-Planner

echo "Creating the Python 3.10 virtual environment..."
python3.10 -m venv .venv310
source .venv310/bin/activate

echo "Updating pip in Python 3.10 virtual environment..."
pip install -q --upgrade pip

echo "Installing EVITA (with the Apollo runner's packages) in Python 3.10 virtual environment..."
pip install -qr requirements.txt
pip install -qe .

echo "Cloning/updating Frenetix-Motion-Planner repository..."
git submodule sync --recursive --quiet
git submodule update --quiet --init --remote --recursive --force

cd Frenetix-Motion-Planner/

# EVITA needs its own FrenetiX fork (e.g. Apollo's vehicle model 101): the remote must be the fork from .gitmodules
FRENETIX_FORK="alessiogambi/Evita-Frenetix-Motion-Planner"
if ! git remote get-url origin | grep -q "$FRENETIX_FORK"; then
    echo "Error: Frenetix-Motion-Planner's remote is $(git remote get-url origin), not EVITA's fork $FRENETIX_FORK" >&2
    echo "       (check the submodule's url in .gitmodules)" >&2
    exit 1
fi

# The latest commit on the fork's main branch
git checkout -q main
git pull -q origin main
echo "Frenetix-Motion-Planner: $(git log -1 --format='%h %ad %s' --date=short) ($FRENETIX_FORK, main)"

echo "Installing Frenetix-Motion-Planner in Python 3.10 virtual environment..."
pip install -qe .
cd ..

# ----- Verification: what EVITA's runs import, so that a missing piece shows now and not hours into a run -----
echo "Verifying the EVITA environment..."
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python - <<'EOF'
import importlib, sys
checks = [("cv2", "OpenCV (system libraries libGL and GLib)"),
          ("cr_scenario_handler.utils.prediction_helpers", "FrenetiX's prediction"),
          ("evita.executors.frenetix_executor", "EVITA's FrenetiX executor"),
          ("evita.executors.apollo_executor", "EVITA's Apollo executor (docker, websocket-client, cyber_record)")]
failed = False
for module, what in checks:
    try:
        importlib.import_module(module)
        print(f"  OK    {what}")
    except Exception as exc:
        failed = True
        print(f"  FAIL  {what}: {type(exc).__name__}: {exc}")
sys.exit(1 if failed else 0)
EOF
echo "EVITA's environment is ready (.venv310)."