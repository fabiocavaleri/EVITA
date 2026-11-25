#!/bin/bash
# Sets up the two Python environments, each with one role:
#   .venv310  EVITA (every generator, FrenetiX and Apollo; the Apollo runner runs with EVITA's Python)
#   .venv38   only DoppelTest's own baseline, with DoppelTest's pinned requirements, so that the baseline runs as published
set -e

# Remove existing virtual environments and Frenetix-Motion-Planner repository directory
echo "Removing existing virtual environments and Frenetix-Motion-Planner repository directory..."
rm -rf .venv3*
rm -rf Frenetix-Motion-Planner

# Check required python versions availability
echo "Checking required python versions availability..."
command -v python3.8 >/dev/null 2>&1 || { echo "python3.8 not found"; exit 1; }
command -v python3.10 >/dev/null 2>&1 || { echo "python3.10 not found"; exit 1; }

echo "Creating virtual environments..."
# ----- Create Python 3.8 venv -----
python3.8 -m venv .venv38

# ----- Create Python 3.10 venv -----
python3.10 -m venv .venv310

# ----- Install DoppelTest in the Python 3.8 virtual environment -----
source .venv38/bin/activate

echo "Updating pip in Python 3.8 virtual environment..."
pip install -q --upgrade pip

echo "Installing DoppelTest in Python 3.8 virtual environment..."
cd EVITA_DoppelTest
pip install -qr requirements.txt
pip install -q "requests<2.32" "urllib3<2"

# ----- Install EVITA, its Apollo runner and Frenetix-Motion-Planner in the Python 3.10 virtual environment -----
cd ..
deactivate

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