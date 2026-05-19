# !/bin/bash
set -e

# Remove existing virtual environments and Frenetix-Motion-Planner repository directory
rm -rf .venv3*
rm -rf Frenetix-Motion-Planner
mkdir Frenetix-Motion-Planner

# Check required python versions availability
command -v python3.8 >/dev/null 2>&1 || { echo "python3.8 not found"; exit 1; }
command -v python3.10 >/dev/null 2>&1 || { echo "python3.10 not found"; exit 1; }

# ----- Create Python 3.8 venv -----
python3.8 -m venv .venv38

# ----- Create Python 3.10 venv -----
python3.10 -m venv .venv310

# ----- Install DoppelTest in python3.8 virtual environment -----
source .venv38/bin/activate

pip install --upgrade pip

cd EVITA_DoppelTest
pip install -r requirements.txt
pip install "requests<2.32" "urllib3<2"

cd conversion_scripts
pip install -r requirements.txt

# ----- Install Frenetix-Motion-Planner and Evita in python3.10 virtual environment -----
cd ../../
deactivate

source .venv310/bin/activate

pip install --upgrade pip

git submodule init
git submodule update --remote

cd Frenetix-Motion-Planner/
git checkout tags/2024.3

# ----- Patch onnxruntime version constraint -----
sed -i 's/onnxruntime = "~1.16.3"/onnxruntime = "<=1.18.1"/' pyproject.toml

# ----- Patch config_sim None check -----
sed -i 's/if config_sim.visualization.ref_path_debug:/if config_sim is not None and config_sim.visualization.ref_path_debug:/' \
cr_scenario_handler/utils/utils_coordinate_system.py

pip install -e .

cd ..
pip install -r requirements.txt

pip install -e .