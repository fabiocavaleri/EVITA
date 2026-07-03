# !/bin/bash
set -e

# Remove existing virtual environments and Frenetix-Motion-Planner repository directory
echo "Removing existing virtual environments and Frenetix-Motion-Planner repository directory..."
rm -rf .venv3*
rm -rf Frenetix-Motion-Planner
mkdir Frenetix-Motion-Planner

# Check required python versions availability
echo "Checking required python versions availability..."
command -v python3.8 >/dev/null 2>&1 || { echo "python3.8 not found"; exit 1; }
command -v python3.10 >/dev/null 2>&1 || { echo "python3.10 not found"; exit 1; }

echo "Creating virtual environments..."
# ----- Create Python 3.8 venv -----
python3.8 -m venv .venv38

# ----- Create Python 3.10 venv -----
python3.10 -m venv .venv310

# ----- Install DoppelTest in python3.8 virtual environment -----
source .venv38/bin/activate

echo "Updating pip in Python 3.8 virtual environment..."
pip install -q --upgrade pip

echo "Installing DoppelTest in Python 3.8 virtual environment..."
cd EVITA_DoppelTest
pip install -qr requirements.txt
pip install -q "requests<2.32" "urllib3<2"

cd conversion_scripts
pip install -qr requirements.txt

# ----- Install Frenetix-Motion-Planner and Evita in python3.10 virtual environment -----
cd ../../
deactivate

source .venv310/bin/activate

echo "Updating pip in Python 3.10 virtual environment..."
pip install -q --upgrade pip

echo "Installing EVITA in Python 3.10 virtual environment..."

pip install -qr requirements-2.txt
pip install -qe .

echo "Cloning Frenetix-Motion-Planner repository..."
git submodule init -q
git submodule update --remote -q

cd Frenetix-Motion-Planner/
git checkout tags/2024.4 -q

# ----- Patch config_sim None check -----
sed -i 's/if config_sim.visualization.ref_path_debug:/if config_sim is not None and config_sim.visualization.ref_path_debug:/' cr_scenario_handler/utils/utils_coordinate_system.py
sed -i 's/evaluate_simulation: False/evaluate_simulation: True/' configurations/simulation/evaluation.yaml

echo "Installing Frenetix-Motion-Planner in Python 3.10 virtual environment..."
pip install -qe .

# ----- Other fixes -----
#
# sed -i 's/            setattr(self, subclasses,  SubConfiguration(config\[subclasses\]))/            if config[subclasses] is not None:\n                setattr(self, subclasses,  SubConfiguration(config[subclasses]))/' cr_scenario_handler/utils/configuration.py

#
# sed -i 's/tr_1 = tr - now_point/tr_1 = np.array(tr) - np.array(now_point)/' .venv310/lib/python3.10/site-packages/prediction/utils/geometry.py
# sed -i 's/            eval_array = np.arange(0, curve_length\[-1\], pixel_dist \* interp_factor)/            if len(curve_length) == 0 or curve_length\[-1\] <= 0 or np.isnan(curve_length\[-1\]):\n                return None\n            eval_array = np.arange(0, curve_length\[-1\], pixel_dist * interp_factor)/' .venv310/lib/python3.10/site-packages/prediction/utils/preprocessing.py
# sed -i 's/    interp_bound_arr = np.concatenate(interp_bound_list, axis=1)/    if not interp_bound_list:\n        return np.zeros((res, res))\n    interp_bound_arr = np.concatenate(interp_bound_list, axis=1)/' .venv310/lib/python3.10/site-packages/prediction/utils/preprocessing.py