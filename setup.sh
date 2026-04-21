#!/bin/bash
set -e

# Check availability
command -v python3.8 >/dev/null 2>&1 || { echo "python3.8 not found"; exit 1; }
command -v python3.10 >/dev/null 2>&1 || { echo "python3.10 not found"; exit 1; }

# ----- Create Python 3.8 venv -----
python3.8 -m venv .venv38

# ----- Create Python 3.10 venv -----
python3.10 -m venv .venv310

# ----- Install DoppelTest -----
source .venv38/bin/activate

pip install --upgrade pip

cd EVITA_DoppelTest
pip install -r requirements.txt
# pip install "requests<2.32" "urllib3<2"

# cd conversion_scripts

# pip install -r requirements.txt

# # ----- Install Evita -----
cd ../ #../
deactivate

source .venv310/bin/activate

pip install --upgrade pip

# git submodule init
git submodule update --init --recursive

ls Frenetix-Motion-Planner/


# cd Frenetix-Motion-Planner/
# git checkout tags/2024.3
# pip install -e .

# cd ..
# pip install -r requirements.txt

# pip install -e .