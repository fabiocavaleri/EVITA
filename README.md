# Replication Package for Automatically Testing Interacting Autonomous Vehicles

This repository contains the code to reproduce the results of our paper using EVITA (EVolutionary Interaction Testing Approach).

# Setup

The code depends on open-source libraries that we had to patch to introduce missing features. We install them directly from github

## Python Version

Check that you have both python 3.8 and 3.10

> Note: For the development of EVITA we used python 3.8.10 and 3.10.14

## Setup environment

From the repository root folder execute the script:

```
./setup.sh
```

This script will:
- Create two virtual environments `.venv38` and `.venv310`
- Install DoppelTest requirements in `.venv38`
- Clone and install `Frenetix-Motion-Planner` submodule
- Install requirements for EVITA

Install Baidu Apollo with the following steps from DoppelTest README file:

1. Download the DoppelTest version of Apollo 7.0 from https://doi.org/10.5281/zenodo.7622089

> In this forked version, we made slight adjustments that are not related to the AD stack

2. At the root directory of Baidu Apollo, create directories `data`, `data/log`, `data/bag`, `data/core` by running `mkdir data data/log data/bag data/core`

> This step is necessary for DoppelTest running on the host machine to delete Apollo's log files. Our framework restarts modules being tested after every scenario, which creates a large number of unnecessary log files.

> Since a lot of commands are executed as root inside of the Docker container, if those directories are created inside of the container, DoppelTest may not be able to remove those directories.

3. At the root directory of Baidu Apollo, start up Apollo container via `./docker/scripts/dev_start.sh -l`

4. Find the name of the container via `docker ps -a`

5. Enter the container in root mode via `docker exec -it apollo_dev_your_name /bin/bash`

> Remember to replace `apollo_dev_your_name` with the container's actual name

6. In the container, build Apollo via `./apollo.sh build`

# Smoke test installations

Test the setup for both EVITA and DoppelTest + BaiduApollo

## Test EVITA installation

Activate the python 3.10 environment:

```
source .venv10/bin/activate
```

In the active virtual environment, from the root of the project, run:

```
(.venv10) evita --help
```

Check that the result looks like:

```
Usage: evita [OPTIONS] COMMAND [ARGS]...

Options:
  -d, --debug
  --output-folder-name TEXT
  --output-folder DIRECTORY
  --help

Commands:
  analyse  Run all the data analyses and by loading data from the given...
  replay
  run
```

## Test DoppelTest installation

Activate the python 3.8 environment:

```
source .venv38/bin/activate
```

Enter `EVITA_DoppelTest` folder an test the tool:

```
(.venv38) cd EVITA_DoppelTest
(.venv38) python test_main.py
```

Check that the result looks like:

```
apollo_dev_ROUTE_0: starting
Dreamview at http://172.17.0.3:8888
apollo_dev_ROUTE_1: starting
Dreamview at http://172.17.0.4:8888
apollo_dev_ROUTE_2: starting
Dreamview at http://172.17.0.5:8888


2026-04-22 09:52:18,365 - ScenarioRunner - INFO - 3 agents running a scenario G0S0.
2026-04-22 09:52:49,546 - Scenario - INFO - Scenario time: 30.0.
```

# Run the implemented generators:

## DoppelTest + BaiduApollo

Activate the python 3.8 environment and enter `EVITA_DoppelTest` folder:

```
source .venv38/bin/activate
(.venv38) cd EVITA_DoppelTest
```

Check the generator configuration file `config.py`.
Most relevant parameters are:
- `APOLLO_ROOT`: Root directory of Apollo 7.0.
- `DT_ROOT`: Root directory of DoppelTest. This path should point to the `EVITA_DoppelTest` folder you are currently in.
- `RECORDS_DIR_BASE`: Desired directory to save record files produced by the execution of scenarios. The default value is `f'{DT_ROOT}/data/records'`.
- `SCENARIO_UPPER_LIMIT`: Scenarios duration (in seconds).
- `MAX_ADC_COUNT`: Number of maximum AVs instances in the generated scenarios.
- `RUN_FOR_HOUR`: Number of hours you wish to run the tool for.
- `MAPS_ROOT_FOLDER`: Root folder with all the DoppelTest maps. Default value is `f'{DT_ROOT}/data/maps/'`.
- `HD_MAP_PATH`: The map used for the scenarios generation.

Start scenarios generation with:

```
(.venv38) python main_ga.py
```

After the execution of DoppelTest you can convert the generated scenarios to CommonRoad format using the script `convert_analyze_experiment.py`.

> Note: The execution of this script requires `.venv10` virtual environment

```
source .venv310/bin/activate
(.venv310) cd EVITA_DoppelTest
(.venv310) python convert_experiments.py <record_files_folder> <experiment_name> <output_files_folder> [max_workers]
```

For example, if the `main_ga.py` script created the experiment `MAX_ADC_COUNT_5_RUN_FOR_HOUR_12_SCENARIO_MAP_borregas_ave_20260421200506`, the call to the conversion script should look like:

```
python convert_analyze_experiment.py data/records/ MAX_ADC_COUNT_5_RUN_FOR_HOUR_12_SCENARIO_MAP_borregas_ave_20260421200506 data/evita-records/
```

> Note: You can also specify the number of parallel processes with `max_workers` (default value is 5) to speed up the conversion and analysis process

Specifically, this script will:
- Convert the executed scenario from .record files to a CommonRoad .xml scenario file.
- Extract the vehicles' planned trajectories
- Extract the vehicles' perception messages regarding other cars or traffic lights
- Extract the stop signs, traffic lights and speeding violations detected by the DoppelTest oracles
- Extract the collisions detected by the DoppelTest oracles and compute additional collision features
- Compute the vehicles interactions

### Dataset generation

The replication package also includes a script to run multiple consecutive generations.

Before executing the script you can edit the following variables in it:
- `TOTAL_RUNS` = 20
- `LOG_DIR` = dataset_generation_logs/DoppelTest

```
source .venv38/bin/activate
(.venv38) cd EVITA_DoppelTest
(.venv38) ./generate_DoppelTest_dataset.sh
```

## EVITA + BaiduApollo

Activate the python 3.10 environment and enter `EVITA_DoppelTest` folder:

```
source .venv310/bin/activate
(.venv310) cd EVITA_DoppelTest
```

Check the generator configuration file `config.py`.

Start scenarios generation with:

```
(.venv310) python main_ga_evita.py
```

The execution of `main_ga_evita.py` will automatically create a new folder storing the Apollo scenario converted in CommonRoad format, the planned trajectories, the perception messages, the vehicles' violations and the detected interactions and collisions.
For example for a scenario with folder path `data/records/Experiment_1/Generation_1/Scenario_1/` the tool will create the folder `data/evita-records/Experiment_1/Generation_1/Scenario_1/` 

### Dataset generation

The replication package also includes a script to run multiple consecutive generations.

Before executing the script you can edit the following variables in it:
- `TOTAL_RUNS` = 20
- `LOG_DIR` = dataset_generation_logs/EVITA

```
source .venv310/bin/activate
(.venv310) cd EVITA_DoppelTest
(.venv310) ./generate_EVITA_dataset.sh
```

## EVITA + Frenetix

All the generators must be executed from the root of the project after activating the virtual environment.

> Note: Add the --max-duration N to limit the execution of each scenario to N steps (useful for debug and play around with the execution)

The execution of EVITA with Frenetix-Motion-Planner requires python 3.10:

```
source .venv310/bin/activate
```

### Random Generation

It generates random scenarios as follows:
- Pick a number between min and max AV, N.
- Add N random AVs to the scenario, validate the scenario.
- If the scenario is valid execute it, otherwise, try again adding N random AVs.
- If after 1000 iterations, no scenario with N AVs can be found. Decrease N and repeat.

```
(.venv10) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.random_generator \
--scenario-generator-class RandomScenarioGenerator \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

### 1+1 EA

This is a simple Single Objective generator. It takes the fitness function to maximize as input.

```
(.venv10) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.basic_generator \
--scenario-generator-class EAOnePlusOneGenerator_MaxDanger \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

```
(.venv10) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.basic_generator \
--scenario-generator-class EAOnePlusOneGenerator_MaxInteractions \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml 
```

### EVITA

For the moment, EVITA runs on pymoo which is not well integrated in the orchestrator, so we invoke this generator directly.
To run it invoke:

```
(.venv10) cd evita
(.venv10) python generator/nsgaII_search_no_generator.py 10 3 ../base/scenarios/showcase-scenario.xml
```

## DoppelTest + Frenetix

Please refer to the ICSE 2023 paper for the description of the algorithm.

> Note: We had to adapt the original implementation that was tightly integrated with Apollo to fit our execution environment

```
(.venv10) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.doppeltest_generator \
--scenario-generator-class DoppelTestScenarioGenerator \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

# Aggregate Experiments Results
We can aggregate the analysis metrics generated using `DoppelTest + BaiduApollo` and `EVITA + BaiduApollo` using the following script:

```
source .venv310/bin/activate
(.venv310) cd EVITA_DoppelTest/analysis_scripts
(.venv310) python analysis.py <experiment_folder_path> [<max_workers>]
```
> Note: `experiment_folder_path` refers to the directory containing the CommonRoad version of the generated scenarios

> Note: You can also specify the number of parallel processes with `max_workers` (default value is 5) to speed up the conversion and analysis process

This script aggregates the values of the analysis metrics for the scenarios creating a new folder `experiments_analysis`

# Generate Comparison Plots
We can compare the aggregated analysis metrics of `DoppelTest + BaiduApollo` and `EVITA + BaiduApollo` experimetns generated using the `analysis.py` script:

```
source .venv310/bin/activate
(.venv310) cd EVITA_DoppelTest/analysis_scripts
(.venv310) python generate_plots.py <evita_borregas_folder_path> <doppeltest_borregas_folder_path> <output_folder_path>"
```
> Note: `evita_borregas_folder_path` and `doppeltest_borregas_folder_path` refer to the directories containing the CommonRoad version of the generated scenarios

This script aggregates the values of the analysis metrics for the scenarios creating a new folder `experiments_analysis`

# Scenarios Animation

We can animate the scenarios generated in the steps `DoppelTest + BaiduApollo` and `EVITA + BaiduApollo` using the following script:

```
source .venv310/bin/activate
(.venv310) cd EVITA_DoppelTest/conversion_scripts
(.venv310) python animate_scenarios.py <experiment_folder_path> [<max_workers>]
```

> Note: `experiment_folder_path` refers to the directory containing the CommonRoad version of the generated scenarios

> Note: You can also specify the number of parallel processes with `max_workers` (default value is 3) to speed up the conversion and analysis process

This script will first plot each simulation frame and then creates a gif to animate them.
The files will be saved in each CommonRoad scenario's folder.