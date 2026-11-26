# EVITA (EVolutionary Interaction Testing Approach)

This repository contains the replication package for the paper *Automatic Testing of Interacting Autonomous Vehicles*.
We introduce EVITA (EVolutionary Interaction Testing Approach): a scenario-generation approach to automatically generate critical testing scenarios triggering interactions among multiple autonomous vehicles in highway and urban traffic.

[![ArXiv Paper](https://img.shields.io/badge/ArXiv-Paper-red?logo=arXiv&logoColor=red)](https://doi.org/10.48550/arXiv.2607.12452)
[![Frenetix Motion Planner](https://img.shields.io/badge/Frenetix-2024.3-blue?logo=github)](https://github.com/TUM-AVS/Frenetix-Motion-Planner/)
[![Baidu Apollo](https://img.shields.io/badge/BaiduApollo-7.0-green?logo=github)](https://github.com/ApolloAuto/apollo)
[![DoppelTest](https://img.shields.io/badge/DoppelTest-Replication_Package-purple?logo=zenodo)](https://zenodo.org/records/7657587)
![Python 3.10](https://img.shields.io/badge/Python-3.10-orange?logo=python)
![Python 3.8](https://img.shields.io/badge/Python-3.8-yellow?logo=python)

# What is included

- `evita`: the EVITA framework. Every generator (EVITA, Random, MaxDanger, MaxInteractions, and DoppelTest's generator
  on FrenetiX) runs with either simulator, **FrenetiX** or **Baidu Apollo**, selected with `--simulator`, and is
  evaluated by the same evaluator.
- `Frenetix-Motion-Planner`: EVITA's fork of the Frenetix motion planner
  ([alessiogambi/Evita-Frenetix-Motion-Planner](https://github.com/alessiogambi/Evita-Frenetix-Motion-Planner), a git
  submodule; it adds, e.g., Apollo's vehicle model).
- `EVITA_DoppelTest`: DoppelTest extended for EVITA: the Apollo runner (`apollo_scenario_runner.py`), Apollo's
  start-up script (`apollo_runner_startup.py`), the record export (`record_export.py`), DoppelTest's HD maps and their
  CommonRoad conversions (`data/`), and DoppelTest's own search, the DoppelTest baseline on Apollo (`main_ga.py`).
- `DoppelTest`: the original version of the DoppelTest framework, for reference.
- `base-scenarios`: the CommonRoad maps of the FrenetiX experiments.
- `scripts`: the replication scripts, the analysis and plotting scripts, and Apollo tools.

# Requirements

- Ubuntu (the paper was developed on Ubuntu 25.04), `git`, Python 3.10 (3.10.14 in the paper).
- For the Apollo experiments: Docker and Baidu Apollo 7.0 as distributed by DoppelTest (see [Baidu Apollo](#baidu-apollo)).
- Only for DoppelTest's own baseline on Apollo: Python 3.8 (3.8.10 in the paper).
- The system libraries OpenCV needs (`libgl1`, `libglib2.0-0`): `setup.sh` checks them and installs the missing ones.

# Setup

From the repository root:

```bash
bash setup.sh                                  # EVITA
bash setup.sh --with-doppeltest-baseline       # EVITA, and DoppelTest's own baseline
```

The project uses two Python environments, each with one role:

| Environment | Python | Used for |
|---|---|---|
| `.venv310` | 3.10 | EVITA: every generator, FrenetiX and Apollo (the Apollo runner runs with EVITA's Python) |
| `.venv38` | 3.8 | only DoppelTest's own baseline (`EVITA_DoppelTest/main_ga.py`), with DoppelTest's pinned requirements, so that it runs as published |

`setup.sh`:
- checks the system libraries OpenCV needs and installs the missing ones (`apt-get`, as root or through `sudo`);
- with `--with-doppeltest-baseline`: creates `.venv38` with DoppelTest's requirements (without the option, an existing
  `.venv38` is left as it is);
- creates `.venv310` with EVITA's requirements (pinned with `constraints.txt`), including the Apollo runner's
  packages (`docker`, `websocket-client`, `cyber_record`);
- installs `Frenetix-Motion-Planner` from EVITA's fork, at the latest commit of its `main` branch, and prints that
  commit: record it with your results;
- verifies the environment by importing OpenCV, FrenetiX's prediction, and EVITA's FrenetiX and Apollo executors.

## Baidu Apollo

Follow the Apollo setup instructions from the DoppelTest replication package:

1. Download the DoppelTest version of Apollo 7.0 from https://doi.org/10.5281/zenodo.7622089

2. In the Baidu Apollo root directory, create the folders `data`, `data/log`, `data/bag`, and `data/core` with `mkdir data data/log data/bag data/core`

> This step is necessary for DoppelTest running on the host machine to delete Apollo's log files. Our framework restarts modules being tested after every scenario, which creates a large number of unnecessary log files.

3. At the root directory of Baidu Apollo, start up Apollo container via `./docker/scripts/dev_start.sh -l`

4. Find the name of the container via `docker ps -a`

5. Enter the container in root mode with `docker exec -it apollo_dev_your_name /bin/bash`

> Remember to replace `apollo_dev_your_name` with the container's actual name

6. In the container, build Apollo via `./apollo.sh build`

Then configure `EVITA_DoppelTest/config.py`:

- `APOLLO_ROOT`: path to the root directory of Apollo 7.0.
- `DT_ROOT`: path to the `EVITA_DoppelTest` folder.
- `HD_MAP_PATH`: the HD map Apollo runs (default: `data/maps/borregas_ave/base_map.bin`).
- `SCENARIO_UPPER_LIMIT` (scenario length, seconds) and `MAX_ADC_COUNT` (vehicles per scenario): EVITA takes both
  from here when it runs on Apollo.

EVITA starts Apollo's containers itself at the beginning of every run (with `apollo_runner_startup.py`): one
container per vehicle, `apollo_dev_ROUTE_0` to `apollo_dev_ROUTE_<MAX_ADC_COUNT - 1>`.

### Maps

EVITA works on CommonRoad maps. For Apollo, each DoppelTest HD map has a CommonRoad conversion in
`EVITA_DoppelTest/data/converted_maps/<map>/<map>.xml` (with the id files linking CommonRoad and Apollo ids, and the
map's feasible paths). On Apollo, **the converted map given to EVITA must be the conversion of the HD map in
`HD_MAP_PATH`**; EVITA checks this at start-up. To convert the maps again (every map in `data/maps`):

```bash
source .venv310/bin/activate
cd EVITA_DoppelTest/data
python map_converter.py
```

# Smoke Test

Activate EVITA's environment, from the root of the project:

```bash
source .venv310/bin/activate
```

Check EVITA's command line:

```bash
(.venv310) evita --help
```

```bash
Usage: evita [OPTIONS] COMMAND [ARGS]...

Options:
  -d, --debug
  --output-folder-name TEXT
  --help

Commands:
  run
```

# Running the experiments

All of EVITA's generators run from the root of the project in `.venv310`, with either simulator:

- `--simulator frenetix` (default): the maps in `base-scenarios/`.
- `--simulator apollo`: the converted map of Apollo's HD map (see [Maps](#maps)). `--delete-records` deletes each scenario's `.record` files once its
  trace is written (by default they are kept: tens of MB per scenario).

> Note: add `--force-max-duration N` to limit each scenario to N steps (useful for debugging).

## Replication scripts

`scripts/` has one replication script per generator; each runs repetitions on one or more maps and writes every run's
results and log:

| Script | Generator |
|---|---|
| `EVITA_replication_custom.sh` | EVITA |
| `Random_replication_custom.sh` | Random |
| `TIAV_replication_custom.sh` | (1+1) EA, MaxDanger |
| `MaxInteractions_replication_custom.sh` | (1+1) EA, MaxInteractions |
| `DoppelTest_replication_custom.sh` | DoppelTest's generator, FrenetiX only |

```bash
[HOURS=<hours>] [CPUS=<cpu-list>] [SIMULATOR=frenetix|apollo] [DELETE_RECORDS=1] \
    bash scripts/<Generator>_replication_custom.sh <start_repetition> <number_of_repetitions> <map_path1> [map_path2 ...]
```

- `HOURS`: duration of each run (default 3; decimals allowed, e.g. `HOURS=0.33` for 20 minutes).
- `CPUS`: pins every run to the given cores (e.g. `CPUS=0-1`) and limits the numerical libraries to that many threads;
  give each terminal its own, disjoint list.
- `SIMULATOR`: `frenetix` (default) or `apollo`; `DELETE_RECORDS=1` (Apollo only) as `--delete-records`.
- EVITA's script also takes `TH_ED=<m>` and `N_BINS=<n>`.

A repetition whose completed run already exists (same hours, map and repetition) is skipped, and an interrupted one is
moved to `_incomplete/` and repeated: a script can be restarted safely. To rerun experiments from scratch, move their
results folder aside first. For example, the MaxDanger (TIAV) experiments on five maps, one terminal per map with two:

```bash
CPUS=0-1 bash scripts/TIAV_replication_custom.sh 1 10 base-scenarios/CHN_Cho-2_2_I-1-1.cr.xml       # terminal 1
CPUS=2-3 bash scripts/TIAV_replication_custom.sh 1 10 base-scenarios/DEU_Cologne-79_2_I-1.cr.xml    # terminal 2
CPUS=4-5 bash scripts/TIAV_replication_custom.sh 1 10 base-scenarios/USA_US101-15_2_T-1.xml         # terminal 3
CPUS=6-7 bash scripts/TIAV_replication_custom.sh 1 10 base-scenarios/USA_US101-23_1_T-1.xml         # terminal 4
CPUS=8-9 bash scripts/TIAV_replication_custom.sh 1 10 base-scenarios/USA_US101-9_1_T-1.xml          # terminal 5
```

## Generators, run directly

### Random Generation

It generates random scenarios as follows:
- Pick a number between min and max AV, N.
- Add N random AVs to the scenario, validate the scenario.
- If the scenario is valid execute it, otherwise, try again adding N random AVs.
- If after 1000 iterations, no scenario with N AVs can be found. Decrease N and repeat.

```bash
(.venv310) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.random_generator \
--scenario-generator-class RandomScenarioGenerator \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

`--time-budget` is in seconds. On Apollo: `--driving-agent-name apollo --simulator apollo` and the converted map, e.g.
`--map-file EVITA_DoppelTest/data/converted_maps/borregas_ave/borregas_ave.xml`.

### 1+1 EA

This is a simple Single Objective generator. It takes the fitness function to maximize as input.

```bash
(.venv310) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.basic_generator \
--scenario-generator-class EAOnePlusOneGenerator_MaxDanger \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

```bash
(.venv310) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.basic_generator \
--scenario-generator-class EAOnePlusOneGenerator_MaxInteractions \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

### EVITA

EVITA runs on pymoo and has its own entry point; `--time-budget` is in minutes:

```bash
(.venv310) python -m evita.generators.nsgaII_search_no_generator --population-size 10 --time-budget 180 \
    --scenario-seed-file base-scenarios/showcase-scenario.xml [--simulator apollo] [--seed N] [--threshold M] [--n-bins N]
```

### DoppelTest + Frenetix

Please refer to the ICSE 2023 paper for the description of the algorithm.

> Note: We had to adapt the original implementation that was tightly integrated with Apollo to fit our execution environment

```bash
(.venv310) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.doppeltest_generator \
--scenario-generator-class DoppelTestScenarioGenerator \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

This generator runs on FrenetiX only; on Apollo, the DoppelTest baseline is DoppelTest itself (next section).

## DoppelTest + Baidu Apollo (the DoppelTest baseline)

DoppelTest's own search (`main_ga.py`), in its own environment (`bash setup.sh --with-doppeltest-baseline`).
`generate_DoppelTest_dataset.sh` runs repetitions of it, like the replication scripts:

```bash
source .venv38/bin/activate
[HOURS=<hours>] [MAP=<map>] bash EVITA_DoppelTest/generate_DoppelTest_dataset.sh <start_repetition> <number_of_repetitions>
```

- `HOURS`: duration of each run (default: `RUN_FOR_HOUR` in `config.py`; decimals allowed).
- `MAP`: DoppelTest's map, a folder of `EVITA_DoppelTest/data/maps` (default: the map of `HD_MAP_PATH`). It must be the
  map Apollo loads (`--map_dir` in Apollo's `modules/common/data/global_flagfile.txt`): checked before every run.
- `MAX_ADC_COUNT` and `SCENARIO_UPPER_LIMIT` stay as in `config.py` (EVITA takes them from there too, on Apollo).

Each repetition gets `results/Main-Experiment/Apollo/DoppelTest/<map>/DoppelTest_RUN_FOR_<h>_HOURS_SCENARIO_MAP_<map>_REPETITION_<k>_<time>/`
(main_ga.py's experiment folder with the records, `run_parameters.json`, main_ga.py's `log.bin` and `hof.bin`,
`run_completed.json`) and its log in `logs/Main-Experiment/Apollo/DoppelTest/<map>/`. Completed repetitions are skipped
and interrupted ones repeated, as in the replication scripts. The script runs DoppelTest through
`EVITA_DoppelTest/run_main_ga.py`, which sets the run's values on `config` before DoppelTest's modules are imported.

> TODO: DoppelTest's runs produce `.record` files; their export to traces and evaluation with EVITA's evaluator
> (`scripts/apollo_evaluate.py`) is to be documented once the record export for DoppelTest's runs is restored.

## How Apollo runs a scenario

On Apollo, every scenario runs in `<run folder>/<scenario id>/`: DoppelTest's JSON (`c.json`), the scenario
(`scenario.xml`), the runner's log (`runner.log`), the trace (`trace.json.gz`) and the records (`records/`). Each run
is health-checked: a run where a vehicle did not run, lost its planning or never moved is retried once (the failed
attempt is kept in `failed_attempt_1/`); a scenario that stays unusable, or that DoppelTest rejects, is invalid and
its folder is moved to `<run folder>/debug/`. A progress bar shows each scenario's execution (setting up, running,
saving records and trace).

# Outputs

Every run folder contains, besides the scenario folders:

- the report: `behavior.csv`, the interaction and collision maps (`*.npy`), `requirements_violations.csv`;
- `metadata.csv` (generator, map, budget, simulator), `performance.csv`;
- `task_times.csv`, `task_times_per_scenario.csv`: the time spent in each task (Apollo's phases included);
- `scenario_outcomes.csv`: one row per generated scenario (executed, valid, and why not).

# Analysis

## Scenario outcomes

Counts of the runs' scenarios by outcome (usable, refused by validation, skipped, non-interesting, simulator failure,
with their causes), per generator and simulator:

```bash
(.venv310) python scripts/scenario_outcomes_analysis.py results/Main-Experiment [--out summary.csv]
```

## Aggregate Experiments Results

```bash
(.venv310) python scripts/analysis.py <experiments_folder_path> [<max_workers>]
```

`scripts/analysis.sh <experiments_folder_path> <analysis|plot|both> [max_workers]` runs the aggregation and the plots together.

## Generate Comparison Plots

```bash
(.venv310) python scripts/generate_plots.py --experiments <folder> [<folder> ...] --labels <label> [<label> ...] --output <output_folder>
```

## Interactions and false positives

- `scripts/interactions_extraction.py` (`extract`, `criticality`, `maneuvers`): the interactions of the experiments,
  their criticality and maneuvers.
- `scripts/false_positive_analysis.py` (`sample`, `execute`, `analyze`): interacting vehicles removed from scenarios,
  to measure false positives of the interaction detection.

## Apollo tools

- `scripts/apollo_evaluate.py --traces <folder> --map <converted map.xml> --out <run folder>`: evaluates Apollo traces
  again (e.g. after a change of the evaluator) and writes the report, without running Apollo.
- `scripts/apollo_check.py trace|convert|spec`: inspects a trace, converts it into the executed scenario, or builds a
  scenario's DoppelTest JSON.