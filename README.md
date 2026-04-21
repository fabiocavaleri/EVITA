# Replication Package for Automatically Testing Interacting Autonomous Vehicles

This repository contains the code to reproduce the results of our paper on Automatically Testing Interacting Autonomous Vehicles.

# Setup

The code depends on open-source libraries that we had to patch to introduce missing features. We install them directly from github

> Note: This will not work for those who are NOT contributors to the project.

## Python Version

Check that you have python 3.10

## Setup venv

Create the venv using:

```
python -m venv .venv
```

Activate the .venv (Linux style)

```
. .venv/bin/activate
```
## Install deps

Upgrade pip

```
(.venv)  pip install --upgrade pip
```

Initialize and update the Evita-Frenetix-Motion-Planner git submodul

```
(.venv)  git submodule init
(.venv)  git submodule update --remote
```

Install the Evita-Frenetix-Motion-Planner 

```
(.venv) cd Frenetix-Motion-Planner
(.venv/Frenetix-Motion-Planner) pip install -e .
```

Install the requirements 

```
(.venv/Frenetix-Motion-Planner) cd ..
(.venv) pip install -r requirements.txt
```

Install Evita

```
(.venv) pip install -e .
```

Install DoppelTest.

> NOTE: This might require old versions of numpy and pandas, double check if they are really necessary

```
(.venv) cd DoppelTest
(.venv/DoppelTest) pip install -r requirements.txt
(.venv/DoppelTest) pip install -e .
```

> NOTE: This fails because numpy and pandas versions are not avilable for this version of python ... but it should work nevertheless?

# Smoke test the installation

In the active virtual environment, from the root of the project, run:

```
(.venv) evita --help
```

Check that the result looks like:

```
Usage: evita [OPTIONS] COMMAND [ARGS]...

Options:
  -d, --debug
  --help       Show this message and exit.

Commands:
  run
  
```

# Run the implemented generators:

All the generators must be executed from the root of the project after activating the virtual environment.

> Note: Add the --max-duration N to limit the execution of each scenario to N steps (useful for debug and play around with the execution)

## Random Generation

It generates random scenarios as follows:
- Pick a number between min and max AV, N.
- Add N random AVs to the scenario, validate the scenario.
- If the scenario is valid execute it, otherwise, try again adding N random AVs.
- If after 1000 iterations, no scenario with N AVs can be found. Decrease N and repeat.

```
(.venv) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.random_generator \
--scenario-generator-class RandomScenarioGenerator \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```

## 1+1 EA

This is a simple Single Objective generator. It takes the fitness function to maximize as input.

```
(.venv) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.basic_generator \
--scenario-generator-class EAOnePlusOneGenerator_MaxDanger \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```


```
(.venv) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.basic_generator \
--scenario-generator-class EAOnePlusOneGenerator_MaxInteractions \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml 
```

## EVITA

For the moment, EVITA runs on pymoo which is not well integrated in the orchestrator, so we invoke this generator directly.
To run it invoke:

```
(.venv) cd evita
(.venv) python generator/nsgaII_search_no_generator.py 10 3 ../base/scenarios/showcase-scenario.xml
```

## DoppelTest

Please refere to the ICSE 2023 paper for the description of the algorithm.

> Note: We had to adapt the original implementation that was tightly integrated with Apollo to fit our execution environment

```
(.venv) evita run --scenario-generator-path . \
--scenario-generator-module evita.generators.doppeltest_generator \
--scenario-generator-class DoppelTestScenarioGenerator \
--driving-agent-name frenetix \
--time-budget 3600 \
--map-file base-scenarios/showcase-scenario.xml
```
