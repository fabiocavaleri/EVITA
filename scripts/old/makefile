SHELL = /bin/bash

# Utility functions

word-dot = $(word $2,$(subst _._, ,$1))

# Private variables

_mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
# the name of the current dir
_current_dir := $(notdir $(patsubst %/,%,$(dir $(_mkfile_path))))
# the home of the project
_evita_home := $(abspath $(dir $(_current_dir)))
# the standard results folder
_result_folder = $(_evita_home)/results
# the standard folder containing the scenarios
_scenarios_folder = $(_evita_home)/base-scenarios
# our python
_python= $(_evita_home)/.venv/bin/python

# Variables that can be overwritten by the user using the CLI 

# --debug
EVITA_OPTS = 

# Generation budget in seconds
GENERATION_BUDGET = 10800

# Repetitions to generate for each scenario/generator execution
N_RUNS = 5

SCENARIOS = CHN_Cho-2_2_I-1-1.cr.xml  DEU_Cologne-79_2_I-1.cr.xml  USA_US101-15_2_T-1.xml  USA_US101-23_1_T-1.xml  USA_US101-9_1_T-1.xml

# All of them
# # evita.generators.random_generator RandomScenarioGenerator" "evita.generators.basic_generator EAOnePlusOneGenerator_MaxDanger" "evita.generators.basic_generator EAOnePlusOneGenerator_MaxInteractions"

# GENERATORS = evita.generators.doppeltest_generator_._DoppelTestScenarioGenerator evita.generators.random_generator_._RandomScenarioGenerator evita.generators.nsgaii_._NSGAIIproblem
GENERATORS = evita.generators.nsgaii_._NSGAIIproblem

# Derived variables
# the sequence of runs
_runs := $(shell seq -w 01 $(N_RUNS))

# Generate the experiment folders name
_expected_experiments := $(addsuffix /collision_map.npy, $(addprefix $(_result_folder)/, $(shell for G in $(GENERATORS); do for S in $(SCENARIOS); do for R in $(_runs); do echo "$${S}_._$${G}_._$${R}"; done; done; done)))


######## Make Targets

# Run all the experiments
all: $(_expected_experiments)


# Run a single experiment
$(_result_folder)/%/collision_map.npy:
# Extract the inputs from the target and prepare the script for the execution
	$(eval output_folder_name := $*)
	$(eval scenario_file := $(addprefix $(_scenarios_folder)/, $(call word-dot,$*,1)))
	$(eval generator_module := $(call word-dot,$*,2))
	$(eval generator_name := $(call word-dot,$*,3))
	$(eval run_id := $(call word-dot,$*,4))
# Output some debug
	@echo "--------------------------------------------------------------------------"
	@echo "    $(output_folder_name)"
	@echo "--------------------------------------------------------------------------"
# Prepare the environment by deleting any old folder and recreating it
	@if [ -e $(_result_folder)/$(output_folder_name) ]; then mv -v $(_result_folder)/$(output_folder_name) $(_result_folder)/$(output_folder_name)_BKP; fi
	@mkdir -p $(_result_folder)/$(output_folder_name)
# # Start the execution from the expected folder
	@cd $(_evita_home); \
	if [ "$(generator_name)" == "NSGAIIproblem" ]; then \
		cd evita; \
		$(_python) generators/nsgaII_search_no_generator.py 10 $(GENERATION_BUDGET) $(scenario_file) $(output_folder_name) 2>&1 | \
			tee "$(_result_folder)/$(output_folder_name)/execution.log"; \
	else \
		$(_python) -m evita $(EVITA_OPTS) --output-folder-name $(output_folder_name) \
			run --scenario-generator-path . \
				--scenario-generator-module $(generator_module) \
				--scenario-generator-class  $(generator_name) \
				--driving-agent-name frenetix \
				--time-budget $(GENERATION_BUDGET) \
				--map-file $(scenario_file) 2>&1 | \
					tee "$(_result_folder)/$(output_folder_name)/execution.log"; \
	fi
