#!/bin/bash

# Run all the generators once for the given amount of seconds

GENERATION_BUDGET=10800 # 3 hours

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ROOT_FOLDER=$(dirname -- ${SCRIPT_DIR})

SCENARIOS_FOLDER="${ROOT_FOLDER}/base-scenarios"

# "evita.generators.doppeltest_generator DoppelTestScenarioGenerator" "evita.generators.random_generator RandomScenarioGenerator" "evita.generators.basic_generator EAOnePlusOneGenerator_MaxDanger" "evita.generators.basic_generator EAOnePlusOneGenerator_MaxInteractions"
for i in "evita.generators.random_generator RandomScenarioGenerator"
do

    for SCENARIO_NAME in $(find ${SCENARIOS_FOLDER} -iname "*.xml" -exec basename {} \;)
    do
        SCENARIO_FILE="${SCENARIOS_FOLDER}/${SCENARIO_NAME}"

        pushd $ROOT_FOLDER > /dev/null
            set -- $i # Convert the "tuple" into the param args $1 $2...

	    # Define the output folder
            OUTPUT_FOLDER_NAME="${SCENARIO_NAME}_$1.$2_$(date +%Y%m%d%H%M%S)"

	    # Creating the output folder into results folder
	    mkdir -p "results/${OUTPUT_FOLDER_NAME}"

            echo "--------------------------------------------------------------------------"
            echo "    $OUTPUT_FOLDER_NAME"
            echo "--------------------------------------------------------------------------"

            evita --output-folder-name $OUTPUT_FOLDER_NAME \
                    run --scenario-generator-path . \
                        --scenario-generator-module $1 \
                        --scenario-generator-class  $2 \
                        --driving-agent-name frenetix \
                        --time-budget ${GENERATION_BUDGET} \
                        --map-file ${SCENARIO_FILE} 2>&1 | tee "results/${OUTPUT_FOLDER_NAME}/execution.log"
        popd > /dev/null
    done

done
