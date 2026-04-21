#!/bin/bash

# Run all the generators once for the given amount of seconds

GENERATION_BUDGET=10800 # 3 Hours

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ROOT_FOLDER=$(dirname -- ${SCRIPT_DIR})

SCENARIOS_FOLDER="${ROOT_FOLDER}/base-scenarios"

# Run two times random.
for SCENARIO_NAME in "USA_US101-9_1_T-1.xml" "USA_US101-9_1_T-1.xml" "USA_US101-9_1_T-1.xml" "USA_US101-9_1_T-1.xml" "USA_US101-9_1_T-1.xml"
do

    SCENARIO_FILE="${SCENARIOS_FOLDER}/${SCENARIO_NAME}"
    for i in "evita.generators.basic_generator EAOnePlusOneGenerator_MaxInteractions"
    do
        pushd $ROOT_FOLDER
            set -- $i # Convert the "tuple" into the param args $1 $2...

            # Store the results in a folder that contains the scenario name, the generator name, and some salt
            # Checkpoint results every hour or so
            OUTPUT_FOLDER_NAME="${SCENARIO_NAME}_$1.$2_$(date +%Y%m%d%H%M%S)"

            echo "--------------------------------------------------------------------------"
            echo "    $OUTPUT_FOLDER_NAME"
            echo "--------------------------------------------------------------------------"

            evita --output-folder-name $OUTPUT_FOLDER_NAME \
                run --scenario-generator-path . \
                    --scenario-generator-module $1 \
                    --scenario-generator-class  $2 \
                    --driving-agent-name frenetix \
                    --time-budget ${GENERATION_BUDGET} \
                    --map-file ${SCENARIO_FILE} 2>&1 | tee "results/${OUTPUT_FOLDER_NAME}.txt"

        popd
    done

done
