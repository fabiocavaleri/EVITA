#!/bin/bash

# Run all the generators once for the given amount of seconds

GENERATION_BUDGET=180 # 3 ore. This uses minutes instead of seconds... 30 minutes - This is a test

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ROOT_FOLDER=$(dirname -- ${SCRIPT_DIR})

SCENARIOS_FOLDER="${ROOT_FOLDER}/base-scenarios"

for SCENARIO_NAME in "CHN_Cho-2_2_I-1-1.cr.xml" "DEU_Cologne-79_2_I-1.cr.xml" "showcase-scenario.xml" "test_scenario_6_planners.xml" "USA_US101-9_1_T-1.xml" "USA_US101-15_2_T-1.xml" "USA_US101-23_1_T-1.xml" "CHN_Cho-2_2_I-1-1.cr.xml" "DEU_Cologne-79_2_I-1.cr.xml" "showcase-scenario.xml" "test_scenario_6_planners.xml" "USA_US101-9_1_T-1.xml" "USA_US101-15_2_T-1.xml" "USA_US101-23_1_T-1.xml" "CHN_Cho-2_2_I-1-1.cr.xml" "DEU_Cologne-79_2_I-1.cr.xml" "showcase-scenario.xml" "test_scenario_6_planners.xml" "USA_US101-9_1_T-1.xml" "USA_US101-15_2_T-1.xml" "USA_US101-23_1_T-1.xml" "CHN_Cho-2_2_I-1-1.cr.xml" "DEU_Cologne-79_2_I-1.cr.xml" "showcase-scenario.xml" "test_scenario_6_planners.xml" "USA_US101-9_1_T-1.xml" "USA_US101-15_2_T-1.xml" "USA_US101-23_1_T-1.xml" "CHN_Cho-2_2_I-1-1.cr.xml" "DEU_Cologne-79_2_I-1.cr.xml" "showcase-scenario.xml" "test_scenario_6_planners.xml" "USA_US101-9_1_T-1.xml" "USA_US101-15_2_T-1.xml" "USA_US101-23_1_T-1.xml"
do

    SCENARIO_FILE="${SCENARIOS_FOLDER}/${SCENARIO_NAME}"
    
    for i in "evita.generators.nsgaii NSGAIIproblem"
    do

        pushd $ROOT_FOLDER

            set -- $i # Convert the "tuple" into the param args $1 $2...

            # Store the results in a folder that contains the scenario name, the generator name, and some salt
            # Checkpoint results every hour or so
            OUTPUT_FOLDER_NAME="${SCENARIO_NAME}_$1.$2_$(date +%Y%m%d%H%M%S)"

            echo "--------------------------------------------------------------------------"
            echo "    $OUTPUT_FOLDER_NAME"
            echo "--------------------------------------------------------------------------"

            pushd evita

                python generators/nsgaII_search_no_generator.py 10 ${GENERATION_BUDGET} ${SCENARIO_FILE} ${OUTPUT_FOLDER_NAME} 2>&1 | tee "${ROOT_FOLDER}/results/${OUTPUT_FOLDER_NAME}.txt"

            popd

        popd
    done

done

