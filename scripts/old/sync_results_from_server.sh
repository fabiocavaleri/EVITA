#!/bin/bash

# Sync the results from the server to this machineall the generators once for the given amount of seconds
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ROOT_FOLDER=$(dirname -- ${SCRIPT_DIR})

ORIGIN_RESULT_FOLDER="/home/gambia/icse25/results"
DESTINATION_RESULT_FOLDER="${ROOT_FOLDER}/results-alien"

# Compress, all files, show progress
echo "SYNCH alien:${ORIGIN_RESULT_FOLDER} to ${DESTINATION_RESULT_FOLDER}"
 
rsync -azP  alien:${ORIGIN_RESULT_FOLDER} ${DESTINATION_RESULT_FOLDER}