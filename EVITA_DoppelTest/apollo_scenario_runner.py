import os
import sys
import json
from pathlib import Path

from hdmap.MapParser import MapParser
from config import HD_MAP_PATH, RuntimeConfig
from apollo.ApolloContainer import ApolloContainer
from framework.scenario import Scenario as ApolloScenario
from framework.scenario.ScenarioRunner import ScenarioRunner

from conversion_scripts.scenario_converter import scenario_converter
from conversion_scripts.utils.RecordFileError import RecordFileError

from config import APOLLO_ROOT, MAX_ADC_COUNT

MAX_TRIALS = 3

def main(json_file_path: str, generation_number: str, scenario_number: str):
    """
    Main function to run the Apollo scenario and convert the executed scenario data
    """
    
    # Get Apollo containers
    containers = [ApolloContainer(APOLLO_ROOT, f'ROUTE_{x}') for x in range(MAX_ADC_COUNT)]

    # Load the HD map
    mp = MapParser(HD_MAP_PATH)
    
    # Load and convert the scenario from .json format to ApolloScenario object
    apollo_scenario = ApolloScenario.from_json(json_file_path)

    ##### Additional code to check the scenario routing correctness #####
    '''
    for agent in apollo_scenario.ad_section.adcs:
        start_lane = agent.routing[0]
        final_lane = agent.routing[-1]

        print(agent.routing)
        print(start_lane)
        print(final_lane)

        while True:
            allowed_routing = mp._MapParser__get_reachable_from(start_lane, 10)
            if len(allowed_routing) > 0:
                # routing = choice(allowed_routing)
                ok = False
                for route in allowed_routing:
                    if route[-1] == final_lane:
                        print("Start Final OK")
                        ok = True
                        break
                if agent.routing in allowed_routing:
                    print("Route OK")

                if not ok:
                    print(allowed_routing)
    
                break

        print("=" * 50)

    print("\n")
    print("-" * 50)
    '''
    #####
    
    # Define generation and scenario names
    g_name = f"Generation_{generation_number}"
    s_name = f"Scenario_{scenario_number}"
    
    completed = False
    trials = 0
    # Try to run the scenario and convert the executed scenario data
    # If conversion fails, reexecute the scenario until conversion succeeds or max trials is reached
    while not completed and trials < MAX_TRIALS:
        # Run the scenario in Apollo and get the executed scenario data
        srunner = ScenarioRunner(containers)
        srunner.set_scenario(apollo_scenario)
        srunner.init_scenario()
        runners = srunner.run_scenario(g_name, s_name, True)

        # Build map between Apollo container IDs and runners ids
        ids_map = {}
        for a, _ in runners:
            ids_map[a.nid] = a.container.container_name
        
        # Build record files folder path
        record_files_path = os.path.join(RuntimeConfig.RECORDS_DIR, g_name, s_name)

        # Extract scenario map name
        scenario_map = Path(HD_MAP_PATH).parent.name

        # Create folder for CommonRoad converted scenario and analysis results
        executed_scenario_folder = record_files_path.replace("/records/", "/evita-records/")
        os.makedirs(executed_scenario_folder, exist_ok = True)
        
        # Save the map between Apollo container IDs and runners ids
        ids_map_file_path = os.path.join(executed_scenario_folder, "ids_map.json")
        with open(ids_map_file_path, "w") as f:
            json.dump(ids_map, f, indent = 4)

        # Try scenario conversion to CommonRoad format
        # If conversion fails, reexecute the scenario until conversion succeeds or max trials is reached
        try:
            scenario_converter(record_files_path, executed_scenario_folder, scenario_map, ids_map)
            completed = True
        except RecordFileError:
            sys.exit(10)
        except AttributeError:
            # Rexecute the scenario if conversion fails (e.g., Record files opening error, all vehicles without history data)
            trials += 1
            if trials >= MAX_TRIALS:
                sys.exit(10)
                # raise

if __name__ == "__main__":
    # Extract arguments    
    if len(sys.argv) != 5:
        print("Usage: python apollo_scenario_runner.py <json_file_path> <generation_number> <scenario_number> <records_dir>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    generation_number = sys.argv[2]
    scenario_number = sys.argv[3]
    RuntimeConfig.RECORDS_DIR = sys.argv[4]

    main(json_file_path, generation_number, scenario_number)