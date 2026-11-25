"""
Record the outcome of every generated scenario, one row each, in <run folder>/scenario_outcomes.csv,
so that a run's scenarios can be counted by cause afterwards:

    scenario    generation id
    simulator   frenetix / apollo
    vehicles    number of vehicles (planning problems)
    executed    whether the scenario was given to the simulator (valid when validated, budget not over)
    valid       whether it ended valid (executed and usable: evaluated)
    message     why it is not valid: the validator's, the executor's, or the budget's message (empty if valid)

A scenario refused by validation is not executed; an executed scenario can still end invalid (an execution error, a
FrenetiX agent that does not start, an Apollo run that cannot be used). Rows are appended as scenarios finish, so the
file is complete even if a run stops early.
"""
import os
import csv

OUTCOMES_FILE = "scenario_outcomes.csv"
OUTCOME_COLUMNS = ["scenario", "simulator", "vehicles", "executed", "valid", "message"]


def record_outcome(output_folder: str, scenario, planning_problem_set, executed: bool, simulator: str) -> None:
    """
    Append the scenario's outcome to <output_folder>/scenario_outcomes.csv (written with its header the first time)
    """
    path = os.path.join(output_folder, OUTCOMES_FILE)
    new_file = not os.path.exists(path)

    valid = bool(getattr(scenario, "is_valid", False))
    if valid:
        message = ""
    else:
        message = str(getattr(scenario, "validation_message", "") or "")
    
    with open(path, "a", newline = "") as csvfile:
        writer = csv.writer(csvfile)
        if new_file:
            writer.writerow(OUTCOME_COLUMNS)
        writer.writerow([getattr(scenario, "generation_id", ""), simulator, len(planning_problem_set.planning_problem_dict),
                         int(bool(executed)), int(valid), " ".join(message.split())])
