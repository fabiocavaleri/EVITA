# This is the main entry point for EVITA (i.e., dodge/avoid) : An Environment for Vehicle Interaction Testing Automation

# The system comprises four main components: the orchestrator, the scenario generator, the scenario executor,
# and the scenario evaluator.
#
# Besides the orchestrator, which implements a fixed logic, all the other components are dynamically loaded
# based on input arguments, ensuring the platform can be easily extended

# Available Commands:
# = run

# References:
# - https://realpython.com/python-click/#adding-arguments-to-a-click-app
# - https://stackoverflow.com/questions/75387904/how-to-exclude-tests-folder-from-the-wheel-of-a-pyproject-toml-managed-lib

import logging
# Try to disable shapely's runtime warning
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)


# Import the commands from orchestrator.py
# - https://click.palletsprojects.com/en/6.x/quickstart/#nesting-commands
from .evita import cli

# Import the commands
from .orchestrator import run as run_orchestrator, replay
from .data_analysis import analyse

# Build the app
cli.add_command(run_orchestrator)
cli.add_command(replay)
cli.add_command(analyse)

# Run the app
if __name__ == "__main__":
    cli()