"""
Main entry point of EVITA (EVolutionary Interaction Testing Approach).

The system comprises four components: the orchestrator, the scenario generator, the scenario executor and the
scenario evaluator. The orchestrator implements a fixed logic; the other components are loaded dynamically from
the command-line arguments, so the platform can be easily extended.

Command: run.
"""
import logging

from .evita import cli
from .orchestrator import run as run_orchestrator

# Silence shapely's runtime warnings
logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

cli.add_command(run_orchestrator)

if __name__ == "__main__":
    cli()