# Module contains the actual app, so it can be imported in the tests

############################################################################
# Main Entry Point
############################################################################

import os
import click

from .utils import RESULTS_FOLDER, get_main_logger, initialize_main_logger


@click.group()
@click.option('-d', '--debug', is_flag = True, show_default = True, default = False)
@click.option('--output-folder-name', is_flag = False, type = str, required = False, multiple = False)
@click.option('--output-folder', is_flag = False, type = click.Path(exists = False, dir_okay = True, file_okay = False), required = False, multiple = False)
@click.pass_context
def cli(ctx, debug, output_folder_name, output_folder):
    """
    EVITA: EVolutionary Interaction Testing Approach.
    """
    # Options shared by the commands
    ctx.obj = {"debug": debug}

    # Name of the output folder of a new run (used by `run`)
    if output_folder_name:
        ctx.obj["output_folder_name"] = output_folder_name

    # Existing output folder (used by `replay` and `analyse`)
    if output_folder:
        ctx.obj["output_folder"] = output_folder

    # Make sure the results folder exists
    os.makedirs(RESULTS_FOLDER, exist_ok = True)

    # Initialize the main logger once for all commands
    initialize_main_logger(debug)

    logger = get_main_logger()
    logger.info("EVITA: Starting the execution")
