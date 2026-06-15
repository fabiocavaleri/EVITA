# Module contains the actual app, so it can be imported in the tests

############################################################################
# Main Entry Point
############################################################################

from .utils import initialize_main_logger, get_main_logger, RESULTS_FOLDER

import click
import os

@click.group()
@click.option('-d', '--debug', is_flag=True, show_default=True, default=False)
@click.option('--output-folder-name', is_flag=False, type=str, required=False, multiple=False)
@click.option('--output-folder', is_flag=False, type=click.Path(exists=False, dir_okay=True, file_okay=False), required=False, multiple=False)
@click.pass_context
def cli(ctx, debug, output_folder_name, output_folder):
    # Global Variables
    ctx.obj = dict()

    # Debug flag
    ctx.obj["debug"] = debug
    
    # Name of the output folder - Used by run - TODO Move it to run command?
    if output_folder_name:
        ctx.obj["output_folder_name"] = output_folder_name
    
    # Actual output folder - Used by replay and analyse- TODO Move it to replay/analyse commands?
    if output_folder:
        ctx.obj["output_folder"] = output_folder

    # Setup folder structure by ensuring that the basic folder structure is there, i.e., "results".
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Make sure we initialize the main logger once for all
    initialize_main_logger(debug)

    # Print a BANNER
    logger = get_main_logger()
    opening_msg = " EVITA: Starting the execution "
    logger.info(" " * len(opening_msg))
    logger.info(" " * len(opening_msg))
    logger.info(opening_msg)
    logger.info(" " * len(opening_msg))
    logger.info(" " * len(opening_msg))