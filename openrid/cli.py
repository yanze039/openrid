#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Openrid command-line interface (cli)

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

import argparse
from openrid.openrid import ReinforcedDynamics


# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================


# TODO: Add optional arguments that we can use to override sys.argv for testing purposes.
def main(argv=None):
    # Parse arguments.
    parser = argparse.ArgumentParser(description='OpenRID command-line interface.')
    parser.add_argument("YAML", help="YAML file containing the simulation parameters.")
    parser.add_argument('--version', action='version', version='%(prog)s ' + '0.0.1')
    args = parser.parse_args()

    # Run the main function.
    rid = ReinforcedDynamics(config="./ala2.yaml")
    rid.run()

    