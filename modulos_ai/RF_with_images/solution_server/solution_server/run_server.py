# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the command line interface for the solution app.
"""
import argparse
import os
import sys
import yaml

from solution_server import app as solution_app
from solution_server import server_utils as su


class CustomTypes:

    @staticmethod
    def custom_dir_path_str(value: str) -> str:
        abs_path = os.path.abspath(value)

        if not os.path.isdir(abs_path):
            sys.exit(f"\n\tERROR:  The path {abs_path} is not a valid "
                     "directory.\n")

        return abs_path

    @staticmethod
    def custom_file_path_str(value: str) -> str:
        abs_path = os.path.abspath(value)

        if not os.path.isfile(abs_path):
            sys.exit(f"\n\tERROR:  The path {abs_path} is not a valid file.\n")

        return abs_path


def set_argument_parser(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments and a description to an argument parser object.

    Args:
        parser (argparse.ArgumentParser): Argument parser object to modify.

    Returns:
        (argparse.ArgumentParser): Modified argument parser object.
    """
    parser.description = ("Run a rest server that serves the prediction API "
                          "of a modulos solution.")
    parser.add_argument("--src", type=CustomTypes.custom_dir_path_str,
                        required=False, default="./src",
                        help="Path to the src directory of the downloaded "
                        "solution that is used to make predictions. The src "
                        "directory contains the modules, the metadata, and "
                        "the dataset history file")
    parser.add_argument("--config", type=CustomTypes.custom_file_path_str,
                        required=False, default="./solution_server_config.yml",
                        help="Path to the directory of the downloaded "
                        "solution that is used to make predictions.")
    return parser


def main(args: argparse.Namespace) -> None:
    """Main function, that calls the `run` of the flask app.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    # Load api config file to get the port and the ip for the app.run().
    su.check_config_file(args.config, args.src)
    with open(args.config, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    port = config_dict["port"]
    host = config_dict["host"]

    app = solution_app.create_app(args.config, args.src)
    app.run(port=port, host=host)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = set_argument_parser(parser)
    args = parser.parse_args()
    main(args)
