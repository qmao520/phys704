# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""The file defines the usage of the modulos solution package.
"""
import argparse

from solution_server import run_server


def _get_argument_parser() -> argparse.ArgumentParser:
    """Get the argument parser of the modulos solution command line tool.

    Returns:
        argparse.ArgumentParser: Argument parser object with arguments defined.
    """
    parser = argparse.ArgumentParser(
        prog="Modulos Solution Server",
        description="Package to run a server on the localhost that serves a "
                    "the prediction REST API of a modulos solution.")

    subparsers = parser.add_subparsers(
        dest="action",
        help="Choose what action to take.")
    subparsers.required = True

    start_help = ("Start a server on the localhost that serves a prediction "
                  "API.")

    start_parser = subparsers.add_parser("start", help=start_help)
    start_parser.set_defaults(which="start")
    run_server.set_argument_parser(start_parser)
    return parser


def main() -> None:
    """Main function of the modulos solution server package.
    """
    parser = _get_argument_parser()
    arguments = parser.parse_args()
    run_server.main(arguments)
    return None


if __name__ == "__main__":
    main()
