# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Load the modulos_utils config."""
import logging
import os
from typing import Dict
import yaml


MODULOS_UTILS_CONFIG = "config.yml"


def load_config_file() -> Dict:
    """Load the modulos_utils config file as a dict.

    Returns:
        Dict: The loaded config.
    """
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(
                __file__)), MODULOS_UTILS_CONFIG)) as config_file:
            config = yaml.safe_load(config_file.read())
    except FileNotFoundError as err:
        msg = "Modulos_utils config file does not exist!"
        logging.error(msg)
        raise FileNotFoundError(f"{err}\n{msg}")

    return config


def get_datetime_heuristic_status() -> bool:
    """Return if the automatic datetime heuristic is switched on or off.

    Returns:
        bool: If true, automatic datetime heuristic is on.
    """
    config = load_config_file()
    if "AUTOMATIC_DATETIME_HEURISTIC_ON" not in config:
        msg = ("Keyword 'AUTOMATIC_DATETIME_HEURISTIC_ON' missing in the "
               "modulos_utils config file!")
        logging.error(msg)
        raise KeyError(msg)
    return config["AUTOMATIC_DATETIME_HEURISTIC_ON"]
