# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
This module can be used to check if a dataset_structure file has the right
format, the given keys are good and will tell you about the default values
for the missing ones.
"""

import argparse
import os
from typing import Tuple

from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_and_structure import DSSFInterpreter
from modulos_utils.dssf_and_structure import structure_logging as struc_log

client_logging = struc_log.LoggingPurpose.CLIENT


def check_my_dssf(
        dssf_path: str, logging_purpose:
        struc_log.LoggingPurpose = struc_log.LoggingPurpose.INTERNAL) \
        -> Tuple[bool, str]:
    """Read in a dssf file and check if it is valid (without looking at the
    data). It will look for all required values and check their type, look
    through the optional ones and will replace the wrong ones and will
    inform about the chosen default values for the missing ones.

    Args:
        dssf_path (str): Path to the dssf file.
        logging_purpose (struc_log.LoggingPurpose): Whether to log for the
            batch client.

    Returns:
        Tuple[bool, str]: Whether DSSF is valid and the report of the check.
    """
    interpreter = DSSFInterpreter.DSSFileInterpreter(dssf_path)
    try:
        interpreter.initialize_interpreter()
    except (DSSFErrors.DSSFFormatError,
            DSSFErrors.DSSFEntryError,
            DSSFErrors.DSSFVersionError) as err:
        if logging_purpose != client_logging:
            print(err)
        return False, str(err)
    (new_info, old_info, (ignored_keys, wrong_type_keys)) = \
        interpreter.initialize_components(os.path.dirname(dssf_path), 10)

    if new_info == old_info:
        report = "The dataset_structure.json can be used.\n"
        report += ("All the optional values are given"
                   " and their types are correct.\n")
    else:
        report = ""
        msg = "|Component: '{0}' |Key: '{1}' |Value: '{2}' | "
        if (any(list(wrong_type_keys.values())) or
                any(list(ignored_keys.values()))):
            report += "    Error!:\n"
            for name in wrong_type_keys:
                for key in wrong_type_keys[name]:
                    report += (msg.format(name, key, old_info[name][key]) +
                               "The value is not valid for this component!\n")
            for name in ignored_keys:
                for key in ignored_keys[name]:
                    report += (msg.format(name, key, old_info[name][key]) +
                               "Key is not defined for this component.\n")
            return False, report

        report = "The dataset_structure.json can be used.\n"
        hints = False
        for name in new_info:
            for key in new_info[name]:
                if key not in old_info[name]:
                    if not hints:
                        report += "    Hints:\n"
                        hints = True
                    report += (msg.format(name, key, "-") +
                               "The optional keyword is missing for this "
                               "component and therefore set to the default "
                               f"value ('{str(new_info[name][key])}').\n")
    if logging_purpose != client_logging:
        print(report)
    return True, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check my DSSF!")
    parser.add_argument("--json", type=str, required=True,
                        help="Path to the dataset_structure.json to test.")
    args = parser.parse_args()
    dssf_path = args.json
    check_my_dssf(dssf_path)
