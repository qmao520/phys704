# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Converter function from old DSSF format to DSSF version 0.1.
"""
import argparse
import copy
import json
import os
import sys
import logging
from typing import Dict, Any, List, Union

from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_and_structure import dssf_utils


VERSION_KEY = "_version"


def v01_to_v02(legacy_dssf: List) -> List:
    """Convert a dssf, that is in the version v0.1, to the version v0.2,

    Args:
        legacy_dssf (List): Version v.0.1 DSSF.

    Returns:
        List: Version v.0.2 DSSF.
    """
    # DSSF 0.1 keys
    column_categorical_key = "column_categorical"
    column_datetime_key = "datetime_column"
    node_categorical_key = "categorical"
    node_datetime_key = "datetime"
    # DSSF 0.2 keys that are used to ensure backward compatibility. These keys
    # are not known by the user.
    not_categorical_key = "not_categorical"
    not_datetime_key = "not_datetime"

    # Keys that are in 0.1 and 0.2.
    optional_info_key = "optional_info"
    feature_type_key = "feature_type"
    table_key = "table"

    if legacy_dssf[-1][VERSION_KEY] != "0.1":
        raise ValueError("This function assumes that the input dssf has "
                         "version 0.1, but the given dssf has version "
                         f"{legacy_dssf[-1][VERSION_KEY]}")
    legacy_dssf_copy = copy.deepcopy(legacy_dssf)
    for comp_index, comp in enumerate(legacy_dssf[:-1]):
        optional_info: Dict[str, Dict] = comp.get(optional_info_key, None)
        if comp["type"] == table_key:
            if optional_info is None:
                continue
            feature_type_table = optional_info.get(feature_type_key, {})
            not_datetime_table = []
            not_categorical_table = []
            if column_categorical_key in optional_info:
                for k, v in optional_info[column_categorical_key].items():
                    if k in feature_type_table:
                        continue
                    if dssf_utils.str_bools_to_booleans(v):
                        feature_type_table[k] = "categorical"
                    else:
                        not_categorical_table.append(k)
            if column_datetime_key in optional_info:
                for k, v in optional_info[column_datetime_key].items():
                    if k in feature_type_table:
                        continue
                    if dssf_utils.str_bools_to_booleans(v):
                        feature_type_table[k] = "datetime"
                    else:
                        not_datetime_table.append(k)
            if feature_type_table != {}:
                legacy_dssf_copy[comp_index][optional_info_key][
                    feature_type_key] = feature_type_table
            if not_categorical_table != []:
                legacy_dssf_copy[comp_index][optional_info_key][
                    not_categorical_key] = not_categorical_table
            if not_datetime_table != []:
                legacy_dssf_copy[comp_index][optional_info_key][
                    not_datetime_key] = not_datetime_table
        else:
            comp_type = comp["type"]
            if comp_type == "datetime":
                legacy_dssf_copy[comp_index]["type"] = "str"
                if optional_info_key not in legacy_dssf_copy[comp_index]:
                    legacy_dssf_copy[comp_index][optional_info_key] = {}
                legacy_dssf_copy[comp_index][optional_info_key][
                    feature_type_key] = "datetime"
            if optional_info is None:
                continue
            feature_type = str(optional_info.get(feature_type_key, ""))
            if node_categorical_key in optional_info:
                node_cat = optional_info[node_categorical_key]
                if dssf_utils.str_bools_to_booleans(node_cat) \
                        and feature_type == "":
                    feature_type = "categorical"
                else:
                    legacy_dssf_copy[comp_index][optional_info_key][
                        not_categorical_key] = True
            if feature_type != "":
                legacy_dssf_copy[comp_index][optional_info_key][
                    feature_type_key] = feature_type

        legacy_dssf_copy[comp_index][optional_info_key].pop(
            column_categorical_key, None)
        legacy_dssf_copy[comp_index][optional_info_key].pop(
            column_datetime_key, None)
        legacy_dssf_copy[comp_index][optional_info_key].pop(
            node_categorical_key, None)
        legacy_dssf_copy[comp_index][optional_info_key].pop(
            node_datetime_key, None)
    legacy_dssf_copy[-1][VERSION_KEY] = "0.2"
    return legacy_dssf_copy


def v02_to_v03(legacy_dssf: List) -> List:
    """Convert a dssf, that is in the version v0.2, to the version v0.3.
    As there is only one new optional keyword, the old dssf should still work
    for the new version.

    Args:
        legacy_dssf (List): Version v.0.2 DSSF.

    Returns:
        List: Version v.0.3 DSSF.
    """
    if legacy_dssf[-1][VERSION_KEY] != "0.2":
        raise ValueError("This function assumes that the input dssf has "
                         "version 0.2, but the given dssf has version "
                         f"{legacy_dssf[-1][VERSION_KEY]}")
    # Check that the `datetime_format` key is not used in the old version as it
    # is newly introduced in version 0.3 and would not be valid before.
    for comp in legacy_dssf[:-1]:
        if ("optional_info" in comp and
                "datetime_format" in comp["optional_info"]):
            msg = ("The keyword `datetime_format` is only valid for dssf "
                   "version 0.3 and later, but it is used in this older dssf "
                   f"version in component {comp['name']}. Please convert your "
                   "dssf manually into version 0.3 if you want to use the new "
                   "key `datetime_format`.")
            raise DSSFErrors.DSSFKeywordNotValidInVersion(msg)

    legacy_dssf_copy = copy.deepcopy(legacy_dssf)
    legacy_dssf_copy[-1][VERSION_KEY] = "0.3"
    return legacy_dssf_copy


def convert_legacy_to_latest_stable_dssf(
        legacy_dssf: Union[List, Dict]) -> List[Dict[str, Any]]:
    """Try to convert the legacy DSSF format to the latest version and print
    it. If it is not in the legacy format, it will throw an error.

    Args:
        legacy_dssf (Dict[str, Dict[str, Any]]): DSSF in legacy format.

    Raises:
        DSSFFormatError: Raised if the type is not specified in the legacy
            version.
        DSSFVersionError: Raised if the DSSF is not in a known format.
    """
    if (isinstance(legacy_dssf, dict) and list(legacy_dssf.keys()) == ["{id}"]
            and isinstance(legacy_dssf["{id}"], dict)):
        dssf = legacy_dssf["{id}"]
        dssf_list: List[Dict[str, Any]] = []
        for component in dssf:
            dssf_dict_new: Dict[str, Any] = {}
            if ("node_name" in dssf[component] and
                    isinstance(dssf[component]["node_name"], str)):
                name = dssf[component]["node_name"]
                dssf_dict_new["name"] = name
            else:
                dssf_dict_new["name"] = "Unknown_" + component
                logging.warning(f"No name for node '{component}'. A "
                                "default name is used.")
            path = dssf[component]["path"]
            dssf_dict_new["path"] = path
            # Find the type.
            if "num" in component:
                node_type = "num"
            elif "str" in component:
                node_type = "str"
            elif "mixed" in component:
                node_type = "table"
            else:
                raise DSSFErrors.DSSFFormatError(
                    "Format is not valid -> not able to convert DSSF.")
            dssf_dict_new["type"] = node_type
            if node_type == "table":
                if "{id}" in dssf[component]:
                    sample_id_column = dssf[component]["{id}"]
                    dssf_dict_new["optional_info"] = {"sample_id_column":
                                                      sample_id_column}
            dssf_list.append(dssf_dict_new)
        dssf_list.append({"_version": "0.1"})
        return convert_legacy_to_latest_stable_dssf(dssf_list)
    elif isinstance(legacy_dssf, list) \
            and VERSION_KEY in legacy_dssf[-1]:
        if legacy_dssf[-1][VERSION_KEY] == "0.1":
            dssf_0_2 = v01_to_v02(legacy_dssf)
            return v02_to_v03(dssf_0_2)
        elif legacy_dssf[-1][VERSION_KEY] == "0.2":
            return v02_to_v03(legacy_dssf)
        elif legacy_dssf[-1][VERSION_KEY] == "0.3":
            raise DSSFErrors.DSSFVersionError(
                "DSSF already has newest version (0.3)")
        else:
            raise DSSFErrors.DSSFVersionError("Unknown DSSF format!")
    else:
        raise DSSFErrors.DSSFVersionError("Unknown DSSF format!")


if __name__ == "__main__":

    description = ("Convert legacy DSSF into version '0.1'.\nRead dssf file, "
                   "test the version: if it is version '0.1' -> do nothing; "
                   "elif it is not the legacy version -> throw error; else "
                   "rename the source, if it is called "
                   "'dataset_structure.json', to '*_legacy.json, convert it to"
                   " the version '0.1' and save it in the same directory to "
                   "'dataset_structure.json'.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--legacy_dssf_path", type=str,
                        help="Path to the legacy DSSF.")
    args = parser.parse_args()

    # Read dssf.
    if not os.path.isfile(args.legacy_dssf_path):
        raise FileNotFoundError("There is no file in "
                                f"'{args.legacy_dssf_path}'")
    try:
        with open(args.legacy_dssf_path) as f:
            legacy_dssf = json.load(f)
    except json.JSONDecodeError as e:
        raise DSSFErrors.DSSFFormatError(f"Could not read json: {e}")

    # Convert dssf.
    new_dssf = convert_legacy_to_latest_stable_dssf(legacy_dssf)

    # Find file to save new dssf.
    json_dir = os.path.dirname(args.legacy_dssf_path)
    if os.path.basename(args.legacy_dssf_path) == "dataset_structure.json":
        os.rename(args.legacy_dssf_path,
                  os.path.join(json_dir, "dataset_structure_legacy.json"))
    new_file = os.path.join(json_dir, "dataset_structure.json")

    # Ask user what to do, if there is another file with the same name.
    if os.path.isfile(new_file):
        answered = False
        while not answered:
            answer = input("There is already a `dataset_structure.json`. Do "
                           "you want to overwrite it? [Y/n]").lower()
            choice = (not answer or answer in ["y", "yes"])
            answered = choice
            if not choice:
                if answer not in ["n", "no"]:
                    print("Please respond with 'y' or 'n'.")
                else:
                    answered = True
        if not choice:
            print("Aborted! New DSSF not saved.")
            sys.exit()
    with open(new_file, "w") as f:
        json.dump(new_dssf, f, indent=4)
    print("DSSF successfully converted from 'legacy' to '0.2' and saved in "
          f"'{new_file}'!")
    print("Good bye!\n")
