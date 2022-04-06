# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Helper function for the dssf_and_structure code.
"""

from typing import List, Iterable, Generator, Any
import collections
import json
import os


DSSF_VERSION = "0.3"


def flatten(it_list: Iterable[Any]) -> Generator:
    """ Flatten list of lists.

    Args:
        it_list (Iterable): list

    Returns:
        Generator
    """
    for el in it_list:
        if isinstance(el, collections.Iterable) and not \
                isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def test_categorical(vector: List, threshold1: float = 0.1,
                     threshold2: int = 20) -> bool:
    """Use simple heuristic to test if vector is likely to be categorical.

    Args:
        vector (List): Node data.
        threshold1 (float, optional): Maximal percentage of unique values
            compared to all values to be categorical. Defaults to 0.1.
        threshold2 (int, optional): Maximal number of unique values to
            be categorical. Defaults to 20.

    Returns:
        bool: True if both thresholds are met and therefore is likely to be
            categorical.
    """
    # flatten input
    vector = list(flatten(vector))
    return (len(list(set(vector)))/float(len(vector)) < threshold1 and
            len(list(set(vector))) <= threshold2)


def str_bools_to_booleans(value: Any) -> bool:
    """Check for booleans saved as strings in the key 'categorical' in
    the optional_info and save them as booleans.

    Args:
        value (Any): Value to convert to bool.

    Returns:
        Dict: New optional info.
    """
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            raise ValueError(f"Unsupported value '{value}'")
    elif isinstance(value, bool):
        return value
    else:
        raise ValueError(f"Unsupported value '{value}'")


def create_dssf_template_file_for_tables(
        dataset_path: str, output_dir: str) -> None:
    """Create a dssf file at the location of the output_dir.

    Args:
        dataset_path (str): Path of the dataset which is described in the dssf.
        output_dir (str): Directory in which the dssf is saved.
    """
    name = os.path.splitext(os.path.basename(dataset_path))[0]
    dssf = [{"name": name, "path": dataset_path, "type": "table"},
            {"_version": DSSF_VERSION}]
    with open(os.path.join(output_dir, "dataset_structure.json"), "w") as f:
        json.dump(dssf, f, indent=4)
