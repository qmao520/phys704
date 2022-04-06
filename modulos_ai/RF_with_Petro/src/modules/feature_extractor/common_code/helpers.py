# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import numpy as np
from typing import Dict, List

from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.data_handling import data_utils

DictOfArrays = Dict[str, np.ndarray]
DictOfMetadata = Dict[str, meta_prop.AllProperties]


class IdentityError(Exception):
    """Error in the Identity FE.
    """
    pass


def check_input_batch(input_data: DictOfArrays, metadata: DictOfMetadata,
                      node_list: List) -> None:
    """Check input types and assumptions.

    Args:
        input_data (DictOfArrays): Dictionary of lists.
        metadata (DictOfMetadata): Metadata dict that was saved in the weights.
        node_list(List): List of required node names.

    Raises:
        IdentityInputError: Error, if the input data type is wrong.

    """
    # Check input type.
    if not isinstance(input_data, dict):
        raise IdentityError(
            "Input to Identity Feature Extractor must be a dictionary."
        )
    for key in input_data.keys():
        if not isinstance(key, str):
            raise IdentityError(
                "Input to Identity Feature Extractor must be a dictionary "
                "with string type keys."
            )
    for required_node in node_list:
        if required_node not in input_data:
            raise IdentityError(
                f"Node {required_node} (that was present during training) is "
                "missing in the input."
            )
    for key, value in input_data.items():
        if key not in metadata:
            continue
        if not isinstance(value, np.ndarray):
            raise IdentityError(
                "Input to Identity Feature Extractor must be a dictionary "
                "with string type keys and numpy array values. However node "
                f"{key} is of type {type(value)}."
            )
        numpy_dtype = np.array(value).dtype
        if not data_utils.is_modulos_numerical(numpy_dtype):
            raise IdentityError(
                f"The data type of node {key} is wrong. The identity "
                "feature extractor assumes that all nodes are numerical.")
