# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import math
import numpy as np
from typing import Dict, Generator, List

from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.data_handling import data_utils

DictOfArrays = Dict[str, np.ndarray]
DictOfMetadata = Dict[str, meta_prop.AllProperties]
GenOfDicts = Generator[dict, None, None]


class PCAError(Exception):
    """Error in the PCA FE.
    """
    pass


def check_input_batch(input_data: DictOfArrays, metadata: DictOfMetadata,
                      node_list: List[str]) -> None:
    """Check input types and assumptions.

    Args:
        input_data (DictOfArrays): Dictionary of arrays.
        metadata (DictOfMetadata): Metadata dict that was saved in the weights.
        node_list(List[str]): List of required node names.

    Raises:
        PCAError: Error, if the input data type is wrong.

    """
    # Check input type.
    if not isinstance(input_data, dict):
        raise PCAError(
            "Input to PCA Feature Extractor must be a dictionary."
        )
    for key in input_data.keys():
        if not isinstance(key, str):
            raise PCAError(
                "Input to PCA Feature Extractor must be a dictionary "
                "with string type keys."
            )
    for required_node in node_list:
        if required_node not in input_data:
            raise PCAError(
                f"Node {required_node} (that was present during training) is "
                "missing in the input."
            )
    for key, value in input_data.items():
        if key not in metadata:
            continue
        if not isinstance(value, np.ndarray):
            raise PCAError(
                "Input to PCA Feature Extractor must be a dictionary "
                "with string type keys and numpy array values. However node "
                f"{key} is of type {type(value)}."
            )
        numpy_dtype = np.array(value).dtype
        if not data_utils.is_modulos_numerical(numpy_dtype):
            raise PCAError(
                f"The data type of node {value} is wrong. The PCA "
                "feature extractor assumes that all nodes are numerical.")


def get_all_samples_from_generator(
        input_samples: d_obj.DatasetGenerator) -> dict:
    """Iterate over all samples of generator and create a dictionary containing
    all nodes and all samples for each node.

    Args:
        input_samples (d_obj.DatasetGenerator): A generator of
                dictionaries where the
                keys are the node names and the values are the batched node
                data as lists.

    Returns:
        dict: Dictionary containing the batch size with the key "batch_size"
            and the data (all samples) with the key "all_samples".
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY, THE T-TEST FEATURE SELECTION FAMILY AND
    # ImgIdentityCatIntEnc.

    # Note that we don't perform  checks to raise explicit exceptions in this
    # function because this whole function will be removed in BAS-603 and the
    # checks are performed after this functions call.
    batch_size = 0
    all_samples: DictOfArrays = {}
    for batch in input_samples:
        for key, values in batch.items():
            if key in all_samples:
                all_samples[key] = np.vstack((all_samples[key], values))
            else:
                all_samples[key] = values
                batch_size = len(values)
    return {"all_samples": all_samples, "batch_size": batch_size}


def get_generator_over_samples(all_samples: DictOfArrays,
                               batch_size: int) -> GenOfDicts:
    """Return a generator over batches of samples, given all the data.

    Args:
        all_samples (DictOfArrays): A dictionary with the node names as keys
            and the node data for all samples as values.
        batch_size (int): Batch size.

    Returns:
        Array: Generator over batches of samples.
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY, ImgIdentityCatIntEnc AND PCA.
    n_samples_total = len(list(all_samples.values())[0])
    n_iterations = math.ceil(n_samples_total / batch_size)
    for i in range(n_iterations):
        sample_dict = {}
        for node_name in all_samples.keys():
            sample_dict[node_name] = all_samples[node_name][
                i * batch_size: (i + 1) * batch_size]
        yield sample_dict
