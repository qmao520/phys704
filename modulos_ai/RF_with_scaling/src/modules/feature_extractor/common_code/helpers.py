# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# THIS WHOLE FILE IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE T TEST
# FEATURE SELECTION FAMILY
"""This file contains helper functions for the t test feature selection
common code.
"""
from abc import ABC, abstractmethod
import collections
import math
import numpy as np
import pandas as pd
from typing import Generator, Dict, Any, Tuple, List

from . import encode_categories as ec
from . import scale_numbers as sn
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.data_handling import data_utils


DictOfArrays = Dict[str, np.ndarray]
GenOfDicts = Generator[dict, None, None]
DictOfMetadata = Dict[str, meta_prop.AllProperties]


class TTEstFeatureSelectionError(Exception):
    """Errors in the TTEstFeatureSelection Feature Extractor class.
    """
    pass


def categorize_regression_label_generator(
        label_generator: GenOfDicts, n_bins: int, min_val: float,
        max_val: float, label_name: str) -> GenOfDicts:
    """Categorize numerical labels by binning them. This function works with
    a generator.

    Args:
        label_generator (GenOfDicts): Labels.
        n_bins (int): Number of bins.
        min_val (float): Minimum label value.
        max_val float): Maximum label value.
        label_name (str): Node name of the labels.

    Returns:
        GenOfDicts: Generator over binned labels.
    """
    for batch in label_generator:
        yield categorize_regression_label(
            batch, n_bins, min_val, max_val, label_name)


def categorize_regression_label(
        label_data: DictOfArrays, n_bins: int, min_val: float,
        max_val: float, label_name: str) -> DictOfArrays:
    """Categorize numerical labels by binning them. This function works with
    a generator.

    Args:
        label_data (DictOfArrays): Labels.
        n_bins (int): Number of bins.
        min (float): Minimum label value.
        max float): Maximum label value.
        label_name (str): Node name of the labels.

    Returns:
        DictOfArrays: Binned labels.
    """
    # Arbitrary max val, if min==max. This is just to prevent errors. The
    # binning does not make any sense in this case. But in fact applying
    # t-test feature selection does not make sense to apply in this case.
    if min_val == max_val:
        if min_val == 0:
            max_val = 1.0
        else:
            max_val = 2*min_val
    bins = np.linspace(min_val, max_val, n_bins)
    categories = np.arange(n_bins - 1)
    df = pd.DataFrame({
        label_name: np.array(label_data[label_name]).reshape(-1).tolist()})
    categorized_labels = pd.cut(
        df[label_name], bins=bins, labels=categories,
        include_lowest=True)
    return {label_name: np.array(categorized_labels.values)}


def get_n_samples_classwise_from_generator(
        label_generator: GenOfDicts,
        unique_label_categories: np.ndarray,
        label_name: str) -> Dict[str, int]:
    """Count the number of samples that lie within each label class.

    Args:
        label_generator (GenOfDicts): Generator over labels.
        unique_label_categories (np.ndarray): np.ndarray of unique categories
            of the labels.
        label_name (str): Node name of the labels.

    Returns:
        Dict[str, int]: Dictionary with classes as keys and number of samples
            as values.
    """
    n_samples_classwise: Dict[str, int] = collections.defaultdict(int)
    for label_batch in label_generator:
        labels_array = np.array(label_batch[label_name])
        for c in unique_label_categories:
            n_c_batch = np.sum(labels_array == c)
            n_samples_classwise[c] += n_c_batch
    return n_samples_classwise


def get_n_samples_classwise(
        label_data: DictOfArrays,
        unique_label_categories: np.ndarray,
        label_name: str) -> Dict[str, int]:
    """Count the number of samples that lie within each label class.

    Args:
        label_data (DictOfArrays): Label data.
        unique_label_categories (np.ndarray): np.ndarray of unique categories
            of the labels.
        label_name (str): Node name of the labels.

    Returns:
        Dict[str, int]: Dictionary with classes as keys and number of samples
            as values.
    """
    label_generator = get_generator_over_samples(
        label_data, batch_size=len(list(label_data.values())[0]))
    return get_n_samples_classwise_from_generator(
        label_generator, unique_label_categories, label_name)


def populate_mean_dicts(
        inclass_means: Dict[str, Dict[str, float]],
        overall_means: Dict[str, float], input_batch: DictOfArrays,
        label_batch: DictOfArrays, unique_label_categories: np.ndarray,
        n_samples: int, n_samples_classwise: Dict[str, int],
        new_node_names: List, label_name: str) -> None:
    """Helper function to populate in-class mean and overall mean dictionaries.

    Args:
        inclass_means (Dict[str, Dict[str, float]]): Dictionary where the
            keys are the node names and the values are dictionaries with the
            classes as keys and the means as values.
        overall_means (Dict[str, float]): Dictionary where the keys are the
            node names and the values the means of these nodes.
        input_batch (DictOfArrays): Batch of input data as a dictionary.
        label_batch (DictOfArrays): Batch of label data as a dictionary.
        unique_label_categories (np.ndarray): np.ndarray of unique categories
            of the labels.
        n_samples (int): Total number of samples in the dataset.
        n_samples_classwise (Dict[str, int]): Number of samples for each class.
        new_node_names (List): Node names after scaling and encoding
            columns of the original table (i.e. after the table prep part.)
    """
    labels_array = np.array(label_batch[label_name])
    sample_masks = {c: labels_array == c
                    for c in unique_label_categories}
    sample_indices = {c: np.where(sample_masks[c])[0]
                      for c in sample_masks.keys()}
    for ni, nn in enumerate(new_node_names):
        node_array = np.array(input_batch[nn])
        overall_means[nn] += float(np.sum(node_array) / n_samples)
        for c in unique_label_categories:
            if n_samples_classwise[c] == 0:
                inclass_means[c][nn] += 0
            else:
                inclass_means[c][nn] += \
                    np.sum(node_array[sample_indices[c]]) / \
                    float(n_samples_classwise[c])
    return None


def get_means_from_generator(
        new_node_names: List, unique_label_categories: np.ndarray,
        input_generator: GenOfDicts, label_generator: GenOfDicts,
        n_samples: int, n_samples_classwise: Dict[str, int], label_name: str) \
        -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Compute mean dictionaries (overall means and class-wise means) while
    iterating over generators of input and labels.

    Args:
        new_node_names (List): Node names after table prep
            transformation.
        unique_label_categories (np.ndarray): Unique categories of the labels.
        input_generator (GenOfDicts): Input data generator.
        label_generator (GenOfDicts): Label data generator.
        n_samples (int): Total number of samples.
        n_samples_classwise (Dict[str, int]): Number of samples for each class.
        label_name (str): Node name of the labels.

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, float]]: Class-wise
            means and overall means.
    """
    inclass_means: Dict[str, Dict[str, float]] = {}
    for c in unique_label_categories:
        inclass_means[c] = collections.defaultdict(float)
    overall_means: Dict[str, float] = collections.defaultdict(float)
    for i_batch, l_batch in zip(input_generator, label_generator):
        populate_mean_dicts(
            inclass_means, overall_means, i_batch, l_batch,
            unique_label_categories, n_samples, n_samples_classwise,
            new_node_names, label_name)
    return inclass_means, overall_means


def get_means(
        new_node_names: List, unique_label_categories: np.ndarray,
        input_data: DictOfArrays,
        label_data: DictOfArrays, n_samples: int,
        n_samples_classwise: Dict[str, int], label_name: str) \
        -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Compute mean dictionaries (overall means and class-wise means) without
    generators.

    Args:
        new_node_names (List): Node names after table prep
            transformation.
        unique_label_categories (np.ndarray): Unique categories of the labels.
        input_generator (DictOfArrays): Input data.
        label_generator (DictOfArrays): Label data.
        n_samples (int): Total number of samples.
        n_samples_classwise (Dict[str, int]): Number of samples for each class.
        label_name (str): Node name of the labels.

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, float]]: Class-wise
            means and overall means.
    """
    n_samples = len(list(input_data.values())[0])
    input_generator = get_generator_over_samples(input_data, n_samples)
    label_generator = get_generator_over_samples(label_data, n_samples)
    return get_means_from_generator(
        new_node_names, unique_label_categories, input_generator,
        label_generator, n_samples, n_samples_classwise, label_name)


def populate_sum_of_squares_classwise(
        sum_of_squares_classwise: Dict[str, Dict[str, float]],
        input_batch: DictOfArrays, label_batch: DictOfArrays,
        unique_label_categories: np.ndarray, new_node_names: List,
        inclass_means: Dict[str, Dict[str, float]], label_name: str) -> None:
    """Populate dictionary of class-wise sum_of_squares.

    Args:
        sum_of_squares_classwise (Dict[str, Dict[str, float]]): Dictionary
            to populate. The keys are the node names and the values will be
            dictionaries with the classes as keys.
        input_batch (DictOfArrays): Input data batch.
        label_batch (DictOfArrays): Label data batch.
        unique_label_categories (np.ndarray): Unique categories of the labels.
        new_node_names (List): Node names after table prep
            transformation.
        inclass_means (Dict[str, Dict[str, float]]): In-class mean dictionary.
        label_name (str): Node name of the labels.
    """
    labels_array = np.array(label_batch[label_name])
    sample_masks = {c: labels_array == c
                    for c in unique_label_categories}
    sample_indices = {c: np.where(sample_masks[c])[0]
                      for c in sample_masks.keys()}
    for ni, nn in enumerate(new_node_names):
        for c in unique_label_categories:
            sum_of_squares_classwise[nn][c] += np.sum(
                [(input_batch[nn][s] - inclass_means[c][nn])**2
                    for s in sample_indices[c]])
    return None


def average_sum_of_squares_classwise(
        new_node_names: List,
        sum_of_squares_classwise: Dict[str, Dict[str, float]],
        n_classes: int,
        n_samples: int) -> List:
    """Average class-wise sum-of-squares.

    Args:
        new_node_names (List): Node names after table prep
            transformation.
        sum_of_squares_classwise (Dict[str, Dict[str, float]]): Sum of
            squares for each node and class.
        n_classes (int): Number of classes of the labels.
        n_samples (int): Total number of samples in the dataset.

    Returns:
        List: For each node, the sum-of-squared that was averaged over
            the classes.
    """
    sum_of_squared_averaged: List = []
    for nnn in new_node_names:
        sum_of_squared_averaged.append(
            float(sum(sum_of_squares_classwise[nnn].values())) / n_classes
            / n_samples)
    return sum_of_squared_averaged


def get_sum_of_squares_classwise_from_generator(
        new_node_names: List, input_generator: GenOfDicts,
        label_generator: GenOfDicts, unique_label_categories: np.ndarray,
        inclass_means: Dict[str, Dict[str, float]],
        n_classes: int, n_samples: int, label_name: str) -> List:
    """Get the sum-of-squares for each node and class as a nested dictionary,
    while iterating over generators of input and labels.

    Args:
        new_node_names (List): Node names after table prep
            transformation.
        input_generator (GenOfDicts): Input data generator.
        label_generator (GenOfDicts): Label data generator.
        unique_label_categories (np.ndarray): Unique categories of the labels.
        inclass_means (Dict[str, Dict[str, float]]): Class-wise means for each
            node as a nested dictionary.
        n_classes (int): Number of label classes.
        n_samples (int): Total number of samples in the dataset.
        label_name (str): Node name of the labels.

    Returns:
        List: For each node, the sum-of-squared that was averaged over
            the classes.
    """
    sum_of_squares_classwise: Dict[str, Dict[str, float]] = {}
    for nn in new_node_names:
        sum_of_squares_classwise[nn] = collections.defaultdict(float)
    for input_batch, label_batch in zip(input_generator, label_generator):
        populate_sum_of_squares_classwise(
            sum_of_squares_classwise, input_batch, label_batch,
            unique_label_categories, new_node_names, inclass_means,
            label_name)
    return average_sum_of_squares_classwise(
        new_node_names, sum_of_squares_classwise, n_classes, n_samples)


def get_sum_of_squares_classwise(
        new_node_names: List, input_data: DictOfArrays,
        label_data: DictOfArrays, unique_label_categories: np.ndarray,
        inclass_means: Dict[str, Dict[str, float]],
        n_classes: int, n_samples: int, label_name: str) -> List:
    """Get the sum-of-squares for each node and class as a nested dictionary
    (without generators).

    Args:
        new_node_names (List): Node names after table prep
            transformation.
        input_data (DictOfArrays): Input data.
        label_data (DictOfArrays): Label data.
        unique_label_categories (np.ndarray): Unique categories of the labels.
        inclass_means (Dict[str, Dict[str, float]]): Class-wise means for each
            node as a nested dictionary.
        n_classes (int): Number of label classes.
        n_samples (int): Total number of samples in the dataset.
        label_name (str): Node name of the labels.

    Returns:
        List: For each node, the sum-of-squared that was averaged over
            the classes.
    """
    n_samples = len(list(input_data.values())[0])
    input_generator = get_generator_over_samples(input_data, n_samples)
    label_generator = get_generator_over_samples(label_data, n_samples)
    return get_sum_of_squares_classwise_from_generator(
        new_node_names, input_generator, label_generator,
        unique_label_categories, inclass_means, n_classes, n_samples,
        label_name)


def slice_nodes_from_generator(
        input_generator: GenOfDicts, node_names: np.ndarray) -> GenOfDicts:
    """Slice a generator by selecting nodes, while keeping sample ids.

    Args:
        input_generator (GenOfDicts): Input generator with sample ids.
        node_names (np.ndarray): Node names to keep.

    Returns:
        GenOfDicts: Sliced generator.
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY, THE T-TEST FEATURE SELECTION FAMILY AND
    # ImgIdentityCatIntEnc.
    for batch in input_generator:
        new_batch = {}
        for n in node_names:
            new_batch[n] = batch[n]
        new_batch[dc.SAMPLE_ID_KEYWORD] = batch[dc.SAMPLE_ID_KEYWORD]
        yield new_batch


def check_data_types(input_object: Any) -> None:
    """Check the input data type and raise Exceptions if the types are not
    supported. We check that the input is a dictionary with strings as keys and
    only the following python native data types for the values: str, int,
    float, list. Furthermore we check that the number of samples is consistent
    across all nodes.
    """
    if not isinstance(input_object, dict):
        raise TTEstFeatureSelectionError(
            "Input to the feature extractor must be a dictionary.")
    if input_object == {}:
        raise TTEstFeatureSelectionError(
            "Input to the feature extractor cannot be empty.")
    n_samples = set()
    for key, value in input_object.items():
        if not isinstance(key, str):
            raise TTEstFeatureSelectionError(
                "The keys of the feature extractor's input dictionary must "
                "be python strings.")
        if not (isinstance(value, np.ndarray)
                and (data_utils.is_modulos_numerical(value.dtype)
                     or data_utils.is_modulos_string(value.dtype))):
            raise TTEstFeatureSelectionError(
                "The type of the values of the feature extractor's input "
                "dictionary must be numpy arrays of strings, floats or ints.")
        n_samples.add(len(value))
    if len(list(n_samples)) != 1:
        raise TTEstFeatureSelectionError(
            "The number of samples in the feature extractor input must be the "
            "same for all nodes.")
    if list(n_samples)[0] == 0:
        raise TTEstFeatureSelectionError(
            "The number of samples in the feature extractor input must be at "
            "least 1.")


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
                               batch_size: int) -> d_obj.DatasetGenerator:
    """Return a generator over batches of samples, given all the data.

    Args:
        all_samples (DictOfArrays): A dictionary with the node names as keys
            and the node data for all samples as values.
        batch_size (int): Batch size.

    Returns:
        Array: Generator over batches of samples.
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY, THE T-TEST FEATURE SELECTION FAMILY AND
    # ImgIdentityCatIntEnc.
    n_samples_total = len(list(all_samples.values())[0])
    n_iterations = math.ceil(n_samples_total / batch_size)
    for i in range(n_iterations):
        sample_dict = {}
        for node_name in all_samples.keys():
            sample_dict[node_name] = all_samples[node_name][
                i * batch_size: (i + 1) * batch_size]
        yield sample_dict


class ColumnTransformatorError(Exception):
    """Errors for ColumnTransformators.
    """
    # THIS FUNCTION IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.
    pass


class ColumnTransformator(ABC):
    # THIS CLASS IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.

    @abstractmethod
    def train_transformator(
            self, node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator.

        Args:
            node_data (np.ndarray): Column data.
            metadata (DictOfMetadata): Metadata.
        """
        pass

    @abstractmethod
    def apply_trained_transformator(self, column: np.ndarray) -> np.ndarray:
        """Apply trained transformator.

        Args:
            column (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        pass

    @abstractmethod
    def get_new_node_names(self) -> List:
        """Get node names of transformation result.

        Returns:
            List: List of strings.
        """
        pass


class ColumnTransformatorCategorical(ColumnTransformator):
    # THIS CLASS IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.

    def __init__(
            self,
            node_name: str,
            transformator_dictionary: Dict[str, ec.Encoder],
            transformation_type: ec.CategoryEncoderTypes) -> None:
        """Initialize object.

        Args:
            node_name (str): Node name of the column.
            transformator_dictionary (Dict[str, ec.Encoder]): Dictionary
                containing encoder to train.
            transformation_type (ec.CategoryEncoderTypes): Encoder type.
        """
        self._node_name: str = node_name
        self._transformation_type: ec.CategoryEncoderTypes = \
            transformation_type
        self._new_dimension: int = -1
        self._transformator_dictionary = transformator_dictionary

    def train_transformator(
            self,
            node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator.

        Args:
            node_data (np.ndarray): Column data.
            metadata (DictOfMetadata): Metadata.
        """
        # Get unique values.
        unique_values = np.array(metadata[
            self._node_name].upload_unique_values.get())

        encoder = ec.CategoryEncoderPicker[self._transformation_type]()
        encoder.fit(unique_values)
        self._transformator_dictionary[self._node_name] = encoder

    def apply_trained_transformator(
            self,
            column: np.ndarray) -> np.ndarray:
        """Apply trained transformator.

        Args:
            column (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        # Make sure the column consists of scalars, i.e. remove nested
        # dimensions.
        colunn_scalars = np.array(column).reshape(-1)
        transformed_column = \
            self._transformator_dictionary[self._node_name].transform(
                colunn_scalars)
        if len(transformed_column.shape) == 2:
            self._new_dimension = transformed_column.shape[1]
        else:
            self._new_dimension = 0
        return transformed_column

    def get_new_node_names(self) -> List:
        """Get node names of transformation result.

        Returns:
            List: List of strings.
        """
        if self._new_dimension == 0:
            return [self._node_name]
        elif self._new_dimension > 0:
            return [self._node_name + "_{}".format(i) for i in
                    range(self._new_dimension)]
        else:
            raise ColumnTransformatorError(
                "ColumnTransformator has not been applied yet!")


class ColumnTransformatorNumerical(ColumnTransformator):
    # THIS CLASS IS A COPY PASTE AND OCCURS IN ALL THE HELPERS OF THE TABLE
    # PREP FAMILY AND THE T-TEST FEATURE SELECTION FAMILY.

    def __init__(
            self,
            node_name: str,
            transformator_dictionary: Dict[str, sn.StandardScaler],
            transformation_type: sn.NumberScalingTypes) -> None:
        """Initialize object.

        Args:
            node_name (str): Node name of the column.
            transformator_dictionary: Dict[str, sn.StandardScaler]: Dictionary
                containing scaler object to train.
            transformation_type (ec.CategoryEncoderTypes): Encoder type.
        """
        self._node_name: str = node_name
        self._transformation_type: sn.NumberScalingTypes = \
            transformation_type
        self._transformator_dictionary = transformator_dictionary

    def train_transformator_online(
            self,
            node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator in batches. (successively updating scaler
        weights).

        Args:
            node_data (np.ndarray): Column data.
            node_name (str): Node name.
            metadata (DictOfMetadata): Metadata.
        """
        if self._node_name not in self._transformator_dictionary:
            scaler = sn.NumberScalingPicker[self._transformation_type]()
            self._transformator_dictionary[self._node_name] = scaler
        self._transformator_dictionary[self._node_name].partial_fit(node_data)

    def train_transformator(
            self,
            node_data: np.ndarray, metadata: DictOfMetadata) -> None:
        """Train transformator in one run.

        Args:
            node_data (np.ndarray): Column data.
            metadata (DictOfMetadata): Metadata.
        """
        if self._node_name not in self._transformator_dictionary:
            scaler = sn.NumberScalingPicker[self._transformation_type]()
            self._transformator_dictionary[self._node_name] = scaler
        self._transformator_dictionary[self._node_name].fit(node_data)

    def apply_trained_transformator(self, column: np.ndarray) -> np.ndarray:
        """Apply trained transformator.

        Args:
            column (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Transformed data.
        """
        return self._transformator_dictionary[self._node_name].transform(
            column)

    def get_new_node_names(self) -> List:
        """Get node names of transformation result.

        Returns:
            List: List of strings.
        """
        return [self._node_name]
