# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This Feature Extractor performs an identity operation on purely numerical
datasets.
"""
from copy import deepcopy
import numpy as np
import os
from typing import Dict, List, Optional

from modulos_utils.metadata_handling import metadata_transferer as meta_trans
from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.module_interfaces import feature_extractor as fe_interface
from . import helpers as fe_helpers

DictOfArrays = Dict[str, np.ndarray]
DictOfMetadata = Dict[str, meta_prop.AllProperties]

DEFAULT_DATASET: DictOfArrays = {"node_name": np.array([])}


class IdentityFeatureExtractor(fe_interface.IFeatureExtractor):
    """Feature extractor that performs identity on purely numerical datasets.
    """
    def __init__(self):
        self._node_list: Optional[List[str]] = None
        self._metadata: Optional[DictOfMetadata] = None
        self._transformed_metadata: Optional[DictOfMetadata] = None
        self._weights_loaded: bool = False

    @staticmethod
    def initialize_new(config_choice_path: str) -> "IdentityFeatureExtractor":
        """Initialize a new feature extractor from a config choice
        file.

        Args:
            config_choice_path (str): Path to config choice file.

        Returns:
            IdentityFeatureExtractor: An initialized object of this
                class.
        """
        return IdentityFeatureExtractor()

    @staticmethod
    def initialize_from_weights(weights_folder: str) \
            -> "IdentityFeatureExtractor":
        """Load a feature extractor from weights. These weights are
        generalized weights meaning anything that is saved in the
        training phase and used in the prediction phase. (Therefore the term
        weights is not used in a strict sense as in the parameters of a neural
        network that are optimized during training.)

        Args:
            weights_folder (str): Path to folder containing weights.

        Raises:
            fe_helpers.IdentityConstructorError: Error if weights folder does
                not exist.

        Returns:
            fe_interface.IFeatureExtractor: An initialized instance of this
                class.
        """
        # Check whether weights path exits.
        if not os.path.isdir(weights_folder):
            raise fe_helpers.IdentityError(
                f"Directory {weights_folder} does not exist.")
        result_obj = IdentityFeatureExtractor()
        metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(weights_folder, "input_metadata.bin")
        )
        result_obj._metadata = metadata
        result_obj._node_list = list(metadata.keys())
        result_obj._weights_loaded = True
        return result_obj

    def _compute_transformed_metadata(self) -> None:
        """Compute transformed metadata.

        Raises:
            fe_helpers.IdentityStateError: Error if this function is called
                before the object has been trained.
        """
        if self._transformed_metadata is not None:
            return None
        if self._metadata is None or self._node_list is None:
            raise fe_helpers.IdentityError(
                "IdentityFeatureExtractor object has not been "
                "trained yet.")
        # Create the transformed metadata metadata.
        # Note: node type and node dim are copied from the original input file.
        self._transformed_metadata = \
            meta_trans.DatasetTransferer.from_dict(self._metadata).get_dict()
        return None

    def _transformed_generator(
            self, input_gen: d_obj.DatasetGenerator,
            check_input: bool) -> d_obj.DatasetGenerator:
        """Generator of transformed data.

        Args:
            input_gen (d_obj.DatasetGenerator): Input generator.
            check_input (bool): Whether or not to perform checks the input.

        Raises:
            fe_helpers.IdentityInputError: Error when sample ids are missing in
                batches of input generator.

        Returns:
            output_gen (d_obj.DatasetGenerator): Output generator.
        """
        for batch in input_gen:
            if dc.SAMPLE_ID_KEYWORD not in batch:
                raise fe_helpers.IdentityError(
                    "Input generator must be a generator over dictionaries, "
                    "where every dictionary must contain the sample ids with "
                    f"the key {dc.SAMPLE_ID_KEYWORD}.")
            sample_ids = batch[dc.SAMPLE_ID_KEYWORD]
            batch.pop(dc.SAMPLE_ID_KEYWORD)
            result_data_dict = self.transform(batch, check_input=check_input)
            result_data_dict[dc.SAMPLE_ID_KEYWORD] = sample_ids
            yield result_data_dict

    def save_weights(self, weights_folder: str) -> None:
        """Save feature extractor weights.

        Args:
            weights_folder (str): Path to folder where weights should be saved.

        Raises:
            fe_helpers.IdentitySaveWeightsError: Error if this function is
                called before the FE has been trained.
        """
        if self._metadata is None:
            raise fe_helpers.IdentityError(
                "Generalized weights of this feature extractor cannot be "
                "saved because the feature extractor has not been trained "
                "yet.")
        if not os.path.isdir(weights_folder):
            os.makedirs(weights_folder)
        meta_utils.MetadataDumper().write_all_nodes(
            self._metadata, os.path.join(weights_folder, "input_metadata.bin")
        )

    def fit(self, input_data: DictOfArrays, metadata: DictOfMetadata) \
            -> "IdentityFeatureExtractor":
        """Train a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            IFeatureExtractor: The class itself.
        """
        # Save metadata node list in member variables.
        self._metadata = deepcopy(metadata)
        self._node_list = list(self._metadata.keys())
        return self

    def fit_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) \
            -> "IdentityFeatureExtractor":
        """Train a feature extractor with a generator over batches as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of data. The keys of the dictionaries are
                the node names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            The class itself.
        """
        # The data is not used, because the fit function just populates the
        # object with information from the metadata.
        return self.fit(DEFAULT_DATASET, metadata)

    def fit_transform(
            self, input_data: DictOfArrays, metadata: DictOfMetadata) \
            -> DictOfArrays:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        # The data is not used, because the fit function just populates the
        # object with information from the metadata.
        self.fit(DEFAULT_DATASET, metadata)
        return self.transform(input_data)

    def fit_transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) \
            -> d_obj.DatasetGenerator:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of data. The keys of the dictionaries are
                the node names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            d_obj.DatasetGenerator: Transformed samples, batched as a
            generator.
        """
        # The data is not used, because the fit function just populates the
        # object with information from the metadata.
        self.fit(DEFAULT_DATASET, metadata)
        return self.transform_generator(input_data)

    def transform(self, input_data: DictOfArrays, check_input: bool = False) \
            -> DictOfArrays:
        """Apply a trained feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            check_input (bool): Whether or not to check whether input data
                type/format.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        if self._metadata is None or self._node_list is None:
            raise fe_helpers.IdentityError(
                "IdentityFeatureExtractor object has not been "
                "trained yet.")
        if check_input:
            fe_helpers.check_input_batch(input_data, self._metadata,
                                         self._node_list)
        self._compute_transformed_metadata()
        return input_data

    def transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            check_input: bool = False) -> d_obj.DatasetGenerator:
        """Apply a trained feature extractor with a generator as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                where the keys are the node names and the values are the
                batched node data as lists.
            check_input (bool = False): Whether or not to perform checks on
                the input.

        Returns:
            d_obj.DatasetGenerator: Transformed samples, batched as a
            generator.
        """
        self._compute_transformed_metadata()
        return self._transformed_generator(input_data, check_input=check_input)

    def get_transformed_metadata(self) -> DictOfMetadata:
        """Get transformed metadata after training the Feature Extractor.

        Returns:
            DictOfMetadata: Transformed metadata.

        Raises:
            fe_helpers.IdentityStateError: Error if this function is called
                before transformed metadata has been computed.
        """
        if self._transformed_metadata is None:
            raise fe_helpers.IdentityError(
                "Transformed metadata has not been computed yet. Run one of "
                "the following function: 'transform', 'transform_generator', "
                "'fit_transform', 'fit_transform_generator'.")
        return self._transformed_metadata
