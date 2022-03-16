# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This Feature Extractor performs an PCA operation on purely numerical
datasets.
"""
from copy import deepcopy
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Generator, Optional

from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.module_interfaces import feature_extractor as fe_interface
from . import helpers as fe_helpers
from . import scale_numbers as sn
from . import pca_decomposition as pca_dec

DictOfArrays = Dict[str, np.ndarray]
GenOfDicts = Generator[dict, None, None]
DictOfMetadata = Dict[str, meta_prop.AllProperties]


class PCAFeatureExtractor(fe_interface.IFeatureExtractor):
    """Feature extractor that performs PCA on datasets consisting purely
    of one-dimensional numerical nodes.
    """
    _MIN_COMPONENTS = 3

    def __init__(self):
        self._node_list: Optional[List[str]] = None
        self._metadata: Optional[DictOfMetadata] = None
        self._transformed_metadata: Optional[DictOfMetadata] = None
        self._reduction: Optional[float] = None
        self._scalers: Optional[dict] = None
        self._pca: Optional[pca_dec.PCA_FE] = None
        self._weights_loaded: bool = False

    @staticmethod
    def initialize_new(config_choice_path: str) -> "PCAFeatureExtractor":
        """Initialize a new feature extractor from a config choice
        file.

        Args:
            config_choice_path (str): Path to config choice file.

        Returns:
            PCAFeatureExtractor: An initialized object of this
                class.
        """
        return_obj = PCAFeatureExtractor()
        # Load the config file.
        with open(config_choice_path) as json_data:
            config_choice = json.load(json_data)
        return_obj._reduction = float(config_choice["reduction"])
        return return_obj

    @staticmethod
    def initialize_from_weights(weights_folder: str) \
            -> "PCAFeatureExtractor":
        """Load a feature extractor from weights. These weights are
        generalized weights meaning anything that is saved in the
        training phase and used in the prediction phase. (Therefore the term
        weights is not used in a strict sense as in the parameters of a neural
        network that are optimized during training.)

        Args:
            weights_folder (str): Path to folder containing weights.

        Raises:
            fe_helpers.PCAConstructorError: Error if weights folder does
                not exist.

        Returns:
            fe_interface.IFeatureExtractor: An initialized instance of this
                class.
        """
        # Check whether weights path exits.
        if not os.path.isdir(weights_folder):
            raise fe_helpers.PCAError(
                f"Directory {weights_folder} does not exist.")
        result_obj = PCAFeatureExtractor()
        metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(weights_folder, "input_metadata.bin")
        )
        result_obj._pca = joblib.load(
            os.path.join(weights_folder, "pca.bin"))
        result_obj._scalers = joblib.load(
            os.path.join(weights_folder, "scalers.bin"))
        result_obj._metadata = metadata
        result_obj._reduction = joblib.load(
            os.path.join(weights_folder, "reduction.bin"))
        result_obj._node_list = list(metadata.keys())
        result_obj._weights_loaded = True
        return result_obj

    def save_weights(self, weights_folder: str) -> None:
        """Save feature extractor weights.

        Args:
            weights_folder (str): Path to folder where weights should be saved.
        """
        if self._metadata is None or self._reduction is None \
                or self._node_list is None:
            raise fe_helpers.PCAError(
                "Generalized weights of this feature extractor cannot be "
                "saved because the feature extractor has not been trained "
                "yet.")
        if not os.path.isdir(weights_folder):
            os.makedirs(weights_folder)
        meta_utils.MetadataDumper().write_all_nodes(
            self._metadata, os.path.join(weights_folder, "input_metadata.bin")
        )
        joblib.dump(self._scalers, os.path.join(weights_folder, "scalers.bin"))
        joblib.dump(self._pca, os.path.join(weights_folder, "pca.bin"))
        joblib.dump(self._reduction,
                    os.path.join(weights_folder, "reduction.bin"))

    def fit(self, input_data: DictOfArrays, metadata: DictOfMetadata) \
            -> "PCAFeatureExtractor":
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

        # Loop over all nodes and scale them.
        scaled_data: Dict[str, np.ndarray] = {}
        self._scalers = {}
        for node_name in self._node_list:
            node_data = input_data[node_name]
            scaler = sn.StandardScaler()
            scaler.fit(node_data)
            scaled_data[node_name] = scaler.transform(node_data)
            self._scalers[node_name] = scaler
        if self._reduction is None:
            raise fe_helpers.PCAError("PCA not correctly initialized!")
        if len(self._node_list) <= self._MIN_COMPONENTS:
            raise fe_helpers.PCAError(
                f"The dataset needs atleast {self._MIN_COMPONENTS} nodes! "
                f"This one has only {len(self._node_list)}!"
            )
        n_components = int(len(self._node_list) * self._reduction)
        n_components = (n_components if n_components >= self._MIN_COMPONENTS
                        else self._MIN_COMPONENTS)
        self._pca = pca_dec.PCA_FE(n_components=n_components)
        self._pca.fit(scaled_data)
        return self

    def fit_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) -> "PCAFeatureExtractor":
        """Train a feature extractor with a generator over batches as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of data. The keys of the dictionaries are
                the node names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            The class itself.
        """
        all_samples = fe_helpers.get_all_samples_from_generator(
            input_data)["all_samples"]
        return self.fit(all_samples, metadata)

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
                numpy arrays.
        """
        self.fit(input_data, metadata)
        return self.transform(input_data)

    def fit_transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) -> GenOfDicts:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of data. The keys of the dictionaries are
                the node names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            GenOfDicts: Transformed samples, batched as a generator.
        """
        unfolded_generator = fe_helpers.get_all_samples_from_generator(
            input_data)
        all_samples = unfolded_generator["all_samples"]
        batch_size = unfolded_generator["batch_size"]

        transformed_data = self.fit_transform(all_samples, metadata)
        transformed_data[dc.SAMPLE_ID_KEYWORD] = \
            all_samples[dc.SAMPLE_ID_KEYWORD]

        # Convert the output back to a generator with the same batch size to
        # mock the generator case (hack to be removed in BAS-603)
        return fe_helpers.get_generator_over_samples(transformed_data,
                                                     batch_size)

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
                numpy arrays.
        """
        if (self._scalers is None or self._pca is None or
                self._metadata is None or self._node_list is None):
            raise fe_helpers.PCAError(
                "PCAFeatureExtractor object has not been "
                "trained yet.")
        if check_input:
            fe_helpers.check_input_batch(input_data, self._metadata,
                                         self._node_list)
        self._compute_transformed_metadata()

        scaled_data: Dict[str, np.ndarray] = {}
        for node_name in self._node_list:
            node_data = input_data[node_name]
            scaled_data[node_name] = self._scalers[node_name].transform(
                node_data)
        transformed_data = self._pca.transform(scaled_data)
        return transformed_data

    def transform_generator(self, input_data: d_obj.DatasetGenerator,
                            check_input: bool = False) -> GenOfDicts:
        """Apply a trained feature extractor with a generator as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                where the keys are the node names and the values are the
                batched node data as numpy arrays.
            check_input (bool = False): Whether or not to perform checks on
                the input.

        Returns:
            GenOfDicts: Transformed samples, batched as a generator.
        """
        self._compute_transformed_metadata()
        return self._get_transformed_generator(
            input_data, check_input=check_input)

    def get_transformed_metadata(self) -> DictOfMetadata:
        """Get transformed metadata after training the Feature Extractor.

        Returns:
            DictOfMetadata: Transformed metadata.

        Raises:
            fe_helpers.PCAStateError: Error if this function is called
                before transformed metadata has been computed.
        """
        if self._transformed_metadata is None:
            raise fe_helpers.PCAError(
                "Transformed metadata has not been computed yet. Run one of "
                "the following function: 'transform', 'transform_generator', "
                "'fit_transform', 'fit_transform_generator'.")
        return self._transformed_metadata

    def _compute_transformed_metadata(self) -> None:
        """Compute transformed metadata.

        Raises:
            fe_helpers.PCAError: Error if this function is called
                before the object has been trained.
        """
        if self._transformed_metadata is not None:
            return None
        if self._node_list is None or self._reduction is None:
            raise fe_helpers.PCAError(
                "PCAFeatureExtractor object has not been "
                "trained yet.")
        n_components = int(len(self._node_list) * self._reduction)
        n_components = (n_components if n_components >= self._MIN_COMPONENTS
                        else self._MIN_COMPONENTS)
        new_nodes = [f"pca{i}" for i in range(n_components)]
        self._transformed_metadata = {
            node: meta_prop.AllProperties() for node in new_nodes
        }
        for i in new_nodes:
            self._transformed_metadata[i].node_dim.set(1)
            self._transformed_metadata[i].node_type.set("num")
        # self._transformed_metadata = \
        #     meta_trans.DatasetTransferer.from_dict(self._metadata).get_dict()
        return None

    def _get_transformed_generator(
            self, input_gen: d_obj.DatasetGenerator,
            check_input: bool) -> GenOfDicts:
        """Iterate over a generator and transform each batch.

        Args:
            input_gen (d_obj.DatasetGenerator): Input generator.
            check_input (bool): Whether or not to perform check on the input.

        Raises:
            fe_helpers.PCAError: Error if sample ids are missing.

        Returns:
            GenOfDicts: Output generator.
        """
        for batch in input_gen:
            if dc.SAMPLE_ID_KEYWORD not in batch:
                raise fe_helpers.PCAError(
                    "Input generator must be a generator over dictionaries, "
                    "where every dictionary must contain the sample ids with "
                    f"the key {dc.SAMPLE_ID_KEYWORD}.")
            transformed_batch = self.transform(batch, check_input=check_input)
            transformed_batch[dc.SAMPLE_ID_KEYWORD] = \
                batch[dc.SAMPLE_ID_KEYWORD]
            yield transformed_batch
