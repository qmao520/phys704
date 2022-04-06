# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_return_object as d_obj

DictOfArrays = Dict[str, np.ndarray]
DictOfMetadata = Dict[str, meta_prop.AllProperties]


class FileFormatError(Exception):
    """Error for when a given file has a format that is not supported.
    """
    pass


class IFeatureExtractor(ABC):
    """Abstract Feature Extractor Base Class.
    """

    @staticmethod
    @abstractmethod
    def initialize_new(config_choice_path: str) -> "IFeatureExtractor":
        """Initialize a new (untrained) feature extractor from a config choice
        file.

        Args:
            config_choice_path (str): Path to config choice file.

        Returns:
            IFeatureExtractor: An initialized instance of this class.
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_from_weights(weights_folder: str) -> "IFeatureExtractor":
        """Load a trained feature extractor from weights. These weights are
        generalized weights meaning anything that is saved in the
        training phase and used in the prediction phase. (Therefore the term
        weights is not used in a strict sense as in the parameters of a neural
        network that are optimized during training.) The weights contain all
        the information necessary to reconstruct the Feature Extractor object.

        Args:
            weights_folder (str): Path to folder containing weights.

        Returns:
            IFeatureExtractor: An initialized instance of this class.
        """
        pass

    @abstractmethod
    def save_weights(self, weights_folder: str) -> None:
        """Save feature extractor weights.

        Args:
            weights_folder (str): Path to folder where weights should be saved.
        """
        pass

    @abstractmethod
    def fit(self, input_data: DictOfArrays, metadata: DictOfMetadata) \
            -> "IFeatureExtractor":
        """Train a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            IFeatureExtractor: The class itself.
            (Real typehint not possible in python < 4.0)
        """
        pass

    @abstractmethod
    def fit_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) -> "IFeatureExtractor":
        """Train a feature extractor with a generator over batches as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns
                batches of data. The keys of the dictionaries are the node
                names.
            metadata (dict): Dictionary containing metadata.

        Returns:
            The class itself. (Typehint not possible in python < 4.0)
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def fit_transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata) \
            -> d_obj.DatasetGenerator:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns
                batches of data. The keys of the dictionaries are the node
                names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            d_obj.DatasetGenerator: Transformed samples, batched as a
            generator.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            check_input: bool = False) -> d_obj.DatasetGenerator:
        """Apply a trained feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns
                batches of data. The keys of the dictionaries are the node
                names.
            check_input (bool): Whether or not to check whether input data
                type/format.

        Returns:
            d_obj.DatasetGenerator: Transformed samples, batched as a
            generator.
        """
        pass

    @abstractmethod
    def get_transformed_metadata(self) -> DictOfMetadata:
        """Get transformed metadata after training the Feature Extractor.

        Returns:
            DictOfMetadata: Transformed metadata.
        """
        pass
