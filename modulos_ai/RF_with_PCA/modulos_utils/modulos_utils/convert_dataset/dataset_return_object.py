# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Definition of the dataset object that is used as a return object by the
read functions of the dataset converter class.
"""
import numpy as np
import copy
from typing import Dict, Tuple, Generator, Optional, Callable, Any

from modulos_utils.data_handling import data_handler as dh


class DatasetGenerator:
    """Generator class to iterate over a dataset.
    """

    def __init__(self, wrapper_function: Callable, batch_size: int,
                 dataset_path: str, flatten: bool,
                 kwargs: Dict[str, Any]) -> None:
        """Init for the DatasetGenerator class.
        Args:
            wrapper_function (Callable): Function that returns a generator.
            batch_size (int): Batch size.
            dataset_path (str): Path to hdf5 dataset.
            flatten (bool): Whether to use flattening generator function
                from the dataset_reader.
            kwargs (Dict[str, Any]): Keyword arguments for wrapper
                function.
        """
        self._wrapper_function = wrapper_function
        self._flatten = flatten
        self._batch_size = batch_size
        self._dataset_path = dataset_path
        self._kwargs = kwargs
        return None

    def __iter__(self) -> Generator[Dict, None, None]:
        """Iteration function for the generator.

        Yields:
            Generator[Dict, None, None]: A generator of
                dictionaries where the
                keys are the node names and the values are the batched node
                data as lists.
        """
        if self._flatten:
            generator = self._wrapper_function(
                dh.DatasetReader(
                    self._dataset_path).get_data_in_flattened_batches(
                        self._batch_size),
                **self._kwargs)
        else:
            generator = self._wrapper_function(
                dh.DatasetReader(
                    self._dataset_path).get_data_in_batches(self._batch_size),
                **self._kwargs)
        for batch in generator:
            yield batch


class DatasetReturnObject:
    """Class used as return type of read functions.
    """

    def __init__(self) -> None:
        """Initialize the Dataset object where all the variables are set
        to None per default.
        """
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.data_matrix: Optional[np.ndarray] = None
        self.metadata: Optional[dict] = None
        self.sample_ids: Optional[np.ndarray] = None
        self.data_generator: Optional[DatasetGenerator] = None
        return None

    def add_data(self, data: Dict[str, np.ndarray]) -> None:
        """Populate the data variable.

        Args:
            data (Dict[str, np.ndarray]): Dictionary with node names as keys
                and node data as values.
        """
        self.data = copy.deepcopy(data)
        return None

    def add_data_matrix(self, data_matrix: np.ndarray) -> None:
        """Populate the data matrix variable.

        Args:
            data_matrix (np.ndarray): All nodes stacked into an n x m matrix
                (numpy array) where n is the number of samples and m is the
                number of nodes.
        """
        self.data_matrix = np.copy(data_matrix)
        return None

    def add_metadata(self, metadata: dict) -> None:
        """Populate metadata variable.

        Args:
            metadata (dict): Metadata dictionary.
        """
        self.metadata = copy.deepcopy(metadata)
        return None

    def add_sample_ids(self, sample_ids: np.ndarray) -> None:
        """Populate sample id variable.

        Args:
            sample_ids (np.ndarray): np.ndarray of sample ids which are
                strings.
        """
        self.sample_ids = np.copy(sample_ids)
        return None

    def add_data_generator(
            self, wrapper_function: Callable, batch_size: int,
            dataset_path: str, flatten: bool,
            kwargs: Dict[str, Any] = {}) -> None:
        """Populate the data generator variable.

        Args:
            wrapper_function (Callable): Function that returns a generator.
            batch_size (int): Batch size.
            dataset_path (str): Path to dataset hdf5 file.
            flatten (bool): Whether to use flattening generator function
                from the dataset_reader.
            kwargs (Dict[str, Any]): Optional keyword arguments for wrapper
                function.
        """
        self.data_generator = DatasetGenerator(
            wrapper_function, batch_size, dataset_path, flatten, kwargs)
        return None


class DatasetReturnObjectTuple:
    """Tuple of two DatasetReturnObjects objects.
    """

    def __init__(
            self,
            dataset_return_object_1: DatasetReturnObject,
            dataset_return_object_2: DatasetReturnObject) -> None:
        """Save list of DatasetReturnObject objects.
        """
        self._object_tuple: Tuple[DatasetReturnObject, DatasetReturnObject] = \
            (dataset_return_object_1, dataset_return_object_2)
        return None

    def __iter__(self) -> Generator[DatasetReturnObject, None, None]:
        """Overload the iteration operator, so that the user can do
        `a, b = dataset_return_object_tuple` for an initialized instance
        `dataset_return_object_tuple` of this class.

        Yields:
            Generator[DatasetReturnObject, None, None]: A generator over
                DatasetReturnObjects.
        """
        yield self._object_tuple[0]
        yield self._object_tuple[1]
