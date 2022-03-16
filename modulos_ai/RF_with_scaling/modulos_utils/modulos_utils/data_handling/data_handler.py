# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import math
import os
import copy
from contextlib import ExitStack
from contextlib import contextmanager
from typing import Dict, Generator, List, Tuple, Any, Union, Type

import h5py
import numpy as np

from modulos_utils.datetime_utils import datetime_utils

GenOfDicts = Generator[dict, None, None]
ArrayGen = Generator[np.ndarray, None, None]
_CHUNK_SIZE = 100
SAMPLE_IDS = "__sample_ids__"
GENERATED_SAMPLE_ID_NAME = "sample_ids_generated"
STRING_TYPE_KINDS = {"O", "S"}
STRING_ENCODING = "UTF-8"

# Used to decode the bytestrings returned by h5py to strings.
decode_h5py_bytestrings = np.vectorize(lambda x: x.decode(STRING_ENCODING))


class DatasetNodeDataShapeMismatchError(Exception):
    """Dataset Node Data Shape Mismatch

    This exception is thrown when the shape of the `new_data` does
    not match the shape of the existing data.

    Args:
        msg (str): Human readable string describing the exception.

    Attributes:
        msg (str): Human readable string describing the exception.
    """

    def __init__(self, msg: str) -> None:
        """Init of the Exception Class."""
        self.msg = msg


class DatasetNodeDoesNotExistError(Exception):
    """Dataset Node does not exist error.

    This exception is thrown when one tries to query a dataset node
    that does not exist.

    Args:
        msg (str): Human readable string describing the exception.

    Attributes:
        msg (str): Human readable string describing the exception.
    """


class DatasetSampleIdDoesNotExistError(Exception):
    """Dataset Sample Id does not exist error.

    This exception is thrown when one tries to query a dataset for
    sample id which does not exist in the dataset.

    Args:
        msg (str): Human readable string describing the exception.

    Attributes:
        msg (str): Human readable string describing the exception.
    """


class DatasetNotValidError(Exception):
    """Dataset is not valid.

    This exception is thrown when one tries to construct an invalid
    dataset from an hdf5 file.

    Args:
        msg (str): Human readable string describing the exception.

    Attributes:
        msg (str): Human readable string describing the exception.
    """


class DatasetIndexOutOfBoundsError(Exception):
    """Index of dataset is out of bound error.

    The exception is thrown if one tries to query a dataset for an
    index which is out of bounds.

    """
    pass


class MetaDataDoesNotExistError(Exception):
    """Metadata does not exist in dataset.

    The exception is thrown if one tries to get the metadata of a
    hdf5 file, which has no metadata.
    """
    pass


class WriterWithoutWrapperException(Exception):
    """Writer is used without the wrapper.

    The exception is thrown if a DatasetWriter object is used without
    initiating it with the wrapper in a `with` statement in order to
    be sure that it is closed properly.
    """
    pass


class MultipleWritersException(Exception):
    """Only one DatasetWriter class instance allowed.

    The exception is thrown if a DatasetWriter object is initiated
    while a second one already exists on the same hdf5 file.
    """
    pass


class DatasetBase:
    """Base class for the writer, reader and splitter hdf5 dataset classes.

    The __eq__ operator is implemented in this class for the comparison
    of datasets in the unit tests.

    Attributes:
        hdf5_file_path (str): path to the hdf5 file

    """

    def __init__(self, dataset_file_path: str) -> None:
        """Init of DatasetBase class.

        Args:
            dataset_file_path (str): Path to the hdf5 file.
        """
        self.hdf5_data_path = "/data"
        self.hdf5_metadata_path = "/metadata"
        self.datetime_path = "/datetime"
        self.datetime_metadata_path = "/datetime_metadata"
        self.hdf5_file_path = dataset_file_path

    def __eq__(self, other_dataset: object) -> bool:
        """Compares if two Dataset objects are equal.

        It is comparing the underlying hdf5 files by going recursively
        through all groups and datasets and compares those. It uses
        numpy.allclose for float comparisons and is returning true for
        `NaN`==`NaN`.
        For a comparison on a higher level with additional options, please
        use the function `compare_datasets` in
        `modulos_utils/tarfile_comparison/compare_dataset.py`.

            Args:
                other_dataset (object): other dataset object.

            Returns:
                bool: True if the two objects are equal, otherwise false
        """
        if not isinstance(other_dataset, DatasetBase):
            return NotImplemented
        try:
            with ExitStack() as stack:
                hdf5_file = stack\
                    .enter_context(h5py.File(self.hdf5_file_path, "r"))
                other_hdf5_file = stack\
                    .enter_context(h5py.File(other_dataset.hdf5_file_path,
                                             "r"))
                # Check recursively through the groups and datasets if they
                # are equal.
                return self._compare_groups(
                    hdf5_file["/"], other_hdf5_file["/"],
                )
            return True  # Totally useless statement for mypy.
        except ValueError:
            return False

    def _compare_groups(self, group1: h5py.Group, group2: h5py.Group) -> bool:
        """Compare two hdf5 groups if they are equal.

        This will test if the groups have the same elements, if their datasets
        are the same and will use this function iteratively on their group
        elements to check if they are equal as well.

        Args:
            group1 (h5py.Group): Group from hdf5 dataset from self
            group2 (h5py.Group): Group from other hdf5 dataset

        Returns:
            bool: True if the groups are equal
        """
        elements1 = self._get_elements_of_group(group1)
        elements2 = self._get_elements_of_group(group2)

        # Check if they have the same items.
        if sorted(list(elements1.keys())) != sorted(list(elements2.keys())):
            return False

        # Iterate over all items contained in group 1 and 2.
        for el in elements1:
            # Check that the element is either a group or a dataset in both
            # groups.
            if elements1[el]["hdf5_type"] != elements2[el]["hdf5_type"]:
                return False

            # Compare the attributes.
            if not self._compare_attributes(
                elements1[el]["attr"], elements2[el]["attr"]
            ):
                return False

            # Go through the case that the item is a dataset.
            if elements1[el]["hdf5_type"] == "dataset":
                # Check if the data is of the same type.
                if elements1[el]["dtype"] != elements2[el]["dtype"]:
                    return False

                # Check if the data is the same.
                dataset1 = group1[el]
                dataset2 = group2[el]
                for idx in np.arange(0, len(dataset1), _CHUNK_SIZE):
                    if h5py.string_dtype() == dataset1.dtype:
                        if not (dataset1[idx:idx+_CHUNK_SIZE]
                                == dataset2[idx:idx+_CHUNK_SIZE]).all():
                            return False
                    else:
                        chunk1 = dataset1[idx:idx+_CHUNK_SIZE]
                        chunk2 = dataset2[idx:idx+_CHUNK_SIZE]
                        # Check if values are close to each other.
                        close = np.allclose(chunk1, chunk2)
                        # Check if chunks contain NaN. This is important
                        # to compare entries in the metadata part of the
                        # dataset.
                        both_nan = len(chunk1) == 1 and \
                            self._is_nan(chunk1[0]) and \
                            len(chunk2) == 1 and self._is_nan(chunk2[0])
                        if not (close or both_nan):
                            return False

            # Use recursion to check the sub group
            elif elements1[el]["hdf5_type"] == "group":
                if not self._compare_groups(group1[el], group2[el]):
                    return False
        return True

    def _is_nan(self, value: Any) -> bool:
        """ Test if input is NaN or not.

        Args:
            value (Any): input value

        Returns:
            bool: input is NaN yes/no
        """
        if isinstance(value, float) and np.isnan(value):
            return True
        return False

    def _get_elements_of_group(self, group: h5py.Group) -> Dict:
        """Return the elements of a group as a dictionary.

        The dictionary has the form of
        {name: {'attr': {...}, 'hdf5_type': '...', ...}}.

        Args:
            group (h5py.Group): hdf5 group

        Returns:
            Dict: Dictionary of all elements with name and description
        """
        elements_of_group: Dict[str, Dict] = {}
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                elements_of_group[name] = self._read_dataset(item)
            elif isinstance(item, h5py.Group):
                elements_of_group[name] = self._read_group(item)
        return elements_of_group

    def _read_dataset(self, item: h5py.Dataset) -> Dict[str, Any]:
        """Get the attributes and type of a dataset

        Args:
            item (h5py.Dataset): dataset

        Returns:
            Dict[str, Any]: Dictionary containing the description of the
                dataset
        """
        item_description: Dict[str, Any] = {}
        item_description["attr"] = self._read_attributes(item)
        item_description["hdf5_type"] = "dataset"
        item_description["dtype"] = type(item.dtype)
        return item_description

    def _read_group(self, item: h5py.Group) -> Dict[str, Any]:
        """Get the attributes of a group and mark it as a group

        Args:
            item (h5py.Group): group

        Returns:
            Dict[str, Any]: Description of the group (attributes)
        """
        item_description: Dict[str, Any] = {}
        item_description["attr"] = self._read_attributes(item)
        item_description["hdf5_type"] = "group"
        return item_description

    def _read_attributes(self, item: Any) -> Dict:
        """Read attributes of hdf5 group or dataset

        Args:
            item (Any): h5py.Group or h5py.Dataset

        Returns:
            Dict: Attributes in form {value: type(value)}
        """
        attributes = {}
        for value in item.attrs:
            attributes[value] = type(item.attrs[value])
        return attributes

    def _compare_attributes(self, attribute1: Dict, attribute2: Dict) -> bool:
        """Compare if two attributes are the same.

        This is used to check if two hdf5 groups or datasets are the same.

        Args:
            attribute1 (Dict): Attributes of a dataset or group as dictionary
                with the attribute names as key and their entries as values.
            attribute2 (Dict): Attributes of a dataset or group as dictionary
                with the attribute names as key and their entries as values.

        Returns:
            bool: True if both attribute dictionary contain the same
                attributes.
        """
        # Check if the attributes are the same.
        if (
            sorted(list(attribute1.keys())) !=
            sorted(list(attribute2.keys()))
        ):
            return False

        # Check if the types of the attributes are the same
        for item in attribute1:
            if (attribute1[item] != attribute2[item]):
                return False
        return True


@contextmanager
def get_Dataset_writer(dataset_file_path: str, n_samples: int):
    """Initiate a DatasetWriter object and is responsible for closing
    it after the use of it.
    Use it in a `with` statement.

    Args:
        dataset_file_path (str): Path to the hdf5 file.
        n_samples (int): Number of samples of the dataset.

    Yields:
        DatasetWriter: Writer object to save data in hdf5.
    """
    ds_writer = DatasetWriter(dataset_file_path, n_samples)
    ds_writer.wrapper_used = True
    try:
        yield ds_writer
    finally:
        ds_writer._validate_dataset()
        ds_writer.close_file()


class DatasetWriter(DatasetBase):
    """Creates a hdf5 dataset if the dataset doesn't exist or if it exists
    appends data.

    A data set is a ordered hierarchical collection of dataset features,
    labels and metadata.

    Attributes:
        hdf5_file_path (str): path to the hdf5 file
        hdf5_data_path (str): hdf5 directory for the data
        self.hdf5_metadata_path (str) : hdf5 directory for the metadata
    """
    POINTER = "pointer"
    open_instances: List[str] = []

    def __init__(self, dataset_file_path: str, n_samples: int) -> None:
        """Init of DatasetWriter class.

        Args:
            dataset_file_path (str): Path to the hdf5 file.
            n_samples (int): number of samples
        """
        if dataset_file_path in DatasetWriter.open_instances:
            raise MultipleWritersException(
                "There is already an open DatasetWriter for this HDF5 file. "
                "There is only one writer allowed at the same time!"
            )
        else:
            DatasetWriter.open_instances.append(dataset_file_path)
        super().__init__(dataset_file_path)
        self.n_samples = n_samples
        if not os.path.exists(dataset_file_path):
            self.hdf5_file = h5py.File(dataset_file_path, "w")
            self.hdf5_file.create_group(self.hdf5_data_path)
            self.hdf5_file.create_group(self.hdf5_metadata_path)
            self.hdf5_file[self.hdf5_data_path].attrs["valid"] = False
            self.hdf5_file[self.hdf5_data_path].attrs["n_samples"] = 0
            self.hdf5_file[self.hdf5_data_path].attrs["n_features"] = 0
            self.hdf5_file[self.hdf5_data_path].attrs["validation_msg"] =\
                np.string_("")
        else:
            self.hdf5_file = h5py.File(dataset_file_path, "a")
        self.wrapper_used = False

    def close_file(self) -> None:
        """Close the hdf5 file.

        Returns:
            None
        """
        self.hdf5_file.close()
        DatasetWriter.open_instances.remove(self.hdf5_file_path)
        return None

    def _check_if_wrapped(self) -> None:
        """Check if the DatasetWriter object was initiated correctly.

        Raises:
            WriterWithoutWrapperException: Raised if not initiated correctly.
        Returns:
            None
        """
        if not self.wrapper_used:
            raise WriterWithoutWrapperException(
                "DatasetWriter object used without using the wrapper! "
                "Please initiate your writer object with \n"
                "`with dh.get_Dataset_writer(path, n_samples) as writer:` \n"
                "`    pass`."
            )
        return None

    def add_datetime(self, node_name: str, time_data: np.ndarray) -> None:
        """Add a datetime stamp as string to the hdf5 in a separate hidden
        group. This can be used for the preview or for getting a sample for
        the online client.

        Args:
            node_name (str): Name of the datetime node.
            time_data (np.ndarray): Datetime data in a numpy array as type
                np.datetime64.

        Raises:
            DatasetNodeDataShapeMismatchError: Raised if the shape of the
                input data does not match the already existing data in
                the hdf5.

        Returns:
            None
        """
        self._check_if_wrapped()
        if not time_data.shape:
            time_data = np.atleast_1d(time_data)
        time_data = time_data.reshape(-1)
        # Create the third datetime group in the hdf5.
        if self.datetime_path not in self.hdf5_file:
            self.hdf5_file.create_group(self.datetime_path)
            self.hdf5_file.create_group(self.datetime_metadata_path)
        node_data_path = os.path.join(self.datetime_path, node_name)
        # If there is no data yet for this node name.
        if node_name not in self.hdf5_file[self.datetime_path]:
            dt = h5py.string_dtype()
            node_data = self.hdf5_file.create_dataset(
                node_data_path, (self.n_samples, ), dtype=dt
            )
            # Save the datetime as a string in ISO format.
            node_data[:time_data.shape[0]] = np.datetime_as_string(time_data)
            node_data.attrs[self.POINTER] = time_data.shape[0]
        # Already some of the data is saved.
        else:
            node_data = self.hdf5_file[node_data_path]
            if node_data.shape[1:] != time_data.shape[1:]:
                msg = "Dataset shape mismatch in node/feature: {}.\
                Existing sample shape {} != new data shape {}"\
                .format(node_name, node_data.shape[1:], time_data.shape[1:])
                raise DatasetNodeDataShapeMismatchError(msg)
            node_data[
                node_data.attrs[self.POINTER]:
                    node_data.attrs[self.POINTER] + time_data.shape[0]
            ] = np.datetime_as_string(time_data)
            node_data.attrs[self.POINTER] += time_data.shape[0]
        return None

    def copy_metadata_to_datetime(
            self, node_name: str,
            raise_without_metadata: bool = False) -> None:
        """Move the metadata of a node to the datetime group.

        Args:
            node_name (str): Name of the node whose metadata is moved.
            raise_without_metadata (bool): Whether to raise an error if the
                metadata does not exist.
        """
        original_path = f"{self.hdf5_metadata_path}/{node_name}"
        target_path = f"{self.datetime_metadata_path}/{node_name}"
        if original_path in self.hdf5_file or raise_without_metadata:
            self.hdf5_file.copy(original_path, target_path)
        return None

    def remove_node(self, node_name: str) -> None:
        """Remove a node (including its metadata) from the hdf5 file.

        Args:
            node_name (str): Name of the node to remove.
        """
        data_path = self.hdf5_data_path + "/" + node_name
        meta_path = self.hdf5_metadata_path + "/" + node_name
        if data_path in self.hdf5_file:
            del self.hdf5_file[data_path]
        else:
            raise DatasetNodeDoesNotExistError(
                f"Node '{node_name}' cannot be removed because it does not "
                "exist in the hdf5 file.")
        if meta_path in self.hdf5_file:
            del self.hdf5_file[meta_path]
        return None

    def remove_datetime_node(self, node_name: str) -> None:
        """Remove a node (including its metadata) from the hdf5 file.

        Args:
            node_name (str): Name of the node to remove.
        """
        dt_data_path = self.datetime_path + "/" + node_name
        dt_meta_path = self.datetime_metadata_path + "/" + node_name
        if dt_data_path in self.hdf5_file:
            del self.hdf5_file[dt_data_path]
        else:
            raise DatasetNodeDoesNotExistError(
                f"Datetime node '{node_name}' cannot be removed because it "
                "does not exist in the hdf5 file.")
        if dt_meta_path in self.hdf5_file:
            del self.hdf5_file[dt_meta_path]
        return None

    def add_data_to_node(self, node_name: str, new_data: np.ndarray) -> None:
        """Add a data node (feature column) to the data set.

        A new data node is added under the group `data` in the hdf5 file.
        If the `node_name` already exists in the hdf5 file the `new_data` is
        appended to the data node.

        Args:
            node_name (str): Name of the node/feature.
            new_data (np.ndarray): Data to be added to the node/feature column.
                Has to have shape (n_samples, data_dim).

        Raises:
            DataSetNodeCreatorDataShapeMismatchError: This exception is
                thrown when the shape of the `new_data` does not match the
                shape of the existing data.
            WriterWithoutWrapperException: Raised if not initiated correctly.

        Returns:
            None
        """
        self._check_if_wrapped()
        if not new_data.shape:
            new_data = np.atleast_1d(new_data)
        node_data_path = os.path.join(self.hdf5_data_path, node_name)
        if node_name not in self.hdf5_file[self.hdf5_data_path]:
            if new_data.dtype.char == "U":
                dt = h5py.string_dtype()
            else:
                dt = new_data.dtype
            node_data = self.hdf5_file\
                .create_dataset(node_data_path,
                                (self.n_samples, *new_data.shape[1:]),
                                dtype=dt)
            node_data[:new_data.shape[0]] = new_data
            node_data.attrs[self.POINTER] = new_data.shape[0]
        else:
            node_data = self.hdf5_file[node_data_path]
            if node_data.shape[1:] != new_data.shape[1:]:
                msg = "Dataset shape mismatch in node/feature: {}.\
                Existing sample shape {} != new data shape {}"\
                .format(node_name, node_data.shape[1:], new_data.shape[1:])
                raise DatasetNodeDataShapeMismatchError(msg)
            if (node_data.dtype != new_data.dtype and
                    not (node_data.dtype.char == "O" and
                         new_data.dtype.char == "U")):
                raise TypeError(f"Type of value ({new_data.dtype}) is not "
                                f"the same as the type of the dataset "
                                f"({node_data.dtype})!")
            node_data[
                node_data.attrs[self.POINTER]:
                    node_data.attrs[self.POINTER] + new_data.shape[0]
            ] = new_data
            node_data.attrs[self.POINTER] += new_data.shape[0]
        return None

    def add_value_to_node_at_position(self, value: Any, node_name: str,
                                      position: int) -> None:
        """Add a single value to a node at a specific position in the dataset.

        A new data node is added under the group `data` in the hdf5 file, if
        there is not one with its name. Then the value is added at the position
        defined by `position`.

        Args:
            value (int, float, str, np.ndarray): Data point to be added to the
                node. Has to have shape (data_dim,).
            node_name (str): Name of the node/feature.
            position (int): position in the dataset

        Raises:
            DataSetNodeCreatorDataShapeMismatchError: This exception is
                thrown when the shape of the `new_data` does not match the
                shape of the existing data.
            WriterWithoutWrapperException: Raised if not initiated correctly.

        Returns:
            None
        """
        self._check_if_wrapped()
        if position < 0 or position >= self.n_samples:
            raise DatasetIndexOutOfBoundsError(
                (f"Position arguments (position = {position}) out of bounds "
                 f"(0, {self.n_samples})."))

        value = np.atleast_1d(value)
        node_data_path = os.path.join(self.hdf5_data_path, node_name)
        if node_name not in self.hdf5_file[self.hdf5_data_path]:
            if value.dtype.char == "U":
                dt = h5py.string_dtype()
            else:
                dt = value.dtype
            node_data = self.hdf5_file.create_dataset(
                node_data_path,
                (self.n_samples, *value.shape),
                dtype=dt)
            node_data[position] = value
            node_data.attrs[self.POINTER] = 1
        else:
            node_data = self.hdf5_file[node_data_path]
            if node_data.shape[1:] != value.shape:
                msg = ("Dataset shape mismatch in node/feature: "
                       f"{node_name}. Existing sample shape "
                       f"{node_data.shape[1:]} != new data shape "
                       f"{value.shape}.")
                raise DatasetNodeDataShapeMismatchError(msg)
            if (node_data.dtype != value.dtype and
                    not (node_data.dtype.char == "O" and
                         value.dtype.char == "U")):
                raise TypeError(f"Type of value ({value.dtype}) is not "
                                f"the same as the type of the dataset "
                                f"({node_data.dtype})!")
            node_data[position] = value
            node_data.attrs[self.POINTER] += 1
        return None

    def remove_empty_dimensions(self, nested_array: np.ndarray) \
            -> Union[int, float, str, np.ndarray]:
        """Remove empty dimensions from numpy array.

        Args:
            nested_scalar (np.ndarray): Possibly nested array with empty
                dimensions.

        Raises:
            WriterWithoutWrapperException: Raised if not initiated correctly.

        Returns:
            Union[int, float, str]: Either a python native scalar, or a numpy
                array.
        """
        self._check_if_wrapped()
        squeezed = np.squeeze(nested_array)
        if squeezed.shape == ():
            # If the resulting array is a scalar np.array(x), we convert it to
            # python native (int, str or float).
            return squeezed.tolist()
        else:
            return squeezed

    def get_updated_hdf5_chunk(
            self, node_data_path: str, lower: int, upper: int,
            indices: List[int], data_dict: Dict[int, np.ndarray]) \
            -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Write all values from data_dict into an hdf5 file whose indices are
        between lower and upper. Read a chunk of entries between lower and
        upper from the h5py object to a numpy array, insert the relevant
        values and then put the chunk back into the h5py file. Despite the
        inefficient copying to numpy and back to h5py, this is faster than
        inserting all the values at once, because the h5py library can write
        contiguous chunks much faster than inserting at random places. This
        solution is however not ideal either, since it scales not linearly
        with the number of batches. See BAS-820.

        Args:
            node_data_path (str): Path to the node data inside the hdf5 file.
            lower (int): Lower bound of the chunk.
            upper (int): Upper bound of the chunk.
            indices (List[int]): List of indices.
            data_dict (Dict[int, np.ndarray]): Dictionary where the keys are
                the indices in the hdf5 file and the values the cells of the
                table whose values are added.

        Raises:
            WriterWithoutWrapperException: Raised if not initiated correctly.

        Returns:
            Tuple[np.ndarray, Dict[int, np.ndarray]]: Modified chunk as a numpy
                array and modified input dictionary (deleted the entries that
                were used)
        """
        self._check_if_wrapped()
        # Get an hdf5 file object for the node that is updated.
        node_hdf5 = self.hdf5_file[node_data_path]
        # Copy contiguous chunk to numpy array.
        chunk_numpy = np.copy(node_hdf5[lower: upper])
        # Make sure the function still works, if the array will not be filled
        # with scalars.
        if len(np.squeeze(data_dict[indices[0]]).shape) > 1:
            chunk_numpy = np.zeros((upper - lower,
                                    *np.squeeze(data_dict[indices[0]]).shape))
        # Edit the numpy array by inserting the relevant values.
        for key in indices:
            if key >= lower and key < upper:
                chunk_numpy[key - lower] = self.remove_empty_dimensions(
                    data_dict[key])
                # Delete values, from the master dict, as soon as they have
                # been added.
                del data_dict[key]
        # Return modified chunk as a numpy array and the modified data
        # dictionary.
        return chunk_numpy, data_dict

    def add_dict_to_node(self, data_dict: Dict[int, Any],
                         node_name: str) -> None:
        """Add the values of a dictionary at the position defined by its keys.
        Note that this function is not ideal since it scales non-linearly with
        the number of batches (hence with the number of samples). It is
        however still faster than
        `node_data[list(data.keys())] = np.concatenate(list(data.values()))`S
        for a 5 GB table with 8 million samples and 82 columns. Nevertheless
        it needs to be optimized for asymptotic behavior (see BAS-820).

        Args:
            data_dict (Dict[int, Any]): Data to add with keys as position.
            node_name (str): Name of the node / feature.

        Raises:
            DatasetNodeDataShapeMismatchError: Raised if the values have not
                the same shape.
            TypeError: Raised if the values are of different type.
            WriterWithoutWrapperException: Raised if not initiated correctly.

        Returns:
            None
        """
        self._check_if_wrapped()
        data: Dict[int, np.ndarray] = {pos: np.atleast_1d(data_dict[pos])
                                       for pos in sorted(data_dict)}
        value_dtype = next(iter(data.values())).dtype
        value_shape = next(iter(data.values())).shape
        for value in data.values():
            if value_shape != value.shape:
                msg = ("Dataset shape mismatch in node/feature: "
                       f"{node_name}. Existing sample shape "
                       f"{value_shape} != new data shape "
                       f"{value.shape}.")
                raise DatasetNodeDataShapeMismatchError(msg)
            if (value_dtype != value.dtype and
                    not (value_dtype.char == value.dtype.char)):
                raise TypeError(f"Type of value ({value.dtype}) is not"
                                f" the same as the type of the dataset"
                                f" ({value_dtype})!")
        node_data_path = os.path.join(self.hdf5_data_path, node_name)
        if node_name not in self.hdf5_file[self.hdf5_data_path]:
            if value_dtype.char == "U":
                dt = h5py.string_dtype()
            else:
                dt = value_dtype
            node_data_h5py_obj = self.hdf5_file.create_dataset(
                node_data_path,
                (self.n_samples,),
                dtype=dt)
            node_data_h5py_obj.attrs[self.POINTER] = 0
        node_data_h5py_obj = self.hdf5_file[node_data_path]
        data_indices = list(data.keys())
        min_key = min(data_indices)
        max_key = max(data_indices)
        batch_size = len(data_indices)
        # Slice the range of indices in the batch into sub-batches that are
        # contiguous and have the length 'batch_size'. Then load these
        # sub-batches into RAM, copy to numpy array for faster data
        # manipulation, insert the relevant values, and write them back
        # in the hdf5 file. Despite inefficient copying around, this is
        # faster, than giving the whole list of indices to the h5py
        # object.
        n_iterations = math.ceil((max_key - min_key + 1)
                                 / float(batch_size))
        for i in range(n_iterations):
            lower = min_key + i * batch_size
            upper = min(max_key + 1, min_key + (i + 1) * batch_size)
            node_data_h5py_obj[lower: upper], data = \
                self.get_updated_hdf5_chunk(node_data_path, lower,
                                            upper, data_indices, data)
            # Update this list, because the dictionary 'data' changes its
            # size as already saved elements are deleted.
            data_indices = list(data.keys())
        # Update the count of changed values.
        node_data_h5py_obj.attrs[self.POINTER] += batch_size
        return None

    def add_metadata_to_node(
            self, node_name: str, metadata: Dict[str, Any]) -> None:
        """Add metadata to a node/feature column in the hdf5 file.

        Args:
            node_name (str): Name of the node to which to add the metadata
            metadata (Dict): The metadata eg. key-value pairs that describe the
                node data

        Raises:
            DatasetNodeDoesNotExistError: Exception is thrown when the node
                to which one wants to add the metadata does not exist.
            WriterWithoutWrapperException: Raised if not initiated correctly.

        Return:
            None
        """
        self._check_if_wrapped()
        if node_name in self.hdf5_file[self.hdf5_data_path]:
            metadata_grp = self.hdf5_metadata_path
        elif (self.datetime_path in self.hdf5_file and
              node_name in self.hdf5_file[self.datetime_path]):
            metadata_grp = self.datetime_metadata_path
        else:
            msg = "Node: {} does not exist. \
                Please create the node first by using the method:\
                `add_data_node`".format(node_name)
            raise DatasetNodeDoesNotExistError(msg)
        if node_name not in self.hdf5_file[metadata_grp]:
            self.hdf5_file[metadata_grp].create_group(node_name)
        node_meta_path = os.path.join(metadata_grp, node_name)
        for metadata_name, metadata_value in metadata.items():
            value = np.atleast_1d(metadata_value)
            metadata_dataset_path = os.path.join(
                node_meta_path, metadata_name)
            if value.dtype.char == "U":
                dt = h5py.string_dtype()
            elif 'datetime64' in value.dtype.name:
                dt = h5py.string_dtype()
                value = np.datetime_as_string(value)
            else:
                dt = value.dtype
            # Check if dataset exists already.
            if metadata_dataset_path in self.hdf5_file:
                # Delete dataset.
                del self.hdf5_file[metadata_dataset_path]
            # Create new dataset.
            metadata_dataset = self.hdf5_file.create_dataset(
                metadata_dataset_path, dtype=dt, shape=value.shape
            )
            metadata_dataset[:] = value
        return None

    def _update_data_info(self, data_info: Dict) -> None:
        """Add information about the dataset to the hdf5 file.

        Args:
            metadata (Dict): The information in form key-value pairs.

        Returns:
            None
        """
        for metadata_name, metadata_value in data_info.items():
            self.hdf5_file[self.hdf5_data_path].attrs[metadata_name] = \
                metadata_value
        return None

    def add_samples(self, sample_data: Dict) -> None:
        """Add some samples to the hdf5 dataset.

        Add some samples to the hdf5 dataset. The `sample_data` contains few
        samples and has to have the key `sample_ids`.

        Args:
            sample_data (Dict): Sample data to add to the hdf5 dataset.
        """
        for node_name, node_data in sample_data.items():
            if not hasattr(node_data, "__len__") or type(node_data) == str:
                node_data = np.array([node_data])

            self.add_data_to_node(node_name, node_data)
        return None

    def add_datetime_samples(self, sample_data: Dict) -> None:
        """Add some datetime samples to the hdf5 dataset.

        Add some samples to the hdf5 dataset. The `sample_data` contains few
        samples and has to have the key `sample_ids`.

        Args:
            sample_data (Dict): Sample data to add to the datetime part of
                the hdf5 dataset.
        """
        for node_name, node_data in sample_data.items():
            if not hasattr(node_data, "__len__") or type(node_data) == str:
                node_data = np.array([node_data])

            self.add_datetime(node_name, node_data)
        return None

    def add_sample_ids(self, sample_ids: np.ndarray) -> None:
        """Add a sample_ids to the hdf5 dataset.

        Add the node sample_ids to the hdf5 dataset.

        Args:
            sample_ids (np.ndarray): List of sample_ids
        """
        self.add_data_to_node(SAMPLE_IDS, sample_ids)
        return None

    def _validate_dataset(self) -> None:
        """Check if a dataset is valid, i.e. the data is filled correctly.

        Check that there are sample ids, they are unique and all nodes have
        n_samples. We specifically test only the data and not the metadata,
        as this is filled with the datahandler, but not computed and therefore
        the datahandler should not be responsible for the correctness of this.

        Returns:
            None
        """
        error_info: Dict[str, Union[bool, str]] = {"valid": False}
        node_names = list(self.hdf5_file[self.hdf5_data_path].keys())
        if SAMPLE_IDS not in node_names:
            msg = (f"Node '{SAMPLE_IDS}' is not present in dataset."
                   "Please use method:`add_sample_ids(sample_ids)`"
                   f"to add the '{SAMPLE_IDS}' node.")
            error_info["validation_msg"] = msg
            self._update_data_info(error_info)
            return None
        node_path = os.path.join(self.hdf5_data_path, node_names[0])
        n_samples = self.hdf5_file[node_path].shape[0]
        for node_name in node_names:
            node_path = os.path.join(self.hdf5_data_path, node_name)
            if self.hdf5_file[node_path].shape[0] != n_samples:
                msg = (f"Node:{node_path} contains "
                       f"{self.hdf5_file[node_path].shape[0]} samples instead "
                       f"of {n_samples}.")
                error_info["validation_msg"] = msg
                self._update_data_info(error_info)
                return None
            if self.hdf5_file[node_path].attrs[self.POINTER] != n_samples:
                msg = ("Dataset is not completely filled. It has "
                       f"{self.hdf5_file[node_path].attrs[self.POINTER]} "
                       f"samples saved, but should have {n_samples}.")
                error_info["validation_msg"] = msg
                self._update_data_info(error_info)
                return None
        sample_ids_path = os.path.join(self.hdf5_data_path, SAMPLE_IDS)
        if (len(np.unique(self.hdf5_file[sample_ids_path][:]))
                != n_samples):
            msg = f"Elements in Node `{SAMPLE_IDS}` are not unique."
            error_info["validation_msg"] = msg
            self._update_data_info(error_info)
            return None
        msg = "Dataset valid. Everything looks good."
        data_info = {"n_samples": n_samples,
                     "n_features": len(node_names)-1,
                     "valid": True,
                     "validation_msg": msg}
        self._update_data_info(data_info)
        return None


class DatasetReader(DatasetBase):
    """Reads data from the hdf5 dataset.

    A data set is a ordered hierarchical collection of features,
    labels and metadata.

    Attributes:
        hdf5_file_path (str): path to the hdf5 file
        hdf5_data_path (str): hdf5 directory for the data
        self.hdf5_metadata_path (str) : hdf5 directory for the metadata
    """

    def __init__(self, dataset_file_path: str) -> None:
        """Init of DatasetReader class.

        Args:
        dataset_file_path (str): Path to the hdf5 file.
        """
        if not os.path.exists(dataset_file_path):
            raise FileNotFoundError()

        super().__init__(dataset_file_path)

    def _raise_error_if_invalid(self) -> None:
        """Check if dataset is valid and if not raise an error.

        Raises:
            DatasetNotValidError: Raised if the dataset is not valid.

        Returns:
            None
        """
        info_data = self.get_data_info()
        if info_data["valid"] is False:
            raise DatasetNotValidError(
                "Dataset not valid! " + info_data["validation_msg"]
            )
        return None

    def get_validation_info(self) -> Tuple[bool, str]:
        """Return the valid status and message.

        Returns:
            Tuple[bool, str]: Return the valid status and message.
        """
        info_data = self.get_data_info()
        return info_data["valid"], info_data["validation_msg"]

    def get_n_samples(self) -> int:
        """Return the number of samples of the dataset.

        Returns:
            int: number of samples
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            n_samples = int(hdf5_file[self.hdf5_data_path].attrs["n_samples"])
        return n_samples

    def get_all_metadata(self) -> Dict:
        """Get all the meta data saved nodewise in the hdf5 file.

        For all nodes and all keys return a dictionary containing
        all the metadata information in the form Dict[str, Dict[str, Any]].
        The first string is the node name and the second one the meta data
        key.

        Returns:
            Dict: Contains all the metadata information that is saved nodewise.
        """
        nodes = self._get_all_metadata_nodes_names()
        metadata_dict: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            metadata_dict[node] = self.get_metadata_of_node(node)
        return metadata_dict

    def get_data_info(self) -> Dict:
        """Get the data info of the hdf5 file.

        Returns:
            Dict: Dictionary with the keys of the data info.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            meta_data = {}
            for key, value in hdf5_file[self.hdf5_data_path].attrs.items():
                meta_data[key] = value
            self._convert_metadata_dtype(meta_data)
            return meta_data

    def _convert_metadata_dtype(self, dictionary: Dict) -> None:
        """Convert the metadata numpy dtypes to python types.

        Necessary if we want to store metadata_information in json.

        Args:
            dictionary (Dict): Dictionary that should be transfromed.
        """
        for key, value in dictionary.items():
            if isinstance(value, np.integer):
                dictionary[key] = int(value)
            elif isinstance(value, np.floating):
                dictionary[key] = float(value)
            elif isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()
            elif isinstance(value, str):
                dictionary[key] = str(value)
            elif isinstance(value, dict):
                self._convert_metadata_dtype(value)
        return None

    def get_metadata_of_node(self, node_name: str) -> Dict:
        """Get metadata information of a specific node.

        Args:
            node_name (str): Name of the node for which to get the metadata
                info.

        Raises:
            DatasetNodeDoesNotExistError: Exception is thrown if the node
                does not exist.

        Returns:
            Dict: Containing the metadata information.
        """
        keys: List = self._get_all_metadata_keys_for_node(node_name)
        metadata_dict: Dict[str, Any] = {}
        for key in keys:
            metadata_dict.update(self.get_metadata_of_node_and_key(
                node_name, key))
        return metadata_dict

    def get_metadata_of_node_and_key(self, node: str, key: str) -> Dict:
        """Returns the metadata of a single node and key

        Args:
            node (str): Node
            key (str): Metadata key

        Raises:
            MetaDataDoesNotExistError: Thrown if there was no metadata
                previously added to the node.

        Returns:
            Dict: Metadata in a dictionary in the form {key: ...}
        """
        keys: List = self._get_all_metadata_keys_for_node(node)
        if key not in keys:
            raise MetaDataDoesNotExistError(
                f"There is no meta data for the key {key} in node {node}."
            )
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            try:
                metadata_dataset = hdf5_file[
                    self.hdf5_metadata_path][node][key]
            except KeyError:
                metadata_dataset = hdf5_file[
                    self.datetime_metadata_path][node][key]
            metadata_dict = self._get_data_from_hdf5_dataset(
                metadata_dataset, key,
            )
            # self._convert_metadata_dtype(metadata_dict)
            return metadata_dict

    def _get_data_from_hdf5_dataset(
        self, hdf5_dataset: h5py.Dataset, key: str,
    ) -> Dict[str, np.ndarray]:
        """Return the data as an array inside a dictionary.

        Args:
            hdf5_dataset (h5py.Dataset): data inside the hdf5 file.
            key (str): The metadata key which is used as key for the returned
                dictionary.

        Raises:
            DatasetNotValidError: Raised if the dataset is not valid.

        Returns:
            Dict[str, Any]: Dictionary in the form {key: value}, where value is
                always an array.
        """
        self._raise_error_if_invalid()
        if hdf5_dataset.dtype.kind in STRING_TYPE_KINDS:
            return {key: decode_h5py_bytestrings(hdf5_dataset[:])}
        else:
            return {key: hdf5_dataset[:]}

    def _get_all_metadata_nodes_names(self) -> List:
        """Return the names of all metadata node groups existing in the
        hdf5 file.

        Raises:
            MetaDataDoesNotExistError: Thrown if an hdf5 file has no group
                metadata.

        Returns:
            List: Name of all nodes with existing metadata.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if "metadata" not in hdf5_file:
                raise MetaDataDoesNotExistError(
                    "There is no meta data."
                )
            metadata = hdf5_file["metadata"]
            nodes = list(metadata.keys())
            return nodes

    def _get_all_metadata_keys_for_node(self, node: str) -> List:
        """Returns the metadata keys for an existing metadata node.

        Args:
            node (str): Node name.

        Raises:
            MetaDataDoesNotExistError: Thrown if there is no metadata for
                the specified node.

        Returns:
            List: Names of all metadata keys.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if node in hdf5_file[self.hdf5_metadata_path]:
                metadata = hdf5_file[self.hdf5_metadata_path]
            elif (self.datetime_metadata_path in hdf5_file and
                  node in hdf5_file[self.datetime_metadata_path]):
                metadata = hdf5_file[self.datetime_metadata_path]
            else:
                raise MetaDataDoesNotExistError(
                    f"There is no meta data for node {node}."
                )
            metadata_node = metadata[node]
            keys = sorted(list(metadata_node.keys()))
            return keys

    def get_data_in_batches(self, batch_size: int) -> GenOfDicts:
        """Get an generator that gets all the data contained in the hdf5 file.

        Get an Generator that returns all nodes that are present in the `/data`
        path. The Generator returns the data in batch of size `batch_size`

        Args:
            batch_size (int): Integer for the batch size.

        Returns:
            GenOfDicts: Returns a Generator that returns
                batches of the entire dataset.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            n_samples = hdf5_file[self.hdf5_data_path].attrs["n_samples"]
            for idx in np.arange(0, n_samples, batch_size):
                data: Dict[str, np.ndarray] = {}
                for node in hdf5_file[self.hdf5_data_path].keys():
                    data[node] = self._get_data_of_node_in_range(
                        node, idx, idx + batch_size)
                yield data

    def get_datetime_data_in_batches(self, batch_size: int) -> GenOfDicts:
        """Get an generator that gets all the datetime data contained in the
        hdf5 file.

        Get an Generator that returns all nodes that are present in the `/data`
        path. The Generator returns the data in batch of size `batch_size`

        Args:
            batch_size (int): Integer for the batch size.

        Returns:
            GenOfDicts: Returns a Generator that returns
                batches of dictionaries containing the datetime nodes.
        """
        sample_ids = self.get_sample_ids_all()
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            n_samples = hdf5_file[self.hdf5_data_path].attrs["n_samples"]
            for idx in np.arange(0, n_samples, batch_size):
                batch_sample_ids = sample_ids[idx:idx+batch_size]
                data = self.get_datetime_for_sample_ids(batch_sample_ids)
                yield data

    def get_data_in_flattened_batches(self, batch_size: int) -> GenOfDicts:
        """Get an generator that gets all the data contained and collapses it
        into a vector that can be accessed using `data` as a key.

        Get an Generator that returns all nodes that are present in the `/data`
        path. The Generator returns the data in batches of size `batch_size`.

        Args:
            batch_size (int): Integer for the batch size.

        Returns:
            GenOfDicts: Returns a Generator that returns
                batches of the entire dataset.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            n_samples = hdf5_file[self.hdf5_data_path].attrs["n_samples"]
            for idx in np.arange(0, n_samples, batch_size):
                data: Dict[str, np.ndarray] = {"data": np.array([])}
                for node in hdf5_file[self.hdf5_data_path].keys():
                    node_data = self\
                        ._get_data_of_node_in_range(node, idx,
                                                    idx + batch_size)
                    if node == SAMPLE_IDS:
                        data[SAMPLE_IDS] = node_data
                    else:
                        if data["data"].size == 0:
                            data["data"] = node_data.reshape(len(node_data),
                                                             -1)
                        else:
                            node_re = node_data.reshape(len(node_data), -1)
                            data["data"] = \
                                np.concatenate((data["data"], node_re), axis=1)
                yield data

    def get_data_all(self) -> Dict[str, np.ndarray]:
        """Get all the the data contained in the hdf5 file.

        Get all the data of the dataset e.g. all nodes that are present in the
        `/data`.

        Returns:
            Dict: Returns a dictionary that contains the entire dataset.
        """
        data: Dict[str, np.ndarray] = {}
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            for node in hdf5_file[self.hdf5_data_path]:
                data[node] = self.get_data_of_node_all(node)
        return data

    def get_data_of_node_all(self, node_name: str) -> np.ndarray:
        """Get all data of one specific node.

        Get data of the node with name `node_name` as one batch containing all
        the data.

        Args:
            node_name (str): Name of the node containing the data

        Returns:
            np.ndarray: Returns a numpy array that contains the data of node
                'node_name'.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            n_samples = hdf5_file[self.hdf5_data_path].attrs["n_samples"]
        return self._get_data_of_node_in_range(node_name, 0, n_samples)

    def get_data_from_node_for_ids(
            self, sample_ids: List[str], node_name: str) -> List:
        """Return the data in a dictionary for selected samples.

        Args:
            sample_ids (List[str]): Sample ids of data to retrieve.
            node_name (str): Name of node containing the data.

        Returns:
            List: List of the data in the order of the given sample ids.
        """
        self._raise_error_if_invalid()
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if node_name not in hdf5_file[self.hdf5_data_path]:
                msg = "Node: {} does not exist. \
                    Please create the node first by using the method:\
                    `add_data_node`".format(node_name)
                raise DatasetNodeDoesNotExistError(msg)
            node_path = os.path.join(self.hdf5_data_path, node_name)

            sample_ids_all = self.get_sample_ids_all()
            ids_to_index = {k: v for v, k in enumerate(sample_ids_all)}
            data_dict = {}
            sample_indices = sorted([ids_to_index[id] for id in sample_ids])
            batch_size = len(sample_ids)

            lower = sample_indices[0]
            last_index = sample_indices[-1]
            while lower <= last_index:
                upper = lower + batch_size
                data = hdf5_file[node_path][lower:upper]
                # Make sure that every node has at least two dimensions (as
                # defined in the doc: `doc/modules_data_assumptions.md`,
                # chapter `Data Shape Across the flow of a workflow`.)
                if len(data.shape) == 1 and node_name != SAMPLE_IDS:
                    data = np.expand_dims(data, -1)
                if data.dtype == h5py.string_dtype():
                    data = decode_h5py_bytestrings(data)
                samples_in_batch = [index for index in sample_indices
                                    if index >= lower and index < upper]
                for sample_index in samples_in_batch:
                    data_dict[sample_ids_all[sample_index]] = copy.deepcopy(
                        data[sample_index - lower])
                lower = upper
            data_list = [data_dict[id] for id in sample_ids]
        return data_list

    def get_sample_ids_all(self):
        """Returns all sample_ids of the hdf5.

        Returns:
            numpy.array: sample_ids
        """
        return self.get_data_of_node_all(SAMPLE_IDS)

    def get_data_of_node_in_batches(self, node_name: str,
                                    batch: int) -> ArrayGen:
        """Get data of one specific node in batches.

        Get data of the node with name `node_name`,
        in batches of certain size.

        Args:
            node_name (str): Name of the node containing the data
            batch (int):  Integer for the batch size.

        Raises:
            DatasetNodeDoesNotExistError: Error is thrown if the node
                does not exist.

        Returns:
            ListGen: A generator over lists containing the batch of data for
                the node with name `node_name`.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            n_samples = hdf5_file[self.hdf5_data_path].attrs["n_samples"]
        if node_name in self.get_node_names():
            for idx in np.arange(0, n_samples, batch):
                yield self._get_data_of_node_in_range(
                    node_name, idx, idx + batch)
        else:
            data = np.datetime_as_string(
                self.get_datetime_data_of_node(node_name))
            for idx in np.arange(0, n_samples, batch):
                yield data[idx:idx + batch]

    def get_datetime_data_of_node(self, node_name: str) -> np.ndarray:
        """Return the timestamps of one node in the np.datetime64 format.

        Args:
            node_name (str): Name of the node containing the datetime stamps.

        Raises:
            DatasetNodeDoesNotExistError: Raised if there is no datetime node
                with this name.

        Returns:
            np.ndarray: Datetime stamps in numpy.datetime64 format.
        """
        self._raise_error_if_invalid()
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if node_name not in hdf5_file[self.datetime_path]:
                msg = "Node: {} does not exist. \
                    Please create the node first by using the method:\
                    `add_datetime_to_node`".format(node_name)
                raise DatasetNodeDoesNotExistError(msg)
            node_path = os.path.join(self.datetime_path, node_name)

            data = np.array(hdf5_file[node_path][:], dtype=np.datetime64)
            data = np.expand_dims(data, -1)
        return data

    def get_datetime_for_sample_ids(self, sample_ids: List[str]) -> Dict:
        """Get the timestamps for certain sample_ids.

        Args:
            sample_ids (List[str]): List of the sample_ids of the datetime
                data to return.

        Returns:
            Dict: Dictionary with node names as key and the datetime stamps
                as values.
        """
        self._raise_error_if_invalid()
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            data = {}
            sample_ids_path = os.path.join(self.hdf5_data_path, SAMPLE_IDS)
            all_sample_ids = list(decode_h5py_bytestrings(
                hdf5_file[sample_ids_path][:]))
            all_samples_dict = dict(zip(list(all_sample_ids),
                                        list(range(len(all_sample_ids)))))
            for n, sample_id in enumerate(sample_ids):
                try:
                    idx = all_samples_dict[sample_id]
                except KeyError:
                    msg = "Sample_id: {} does not exist. \
                        Please create the sample id first by using the method:\
                        `add_sample_ids`".format(sample_id)
                    raise DatasetSampleIdDoesNotExistError(msg)
                # Get the timestamp for all nodenames for the current
                # sample_id.
                for key, value in hdf5_file[self.datetime_path].items():
                    stamp = np.array(value[idx], dtype=np.datetime64)
                    if n == 0:
                        data[key] = np.expand_dims(stamp, axis=0)
                    else:
                        data[key] = np.append(data[key], stamp)
            return data

    def get_data_for_sample_ids(self, sample_ids: List[str]) -> Dict:
        """Get data for data points using the sample_ids.

        Args:
            sample_ids (List[str]): Unique identifiers for a list a samples

        Raises:
            DatasetSampleIdDoesNotExistError: Throw error is sample_id
                does not exist in the Dataset.
            DatasetNotValidError: Raised if the dataset is not valid.

        Returns:
            Dict: where the keys are the node names of the dataset and the
                values lists that contain the values corresponding to the
                list of sample_ids.
        """
        self._raise_error_if_invalid()
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            data = {}
            sample_ids_path = os.path.join(self.hdf5_data_path, SAMPLE_IDS)
            all_sample_ids = list(decode_h5py_bytestrings(
                hdf5_file[sample_ids_path][:]))
            all_samples_dict = dict(zip(list(all_sample_ids),
                                        list(range(len(all_sample_ids)))))
            for n, sample_id in enumerate(sample_ids):
                try:
                    idx = [all_samples_dict[sample_id]]
                except KeyError:
                    msg = "Sample_id: {} does not exist. \
                        Please create the sample id first by using the method:\
                        `add_sample_ids`".format(sample_id)
                    raise DatasetSampleIdDoesNotExistError(msg)
                for key, node_data in hdf5_file[self.hdf5_data_path].items():
                    # Read out the entry for the node and sample_id.
                    value: Any = node_data[idx[0]]
                    # Decode it if it is a bytestring.
                    if isinstance(value, bytes):
                        value = value.decode(STRING_ENCODING)
                    elif (isinstance(value, np.ndarray) and
                            value.dtype.kind in STRING_TYPE_KINDS):
                        value = decode_h5py_bytestrings(value)

                    if n == 0 and type(value) == np.ndarray:
                        data[key] = np.expand_dims(value, axis=0)
                    elif n == 0 and type(value) != np.ndarray:
                        data[key] = np.array([value])
                    elif n > 0 and type(value) == np.ndarray:
                        data[key] = np.vstack((
                            data[key], np.expand_dims(value, axis=0)))
                    else:
                        data[key] = np.append(data[key], value)
            return data

    def get_data_of_node_for_sample_ids(
            self, node: str, sample_ids: List[str]) -> np.ndarray:
        """Get the data of a node (normal or datetime) and return the entries
        of the given sample ids.

        Args:
            node (str): Name of the node for which data has to be returned.
            sample_ids (List[str]): List of sample ids for which the data has
                to be returned.

        Returns:
            np.ndarray: Data in a numpy array in the order of the given sample
                ids.
        """
        all_sample_ids = self.get_sample_ids_all()
        indices = [(all_sample_ids == id).nonzero()[0][0]
                   for id in sample_ids]

        data = np.array([])
        # 'Normal' case without datetime.
        if node in self.get_node_names():
            for n in range(len(sample_ids)):
                value = self._get_data_of_node_in_range(
                    node, indices[n], indices[n] + 1)
                data = (value if len(data) == 0
                        else np.append(data, value, axis=0))
        # Retrieve datetime node.
        else:
            values = np.datetime_as_string(
                self.get_datetime_data_of_node(node))
            data = values[indices]
        return data

    def get_node_names(self, include_sample_id: bool = True) -> List:
        """Get the list of node names from a dataset.

        Args:
            include_sample_id (bool = True): Whether to include the
                sample id name or not.

        Returns:
            List: List of nodenames is returned.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            node_names = list(hdf5_file[self.hdf5_data_path].keys())
            if not include_sample_id:
                node_names.remove(SAMPLE_IDS)
            return sorted(node_names)

    def get_upload_node_names(self) -> List:
        """Get the list of node names from the uploaded dataset.

        Returns:
            List: List of nodenames is returned.
    """
        node_names_data = self.get_node_names()
        node_names_time = self.get_datetime_node_names()
        for name in node_names_time:
            # Get all possibly generated datetime features.
            gen_nodes = datetime_utils.get_datetime_node_names(name)
            for node in gen_nodes:
                # Remove it, if it exists.
                if node in node_names_data:
                    node_names_data.remove(node)
        node_names_data.extend(node_names_time)
        return node_names_data

    def get_datetime_node_names(self) -> List[str]:
        """Return the datetime nodes if they exist, else an empty list.

        Returns:
            List: Node names of the datetime nodes.
        """
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if self.datetime_path in hdf5_file:
                return sorted(list(hdf5_file[self.datetime_path].keys()))
            else:
                return []

    def get_gen_datetime_nodes_names_of_node(self, node: str) -> List[str]:
        """Return the new nodes generated from a datetime node. If it is not a
        datetime node, the list will be empty.

        Args:
            node (str): Name of the original node.

        Returns:
            List: Node names of the datetime node.
        """
        datetime_nodes = self.get_datetime_node_names()
        if node in datetime_nodes:
            return datetime_utils.get_datetime_node_names(node)
        else:
            return []

    def get_data_all_as_one_tensor(self) -> Dict:
        """Get all data as one numpy tensor from a dataset.

        This method creates *one* numerical or string numpy tensor for all the
        data present in the dataset. Each sample node is flattened and
        appended. The rows of that matrix are the samples.
        NOTE: For datasets containing mixed types, use
        get_data_of_node function at the moment. BAS-502 has the goal to
        implement another function to simplify this.

        Raises:
            TypeError: Raised if the data type is mixed.

        Returns:
            Dict: A dictionary containing the keys `sample_ids` and
                  `data`. The `data` key contains the entire dataset
                  in either numerical or string form where every node
                  of the data is flattend into a vector.
        """
        out = {}
        all_data = self.get_data_all()
        one_ten_data = np.array([])
        for node_name, data in sorted(all_data.items()):
            if node_name == SAMPLE_IDS:
                out[node_name] = data
            else:
                if len(one_ten_data) == 0:
                    one_ten_data = data.reshape(len(data), -1)
                    if np.issubdtype(one_ten_data.dtype, np.number):
                        tensor_type: Type = np.number
                    else:
                        tensor_type = str
                else:
                    if not np.issubdtype(data.dtype, tensor_type):
                        raise TypeError("Mixing between numerical and "
                                        "string values is not allowed!")
                    one_ten_data = np.hstack((one_ten_data,
                                              data.reshape(len(data), -1)))
        out["data"] = one_ten_data
        return out

    def _get_data_of_node_in_range(self, node_name: str, start: int,
                                   end: int) -> np.ndarray:
        """Private function to get data in a range

        Args:
            node_name (str): Name of the node (feature)
            start (int): start index to retrieve the data
            end (int): end index

        Raises:
            DatasetNodeDoesNotExistError: Throws an error if node_name does not
                exist in the dataset.
            DatasetIndexOutOfBoundsError: Throws an error if start is out of
                bounds or end is smaller or equal than start.
            DatasetNotValidError: Raised if the dataset is not valid.

        Returns:
            np.ndarray: Returns a numpy array that contains
                the data from [start:end] of node node_name.
        """
        self._raise_error_if_invalid()
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if node_name not in hdf5_file[self.hdf5_data_path]:
                msg = "Node: {} does not exist. \
                    Please create the node first by using the method:\
                    `add_data_node`".format(node_name)
                raise DatasetNodeDoesNotExistError(msg)
            node_path = os.path.join(self.hdf5_data_path, node_name)

            n_samples = hdf5_file[self.hdf5_data_path].attrs["n_samples"]
            if start >= n_samples:
                msg = "IndexOutOfBoundsError: Dataset has length {0},\
                    but requested index is {1}!".format(n_samples, start)
                raise DatasetIndexOutOfBoundsError(msg)
            if end <= start:
                msg = "IndexOutOfBoundsError: The selection \
                    [start:end] = [{0}:{1}] does not make sense! end has\
                    to be bigger than start!".format(n_samples, start)
                raise DatasetIndexOutOfBoundsError(msg)
            if hdf5_file[node_path].dtype.kind in STRING_TYPE_KINDS:
                data = decode_h5py_bytestrings(hdf5_file[node_path][start:end])
            else:
                data = hdf5_file[node_path][start:end]
            # Make sure that every node has at least two dimensions (as
            # defined in the doc: `doc/modules_data_assumptions.md`, chapter
            # `Data Shape Across the flow of a workflow`.)
            if len(data.shape) == 1 and node_name != SAMPLE_IDS:
                data = np.expand_dims(data, -1)
        return data
