# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Library for reading and writing datasets. This library can be used by all
modules.
"""
import os
import numpy as np
from typing import Optional, Generator, Dict

from modulos_utils.data_handling import data_handler as dh
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.convert_dataset import dataset_converter_exceptions as dce
from modulos_utils.convert_dataset import dataset_return_object as d_obj

GenOfDicts = Generator[dict, None, None]

SAMPLE_ID_KEYWORD = dh.SAMPLE_IDS
PREDICTIONS_NODE_NAME = "predictions"


class DatasetConverter:
    """Class for reading and writing hdf5 datasets to and from dictionaries and
    generators.
    """

    def __init__(self):
        """Initialize an empty list for the sample ids. We save it in a member
        variable because we want to use the sample ids from the read later in
        the write function.
        """
        self._sample_ids: Optional[np.ndarray] = None
        self._batch_size: int = 1
        self._number_of_samples: Optional[int] = None

    def _create_dataset_return_object(
            self, file_path: str,
            retrieve_sample_ids: Optional[bool] = False,
            retrieve_metadata: Optional[bool] = False,
            as_matrix: Optional[bool] = False) -> d_obj.DatasetReturnObject:
        """Create dataset return object.

        Args:
            file_path (str): Path to dataset in internal dataset format.
            retrieve_sample_ids ([bool]): Whether or not to include sample ids
                in the returned object.
            retrieve_metadata ([bool]): Whether or not to include metadata
                in the returned object.
            as_matrix ([bool]): Whether or not to stack all nodes into a matrix
                omitting the node names.

        Returns:
            Dataset: Dataset object with some of the variables populated and
                some of them at their default values.
        """
        if self._sample_ids is None:
            raise dce.DatasetConverterInternalError(
                "The member variable 'self._sample_ids' must be set.")
        data_reader = dh.DatasetReader(file_path)

        res_obj = d_obj.DatasetReturnObject()
        if retrieve_sample_ids:
            res_obj.add_sample_ids(self._sample_ids)
        if retrieve_metadata:
            res_obj.add_metadata(
                meta_handler.get_metadata_all_nodes(file_path))
        if as_matrix:
            res_obj.add_data_matrix(
                data_reader.get_data_all_as_one_tensor()["data"]
                )
        else:
            data_to_add: Dict[str, np.ndarray] = data_reader.get_data_all()
            if dh.SAMPLE_IDS in data_to_add:
                data_to_add.pop(dh.SAMPLE_IDS)
            res_obj.add_data(data_to_add)
        return res_obj

    def read_data_as_dict(
            self, input_path: str,
            retrieve_metadata: Optional[bool] = False,
            retrieve_sample_ids: Optional[bool] = False,
            ) -> d_obj.DatasetReturnObject:
        """Read an hdf5 dataset to a dictionary.

        Args:
            input_path (str): Path to input dataset.
            retrieve_metadata (bool, optional):  Whether or not to retrieve
                metadata.
            retrieve_sample_ids (bool, optional):  Whether or not to retrieve
                sample_ids.

        Returns:
            Dataset: Dataset object where the variable `data` is populated
                and `data_matrix` and `data_generator` are None.
        """
        return self._read_data(input_path, retrieve_metadata=retrieve_metadata,
                               retrieve_sample_ids=retrieve_sample_ids)

    def read_data_as_matrix(
            self, input_path: str,
            retrieve_metadata: Optional[bool] = False,
            retrieve_sample_ids: Optional[bool] = False,
            ) -> d_obj.DatasetReturnObject:
        """Read an hdf5 dataset to a matrix, where the nodes are the columns,
        ordered alphabetically according to the node names.

        Args:
            input_path (str): Path to input dataset.
            retrieve_metadata (bool, optional):  Whether or not to retrieve
                metadata.
            retrieve_sample_ids (bool, optional):  Whether or not to retrieve
                sample_ids.

        Returns:
            Dataset: Dataset object where the variable `data_matrix` is
                populated and `data` and `data_generator` are None.
        """
        return self._read_data(input_path, retrieve_metadata=retrieve_metadata,
                               retrieve_sample_ids=retrieve_sample_ids,
                               as_matrix=True)

    def _read_data(
            self, input_path: str,
            retrieve_metadata: Optional[bool] = False,
            retrieve_sample_ids: Optional[bool] = False,
            as_matrix: Optional[bool] = False) -> d_obj.DatasetReturnObject:
        """Read an hdf5 dataset to a dictionary.

        Args:
            input_path (str): Path to input dataset.
            retrieve_metadata (bool, optional):  Whether or not to retrieve
                metadata.
            retrieve_sample_ids (bool, optional):  Whether or not to retrieve
                sample_ids.
            as_matrix  (Optional[bool]): Whether or not to stack all the nodes
                alphabetically into an n x m matrix (n: number of samples,
                m: number of nodes).

        Returns:
            Dataset: Dataset object.
        """
        if not os.path.isfile(input_path):
            raise dce.DataFileInexistentError(
                "Dataset cannot be read in. This dataset hdf5 file does not "
                f"exist: {input_path}")

        # Initialize a DatasetReader class.
        data_reader_input = dh.DatasetReader(input_path)
        sample_ids_input = np.array(
            data_reader_input.get_sample_ids_all(), dtype=str)

        self._sample_ids = sample_ids_input
        self._number_of_samples = len(sample_ids_input)

        # Get dataset object to return.
        res_obj = self._create_dataset_return_object(
            input_path, retrieve_sample_ids, retrieve_metadata,
            as_matrix)

        return res_obj

    def read_input_labels_as_dict(
            self, input_path: str, label_path: str,
            retrieve_metadata: Optional[bool] = False,
            retrieve_sample_ids: Optional[bool] = False,
            ) -> d_obj.DatasetReturnObjectTuple:
        """Read input from an hdf5 file and labels from another hdf5 file.

        Args:
            input_path (str): Path to input dataset.
            label_path (str): Path to label dataset.
            retrieve_metadata (bool, optional):  Whether or not to retrieve
                metadata.
            retrieve_sample_ids (bool, optional):  Whether or not to retrieve
                sample_ids.

        Returns:
            d_obj.DatasetReturnObjectTuple:
                A tuple of DatasetReturnObjects (input and labels). For both
                return objects, the variable `data` is populated while
                `data_matrix` and `data_generator` are None.
        """
        return self._read_input_labels(
            input_path, label_path, retrieve_metadata=retrieve_metadata,
            retrieve_sample_ids=retrieve_sample_ids)

    def read_input_labels_as_matrix(
            self, input_path: str, label_path: str,
            retrieve_metadata: Optional[bool] = False,
            retrieve_sample_ids: Optional[bool] = False,
            ) -> d_obj.DatasetReturnObjectTuple:
        """Read input from an hdf5 file and labels from another hdf5 file.

        Args:
            input_path (str): Path to input dataset.
            label_path (str): Path to label dataset.
            retrieve_metadata (bool, optional):  Whether or not to retrieve
                metadata.
            retrieve_sample_ids (bool, optional):  Whether or not to retrieve
                sample_ids.

        Returns:
            d_obj.DatasetReturnObjectTuple:
                A tuple of DatasetReturnObjects (input and labels). For both
                return objects, the variable `data_matrix` is populated while
                `data` and `data_generator` are None.
        """
        return self._read_input_labels(
            input_path, label_path, retrieve_metadata=retrieve_metadata,
            retrieve_sample_ids=retrieve_sample_ids, as_matrix=True)

    def _read_input_labels(
            self, input_path: str, label_path: str,
            retrieve_metadata: Optional[bool] = False,
            retrieve_sample_ids: Optional[bool] = False,
            as_matrix: Optional[bool] = False) \
            -> d_obj.DatasetReturnObjectTuple:
        """Read input from an hdf5 file and labels from another hdf5 file.

        Args:
            input_path (str): Path to input dataset.
            label_path (str): Path to label dataset.
            retrieve_metadata (bool, optional):  Whether or not to retrieve
                metadata.
            retrieve_sample_ids (bool, optional):  Whether or not to retrieve
                sample_ids.
            as_matrix  (Optional[bool]): Whether or not to stack all the nodes
                alphabetically into an n x m matrix (n: number of samples,
                m: number of nodes).

        Returns:
            d_obj.DatasetReturnObjectTuple:
                A tuple of dataset objects (input and labels) or just one
                dataset object (in case) no label path is given.
        """
        if not os.path.isfile(input_path):
            raise dce.DataFileInexistentError(
                "Dataset cannot be read in. This dataset hdf5 file does not "
                f"exist: {input_path}")
        if not os.path.isfile(label_path):
            raise dce.DataFileInexistentError(
                "Dataset cannot be read in. This dataset hdf5 file does not "
                f"exist: {label_path}")

        # Initialize a DatasetReader class.
        data_reader_input = dh.DatasetReader(input_path)
        sample_ids_input = np.array(
            data_reader_input.get_sample_ids_all(), dtype=str)
        # Initialize a DatasetReader class for the labels.
        data_reader_labels = dh.DatasetReader(label_path)
        sample_ids_labels = np.array(
            data_reader_labels.get_sample_ids_all(), dtype=str)

        # Check whether the orders of the sample ids is the same than for
        # the input dataset.
        if (sample_ids_input != sample_ids_labels).any():
            raise dce.SampleIdsInconsistentError(
                "The samples of the input dataset are not in the same "
                "order than the labels.")
        self._sample_ids = sample_ids_input
        self._n_samples = len(sample_ids_input)

        # Get dataset objects to return.
        res_obj_input = self._create_dataset_return_object(
            input_path, retrieve_sample_ids, retrieve_metadata,
            as_matrix)
        res_obj_labels = self._create_dataset_return_object(
            label_path, retrieve_sample_ids, retrieve_metadata,
            as_matrix)

        return d_obj.DatasetReturnObjectTuple(res_obj_input, res_obj_labels)

    # [BAS-759] Remove optional keyword flatten and introduce private function.
    def read_data_as_generator(
            self, input_path: str, batch_size: int,
            retrieve_metadata: Optional[bool] = False,
            flatten: bool = False
            ) -> d_obj.DatasetReturnObject:
        """Read in data from an hdf5 as a generator.

        Args:
            input_path (str): Path to input dataset hdf5 file.
            retrieve_metadata (Optional[bool], optional): Whether or not to
                retrieve metadata.
            flatten (Optional[bool], optional): Whether to flatten the input
                data into one vector.
            batch_size (Optional[int], optional): Batch size of the generator.
                Set to 1 per default.

        Yields:
            Dataset: Dataset object with variable data_generator set.
        """
        if not os.path.isfile(input_path):
            raise dce.DataFileInexistentError(
                "Dataset cannot be read in. This dataset hdf5 file does not "
                f"exist: {input_path}")

        # Initialize a DatasetReader class.
        data_reader_input = dh.DatasetReader(input_path)
        self._number_of_samples = data_reader_input.get_n_samples()

        # Initialize dataset object to return for the input.
        res_obj_input = d_obj.DatasetReturnObject()
        if retrieve_metadata:
            res_obj_input.add_metadata(
                meta_handler.get_metadata_all_nodes(input_path)
            )
            if res_obj_input.metadata == {}:
                raise dce.DatasetMetadataError(
                    "Input dataset does not contain any metadata.")
        # Get generator over samples from DatasetReader class.
        self._batch_size = batch_size
        res_obj_input.add_data_generator(
            wrapper_function=self._iterate_while_setting_sample_ids,
            batch_size=batch_size, dataset_path=input_path,
            flatten=flatten)
        return res_obj_input

    # [BAS-759] Remove optional keyword flatten and introduce private function.
    def read_input_labels_as_generator(
            self, input_path: str, label_path: str, batch_size: int,
            retrieve_metadata: Optional[bool] = False,
            flatten_input: bool = False
            ) -> d_obj.DatasetReturnObjectTuple:
        """Read in input from an hdf5 file and labels from another hdf5 file.

        Args:
            input_path (str): Path to input dataset hdf5 file.
            label_path (str): Path to label dataset hdf5 file.
            retrieve_metadata (Optional[bool], optional): Whether or not to
                retrieve metadata.
            batch_size (int): Batch size of the generator.
                Set to 1 per default.
            flatten_input (bool, optional): Whether to flatten the
                data into one node.

        Yields:
            d_obj.DatasetReturnObjectTuple:
                Tuple containing input dataset object and label dataset object.
        """
        if not os.path.isfile(input_path):
            raise dce.DataFileInexistentError(
                "Dataset cannot be read in. This dataset hdf5 file does not "
                f"exist: {input_path}")
        if not os.path.isfile(label_path):
            raise dce.DataFileInexistentError(
                "Dataset cannot be read in. This dataset hdf5 file does not "
                f"exist: {label_path}")

        # Set number of samples.
        input_reader = dh.DatasetReader(input_path)
        self._number_of_samples = input_reader.get_n_samples()

        # Initialize dataset objects to return for input and labels.
        res_obj_labels = d_obj.DatasetReturnObject()
        res_obj_input = d_obj.DatasetReturnObject()
        if retrieve_metadata:
            res_obj_input.add_metadata(
                meta_handler.get_metadata_all_nodes(input_path)
            )
            res_obj_labels.add_metadata(
                meta_handler.get_metadata_all_nodes(label_path)
            )

        # Get generator over samples from DatasetReader class.
        self._batch_size = batch_size
        res_obj_input.add_data_generator(
            wrapper_function=self._iterate_while_setting_sample_ids,
            batch_size=batch_size, dataset_path=input_path,
            flatten=flatten_input)
        res_obj_labels.add_data_generator(
            wrapper_function=self._iterate_while_checking_sample_ids,
            batch_size=batch_size, dataset_path=label_path,
            flatten=False, kwargs={"batch_size": batch_size})
        return d_obj.DatasetReturnObjectTuple(res_obj_input, res_obj_labels)

    def _iterate_while_setting_sample_ids(
            self, generator: GenOfDicts) -> GenOfDicts:
        """Iterate over a generator while setting the sample id variable.

        Args:
            generator (GenOfDicts): Input dataset generator.

        Yields:
            GenOfDicts: The input generator.

        """
        self._sample_ids = np.array([])
        for element in generator:
            self._sample_ids = np.hstack(
                (self._sample_ids, element[dh.SAMPLE_IDS]))
            element_lists = {}
            for key, value in element.items():
                element_lists[key] = value
            yield element_lists

    def _iterate_while_checking_sample_ids(
            self, generator: GenOfDicts, batch_size: int) -> GenOfDicts:
        """Iterate over a generator while comparing the sample ids to the
        member variable self._sample_ids (to test consistency between labels
        and input).

        Args:
            generator (GenOfDicts): Input dataset generator. (Generator
                of dictionaries)
                batch_size (int): Batch size of the generator.

        Yields:
            GenOfDicts: The input generator.
        """
        if self._sample_ids is None:
            raise dce.DatasetConverterInternalError(
                "The member variable 'self._sample_ids' must be set.")
        for index, element in enumerate(generator):
            if (self._sample_ids[index * batch_size: (index + 1) * batch_size]
                    != element[dh.SAMPLE_IDS]).any():
                raise dce.SampleIdsInconsistentError(
                    "The sample ids of input and label are not consistent.")
            yield element

    def write_dataset_to_hdf5(
            self, data_dict: Dict[str, np.ndarray],
            metadata_dict: Dict[str, meta_prop.AllProperties],
            output_path: str, sample_ids: np.ndarray = np.array([])) -> None:
        """Write a dataset into an hdf5 file. The data and metadata
        are in the form of dictionaries.

        Args:
            data_dict (Dict[str, np.ndarray]): Data dict with nodes as keys and
                node data arrays as values.
            metadata_dict (Dict[str, meta_prop.AllProperties]):
                Metadata dictionary.
            output_path (str): path to new hdf5 file
            sample_ids (np.ndarray): np.ndarray of string sample_ids, if not
                given, it is assumed that the samples are in the same order,
                than they were read in.

        Raises:
            DatasetOutputWriteError: Error if the output is not valid.
        """
        if not os.path.isdir(os.path.dirname(output_path)):
            raise dce.DatasetOutputWriteError(
                "Saving the dataset failed because the parent directory of "
                f"the following output file does not exist: {output_path}")

        if len(sample_ids) == 0:
            if self._sample_ids is None:
                raise dce.DatasetConverterInternalError(
                    "The member variable 'self._sample_ids' must be set.")
            sample_ids = self._sample_ids

        if len(list(data_dict.values())[0]) != len(sample_ids):
            raise dce.DatasetOutputWriteError(
                "The number of data samples does not match the number of "
                "sample ids.")
        if self._sample_ids is not None \
                and (sample_ids != np.array(self._sample_ids)).any():
            raise dce.SampleIdsInconsistentError(
                "The samples ids of the output are not consistent with "
                "the input dataset.")

        # Add data and metadata to hdf5 dataset.
        node_names = list(data_dict.keys())
        with dh.get_Dataset_writer(output_path, len(sample_ids)) as dataset:
            for node_name in node_names:
                dataset.add_data_to_node(
                    node_name,
                    self._at_least_2d(np.array(data_dict[node_name])))
                meta_handler.PropertiesWriter().write_to_ds_writer(
                    metadata_dict[node_name], node_name, dataset
                )
            dataset.add_sample_ids(sample_ids.astype(str))

            # Check whether the written dataset is valid.
        valid, msg = dh.DatasetReader(output_path).get_validation_info()
        if not valid:
            raise dce.DatasetOutputWriteError(
                "The output dataset is not valid: {}".format(msg)
                )

    def write_model_predictions(
            self, predictions: np.ndarray, output_path: str,
            sample_ids: np.ndarray = np.array([])) -> None:
        """Write model predictions to an hdf5 file.

        Args:
            predictions (np.ndarray): Predictions (single node) as numpy array.
            output_path (str): path to new hdf5 file
            sample_ids (np.ndarray): np.ndarray of string sample_ids. If not
                given, it is assumed that the predictions are in the same order
                than the data that was read in, and the sample ids from the
                read data are used.

        Raises:
            MODOutputWriteError: Error if the output is not valid.
        """
        if not os.path.isdir(os.path.dirname(output_path)):
            raise dce.MODOutputWriteError(
                "Saving the predictions failed because the parent directory "
                f"of the following output file does not exist: {output_path}")

        if len(sample_ids) == 0:
            if self._sample_ids is None:
                raise dce.DatasetConverterInternalError(
                    "The member variable 'self._sample_ids' must be set.")
            sample_ids = self._sample_ids

        if len(predictions) != len(sample_ids):
            raise dce.MODOutputWriteError(
                "The number of predictions does not match the number of "
                "sample ids.")

        # Add predictions as single node to hdf5 dataset.
        with dh.get_Dataset_writer(output_path, len(sample_ids)) as dataset:
            dataset.add_data_to_node(
                PREDICTIONS_NODE_NAME,
                self._at_least_2d(np.array(predictions)))
            dataset.add_sample_ids(sample_ids.astype(str))

            # Check whether the written dataset is valid.
        valid, msg = dh.DatasetReader(output_path).get_validation_info()
        if not valid:
            raise dce.MODOutputWriteError(
                "The output dataset is not valid: {}".format(msg)
                )

    def write_model_predictions_generator(
            self, predictions: Generator[np.ndarray, None, None],
            output_path):
        """Write model predictions from a generator.

        Args:
            predictions (Generator[np.ndarray, None, None]): Predictions as a
                generator of numpy arrays.
            output_path ([type]): Path to output hdf5 file.

        Raises:
            dce.MODOutputWriteError: Error if sample ids are not set, path
                does not exist, or output hdf5 is invalid.
        """
        if not os.path.isdir(os.path.dirname(output_path)):
            raise dce.MODOutputWriteError(
                "Saving the predictions failed because the parent directory "
                f"of the following output file does not exist: {output_path}")

        if self._number_of_samples is None:
            raise dce.MODOutputWriteError(
                "Number of samples is not set. This member function can only "
                "be used after the DatasetConverter object has been used to "
                "read in a dataset.")

        # Add predictions as single node to hdf5 dataset.
        with dh.get_Dataset_writer(output_path, self._number_of_samples) as\
                dataset:
            n_predictions = 0
            for batch in predictions:
                batch_reshaped = self._at_least_2d(batch)
                dataset.add_data_to_node(PREDICTIONS_NODE_NAME, batch_reshaped)
                if self._sample_ids is None:
                    raise dce.MODOutputWriteError(
                        "Sample ids are not set. This member function can only"
                        " be used after the DatasetConverter object has been "
                        "used to read in a dataset.")
                sample_ids_batch = self._sample_ids[
                    n_predictions:n_predictions+len(batch)]
                dataset.add_data_to_node(dh.SAMPLE_IDS, sample_ids_batch)
                n_predictions += len(batch)

            # Check whether the written dataset is valid.
        valid, msg = dh.DatasetReader(output_path).get_validation_info()
        if not valid:
            raise dce.MODOutputWriteError(
                "The output dataset is not valid: {}".format(msg)
                )

    def _at_least_2d(self, node: np.ndarray) -> np.ndarray:
        """Reshape a node such that it has at least two dimensions, as of the
        conventions defined in the doc. The first dimension is the number of
        samples and the second dimension the dimension of the node.

        Args:
            node (np.ndarray):  Input node, either a flat array or in the right
                shape already.

        Returns:
            np.ndarray: Reshaped array, such that the number of dimensions is
                at least 2.
        """
        if type(node) != np.ndarray:
            raise dce.MODOutputWriteError(
                "The input to this function must be a numpy array.")
        elif len(node.shape) == 0:
            raise dce.MODOutputWriteError(
                "The input to this function can not be a scalar. However the "
                f"shape is `{node.shape}`.")
        elif len(node) == 0:
            raise dce.MODOutputWriteError(
                "The length of the input array must be at least 1.")
        if len(node.shape) == 1:
            return np.reshape(node, (len(node), -1))
        else:
            return node

    def write_dataset_to_hdf5_generator(
            self, data_generator: GenOfDicts,
            metadata_dict: Dict[str, meta_prop.AllProperties],
            output_path: str, sample_ids: np.ndarray = np.array([])) -> None:
        """Write data to hdf5 batch wise by iterating over a generator. The
        generator contains data, metadata, and sample_ids.

        Args:
            data_generator (GenOfDicts): Generator of dictionaries. The
                dictionaries must contain the keys `data` `metadata` and what
                is defined as the variable SAMPLE_ID_KEYWORD in this file. This
                function checks in runtime, whether these keys are present and
                raises an Error if not.
            metadata_dict (dict): Metadata dict.
            output_path (str): Path to output hdf5 file.
            sample_ids (np.ndarray): Sample ids if self._sample_ids is not
                filled by reading in the dataset with the same object.

        Raises:
            DatasetOutputWriteError: Exception, if output is invalid.
        """
        if not os.path.isdir(os.path.dirname(output_path)):
            raise dce.DatasetOutputWriteError(
                "Saving the dataset failed because the parent directory of "
                f"the following output file does not exist: {output_path}")
        if self._number_of_samples is None:
            raise dce.DatasetConverterInternalError(
                "The member variable 'self._number_of_samples' must be set.")

        # Initialize DatasetWriter class.
        with dh.get_Dataset_writer(output_path, self._number_of_samples) as\
                dataset:
            # Iterate over all batches of the generator.
            for batch_index, batch in enumerate(data_generator):
                if SAMPLE_ID_KEYWORD not in batch:
                    raise dce.DatasetOutputWriteError(
                        "The generator `data_generator` must iterate over "
                        "dictionaries that contain the sample ids with the "
                        f"following key: `{SAMPLE_ID_KEYWORD}`")
                sample_ids = batch[SAMPLE_ID_KEYWORD].astype(str)
                current_index = batch_index * self._batch_size
                if self._sample_ids is not None:
                    reference_sample_ids = self._sample_ids[
                        current_index:current_index+self._batch_size]
                    if (sample_ids != reference_sample_ids).any():
                        raise dce.SampleIdsInconsistentError(
                            "The samples ids of the output are not consistent "
                            "with the input dataset.")
                batch.pop(SAMPLE_ID_KEYWORD)
                node_names = list(batch.keys())

                # Iterate over all nodes and save the data of this batch.
                for node_name in node_names:
                    dataset.add_data_to_node(
                        node_name,
                        self._at_least_2d(np.array(batch[node_name])))
                    # In the first iteration we also write metadata to the hdf5
                    # file.
                    if batch_index == 0 and metadata_dict:
                        meta_handler.PropertiesWriter().write_to_ds_writer(
                            metadata_dict[node_name], node_name, dataset
                        )
                # Add sample ids to hdf5 file.
                dataset.add_sample_ids(sample_ids)

        # Check whether written hdf5 file is valid.
        valid, msg = dh.DatasetReader(output_path).get_validation_info()
        if not valid:
            raise dce.DatasetOutputWriteError(
                "The output dataset is not valid: {}".format(msg)
                )
