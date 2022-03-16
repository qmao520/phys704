# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
Split dataset either by nodes or by sample IDs.
"""
import os
import shutil
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from modulos_utils.data_handling import data_handler as dh
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.metadata_handling import metadata_transferer as meta_trans
from modulos_utils.data_handling import data_utils
from modulos_utils.datetime_utils import datetime_utils

BATCH_SIZE_GB = 10**9


class DatasetSplitterException(Exception):
    """Raised if there is an error in the splitting of the dataset."""
    pass


class DatasetSplitter:
    """ Dataset splitter class.

    Attributes:
        orig_ds_file_path (str): path to original dataset
        orig_ds_reader (dh.DatsetReader): Dataset Handler Reader for
            original dataset
        orig_ds_node_names (List[str]): list of nodes in original dataset
    """

    def __init__(self, dataset_file_path: str):
        """ Init for Dataset Splitter class.

        Args:
            dataset_file_path (str): path to original dataset
        Raises:
            FileNotFoundError: if original dataset does not exist
        """

        self.orig_ds_file_path = dataset_file_path
        if not os.path.isfile(self.orig_ds_file_path):
            raise FileNotFoundError(f"File {self.orig_ds_file_path} "
                                    "does not exist.")

        self.orig_ds_reader = dh.DatasetReader(self.orig_ds_file_path)
        # NOTE: get_node_names also returns sample IDs.
        self.orig_ds_node_names = self.orig_ds_reader.get_node_names()
        self.batch_size = data_utils.compute_batch_size(
            self.orig_ds_file_path, self.orig_ds_reader.get_n_samples())

    @staticmethod
    def _check_dataset_valid(file_path: str) -> None:
        """ Check if a dataset is valid. Raise error
        if it is not.
        Args:
            file_path (str): Path to hdf5 dataset.

        Raises:
            DatasetNotValidError: if dataset is not valid

        Returns:
            None
        """
        is_valid, msg = dh.DatasetReader(file_path).get_validation_info()
        if not is_valid:
            raise dh.DatasetNotValidError(msg)
        return None

    def _check_nodes_subset_original(self, node_names: List[str]) \
            -> Tuple[bool, List]:
        """ Check if the nodes given in node_names are a subset of the nodes
        of the original dataset.

        Args:
            node_names (List[str]): list of nodes names

        Returns:
            bool: nodes in node_names are a subset of the original node names
                yes/no
            List: of node names missing in the original dataset
        """
        if set(node_names) <= set(self.orig_ds_node_names):
            return True, []
        missing_nodes = [k for k in node_names if k not in
                         self.orig_ds_node_names
                         ]
        return False, missing_nodes

    def _check_sample_ids_subset_original(self, sample_ids: List[str]) \
            -> Tuple[bool, List[str]]:
        """ Check if sample IDs are a subset of the samples in the original
        dataset.

        Args:
            sample_ids (List[str]): list of sample IDs

        Returns:
            bool: subset yes/no
            List: of sample IDs which are not in original dataset
        """

        orig_sample_ids = self.orig_ds_reader.get_sample_ids_all()

        if set(sample_ids) <= set(orig_sample_ids):
            return True, []
        missing_sample_ids = [k for k in sample_ids if
                              k not in orig_sample_ids
                              ]
        return False, missing_sample_ids

    def create_new_dataset_by_nodes(self, new_file_path: str,
                                    node_names: List[str]) -> None:
        """ Create a new dataset, which contains a subset of the nodes of
        the original dataset.

        Note: For each node in the new dataset,
        the metadata will be copied from the corresponding original node
        to the new node.

        Args:
            new_file_path (str): path to new dataset
            node_names (List[str]): list of nodes which are meant to be
                transferred

        Returns:
            None

        Raises:
            KeyError: if the nodes provided as input are not a subset of the
            nodes of the original dataset
        """
        # Check nodes.
        is_subset, missing_nodes = \
            self._check_nodes_subset_original(node_names)
        if not is_subset:
            raise KeyError("The nodes of the new dataset must be a subset "
                           "of the original dataset nodes. The following "
                           f"nodes are missing: {missing_nodes}"
                           )

        # Create new dataset.
        nr_samples = self.orig_ds_reader.get_n_samples()
        with dh.get_Dataset_writer(new_file_path, nr_samples) as new_ds:

            # Add sample IDs.
            for data_batch in self.orig_ds_reader.get_data_of_node_in_batches(
                    dh.SAMPLE_IDS, batch=self.batch_size
            ):
                new_ds.add_sample_ids(data_batch)

            # Add data and copy metadata.
            for node_name in node_names:

                # If node list happens to contain sample IDs, continue.
                # Sample IDs are not added again.
                if node_name == dh.SAMPLE_IDS:
                    continue
                # Get data of node in batches and add data.
                for data_batch in self.orig_ds_reader.\
                        get_data_of_node_in_batches(
                            node_name, batch=self.batch_size
                        ):
                    new_ds.add_data_to_node(node_name, data_batch)
                # Get metadata of node. Test if original dataset contains
                # metadata.
                # Note: here we do not use the Metadata Transferer. Since the
                # data itself remains untouched, the computable properties of
                # each node can be copied to the new dataset.
                try:
                    meta_obj = meta_handler.PropertiesReader().\
                        read_from_ds_reader(
                        node_name, self.orig_ds_reader
                    )
                    # Add metadata.
                    meta_handler.PropertiesWriter().write_to_ds_writer(
                        meta_obj, node_name, new_ds
                    )
                except dh.MetaDataDoesNotExistError:
                    # Original node had no metadata to copy.
                    pass

            # Check if there is datetime data and copy if there is.
            date_time_nodes = self.orig_ds_reader.get_datetime_node_names()
            for dt_node in date_time_nodes:
                # Check if the datetime stamp `dt_node` is in this split.
                if not any(name in datetime_utils.get_datetime_node_names(
                        dt_node) for name in node_names):
                    continue
                dt = self.orig_ds_reader.get_datetime_data_of_node(dt_node)
                new_ds.add_datetime(dt_node, dt)
                # Copy the metadata of the datetime node.
                try:
                    metadata = meta_handler.PropertiesReader().\
                        read_from_ds_reader(dt_node, self.orig_ds_reader)
                    meta_handler.PropertiesWriter().write_to_ds_writer(
                        metadata, dt_node, new_ds)
                except dh.MetaDataDoesNotExistError:
                    pass
        # Check if dataset is valid.
        self._check_dataset_valid(new_file_path)
        return None

    def split_dataset_by_nodes(self, new_file_path1: str,
                               new_file_path2: str,
                               node_names1: List[str]) -> None:
        """ Split a dataset by nodes. All nodes of the original dataset
        are either transferred to the new file1 or the new file2.

        Note: For each node in the new dataset, the metadata will be copied
        from the corresponding original node to the new node.

        All nodes not listed in node_names1 will be added to new_file2.

        Args:
            new_file_path1 (str): path to new file 1
            new_file_path2 (str): path to new file 2
            node_names1 (List[str]): list of nodes which are meant to be
                saved in the new file 1

        Returns:
            None

        Raises:
            KeyError: if nodes are not a subset of the original dataset
        """
        # Check input.
        # Make sure input nodes are a subset of the original dataset.
        is_subset, missing_nodes = \
            self._check_nodes_subset_original(node_names1)
        if not is_subset:
            raise KeyError("The nodes of the new datasets must be a subset "
                           "of the original dataset nodes. The following "
                           "nodes are missing from the original "
                           f"dataset: {missing_nodes}"
                           )
        # Determine the nodes which will be part of the second dataset.
        node_names2 = [n for n in self.orig_ds_node_names
                       if n not in node_names1]

        # Create two new datasets.
        self.create_new_dataset_by_nodes(new_file_path1, node_names1)
        self.create_new_dataset_by_nodes(new_file_path2, node_names2)

        return None

    def split_off_example_samples(self, new_file_path: str,
                                  sample_ids: List[str],
                                  dshf_path: str) -> None:
        """ Save a few samples of the original dataset in a new dataset file.
        This is then used for the example samples in the online and batch
        client.

        Note: the metadata will be transferred, i.e. computable properties will
        be set to their default values. Non-computable properties are copied.

        Args:
            new_file_path (str): path to new dataset file
            sample_ids List[str]: samples which are meant to be added to new
                dataset
            dshf_path (str): Path to the dshf file.

        Returns:
            None

        Raises:
            KeyError: if the samples are not part of the original dataset
        """
        dshf = dshf_handler.DSHFHandler(dshf_path)
        with dh.get_Dataset_writer(
                new_file_path, n_samples=len(sample_ids)) as new_ds:
            # Add data for the samples.
            try:
                sample_data = self.orig_ds_reader.get_data_for_sample_ids(
                    sample_ids
                )
                node_names = self.orig_ds_node_names
            except dh.DatasetSampleIdDoesNotExistError:
                raise KeyError(f"Some samples of {sample_ids} do not exist in "
                               "the original dataset.")

            # Check if there are datetime nodes. If yes, remove the
            # generated features and add the original ones.
            datetime_nodes = self.orig_ds_reader.get_datetime_node_names()
            if datetime_nodes:
                sample_datetime = self.orig_ds_reader.\
                    get_datetime_for_sample_ids(sample_ids)
                # Stringify the datetime stamps using the upload format.
                dt_formats = dshf.datetime_format
                sample_datetime = {node: np.array(pd.to_datetime(
                    sample_datetime[node]).strftime(
                        dt_formats[node])).astype("U")
                    if node in dt_formats
                    else np.datetime_as_string(sample_datetime[node])
                    for node in sample_datetime}
                # Get all possible datetime features to delete from sample.
                new_datetime_features = []
                for name in datetime_nodes:
                    new_datetime_features.extend(
                        datetime_utils.get_datetime_node_names(name))
                for new_feature in new_datetime_features:
                    # Remove them if they exist.
                    sample_data.pop(new_feature, None)
                    if new_feature in node_names:
                        node_names.remove(new_feature)
                sample_data.update(sample_datetime)
                node_names.extend(list(sample_datetime.keys()))

            new_ds.add_samples(sample_data)
            # Transfer and adjust metadata for datetime samples.
            for dt_node in datetime_nodes:
                try:
                    metadata = meta_trans.NodeTransferer.from_ds(
                        dt_node, self.orig_ds_file_path).get_obj()
                except dh.MetaDataDoesNotExistError:
                    metadata = meta_handler.meta_prop.AllProperties()
                metadata.upload_node_dim.set([1])
                metadata.node_dim.set([1])
                metadata.upload_node_type.set("datetime")
                metadata.node_type.set("datetime")
                meta_handler.PropertiesWriter().write_to_ds_writer(
                    metadata, dt_node, new_ds)

            # Transfer metadata for rest of the nodes.
            # Note: computable metadata properties will be
            # set to their default values.
            # NOTE: Clean up this mess in BAS-994.
            for node in [node for node in node_names
                         if node not in datetime_nodes]:
                try:
                    meta_obj = meta_trans.NodeTransferer.from_ds(
                        node, self.orig_ds_file_path).get_obj()
                    if not np.isnan(
                            meta_obj.upload_nr_unique_values.get()):
                        meta_obj.nr_unique_values.set(
                            meta_obj.upload_nr_unique_values.get())
                    meta_handler.PropertiesWriter().write_to_ds_writer(
                        meta_obj, node, new_ds)
                except dh.MetaDataDoesNotExistError:
                    pass

        # Check if dataset is valid.
        self._check_dataset_valid(new_file_path)
        return None

    def split_dataset_by_sample_ids(self, new_file_path1: str,
                                    new_file_path2: str,
                                    sample_ids1: List[str],
                                    sample_ids2: List[str]) -> None:
        """ Split a dataset by sample IDs. The samples listed in `sample_ids1`
        are transferred to the new file1, the ones listed in `sample_ids2` to
        file2.

        Note: For each node in the new dataset, the metadata will be
        transferred from the corresponding original node to the new node.
        Computable properties are copied, non-computable properties are set
        to their default values.

        Args:
            new_file_path1 (str): path to new file 1
            new_file_path2 (str): path to new file 2
            sample_ids1 (List[str]): list of samples that are meant to be
                copied to new file 1 in the given order
            sample_ids2 (List[str]): list of samples that are meant to be
                copied to new file 2 in the given order

        Returns:
            None

        Raises:
            KeyError: if sample_ids1 or sample_ids2 contain samples that are
            not part of the original dataset.
        """
        # Ensure that the sample IDs are a subset of the original dataset.
        is_subset, missing_sample_ids = \
            self._check_sample_ids_subset_original(sample_ids1)
        is_subset2, missing_sample_ids2 = \
            self._check_sample_ids_subset_original(sample_ids2)
        if not is_subset or not is_subset2:
            missing_sample_ids.extend(missing_sample_ids2)
            raise KeyError("The following samples are missing in the original "
                           f"dataset: {missing_sample_ids}"
                           )

        # Create new datasets.
        with dh.get_Dataset_writer(
            new_file_path1, len(sample_ids1)
        ) as new_ds1, dh.get_Dataset_writer(
            new_file_path2, len(sample_ids2)
        ) as new_ds2:
            for sample_ids, writer in zip(
                    (sample_ids1, sample_ids2), (new_ds1, new_ds2)):
                training_chunks = [
                    sample_ids[i:i+self.batch_size]
                    for i in range(0, len(sample_ids), self.batch_size)]
                for training_chunk in training_chunks:
                    for node_name in self.orig_ds_reader.get_node_names():
                        data_list = self.orig_ds_reader.\
                            get_data_from_node_for_ids(
                                training_chunk, node_name)
                        data_chunk = np.array(data_list)
                        writer.add_data_to_node(node_name, data_chunk)

            # Get the datetime nodes if they exist and save them.
            if self.orig_ds_reader.get_datetime_node_names() != []:
                datetime_data1 = self.orig_ds_reader.\
                    get_datetime_for_sample_ids(sample_ids1)
                datetime_data2 = self.orig_ds_reader.\
                    get_datetime_for_sample_ids(sample_ids2)
                for node in datetime_data1:
                    new_ds1.add_datetime(node, datetime_data1[node])
                    new_ds2.add_datetime(node, datetime_data2[node])

            # Transfer metadata. Note: computable metadata properties will be
            # set to their default values. Check if metadata exists.
            for node in (self.orig_ds_node_names +
                         self.orig_ds_reader.get_datetime_node_names()):
                try:
                    meta_trans.NodeTransferer.from_ds(
                        node, self.orig_ds_file_path
                    ).save_new_node_with_ds_witer(node, new_ds1)

                    meta_trans.NodeTransferer.from_ds(
                        node, self.orig_ds_file_path
                    ).save_new_node_with_ds_witer(node, new_ds2)

                except dh.MetaDataDoesNotExistError:
                    pass
        # Check if datasets are valid.
        self._check_dataset_valid(new_file_path1)
        self._check_dataset_valid(new_file_path2)
        return None


def generate_split_strategy_datasets(
        train_ids: List[str], val_ids: List[str], path_dict: Dict) -> None:
    """Generate the input, label and transformed label datasets split into
    training and validation.

    Args:
        train_ids (List[str]): Sample ids which are used for the training set.
        val_ids (List[str]): Sample ids which are used for the validation set.
        path_dict (Dict): Dictionary containing the paths of all the datasets.
        dataset_filename (str): Default name of the datasets.
    """
    # Split the input dataset into training and validation.
    tr_in_dirname = os.path.dirname(path_dict["training"]["input"])
    if not os.path.isdir(tr_in_dirname):
        os.makedirs(tr_in_dirname)
    val_in_dirname = os.path.dirname(path_dict["validation"]["input"])
    if not os.path.isdir(val_in_dirname):
        os.makedirs(val_in_dirname)
    dataset_splitter_input = DatasetSplitter(path_dict["input"])
    dataset_splitter_input.split_dataset_by_sample_ids(
        new_file_path1=path_dict["training"]["input"],
        new_file_path2=path_dict["validation"]["input"],
        sample_ids1=train_ids, sample_ids2=val_ids)

    # Split the label dataset into training and validation.
    tr_label_dirname = os.path.dirname(path_dict["training"]["label"])
    if not os.path.isdir(tr_label_dirname):
        os.makedirs(tr_label_dirname)
    val_label_dirname = os.path.dirname(path_dict["validation"]["label"])
    if not os.path.isdir(val_label_dirname):
        os.makedirs(val_label_dirname)
    dataset_splitter_labels = DatasetSplitter(path_dict["label"])
    dataset_splitter_labels.split_dataset_by_sample_ids(
        new_file_path1=path_dict["training"]["label"],
        new_file_path2=path_dict["validation"]["label"],
        sample_ids1=train_ids,
        sample_ids2=val_ids)

    # Split the transformed label dataset into training and validation.
    tr_trans_label_dir = os.path.dirname(
        path_dict["training"]["transformed_labels"])
    if not os.path.isdir(tr_trans_label_dir):
        os.makedirs(tr_trans_label_dir)
    dataset_splitter_labels = DatasetSplitter(path_dict["transformed_labels"])
    tmp_dir = os.path.join(tr_trans_label_dir, "tmp")
    os.makedirs(tmp_dir)
    dataset_splitter_labels.split_dataset_by_sample_ids(
        new_file_path1=path_dict["training"]["transformed_labels"],
        new_file_path2=os.path.join(tmp_dir, "data.hdf5"),
        sample_ids1=train_ids,
        sample_ids2=val_ids)
    shutil.rmtree(tmp_dir)
    return None
