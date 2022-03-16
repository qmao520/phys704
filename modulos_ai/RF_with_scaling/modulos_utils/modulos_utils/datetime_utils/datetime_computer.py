# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the class used to generate the datetime features."""
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple

from modulos_utils.data_handling import data_handler as dh
from modulos_utils.datetime_utils import datetime_utils as dt_utils
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.metadata_handling import metadata_handler as meta_handler


class DatetimeComputer():
    """Compute and save the datetime features.

    Attributes:
        hdf5_file (str): Path to the internal dataset file.
        dataset_folder (str): Path to the folder containing the internal
            dataset file and the history file.
        nr_samples (int): Number of samples in the internal dataset.
        dshf_read_path (str): Path to the history file that is used to read
            from.
        dshf_write_path (str): Path to the history file that is written to.
    """

    def __init__(
            self, hdf5_file: str, dshf_read_path: Optional[str] = None,
            dshf_write_path: Optional[str] = None):
        """Class initialization. It will read the internal dataset to determine
        the number of samples in it.

        Args:
            hdf5_file (str): Path to the internal dataset.
            dshf_read_path (str): Path to the dshf file to read from.
            dshf_write_path (str): Path to the dshf file to write to. Note
                that only in the solution this path differs from
                `dshf_read_path`.
        """
        self.hdf5_file = hdf5_file
        self.nr_samples = dh.DatasetReader(hdf5_file).get_n_samples()
        self.dataset_folder = os.path.dirname(hdf5_file)
        self.dshf_read_path = dshf_read_path or os.path.join(
            self.dataset_folder, dshf_handler.DSHF_FILE_NAME)
        self.dshf_write_path = dshf_write_path or self.dshf_read_path
        return None

    def _get_generated_datetime_feature_names(
            self, is_solution: bool) -> Dict[str, List[str]]:
        """Get a dictionary the contains lists of datetime features for each
        node that is of type datetime (according to the dshf).

        Args:
            is_solution (bool): Whether this is used in the solution, (i.e.
                the datetime features can be read from the dshf) or on the
                platform (i.e. the datetime features are written into the
                dshf).

        Returns:
            Dict[str, List[str]]: Resulting dictionary that contains the
                datetime features for each datetime node.
        """
        dshf = dshf_handler.DSHFHandler(self.dshf_read_path)
        nodewise_features: Dict[str, List[str]] = {}
        if is_solution:
            for n in dshf.generated_nodes:
                nodewise_features.setdefault(
                    dshf.current_to_upload_name[n], []).append(n)
        else:
            hdf5_node_names = dh.DatasetReader(self.hdf5_file).get_node_names()
            hdf5_node_names = [n for n in hdf5_node_names
                               if n != dh.SAMPLE_IDS]
            for node in hdf5_node_names:
                if node in dshf.node_type \
                        and dshf.node_type[node] == "datetime":
                    nodewise_features[node] = \
                        dt_utils.get_datetime_node_names(node)
        return nodewise_features

    def get_generated_node_types(
            self, generated_nodes: List[str]) -> Dict[str, str]:
        """Get the node type of each generated datetime node.

        Args:
            generated_nodes (List[str]): List of generated datetime nodes.

        Returns:
            Dict[str, str]: Dictionary with schema node type for each
                generated datetime feature.
        """
        node_type = {}
        for n in generated_nodes:
            feat = dt_utils.split_datetime_feature_name(n)[1]
            is_cat = dt_utils.TIME_FEATURES_CAT[feat] != []
            node_type[n] = "categorical" if is_cat else "numerical"
        return node_type

    def _update_dshf(
            self, datetime_features_nodewise: Dict[str, List[str]],
            dt_formats: Dict[str, str],
            is_solution: bool) -> None:
        """Update the dshf after saving the generated datetime features and
        removing the parent nodes.

        Args:
            datetime_features_nodewise (Dict[str, List[str]]): Dictionary
                with generated datetime features for each datetime parent
                node.
            dt_formats (Dict[str, str]): The string format of the original
                datetime nodes.
            is_solution (bool): Whether the datetime computer is being used in
                the solution or not.
        """
        nodes_added = [feat for v in datetime_features_nodewise.values()
                       for feat in v]

        # Get new node type dict for the DSFH.
        new_node_type = self.get_generated_node_types(nodes_added)

        # Get the translation dict between generated datetime features and
        # original parent node.
        feature_to_parent_node = {
            feat: k for k, v in datetime_features_nodewise.items()
            for feat in v
        }

        # Update the dshf.
        with dshf_handler.get_dshf_writer(self.dshf_write_path) as dshf_writer:
            dshf_writer.add_to_dshf(
                {dshf_handler.DSHFKeys.current_nodes: nodes_added,
                 dshf_handler.DSHFKeys.node_type: new_node_type,
                 dshf_handler.DSHFKeys.current_to_upload_name:
                    feature_to_parent_node,
                 dshf_handler.DSHFKeys.datetime_format: dt_formats},
                dshf_handler.EntryTypes.nodes_generated,
                description="Generate datetime features.")
            if not is_solution:
                dshf_writer.add_to_dshf(
                    {dshf_handler.DSHFKeys.removed_upload_nodes:
                        list(datetime_features_nodewise.keys())},
                    dshf_handler.EntryTypes.nodes_removed,
                    description="Remove datetime parent nodes.")
        return None

    def compute_and_save_subnodes(self, is_solution: bool = False) -> \
            Dict[str, str]:
        """Read the history file to get the information which datetime features
        have to be created from which nodes. Then iterate over these nodes,
        compute the datetime features and save them. Also move the nodes to the
        datetime group in the internal dataset and remove them from the data
        group.

        Args:
            is_solution (bool): Set to true, if it is used in the solution.
                Defaults to False.

        Returns:
            Dict[str, str]: The datetime format strings of the parsed nodes.
        """
        # Get the datetime features for all datetime nodes.
        datetime_features_nodewise = \
            self._get_generated_datetime_feature_names(is_solution)

        # Get the datetime format strings. This is empty if it is not given in
        # the dssf. Then we have to compute it while opening the data.
        dshf = dshf_handler.DSHFHandler(self.dshf_read_path)
        dt_formats: Dict[str, str] = dshf.datetime_format

        # Generate the datetime features and remove the parent nodes (i.e.
        # delete it from the main group and move it to the datetime group in
        # the data handler.)
        with dh.get_Dataset_writer(self.hdf5_file, self.nr_samples) as writer:
            for dt_node, dt_values in datetime_features_nodewise.items():
                try:
                    # The function _save_datetime_get_format saves the features
                    # and (re)moves the parent node in the hdf5 file.
                    dt_formats[dt_node] = self._save_datetime_get_format(
                        dt_node, dt_values, writer, is_solution, dt_formats)
                except dh.DatasetNodeDoesNotExistError:
                    # Is raised, if it is used as label. We check for
                    # completeness at the end, so we do not check further.
                    continue

                # (Re)move the parent node inside the hdf5 file.
                writer.remove_node(dt_node)

        # Write the updated information into the DSHF.
        self._update_dshf(datetime_features_nodewise, dt_formats, is_solution)
        return dt_formats

    def _save_datetime_get_format(
            self, name: str, sub_features: List[str],
            h5_writer: dh.DatasetWriter,
            is_solution: bool, dt_formats: Dict[str, str]) -> str:
        """Read the column with the time stamp, create new nodes and save them.

        Args:
            name (str): Name of the column or component containing the
                timestamp.
            sub_features (str): List of feature names to generate.
            h5_writer (dh.DatasetWriter): h5 datahandler writer instance.
            is_solution (bool): Whether the datetime computer is being used in
                the solution or not.
            dt_formats (Dict[str, str]): The datetime format strings for all
                datetime nodes. It is given for the solution and if it is
                specified in the dssf. If not, it will be inferred.

        Returns:
            str: The datetime format string of the saved datetime.
        """
        timestamps, dt_format = self._get_time_stamp(name, dt_formats)

        # Add timestamp to internal dataset (in datetime group).
        h5_writer.add_datetime(name, np.array(timestamps))
        h5_writer.copy_metadata_to_datetime(name)

        # Compute and add new features from timestamp to internal dataset.
        df_timestamps = pd.to_datetime(timestamps)
        new_features = dt_utils.compute_features_from_timestamp(
            df_timestamps)
        for subfeature, values in new_features.items():
            node_name = dt_utils.get_datetime_node_name_for_feature(
                name, subfeature)
            if node_name not in sub_features:
                continue
            h5_writer.add_data_to_node(node_name, np.array(values))
            if not is_solution:
                empty_meta = meta_prop.AllProperties()
                meta_handler.PropertiesWriter().write_to_ds_writer(
                    empty_meta, node_name, h5_writer
                )
        return dt_format

    def _get_time_stamp(self, name: str, dt_formats: Dict[str, str]) -> \
            Tuple[List[np.datetime64], str]:
        """Get the time stamp from a column or component using the reader.
        Args:
            name (str): Name of the column or component containing the
                timestamp.
            dt_formats (Dict[str, str]): The datetime format strings for all
                nodes.
        Returns:
            Tuple[List[np.datetime64], str]: List of all time stamps and the
                datetime format string.
        """
        reader = dh.DatasetReader(self.hdf5_file)
        data = reader.get_data_of_node_all(name)
        df = pd.DataFrame({name: data.flatten()})
        # If the format is given in the dt_formats, we use this datetime format
        # string to read it in.
        if name in dt_formats:
            dt_format = dt_formats[name]
            try:
                df[name] = pd.to_datetime(
                    df[name], format=dt_format)
            except ValueError:
                msg = (f"The given datetime format string (`{dt_format}`) is "
                       "not correct.")
                _, format = dt_utils.check_and_infer_format(df, name)
                if format != "":
                    msg += f" The format `{format}` would be valid."
                # Should we raise an error here or a warning?
                raise ValueError(msg)
        else:
            dt_format = dt_utils.parse_datetime_in_dataframe_column(df, name)
        return list(df[name].values), dt_format
