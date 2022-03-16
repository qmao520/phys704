# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the code used for the datetime in assembling the solution
folder."""

import logging
import pandas as pd
from typing import List, Dict

from modulos_utils.datetime_utils import datetime_utils
from modulos_utils.solution_utils import utils as su
from modulos_utils.dshf_handler import dshf_handler


def get_datetime_node_names(label_name: str, dshf_path: str) -> List[str]:
    """Get the list of datetime nodes.

    Args:
        label_name (str): Name of the label node to exclude.
        dshf_path (str): Path to the history file of the training data.

    Returns:
        List[str]: List of node names containing datetime stamps.
    """
    node_names_upload = su.get_input_node_names_from_dshf(
        dshf_path, label_name,  generated=False)
    node_names_internal = su.get_input_node_names_from_dshf(
        dshf_path, label_name, generated=True)
    nodes_only_upload = [name for name in node_names_upload
                         if name not in node_names_internal]
    datetime_nodes = [name for name in nodes_only_upload
                      if any(feature in node_names_internal for feature
                             in datetime_utils.get_datetime_node_names(name))]
    return datetime_nodes


def add_datetime_features(
        sample_dict: Dict, node_names_input: List[str],
        dshf_path: str) -> None:
    """Compute the datetime features of a sample.

    Args:
        sample_dict (Dict): Dictionary of one sample.
        node_names_input (List[str]): Node names of the nodes that are
            downloaded from the platform and included in the solution, i.e.
            the names of the nodes that are fed to the feature extractor.
        dshf_path (str): Path to the history file of the training data.

    Raises:
        Exception: Failing to parse datetime.
    """
    # Note that we assume that all generated nodes are datetime features. When
    # we introduce other generated nodes we will need a lookup (probably in
    # the DSHF) that tells us which nodes to generate and with which
    # function/module.
    dshf = dshf_handler.DSHFHandler(dshf_path)
    parents_of_generated = [dshf.current_to_upload_name[g]
                            for g in dshf.generated_nodes]
    time_nodes = [k for k in sample_dict.keys() if k in parents_of_generated]
    is_single_sample = (type(list(sample_dict.values())[0]) is not list)
    dt_formats = dshf.datetime_format
    for time_node in time_nodes:
        if is_single_sample:
            time_col = pd.DataFrame({time_node: [sample_dict[time_node]]})
        else:
            time_col = pd.DataFrame({time_node: sample_dict[time_node]})
        try:
            if time_node in dt_formats:
                dt_format = dt_formats[time_node]
                time_col[time_node] = pd.to_datetime(time_col[time_node],
                                                     format=dt_format)
            else:
                datetime_utils.parse_datetime_in_dataframe_column(
                    time_col, time_node)
            time_values = list(time_col[time_node].values)
            new_features = datetime_utils.\
                compute_features_from_timestamp(pd.to_datetime(time_values))
        except ValueError as err:
            msg = f"Could not read datetime node {time_node}!\n{err}"
            logging.error(msg)
            raise ValueError(msg)

        for feature in new_features:
            feature_name = datetime_utils.get_datetime_node_name_for_feature(
                time_node, feature)
            if feature_name in node_names_input:
                if is_single_sample:
                    sample_dict.update(
                        {feature_name: new_features[feature][0]})
                else:
                    sample_dict.update(
                        {feature_name: new_features[feature]})
        sample_dict.pop(time_node)
