# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the code used preparing the data for the online client.
At the moment this only contains the datetime preparation."""
import os
from typing import Dict, List

from modulos_utils.solution_utils import datetime
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.data_handling import data_handler as dh
from modulos_utils.solution_utils import utils
from modulos_utils.datetime_utils import datetime_computer as dt_computer
from modulos_utils.solution_utils import utils as su
from modulos_utils.sliding_window import utils as sw_utils


def preprocess_dict(
        sample_dict: Dict, input_node_names: List[str], src_dir: str,
        ignore_sample_ids: bool) -> Dict:
    """Preprocess the data. At the moment only compute the datetime features.

    Args:
        sample_dict (Dict): sample to compute
        input_node_names (List[str]): Node names of the input nodes, i.e. the
            nodes present in the input metadata that is downloaded from the
            platform and included in the solution, i.e. the names
            of the nodes that are fed to the feature extractor.
        src_dir (str): Path to the src directory of the solution.
        ignore_sample_ids: Wether to ignore sample ids or not.
    Returns:
        Dict: Copy of sample_dict with the datetime features added.
    """
    # Add the datetime features to the sample.
    dshf_path = os.path.join(
        src_dir, "dataset_history", dshf_handler.DSHF_FILE_NAME)
    new_dict = dict(sample_dict)

    # Replace the original sample_id name with the internal sample_id name.
    dshf = dshf_handler.DSHFHandler(dshf_path)
    old_name = utils.get_sample_id_column_name(dshf=dshf)
    new_name = dh.SAMPLE_IDS
    if old_name not in new_dict:
        raise KeyError(f"Sample id key `{old_name}` is missing.")
    sample_ids = new_dict[old_name]
    if ignore_sample_ids:
        new_sample_ids = list(range(len(sample_ids)))
        if type(sample_ids[0]) is str:
            sample_ids = [str(s) for s in new_sample_ids]
        else:
            sample_ids = new_sample_ids
    new_dict[new_name] = sample_ids
    del new_dict[old_name]

    # The lagged nodes are suffixed with the string __shifted_by_. We
    # manually split this off, to get the original name, but this should
    # eventually be done by the dshf: TODO: PRB-422. Note that the splitting
    # is not dangerous, since we reject column names or component names (for
    # single node components), that contain this string, in the upload.
    suffix = sw_utils.GENERATED_NODES_SUFFIX
    input_node_names = [n.split(suffix)[0] if suffix in n else n
                        for n in input_node_names]
    datetime.add_datetime_features(new_dict, input_node_names, dshf_path)
    return new_dict


def preprocess_forecast_input_dict(
        sample_dict: Dict, input_node_names: List[str], label_node_name: str,
        src_dir: str, ignore_sample_ids: bool) -> Dict:
    """Preprocess the input data dictionary of the forecast client.
    It calls the preprocess function of the online client (that adds generated
    datetime features) ands a dummy value for the label to make sure that
    all nodes have equal length (see comment in the code below).

    Args:
        sample_dict (Dict): sample to compute
        input_node_names (List[str]): Node names of the input nodes, i.e. the
            nodes present in the input metadata that is downloaded from the
            platform and included in the solution, i.e. the names
            of the nodes that are fed to the feature extractor.
        label_node_name (str): Node name of the label.
        src_dir (str): Path to the src directory of the solution.
        ignore_sample_ids: Wether to ignore sample ids or not.
    Returns:
        Dict: Copy of sample_dict with the datetime features added.
    """
    preprocessed_dict = preprocess_dict(
        sample_dict, input_node_names, src_dir, ignore_sample_ids)

    # The label node must be included in the forecast client, however only
    # up to t-1 since we are predicting time step t. However, the dictionary
    # is temporarily saved in an hdf5 file in the forecast client. To make
    # sure the hdf5 file saving does not fail du to different lengths of the
    # nodes, we add a dummy value for the label at time step t. The value
    # is a dummy value because it is not used anywhere: The window sliding,
    # that is applied to the hdf5 file, only takes up to t-1 values of the
    # label and therefore removes the added value.
    original_label_name = label_node_name.split(
        sw_utils.GENERATED_NODES_SUFFIX)[0].split(
            sw_utils.MULTI_OUTPUT_LABEL_SUFFIX)[0]
    if len(preprocessed_dict[original_label_name]) == len(
                preprocessed_dict[dh.SAMPLE_IDS]) - 1:
        preprocessed_dict[original_label_name].append(
            preprocessed_dict[original_label_name][-1])
    return preprocessed_dict


def preprocess_hdf5(
        hdf5_path: str, dshf_path: str, tmp_dshf_file: str,
        tmp_dirs=List[str], verbose: bool = True,
        keep_tmp: bool = False) -> None:
    """Apply the same preprocessing steps, that were performed on-platform, to
    an hdf5 file.

    Args:
        hdf5_path (str): Path to the hdf5 file that is modified.
        dshf_path (str): Path to the dataset history file that contains the
            information of which preprocessing steps were preformed on the
            platform.
        tmp_dshf_file (str): Path to temporary dshf file where steps, that
            are performed to the new dataset, are logged.
    """
    try:
        # Generate all datetime nodes that were generated on-platform.
        dt_comp = dt_computer.DatetimeComputer(
            hdf5_path, dshf_read_path=dshf_path, dshf_write_path=tmp_dshf_file)
        dt_comp.compute_and_save_subnodes(is_solution=True)
    except Exception as e:
        err_empty_trimmed = ValueError(
            "Error while generating datetime features for the input dataset.")
        su.exit_with_error(tmp_dirs, keep_tmp, err_empty_trimmed,
                           e, verbose)

    # Note that we don't have to replicate the removal of zero variance nodes,
    # since these are prevented from being saved: In the function
    # 'check_and_convert_dataset', we retrieve the node names from the
    # downloaded metadata, i.e. the removed nodes are not included in the
    # nodes to be saved.
    return None
