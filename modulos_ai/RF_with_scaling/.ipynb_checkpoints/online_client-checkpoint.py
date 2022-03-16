# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.

import argparse
import json
import os
import shutil
from typing import Any, Dict

from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.solution_utils import utils as su
from modulos_utils.solution_utils import preprocessing


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODULES_DIR = os.path.join(BASE_DIR, "src/modules")
TMP_DIR = os.path.join(BASE_DIR, "tmp_data_dir")
METADATA_PATH = os.path.join(SRC_DIR, "metadata/input_metadata.bin")

# Simple Example, feel free to use your data here.
SAMPLE_DICT: Dict[str, Any] = {'gi': 1.2780000000000022,
                               'gk': 4.847000000000001,
                               'gr': 0.7530000000000001,
                               'gw1': 6.8450000000000015,
                               'gw2': 7.9849999999999985,
                               'gz': 1.7840000000000025,
                               'ij': 1.6529999999999987,
                               'ik': 3.5689999999999986,
                               'iw1': 5.5669999999999975,
                               'iw2': 6.706999999999997,
                               'iz': 0.5060000000000002,
                               'jw1': 3.914,
                               'jw2': 5.053999999999999,
                               'kw1': 1.998,
                               'kw2': 3.137999999999998,
                               'ri': 0.5250000000000021,
                               'rw1': 6.0920000000000005,
                               'rw2': 7.231999999999997,
                               'rz': 1.0310000000000024,
                               'sample_ids_generated': '210',
                               'ug': 1.4840000000000049,
                               'ui': 2.7620000000000084,
                               'uj': 4.415000000000006,
                               'uk': 6.331000000000008,
                               'ur': 2.237000000000005,
                               'uw1': 8.329000000000006,
                               'uw2': 9.469000000000005,
                               'uz': 3.268000000000008,
                               'w1w2': 1.1399999999999988,
                               'zj': 1.1469999999999985,
                               'zk': 3.062999999999999,
                               'zw1': 5.060999999999998,
                               'zw2': 6.200999999999998}


def main(sample_dict: dict, tmp_dir: str, verbose: bool = False,
         keep_tmp: bool = False, ignore_sample_ids=False) -> dict:
    """Main function of the online client. Takes a sample as input and returns
    the predictions.

    Args:
        sample_dict (dict): Sample for which to predict.
        tmp_dir (str): Path to temporary data dir.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep the temporary data.
        ignore_sample_ids (bool = False): Whether to require unique sample_ids.

    Returns:
        dict: Dictionary containing prediction for this sample.
    """

    # Preprocess the data.
    input_node_names = list(meta_utils.MetadataDumper().load_all_nodes(
        METADATA_PATH).keys())
    prep_sample_dict = preprocessing.preprocess_dict(
        sample_dict, input_node_names, SRC_DIR, ignore_sample_ids)
    # Check whether required nodes are present. This raises an Exception if one
    # node is missing. Additional nodes are ignored.
    su.check_input_node_names(prep_sample_dict, METADATA_PATH, [tmp_dir],
                              verbose=verbose, keep_tmp=keep_tmp)

    # Create tmp data directory
    su.create_tmp_dir(tmp_dir, verbose=verbose, keep_tmp=keep_tmp)

    # Save sample dict for the sake of reproducibility.
    with open(os.path.join(tmp_dir, "sample.json"), "w") as f:
        json.dump(prep_sample_dict, f)

    # Create HDF5 version of the dataset. (TO BE REMOVED)
    prediction_input = su.prepare_sample_for_prediction(
        prep_sample_dict, BASE_DIR, [tmp_dir], verbose=verbose,
        keep_tmp=keep_tmp)
    sample_hdf5_path = os.path.join(tmp_dir, "sample.hdf5")
    su.convert_sample_to_hdf5(prediction_input, sample_hdf5_path, [tmp_dir],
                              verbose=verbose, keep_tmp=keep_tmp)

    # Run Feature Extractor.
    su.run_feature_extractor(sample_hdf5_path, tmp_dir, SRC_DIR, [tmp_dir],
                             verbose=verbose, keep_tmp=keep_tmp)

    # Run Model.
    su.run_model(SRC_DIR, tmp_dir, tmp_dir, [tmp_dir],
                 verbose=verbose, keep_tmp=keep_tmp)

    # Read Predictions (TO BE REMOVED)
    predictions_path = os.path.join(tmp_dir, "predictions.hdf5")
    predictions_raw = su.read_predictions(predictions_path, [tmp_dir],
                                          verbose=verbose, keep_tmp=keep_tmp)

    # Postprocessing the predictions.
    predictions = su.post_process(predictions_raw, SRC_DIR)

    # Remove the temporary data folder.
    if (not keep_tmp) and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online client.")
    parser.add_argument("--verbose", action="store_true",
                        help="Use this flag to more verbose error messages.")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="If this flag is used, the batch client does "
                        "not delete the temporary data.")
    parser.add_argument("--debug", action="store_true",
                        help="Use this flag to run the batch client in debug "
                        "mode. This is equivalent to activating both the "
                        "flags '--verbose' and '--keep-tmp'.")
    args = parser.parse_args()

    # Run Main Script.
    verbose_flag = args.verbose or args.debug
    keep_tmp_flag = args.keep_tmp or args.debug
    prediction = main(SAMPLE_DICT, TMP_DIR, verbose_flag, keep_tmp_flag)
    su.print_predictions(prediction)
