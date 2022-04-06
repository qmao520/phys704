# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This code runs a trained solution to get predictions.
"""
import os
import shutil
from typing import Dict

from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.solution_utils import utils as su
from modulos_utils.solution_utils import preprocessing


class SampleInputError(Exception):
    """Errors for when the input sample is wrong, i.e. either the node names,
    node types or node values are invalid.
    """
    pass


class FeatureExtractorError(Exception):
    """Errors in the feature extractor.
    """
    pass


class ModelError(Exception):
    """Errors in the model.
    """
    pass


def predict(
        sample_dict: Dict, tmp_dir: str, src_dir: str) -> Dict:
    """Top level prediction function. It calls all the steps of the pipeline,
    i.e. pre-processing, FE, model and post-processing.

    Note: This function accepts only one single sample!

    Args:
        sample_dict (Dict): Sample for which to predict.
        tmp_dir (str): Path to temporary data dir.
        src_dir (str): Path to the src dir inside a solution folder,
            where the modules and the metadata are saved.

    Returns:
        Dict: Dictionary containing the prediction for the input sample.
    """

    # Preprocess the data.
    metadata_path = os.path.join(src_dir, "metadata/input_metadata.bin")
    metadata = meta_utils.MetadataDumper().load_all_nodes(
        metadata_path)
    input_node_names = list(metadata.keys())

    try:
        prep_sample_dict = preprocessing.preprocess_dict(
            sample_dict, input_node_names, src_dir, ignore_sample_ids=False)

        # Check whether required nodes are present. This raises an Exception if
        # one node is missing. Additional nodes are ignored.
        su.check_input_node_names(prep_sample_dict, metadata_path, [tmp_dir])

        # Create tmp data directory
        su.create_tmp_dir(tmp_dir)

        # Create HDF5 version of the dataset. (TO BE REMOVED)
        prediction_input = su.prepare_sample_for_prediction(
            prep_sample_dict, os.path.dirname(src_dir), [tmp_dir])
        sample_hdf5_path = os.path.join(tmp_dir, "sample.hdf5")
        su.convert_sample_to_hdf5(prediction_input, sample_hdf5_path,
                                  [tmp_dir])
    except Exception as e:
        raise SampleInputError(e)

    try:
        # Run Feature Extractor.
        su.run_feature_extractor(sample_hdf5_path, tmp_dir, src_dir,
                                 [tmp_dir])
    except Exception as e:
        raise FeatureExtractorError(e)

    try:
        # Run Model.
        su.run_model(src_dir, tmp_dir, tmp_dir, [tmp_dir])
    except Exception as e:
        raise ModelError(e)

    # Read Predictions (TO BE REMOVED)
    predictions_path = os.path.join(tmp_dir, "predictions.hdf5")
    predictions_raw = su.read_predictions(predictions_path, [tmp_dir])

    # Post-processing the predictions.
    predictions = su.post_process(predictions_raw, src_dir)

    # Remove the temporary data folder.
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    return predictions
