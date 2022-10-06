# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file includes duplicated code."""
import copy
from datetime import datetime
import itertools
import json
import tarfile
import os
import shutil
from subprocess import Popen, PIPE
import sys
import threading
import time
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np

from modulos_utils.dssf_and_structure import dssf_utils
from modulos_utils.dssf_and_structure import read_data as rd
from modulos_utils.solution_utils import convert_hdf5
from modulos_utils.data_handling import data_handler as dh
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.dssf_validation import check_my_dataset as check_ds
from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_and_structure import structure_logging as struc_log
from modulos_utils.sliding_window import sliding_window, utils as sw_utils


FINISH_ANIMATION = False
MIN_SAMPLE_SIZE = 1
BATCH_SIZE_GB = 10**9


class DatasetNotValidError(Exception):
    """Error if check_my_dataset fails on solution input dataset."""
    pass


class DSSFNotFoundError(Exception):
    __module__ = Exception.__module__


class FeatureExtractorRunError(Exception):
    pass


class ModelRunError(Exception):
    pass


class DataConversionError(Exception):
    __module__ = Exception.__module__


class TmpDataExistsError(Exception):
    __module__ = Exception.__module__


class ComponentsNotConsistentError(Exception):
    __module__ = Exception.__module__


class NodesNotConsistentError(Exception):
    __module__ = Exception.__module__


class DatasetUntarError(Exception):
    __module__ = Exception.__module__


class PredictionsDoNotExistError(Exception):
    __module__ = Exception.__module__


class PredictionsReadInError(Exception):
    __module__ = Exception.__module__


class SampleNotValidError(Exception):
    __module__ = Exception.__module__


class SamplePreparationError(Exception):
    __module__ = Exception.__module__


class DSSFParsingError(Exception):
    __module__ = Exception.__module__


def exit_with_error(dirs_to_delete: List[str], keep_tmp: bool,
                    exception: Exception, verbose_exception: Exception,
                    verbose: bool) -> None:
    """Raise an error message (either user friendly or fully verbose).

    Args:
        dirs_to_delete (str): Path to folders that should be deleted.
        keep_tmp (str): Whether to keep the temporary data or to delete it.
        exception (Exception): The Error to raise if verbose is False.
        verbose_exception (Exception): The verbose Error to raise if verbose
            is true.
        verbose (bool): Whether to display the full error or not.
    """
    if not keep_tmp:
        for d in dirs_to_delete:
            if os.path.isdir(d):
                shutil.rmtree(d)
    if verbose:
        raise verbose_exception
    else:
        # TODO [REB-480] The sys.exit() call here is used to only display
        # the error message without the stackTrace. In Ticket REB-480 one
        # should investigate other ways of preventing the stackTrace from
        # showing.
        # Do not call sys.exit() when running a function in the core!
        # print(f"\n\nExecution of the solution failed: \n{exception}\n")
        # sys.exit()
        raise verbose_exception


def read_in_label_metadata(path: str) -> dict:
    """Read in label metadata json file.

    Args:
        path (str): Path to json file.

    Returns:
        dict: Metadata dictionary.
    """
    with open(path, "r") as f:
        label_metadata = json.load(f)
    return label_metadata


def get_predictions_dir_path(dir_path: str) -> str:
    """ Get file path to predictions directory. Check if predictions
    directory exists already. If yes, create new one with index + 1.

    Args:
        dir_path (str): path to base directory

    Returns:
        str: Path to newly created predictions directory.
    """

    index = 0
    is_new_dir = False
    path = ""

    while is_new_dir is False:
        path = os.path.join(dir_path, "predictions_" + str(index))
        if os.path.isdir(path):
            index += 1
        else:
            is_new_dir = True
            os.mkdir(path)

    return path


def get_predictions_output_path(output_dir: str) -> str:
    """Get predictions dir name.

    Args:
        output_dir (str): Output directory path.

    Returns:
        str: Path to output dir for predictions.
    """
    base_dir_name = "predictions"
    predictions_dir_name = base_dir_name
    version_index = 0
    while os.path.isdir(os.path.join(output_dir, predictions_dir_name)):
        version_index += 1
        predictions_dir_name = f"{base_dir_name}_{version_index}"
    return os.path.join(output_dir, predictions_dir_name)


def prepare_upload_file_for_validation(
        dataset_path: str, output_path: str, verbose: bool,
        keep_tmp: bool, tmp_dirs: Optional[List[str]] = None) -> None:
    """Untar a Dataset and check that it includes the DSSF if the data
    is tarred, else add a DSSF if it is a csv file.

    Args:
        dataset_path (str): Path to the dataset .tar or .csv file.
        output_path (str): Output path of the prepare_data function.
        verbose (bool): Whether to raise verbose error messages.
        keep_tmp (bool): Whether to keep temporary data or to delete it.
        tmp_dirs (Optional[List[str]]): Path to temporary directories.

    Raises:
        DatasetUntarError: Error if dataset path does not point to a tar file
            or if untaring fails for some other reason.
        DSSFNotFoundError: Error if the DSSF cannot be found.
    """
    if tmp_dirs is None:
        tmp_dirs = [output_path]
    dataset_ext = os.path.splitext(dataset_path)[-1]
    if dataset_ext not in (".tar", ".csv"):
        err = DatasetUntarError(
            "The dataset path must point to a .tar or .csv file.")
        exit_with_error(tmp_dirs, keep_tmp, err, err, verbose)
    if dataset_ext == ".csv":
        rel_csv_path = os.path.basename(dataset_path)
        shutil.copyfile(
            dataset_path, os.path.join(output_path, rel_csv_path))
        dssf_utils.create_dssf_template_file_for_tables(
            rel_csv_path, output_path)
    else:
        try:
            with tarfile.open(dataset_path, "r:") as tar:
                # Check if there is a top-level directory.
                found = False
                dir_name = ""
                for f in tar.getnames():
                    file_path, file_name = \
                        os.path.split(os.path.normpath(f))
                    if file_name == check_ds.DSSF_FILENAME:
                        dir_name = file_path
                        found = True
                        break

                if not found:
                    raise DSSFNotFoundError(
                        "The input dataset must contain a "
                        f"file `{check_ds.DSSF_FILENAME}` at "
                        "the root \nlevel of the tarred "
                        "directory.")
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, output_path)

                # If top level directory exists, move files.
                if dir_name:
                    for f in os.listdir(os.path.join(output_path,
                                                     dir_name)):
                        shutil.move(os.path.join(output_path, dir_name, f),
                                    output_path)
                    # Remove top level directory.
                    os.rmdir(os.path.join(output_path, dir_name))
        except DSSFNotFoundError as e:
            exit_with_error(tmp_dirs, keep_tmp, e, e, verbose)
        except Exception as exception:
            err_verbose = DatasetUntarError(
                "Untaring the input dataset failed with the following error "
                f"message: \n{exception}")
            err_trimmed = DatasetUntarError(
                "Untaring the input dataset failed.")
            exit_with_error(tmp_dirs, keep_tmp, err_trimmed, err_verbose,
                            verbose)


def get_input_node_names_from_metadata(
        meta_data_path: str, generated=False) -> List:
    """Get input node names from the metadata file.
    As the online client uses the actually used nodes (generated=True) and
    the batch client the ones, which have to be uploaded (generated=False),
    there can be a difference between the output of the two and therefore
    the option `generated` was introduced. This applies, for example, to
    datetime columns: During upload we generate new nodes, i.e. the uploaded
    node names (needed by the batch client) differ from the ones that are
    being used for training (and the online client).

    Args:
        meta_data_path (str): Path to the metadata file.
        generated (bool): If true, it returns the nodes of the internal
            dataset, else it returns the nodes of the upload, i.e. the
            input.

    Returns:
        List: List of node input node names.
    """
    meta_dict = meta_utils.MetadataDumper().load_all_nodes(meta_data_path)
    if generated:
        node_names = list(meta_dict.keys())
    else:
        node_names = list(set([meta_dict[metadata].upload_node_name.get()
                               for metadata in meta_dict]))
    if dh.SAMPLE_IDS in node_names:
        node_names.remove(dh.SAMPLE_IDS)
    return node_names


def get_input_node_names_from_dshf(
        dshf_path: str, label_name: str, generated=False) -> List:
    """Get input node names from the history file.
    As the online client uses the actually used nodes (generated=True) and
    the batch client the ones, which have to be uploaded (generated=False),
    there can be a difference between the output of the two and therefore
    the option `generated` was introduced. This applies, for example, to
    datetime columns: During upload we generate new nodes, i.e. the uploaded
    node names (needed by the batch client) differ from the ones that are
    being used for training (and the online client).

    Args:
        dshf (str): Path to the dataset history file.
        label_name (str): Name of the label node to exclude it from the input
            node.
        generated (bool): If true, it returns the nodes of the internal
            dataset, else it returns the nodes of the upload, i.e. the
            input.

    Returns:
        List: List of node input node names.
    """
    dshf = dshf_handler.DSHFHandler(dshf_path)
    if generated:
        node_names = dshf.current_nodes
    else:
        node_names = list(set(dshf.current_to_upload_name.values()))
    if label_name in node_names:
        node_names.remove(label_name)
    return node_names


def get_input_component_names(meta_data_path: str) -> List:
    """Get input component names from the metadata file.

    Args:
        meta_data_path (str): Path to the metadata file.

    Returns:
        List: List of component names of the input nodes.
    """
    meta_dict = meta_utils.MetadataDumper().load_all_nodes(meta_data_path)
    comp_names = set()
    for node_obj in meta_dict.values():
        comp_names.add(node_obj.dssf_component_name.get())
    return list(comp_names)


def check_component_names(dssf_path: str, orig_comp_names: list) -> List[str]:
    """Check whether a set of component names is contained in a DSSF and
    return the ones that are missing.

    Args:
        dssf_path (str): Path to the dssf file that is being checked.
        orig_comp_names (list): List of nodes to check for.

    Returns:
        List[str]: List of missing components (empty list if there are no
            missing components)
    """
    with open(dssf_path, "r") as f:
        dssf = json.load(f)
    dssf_comp_names = [comp["name"] for comp in dssf if "_version" not in comp]
    missing_comps: List[str] = []
    for comp_name in orig_comp_names:
        if comp_name not in dssf_comp_names:
            missing_comps.append(comp_name)
    return missing_comps


def animate_rotating_line(client_message: str) -> None:
    """Animate a rotating line in the console output.

    Args:
        client_message (str): Message to print next to animation.
    """
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if FINISH_ANIMATION:
            break
        print(f"{client_message} " + c, end="\r")
        time.sleep(0.1)
    return None


def get_python_interpreter() -> str:
    """Get the python interpreter that is currently executing this script.
    If it is not retrievable, then just use `python`.

    Returns:
        str: Path to the python executable.
    """
    interpreter = sys.executable

    if not interpreter:
        interpreter = "python"
    return interpreter


def run_python_script(
        command: str, script_path: str, keyword_args: Dict[str, str],
        flags: List[str], client_logging: bool = False,
        process_name: str = None) -> Dict:
    """ Run python script.

    Args:
        command (str): Command that is used to run the script.
        script_path (str): Path that to the script.
        keyword_args (Dict[str, str]): Dictionary of keyword arguments for
            the script.
        flags: List[str]: List of flags for the script. The convention is, that
            the prefix "--" must be included in the flag name (i.e. in each
            entry of this list).
        client_logging (bool): Whether to print logging for the user (if this
            function is used in the batch client).
        process_name (str): Name of the process that is being run (used for
            client logging).

    Returns:
        Dict: Dictionary that contains the information, if the run was
            successful.
    """

    if " -m" in command:

        module_path, module_ext = os.path.splitext(script_path)
        module_path = module_path.replace("/", ".")
        # Split the `-m` of the command.
        cmd_list = [*command.split(), module_path]
    else:
        cmd_list = [*command.split(), script_path]

    for key in keyword_args:
        cmd_list.append(key)
        cmd_list.append(keyword_args[key])
    for flag in flags:
        cmd_list.append(flag)

    if client_logging:
        # Animate a rotating line to give the impression, that the computation
        # is in progress.
        client_message = ("Process running ..." if process_name is None
                          else f"Running {process_name} ... ")
        global FINISH_ANIMATION
        FINISH_ANIMATION = False
        t = threading.Thread(target=animate_rotating_line,
                             args=[client_message])
        t.start()

    with Popen(cmd_list, stdout=PIPE, stderr=PIPE) as process:
        stdout, stderr = process.communicate()
        if client_logging:
            # Stop rotating line animation for client logging.
            FINISH_ANIMATION = True
            t.join()
            if process.returncode == 0:
                print(f"{client_message} Done.\n", end="\r")
            else:
                print(f"{client_message} Failed.\n", end="\r")
        if process.returncode != 0:
            return {"status": "failed", "output": stderr.decode(),
                    "debug": stdout.decode()}

    return {"status": "success", "output": stdout.decode()}


def check_and_convert_dataset(
        dssf_path: str, tmp_dir: str, nodes_to_save: List[str],
        logging_purpose: struc_log.LoggingPurpose, verbose: bool,
        keep_tmp: bool, dshf_read_path: str, dshf_write_path: str) -> str:
    """Convert an untared dataset to internal format by use of the
    check_my_dataset function. (We run the check function and keep the output).

    Args:
        dssf_path (str): Path DSSF.
        tmp_dir (str): Path to temporary data for the the check_my_dataset
            function.
        nodes_to_save (List[str]): Which nodes should be saved.
        logging_purpose (struc_log.LoggingPurpose): Logging purpose (internal
            or for the client).
        verbose (bool): Whether to be verbose.
        keep_tmp (bool): Whether to keep temporary data.
        dshf_read_path (str): Path to the dshf path of the training data.
        dshf_write_path (str): Path to the temporary dshf file that the
            solution code writes to.

    Returns:
        str: Path to hdf5 dataset file.
    """
    invalid_error = DatasetNotValidError(
            "The format of the input dataset is not supported by Modulos "
            "AutoML. Run with the flag `--verbose` to get more details.")
    try:
        dataset_valid, report = check_ds.check_my_dataset(
            dssf_path, output_parent_dir=tmp_dir,
            nodes_to_save=nodes_to_save, logging_purpose=logging_purpose,
            verbose=False, keep_tmp=keep_tmp, min_sample_size=MIN_SAMPLE_SIZE,
            with_metadata=False, dshf_read_path=dshf_read_path,
            dshf_write_path=dshf_write_path, is_solution=True)
    except DSSFErrors.NodesMissingError as err_verbose:
        nodes_missing_err = DatasetNotValidError(
            "The dataset is not compatible with the one uploaded to the \n"
            "platform (and used to generate this solution). It should \n"
            f"contain the following features: \n'{nodes_to_save}'. \n"
            "The following features are missing: \n"
            f"'{err_verbose.missing_nodes}'.")
        exit_with_error([tmp_dir], keep_tmp, nodes_missing_err, err_verbose,
                        verbose)
    except DSSFErrors.DSSFNodesMissing as err_verbose:
        nodes_missing_err = DatasetNotValidError(
            "There is an error in the optional info of the DSSF. "
            "Run with the flag `--verbose` to get more details.")
        exit_with_error([tmp_dir], keep_tmp, nodes_missing_err, err_verbose,
                        verbose)
    except Exception as err_verbose:
        exit_with_error([tmp_dir], keep_tmp, invalid_error, err_verbose,
                        verbose)
    if not dataset_valid:
        error_verbose = DatasetNotValidError("\n\n" + report + "\n")
        exit_with_error([tmp_dir], keep_tmp, invalid_error, error_verbose,
                        verbose)
    return os.path.join(tmp_dir, check_ds.OUT_DIR_NAME,
                        check_ds.DATASET_FILENAME)


def compare_categories_to_upload(
        upload_categories: Union[List[int], List[str], List[float]],
        input_categories: Union[List[int], List[str], List[float]]) -> bool:
    """Compare whether input categories were all present in the uploaded
    dataset.

    Args:
        upload_categories (Union[List[int], List[str], List[float]]): Cats
            that were in the uploaded dataset.
        input_categories (Union[List[int], List[str], List[float]]): Input
            values. Assumed to be categorical.

    Returns:
        bool: True if everything is alright. False if there is a new category
            in the input that was not present in the uploaded dataset.
    """
    # We convert all values to strings to compare them. If they are convertible
    # to float, we first convert to float, to avoid that the same number
    # once represented as int and once as float get converted to different
    # strings (e.g. 1 vs 1.0).
    in_prep: Union[List[int], List[float], List[str]]
    up_prep: Union[List[int], List[float], List[str]]
    try:
        in_prep = list(map(float, input_categories))
        up_prep = list(map(float, upload_categories))
    except Exception:
        in_prep = input_categories
        up_prep = upload_categories
    in_str = list(map(str, in_prep))
    up_str = set(map(str, up_prep))
    return all(elem in up_str for elem in in_str)


def run_feature_extractor(dataset_path: str, output_dir: str,
                          src_dir: str, tmp_dirs: List[str],
                          verbose: bool = False, keep_tmp: bool = False,
                          animate_progress: bool = False) -> Dict:
    """Run the feature extractor in the online-solution.
    The feature extractor parameters are hardcoded.
    As the directory structure of the solution folder is fixed.
    This function needs to be adapted see [BAS-478].

    Args:
        dataset_path (str): Path to dataset hdf5 file.
        output_dir (str): Path to the output dir of the feature extractor.
        src_dir (str): Path to the src folder that contains the modules.
        tmp_dirs (List[str]): List of temporary directories.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.
        animate_progress (bool = False): Whether to animate in the console
            to give a sense of progress.

    Returns:
        Dict: Dictionary that contains the information, if the run was
            successful.
    """
    metadata_upload = meta_utils.MetadataDumper().load_all_nodes(
        os.path.join(src_dir, "metadata", "input_metadata.bin"))

    for node_name, node_meta in metadata_upload.items():
        if not node_meta.is_categorical():
            continue
        uniques = metadata_upload[node_name].upload_unique_values.get()
        node_data = dh.DatasetReader(dataset_path).get_data_of_node_all(
            node_name)[:, 0]
        if not compare_categories_to_upload(uniques, node_data):
            err_trimmed_cat = DatasetNotValidError(
                "There are unknown values for categorical features in this "
                "dataset which were not present in the dataset uploaded to "
                "Modulos AutoML.")
            err_verbose_cat = DatasetNotValidError(
                "The solution can only understand categories, that "
                "were present in the training dataset, however the node "
                f"`{node_name}` contains categories that did not occur in "
                "the original dataset that was uploaded to the platform "
                "(and used to generate this solution).")
            exit_with_error(
                tmp_dirs, keep_tmp, err_trimmed_cat, err_verbose_cat,
                verbose)

    module_dir = os.path.join(src_dir, "modules",
                              "feature_extractor")
    fe_run_parameters = {
        "--config-choice-file": os.path.join(
            module_dir, "weights/fe_config_choice.json"),
        "--input-data-file": dataset_path,
        "--weights-dir": os.path.join(module_dir, "weights"),
        "--output-data-file": os.path.join(
            output_dir, "transformed_dataset.hdf5")
    }
    fe_script_path = os.path.join(module_dir, "transform.py")
    interpreter = get_python_interpreter()
    output = run_python_script(
        f"{interpreter}", fe_script_path, fe_run_parameters, [],
        client_logging=animate_progress, process_name="feature extractor")
    if output["status"] != "success":
        fe_log = output["output"]
        err_verbose = FeatureExtractorRunError(
            "Running the trained feature extractor failed with the "
            f"following message: \n{fe_log}")
        err_trimmed = FeatureExtractorRunError(
            "Running the trained feature extractor failed.")
        exit_with_error(tmp_dirs, keep_tmp, err_trimmed, err_verbose, verbose)
    return output


def run_model(src_dir: str, input_dir: str, output_dir: str,
              tmp_dirs: List[str], convert_to_original_format: bool = False,
              verbose: bool = False, keep_tmp: bool = False,
              animate_progress: bool = False) -> Dict:
    """Run the model in the online-solution.

    The model parameters are hardcoded.
    As the directory structure of the solution folder is fixed.
    This function needs to be adapted see [BAS-478].

    Args:
        src_dir (str): Path to the src folder that contains the modules.
        input_dir (str): Path to the input dir of the model.
        tmp_dirs (List[str]): Path to temporary files folder.
        output_dir (str): Path to the folder where the predictions are
            saved.
        convert_to_original_format (bool = False): Whether or not to convert
            the predictions back to original label format.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.
        animate_progress (bool = False): Whether to animate in the console
            to give a sense of progress.

    Returns:
        Dict: Dictionary that contains the information, if the run was
            successful.

    Raises:
        ModelRunError: Raised if model script failed.
        DataConversionError: Raised if converting to original format failed.
    """
    dshf = dshf_handler.DSHFHandler(os.path.join(
        src_dir, "dataset_history", dshf_handler.DSHF_FILE_NAME))

    module_dir = os.path.join(src_dir, "modules", "model")
    model_run_parameters = {
        "--weights-dir": os.path.join(
            module_dir, "weights"),
        "--input-data-file": os.path.join(
            input_dir, "transformed_dataset.hdf5"),
        "--output-data-file": os.path.join(input_dir, "predictions.hdf5")
    }
    model_script_path = os.path.join(module_dir, "predict.py")
    interpreter = get_python_interpreter()
    output = run_python_script(
        f"{interpreter}", model_script_path, model_run_parameters, [],
        client_logging=animate_progress, process_name="model")
    if output["status"] != "success":
        model_log = output["output"]
        err_model_verbose = ModelRunError(
            "Running the trained model failed with the following message: \n"
            f"{model_log}")
        err_model_trimmed = ModelRunError("Running the trained model failed.")
        exit_with_error(tmp_dirs, keep_tmp, err_model_trimmed,
                        err_model_verbose, verbose)

    if convert_to_original_format:
        output_path = get_predictions_output_path(output_dir)
        label_metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(src_dir, "metadata/label_metadata.bin")
        )
        try:
            convert_hdf5.save_predictions_to_label_format(
                model_run_parameters["--output-data-file"], label_metadata,
                output_path, dshf)
        except Exception as exception:
            err_conversion_verbose = DataConversionError(
                "Converting the predictions failed with the following "
                f"message: \n{exception}")
            err_conversion_trimmed = DataConversionError(
                    "Converting the predictions failed.")
            exit_with_error(tmp_dirs, keep_tmp, err_conversion_trimmed,
                            err_conversion_verbose, verbose)
    return output


def prepare_sample_for_prediction(
        sample_dict: Dict, base_dir: str, tmp_dirs: List[str],
        verbose: bool = False, keep_tmp: bool = False) -> Dict:
    """If sample dictionary contains a file path, read in the data
       and return a dictionary that contain the actual data.

    Args:
        sample_dict (Dict): Sample data dictionary
        base_dir (str): Base of the solutions dir.
        tmp_dirs (List[str]): List of temporary folders.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.

    Returns:
        Dict: Sample Dictionary that contains the real.
    """
    try:
        new_sample_dict = copy.deepcopy(sample_dict)
        for node_name, node_data in new_sample_dict.items():
            new_data = []
            if type(node_data) is not list:
                node_data = [node_data]
            for sample in node_data:
                node_path = os.path.join(base_dir, str(sample))
                if os.path.exists(node_path):
                    data = np.array(rd.read_data(node_path).get_data())
                    new_data.append(data)
                else:
                    new_data = node_data
                    break
            new_sample_dict.update({node_name: np.array(new_data)})
    except Exception as e:
        err_verbose = SamplePreparationError(
            "Preparing the sample for predictions failed with the "
            f"following \nerror message: \n{e}")
        err_trimmed = SamplePreparationError(
            "Tensors of the sample could not be read in.")
        exit_with_error(tmp_dirs, keep_tmp, err_trimmed, err_verbose, verbose)
    return new_sample_dict


def setup_directories(dataset_path: str, output_dir: str,
                      default_output_dir: str, verbose: bool = False,
                      keep_tmp: bool = False) -> Tuple[str, str]:
    """Make sure all files and directories, that are necessary for the batch
    client, exist.

    Args:
        dataset_path (str): Path to dataset file.
        output_dir (str): Path to output dir to generate.
        default_output_dir (str): Output dir to generate, if output_dir is "".
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep the temporary data.

    Returns:
        Tuple[str, str]: Path to tmp directory, that was created and path to
            output dir, that is used in the batch client (either the one
            given by the user or the default defined in the batch client.)

    Raises:
        TmpDataExistsError: Raised if tmp dir exists already.
        FileNotFoundError: Raised if dataset file does not exist.
        OSError: Raised if output dir is given but points to an inexistent
            folder.
    """
    if not os.path.exists(dataset_path):
        err_1 = FileNotFoundError(f"Dataset path \n'{dataset_path}' \npoints "
                                  "to an inexistent file.")
        exit_with_error([], keep_tmp, err_1, err_1, verbose)

    if output_dir == "":
        if not os.path.isdir(default_output_dir):
            os.makedirs(default_output_dir)
        out_dir = default_output_dir
    else:
        if os.path.isdir(output_dir):
            out_dir = output_dir
        else:
            err_2 = OSError(f"Output dir \n'{output_dir}' \ndoes not exist.")
            exit_with_error([], keep_tmp, err_2, err_2, verbose)

    tmp_dir = os.path.join(out_dir, "tmp_data_dir")
    create_tmp_dir(tmp_dir, [tmp_dir, default_output_dir],
                   verbose=verbose, keep_tmp=keep_tmp)
    return tmp_dir, out_dir


def create_tmp_dir(tmp_dir: str, dirs_to_delete: List[str] = None,
                   verbose: bool = False, keep_tmp: bool = False) -> None:
    """Create tmp dir.
    Args:
        tmp_dir (str): Path to tmp dir to generate.
        dirs_to_delete: List[str]: List of directories to delete in case this
            function fails.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep the temporary data.
    Raises:
        TmpDataExistsError: Raised if tmp dir exists already.
    """
    if dirs_to_delete is None:
        dirs_to_delete = [tmp_dir]
    if os.path.exists(tmp_dir):
        if keep_tmp:
            err = TmpDataExistsError(
                "The temporary data folder exists already. Probably the "
                "client has failed in a previous run. Please delete the "
                f"folder {tmp_dir} and try again.")
            exit_with_error(dirs_to_delete, keep_tmp, err, err, verbose)
        else:
            shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


def clean_up(tmp_dir: str, keep_tmp: bool = False) -> None:
    """Delete temporary files.

    Args:
        tmp_dir (str): Path to tmp dir.
        keep_tmp (bool): Whether or not to keep the temporary folders.
    """
    if (not keep_tmp) and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)


def convert_to_internal_dataset_format(
        untarred_path: str, download_path: str, dshf_path: str,
        tmp_dshf_path: str, tmp_dirs: List[str], verbose: bool,
        keep_tmp: bool) -> str:
    """Convert the untared input dataset to the internal format (hdf5).

    Args:
        untared_path (str): Path to the untared dataset directory. The folder
            must contain the DSSF.
        download_path (str): Path to the download files containing at least
            the metadata.
        dshf_path (str): Path to the dataset history file that was downloaded
            from the platform.
        tmp_dshf_path (str): Path to temporary dshf file.
        tmp_dirs (List[str]): List of temporary folders.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.

    Returns:
        str: Path to converted dataset hdf5 file.

    Raises:
        DataConversionError: Raised if the data cannot be converted to hdf5.
        KeyError: Can be raised inside the function 'check_component_names'.
            (If the format of DSSF is wrong.)
        AttributeError: Can be raised inside the function
            'check_component_names'. (If the format of DSSF is wrong.)
        ComponentsNotConsistentError: Raised if components of input dataset
            are not consistent with the dataset that was uploaded to the
            platform.
    """
    # DSHF Parse
    input_metadata = os.path.join(download_path,
                                  "src/metadata/input_metadata.bin")
    nodes_to_be_saved = get_input_node_names_from_metadata(
        input_metadata)

    # Check whether the input dataset contains the necessary component names.
    # Note that this is not a full consistency check, because the nodes in a
    # table component could still be different.

    # Check whether dssf exists in the untared path.
    dssf_path = os.path.join(untarred_path, check_ds.DSSF_FILENAME)
    if not os.path.isfile(dssf_path):
        err_dssf = DatasetUntarError(
            "Dataset structure file is missing in the untared dataset.")
        exit_with_error(tmp_dirs, keep_tmp, err_dssf, err_dssf, verbose)

    # Convert untared dataset to and internal dataset file (hdf5) while
    # checking the format (with the function check_my_dataset).
    try:
        print("\nConverting dataset ...")
        ds_hdf5_path = check_and_convert_dataset(
            os.path.join(untarred_path, check_ds.DSSF_FILENAME),
            tmp_dir=untarred_path, nodes_to_save=nodes_to_be_saved,
            logging_purpose=struc_log.LoggingPurpose.CLIENT,
            verbose=verbose, keep_tmp=keep_tmp, dshf_read_path=dshf_path,
            dshf_write_path=tmp_dshf_path)
    except DSSFErrors.NodesMissingError as nodes_missing_err:
        err_empty_trimmed = ValueError(
            "The dataset is not compatible with the one uploaded to the \n"
            "platform (and used to generate this solution). It should \n"
            f"contain the following features: \n'{nodes_to_be_saved}'. \n"
            "The following features are missing: \n"
            f"'{nodes_missing_err.missing_nodes}'.")
        exit_with_error(tmp_dirs, keep_tmp, err_empty_trimmed,
                        nodes_missing_err, verbose)

    return ds_hdf5_path


def convert_sample_to_hdf5(
        sample_dict: dict, sample_hdf5_path: str, tmp_dirs: List[str],
        verbose: bool = False, keep_tmp: bool = False) -> None:
    """Convert sample to hdf5.

    Args:
        sample_dict (dict): Dictionary representing one input sample. The keys
            are the node names.
        sample_hdf5_path (str): Path to output hdf5 file.
        tmp_dirs (List[str]): List of temporary folders.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.
    """
    conversion_succeeded = True
    try:
        n_samples = len(list(sample_dict.values())[0])
        with dh.get_Dataset_writer(sample_hdf5_path, n_samples) as dataset:
            dataset.add_samples(sample_dict)
        valid, msg = dh.DatasetReader(sample_hdf5_path).get_validation_info()
        if not valid:
            err_verbose = SampleNotValidError(
                "Converting the sample to the internal data format "
                "resulted \n in an invalid dataset. The data handler raised "
                f"the following exception: \n{msg}")
            conversion_succeeded = False
    except Exception as e:
        err_verbose = SampleNotValidError(
            "Saving sample in internal file format failed with the "
            f"following \nerror message: \n{e}")
        conversion_succeeded = False

    if not conversion_succeeded:
        err_trimmed = SampleNotValidError(
            "Saving sample in internal file format failed."
        )
        exit_with_error(tmp_dirs, keep_tmp, err_trimmed, err_verbose, verbose)
    return None


def read_predictions(
        predictions_path: str, tmp_dirs: List[str], verbose: bool = False,
        keep_tmp: bool = False) -> dict:
    """Read in one-sample predictions in hdf5 format and convert them to
    dictionary.

    Args:
        predictions_path (str): Path to prediction hdf5.
        tmp_dirs (List[str]): List of temporary folders.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.

    Returns:
        dict: Sample prediction dictionary.
    """
    if not os.path.isfile(predictions_path):
        err_1 = PredictionsDoNotExistError("The model did not produce any "
                                           "predictions.")
        exit_with_error(tmp_dirs, keep_tmp, err_1, err_1, verbose)
    try:
        prediction_reader = dh.DatasetReader(predictions_path)
        predictions = prediction_reader.get_data_all()
    except Exception as e:
        err_2_verbose = PredictionsReadInError(
            "Reading in the predictions failed with the following error "
            f"message: \n{e}")
        err_2_trimmed = PredictionsReadInError(
            "The predictions output by the model are in the wrong format.")
        exit_with_error(tmp_dirs, keep_tmp, err_2_trimmed, err_2_verbose,
                        verbose)
    return predictions


def convert_to_builtin_types(input_array: np.ndarray) -> List:
    """
    Convert a numpy array to a python builtin type.
    Attention: This function is very slow for large nested arrays.

    Args:
        input_array (np.ndarray): A numpy array.

    Returns:
        List: An arbitrarily nested list of builtin types, e.g. str, int or
            float.
    """
    return input_array.tolist()


def post_process(predictions: Dict, src_dir: str, **kwargs) -> Dict:
    """Convert the predictions back to the uploaded state. In the online client
    this function converts unixtime to a real datetime if the label on the
    platform was a label. In the forecast client, this function also formats
    the forecast vector (in case of forecast_horizon > 1) to a human readable
    format that can be printed to the console.

    Args:
        predictions (Dict): A single prediction in a dictionary.
        src_dir (str): Directory of the src dir inside the solution folder.

    Keyword Args:
        ts_setting (Dict): The time series config json that was edited by
            the user.
        sample_dict (Dict): Dictionary containing the input samples. This is
            used in the time series case for retrieving the time feature.

    Returns:
        Dict: Prediction dictionary with converted prediction value.
    """
    # Get original name of the sample ids.
    is_single_sample = (len(predictions[dh.SAMPLE_IDS]) == 1)
    label_metadata = meta_utils.MetadataDumper().load_all_nodes(
        os.path.join(src_dir, "metadata/label_metadata.bin"))
    label_node_name = list(label_metadata.keys())[0]
    dshf = dshf_handler.DSHFHandler(os.path.join(
        src_dir, "dataset_history", dshf_handler.DSHF_FILE_NAME))

    # Get the time series config json. The variable ts_settings defaults to
    # None, if there is no config json (i.e. the task is not a forecasting
    # task)
    ts_settings = kwargs.get("ts_setting")
    # Remove empty batch dimension, because n_sample = 1. For scalars, also
    # remove the empty node dimension.
    if is_single_sample and ts_settings is None:
        predictions_mod = {}
        for key, value in predictions.items():
            value_builtin = convert_to_builtin_types(value)
            if key != dh.SAMPLE_IDS and len(value[0]) == 1:
                predictions_mod[key] = value_builtin[0][0]
            else:
                predictions_mod[key] = value_builtin[0]
        if label_node_name in dshf.generated_nodes:
            predictions_mod["predictions"] = \
                datetime.fromtimestamp(
                    np.rint(predictions_mod["predictions"])).isoformat()
    elif ts_settings is not None:
        # In this case, we always show only one sample independent on the
        # number of predictions.
        forecast_step = ts_settings.get("forecast_step")
        forecast_horizon = ts_settings.get("forecast_horizon")
        predictions_mod = {"predictions": {}}
        predictions_builtin = convert_to_builtin_types(
            predictions["predictions"])
        for i in range(forecast_horizon):
            # If forecast horizon > 1, there is an additional dimension after
            # the sample dimension, i.e.
            # [nr_samples, forecast_horizon, *original_label_dim]. For
            # forecast_horizon == 1, this additional dimension is missing, i.e.
            # [nr_samples, *original_label_dim].
            # Furthermore, we return only one sample, hence we can just take
            # the last element of the list.
            pred_i = predictions_builtin[-1][i] if forecast_horizon > 1 \
                else predictions_builtin[-1]
            # A) If the original label was scalar, the variable `pred_i` now
            #    has shape [1], e.g.: `pred_i = [23.456]`.
            # B) If the original label dim was a tensor, e.g. an RGB image with
            #    shape [100, 100, 3], `pred_i` now has shape [100, 100, 3].
            if len(pred_i) == 1 and not isinstance(pred_i[0], list):
                # This if checks, if the prediction is a scalar, which would
                # be the case for original_label_dim = [1] and
                # forecast_horizon = 1 (case A of the example above).
                pred_i = pred_i[0]
            predictions_mod["predictions"][f"t + {i + forecast_step}"] = pred_i
        predictions.pop("predictions")
        # Add sample ids.
        sample_ids = convert_to_builtin_types(predictions[dh.SAMPLE_IDS])
        predictions_mod[dh.SAMPLE_IDS] = sample_ids[-1]
        # Add time feature name and value at 0.
        predictions_mod["time_feature"] = ts_settings.get("time_feature")
        sample_dict = kwargs["sample_dict"]
        t0 = sample_dict[predictions_mod["time_feature"]][-1]
        predictions_mod["t0"] = t0
    else:
        predictions_mod = {}
        for key, value in predictions.items():
            if value.shape == (len(value), 1):
                value = value.reshape(-1)
            value_builtin = convert_to_builtin_types(value)
            predictions_mod[key] = value_builtin
        if label_node_name in dshf.generated_nodes:
            predictions_mod["predictions"] = \
                [datetime.fromtimestamp(np.rint(j)).isoformat()
                 for j in predictions["predictions"]]

    return predictions_mod


def print_predictions(predictions: Dict) -> None:
    """Print predictions of one single sample.

    Args:
        predictions (Dict): Dictionary containing the predictions for a
            sample.
    """
    prediction_value = predictions.pop("predictions")
    sample_id = list(predictions.values())[0]
    print()
    print(f"    Sample: {sample_id}")
    print(f"Prediction: {prediction_value}")
    print()
    return None


def print_forecast(predictions: Dict) -> None:
    """Print the forecast of a time series.

    Args:
        predictions (Dict): Dictionary containing the predictions for a
            forecast.
    """
    prediction_forecasts = predictions.pop("predictions")
    sample_id = predictions.pop(dh.SAMPLE_IDS)
    time_feature = predictions.pop("time_feature")
    time_of_prediction = predictions.pop("t0")

    print("\n" + len(time_feature)*" " + f"         Sample: {sample_id}")
    print(f"Time Feature ({time_feature}): t = {time_of_prediction}")
    print(len(time_feature) * " " + "     Prediction:", end=" ")
    indent = len(time_feature) + 17
    print(json.dumps(prediction_forecasts, indent=indent).replace(
        "{\n" + indent * " ", "").replace("}", "").replace("\"", ""))
    return None


def check_input_node_names(
        sample_dict: dict, metadata_path: str, tmp_dirs: List[str],
        time_series_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True, keep_tmp: bool = True) -> None:
    """Raise an Error if necessary node names (according to the downloaded
    input metadata file) are not present in input sample_dict.

    Args:
        sample_dict (dict): Dictionary representing one sample, where for
        tensors it can contain either the path or the loaded tensor.
        metadata_path (str): Path to downloaded input metadata file.
        tmp_dirs (List[str]): List of temporary folders.
        time_series_config (Optional[Dict[str, Any]]): Time series
            configuration, if it is a time series solution.
        verbose (bool = False): Whether to raise verbose error messages.
        keep_tmp (bool = False): Whether to keep temporary data or to
            delete it.

    Raises:
        NodesNotConsistentError: Raised if node names of input sample are not
            consistent with the dataset that was uploaded to the system.
    """
    sample_nodes = list(sample_dict.keys())
    # Raise an error if node name are not all strings.
    for key in sample_nodes:
        if not isinstance(key, str):
            err_1 = ValueError("Feature names of input data must be strings.")
            exit_with_error(tmp_dirs, keep_tmp, err_1, err_1, verbose)

    required_nodes = get_input_node_names_from_metadata(
        metadata_path, generated=True)

    # We only raise an Exception if a node is missing. Additional nodes are
    # ignored without an exception.
    for node in required_nodes:
        if sw_utils.GENERATED_NODES_SUFFIX in node:
            continue
        if node not in sample_nodes:
            err_2 = NodesNotConsistentError(
                "At least one of the features, which were present during "
                "training, is missing in \nthe input sample.")
            exit_with_error(tmp_dirs, keep_tmp, err_2, err_2, verbose)
        if dh.SAMPLE_IDS not in sample_nodes:
            err_3 = NodesNotConsistentError(
                "The sample dictionary must contain a sample id entry with "
                f"the key \n'{dh.SAMPLE_IDS}'.")
            exit_with_error(tmp_dirs, keep_tmp, err_3, err_3, verbose)


def get_sample_id_column_name(dshf: dshf_handler.DSHFHandler) -> str:
    """Get the original sample id name from the dataset history object.
    If there are multiple sample_id names take the one first one in
    alphabetical order. If there are no sample id names use `sample_ids`

    Args:
        dshf (dshf_handler.DSHFHandler): Dataset history file object.

    Returns:
        str: original name of the sample id.
    """
    original_sample_id_names_list = []
    for comp in dshf.dssf_info.values():
        if "sample_id_column" in comp:
            original_sample_id_names_list.append(comp["sample_id_column"])
    if len(original_sample_id_names_list) == 0:
        original_sample_id_name = dh.GENERATED_SAMPLE_ID_NAME
    else:
        original_sample_id_name = str(original_sample_id_names_list[0])
    return original_sample_id_name


def add_window_sliding(
        hdf5_path: str, tmp_dir: str, sw_config_file_path: str) -> None:
    """Add the shifted features to the data which have to be predicted.

    Args:
        hdf5_path (str): Path to the samples in a hdf5.
        tmp_dir (str): Temporary directory for the temporary dataset.
        sw_config_file_path (str): Path to the config file that can be given
            as input to the sliding window module.
    """
    with open(sw_config_file_path, "r") as f:
        sw_config_dict = json.load(f)

    temporary_dataset_path = os.path.join(tmp_dir, "data_slided.h5")
    if os.path.exists(temporary_dataset_path):
        os.remove(temporary_dataset_path)
    sliding_window.run(
        hdf5_path, temporary_dataset_path, sw_config_dict)
    os.remove(hdf5_path)
    shutil.move(temporary_dataset_path, hdf5_path)
    return None
