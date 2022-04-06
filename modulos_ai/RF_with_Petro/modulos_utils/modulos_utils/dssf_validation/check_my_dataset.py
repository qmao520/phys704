# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Module to check if a dataset is in the right form and will be accepted
by the AutoML."""
import argparse
import os
import shutil
from typing import List, Tuple, Optional

from modulos_utils.solution_utils import utils as su
from modulos_utils.dssf_and_structure import DSSFSaver
from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_validation import check_my_dssf
from modulos_utils.dssf_and_structure import structure_logging as struc_log


DSSF_FILENAME = "dataset_structure.json"
DATASET_FILENAME = "temp.hdf5"
OUT_DIR_NAME = "check_my_dataset_output"


client_logging = struc_log.LoggingPurpose.CLIENT


ERROR_TO_PRINT = {
        FileNotFoundError: "\nFile not found!",
        DSSFErrors.WrongExtensionError: "\nType Error!",
        DSSFErrors.WrongTypeError: "\nType Error!",
        DSSFErrors.DataShapeMissmatchError: "\nData shape Error!",
        DSSFErrors.MissingSampleError:
        "Number of samples are not the same for all nodes!",
        DSSFErrors.SampleIDError: "\nSample ID Error!",
        DSSFErrors.ColumnHeaderError: "\nCSV Header Error!",
        DSSFErrors.DataShapeError: "\nCSV Shape Error!",
        DSSFErrors.ColumnOverflowException:
        "\nDataset contains too many features!",
        DSSFErrors.DatasetTooBigException: "",
        DSSFErrors.NotEnoughSamplesError: "\nNot enough samples in dataset!",
        DSSFErrors.DSSFNodesMissing:
        "\nError in the nodes given in the optional info!",
    }


def clean_up(keep_tmp: bool, output_dir: str,
             verbose: bool = True) -> None:
    """If the temp dir was given from outside this function, we only delete it
    in case of an error.

    Args:
        keep_tmp (bool): Whether to keep the temporary data.
        output_dir (str): Path to created output dir.
        verbose (bool): Whether to print cleaning message.
    """
    if keep_tmp or not os.path.isdir(output_dir):
        return None
    if verbose:
        print("Cleaning up ...")
    shutil.rmtree(output_dir)
    return None


def check_my_dataset(
        dataset_path: str, output_parent_dir: str = None,
        nodes_to_save: List[str] = None, logging_purpose:
        struc_log.LoggingPurpose = struc_log.LoggingPurpose.INTERNAL,
        verbose: bool = False, keep_tmp: bool = False,
        min_sample_size: Optional[int] = None, with_metadata=True,
        dshf_read_path: Optional[str] = None,
        dshf_write_path: Optional[str] = None,
        is_solution=False)\
        -> Tuple[bool, str]:
    """Perform checks whether a given dataset is supported by Modulos AutoML
    or not.

    Args:
        dataset_path (str): Path to the dataset file. If this path points to
            a tar file, the dataset is first untared before being checked. If
            it points to a DSSF, the checks can be performed directly.
        output_parent_dir (str, optional): Path to directory, where the output
            dir of the check_my_dataset function is created.
        nodes_to_save (List[str]): List of nodes to save. If the argument is
            not given, all nodes are saved.
        logging_purpose (struc_log.LoggingPurpose): Whether to log for the
            batch client.
        verbose (bool, optional): Whether to show verbose errors and outputs.
            Defaults to False.
        keep_tmp (bool, optional): Whether to keep temporary data.
            Defaults to False.
        min_sample_size (Optional[int], optional): Set minimum number of
            samples. Defaults to None.
        is_solution (bool): Whether the function is being used in the solution
            or not.

    Raises:
        OSError: If the dataset_path does not exist.
        ValueError: If the argument `dataset_path` has the wrong value (Has
            to either point to a tar file or a DSSF.)

    Returns:
        Tuple[bool, str]: Bool whether dataset is valid and checking report.
    """
    if not os.path.exists(dataset_path):
        raise OSError(f"Dataset path '{dataset_path}' points to an inexistent "
                      "file.")
    file_ext = os.path.splitext(dataset_path)[-1]
    basename = os.path.basename(dataset_path)
    if file_ext != ".tar" and basename != DSSF_FILENAME:
        raise ValueError("Argument `dataset_path` has wrong value. It must "
                         "either point to a tar file or a dataset structure "
                         "file.")
    if output_parent_dir:
        output_dir = os.path.join(output_parent_dir, OUT_DIR_NAME)
    else:
        output_dir = os.path.join(os.path.dirname(dataset_path), OUT_DIR_NAME)
    if os.path.isdir(output_dir):
        # As long as this is only used internally, raising an exception is
        # fine. It should however be replaced by more user friendly behavior
        # (for example changing the name to something that does not exist yet.)
        raise OSError("Output dir already exists!")
    os.makedirs(output_dir)

    if file_ext in (".tar", ".csv"):
        su.prepare_upload_file_for_validation(
            dataset_path, output_dir, verbose=verbose, keep_tmp=keep_tmp)
        dssf_file = os.path.join(output_dir, DSSF_FILENAME)
    else:
        dssf_file = dataset_path

    # Check if the dssf is correctly set up.
    report = "\n---------------------------------------------------"
    report += "\nCheck the DSSF:"
    dssf_valid, dssf_report = check_my_dssf.check_my_dssf(
        dssf_file, logging_purpose=logging_purpose)
    report += "\n" + dssf_report
    if not dssf_valid:
        if verbose or logging_purpose != client_logging:
            print(report)
        clean_up(keep_tmp, output_dir)
        return False, report

    report += "---------------------------------------------------"
    report += "\nRead the data:\n"

    is_successful: bool = False
    try:
        DSSFSaver.DSSFSaver(
            dssf_file, os.path.dirname(dssf_file),
            os.path.join(output_dir, DATASET_FILENAME),
            nodes_to_be_saved=nodes_to_save,
            to_be_shuffled=False,
            logging_purpose=logging_purpose,
            min_number_of_samples=min_sample_size,
            dshf_path=dshf_write_path).main()

        # Do not import and use in solutions (uses base).
        if with_metadata:
            h5_path = os.path.join(output_dir, DATASET_FILENAME)
            from modulos_utils.dssf_and_structure import compute_metadata
            computer = compute_metadata.NewMetadataComputer.from_ds_path(
                h5_path
            )
            computer.compute()
            computer.save(h5_path)

        is_successful = True
    except DSSFErrors.NodesMissingError as err:
        # This error needs to be caught, when `check_my_dataset` is not
        # used as a standalone tool. It will occur if none of the nodes in
        # `nodes_to_be_saved` is present in the dataset.
        clean_up(keep_tmp, output_dir)
        raise err
    except Exception as err:
        if type(err) in ERROR_TO_PRINT:
            report += ERROR_TO_PRINT[type(err)]
            report += "\n" + str(err)
        else:
            report += "Unknown Error!"
            report += "\n" + str(err)
            report += "\nPlease contact Modulos support."
        if verbose:
            clean_up(keep_tmp, output_dir)
            raise
    else:
        report += "\nDataset is correct!"

    if verbose or logging_purpose != client_logging:
        print(report)
    if output_parent_dir is None:
        clean_up(keep_tmp, output_dir)
    return is_successful, report


if __name__ == "__main__":
    description = ("Take a dataset in tar format and check if there is "
                   "DSSF, if it is in the correct format and if the "
                   "data is correct and matches.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("dataset", type=str, help="Dataset in tar format.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="If this flag is set, it will be more verbose. "
                        "If there is an error in testing, then it is raised "
                        "if verbose is true.")
    args = parser.parse_args()

    check_my_dataset(args.dataset, verbose=args.verbose)
