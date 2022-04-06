# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import argparse
import os

from modulos_utils.solution_utils import utils as su
from modulos_utils.solution_utils import preprocessing as prep
from modulos_utils.dshf_handler import dshf_handler


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(FILE_DIR, "output_batch_client")
DSHF_PATH = os.path.join(
    FILE_DIR, "src/dataset_history", dshf_handler.DSHF_FILE_NAME)
SRC_DIR = os.path.join(FILE_DIR, "src")


def main(dataset_path: str, output_dir_user: str,
         verbose: bool, keep_tmp: bool) -> None:
    """Run the batch prediction pipeline.

    Args:
        dataset_path (str): Path to the `.csv` or `.tar` dataset.
        output_dir_user (str): Path to the output directory.
        verbose (bool): Whether or not to show full error messages.
        keep_tmp (bool): Whether to keep the temporary data or to delete it.

    Raises:
        OSError: Raised if files/folders don't exist.
    """
    # Check that dataset_path exists:
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_path):
        raise OSError("File not found.")

    # Create tmp data directory.
    tmp_dir, out_dir = su.setup_directories(
        dataset_path, output_dir_user, DEFAULT_OUTPUT_DIR,
        verbose=verbose, keep_tmp=keep_tmp)
    tmp_dshf_path = os.path.join(tmp_dir, dshf_handler.DSHF_FILE_NAME)

    # Untar Dataset for a tar input or add a default dssf for a csv file.
    su.prepare_upload_file_for_validation(
        dataset_path, tmp_dir, verbose=verbose, keep_tmp=keep_tmp)

    # Convert the input dataset to the internal format.
    ds_hdf5_path = su.convert_to_internal_dataset_format(
        untarred_path=tmp_dir, download_path=FILE_DIR, dshf_path=DSHF_PATH,
        tmp_dshf_path=tmp_dshf_path, tmp_dirs=[tmp_dir],
        verbose=verbose, keep_tmp=keep_tmp)

    # Perform the same preprocessing steps that were performed on-platform.
    prep.preprocess_hdf5(
        ds_hdf5_path, DSHF_PATH, tmp_dshf_path, tmp_dirs=[tmp_dir],
        verbose=verbose, keep_tmp=keep_tmp)

    # Run Feature Extractor.
    print("\n")
    su.run_feature_extractor(
        ds_hdf5_path, tmp_dir, SRC_DIR, [tmp_dir],
        verbose=verbose, keep_tmp=keep_tmp, animate_progress=True)

    # Run Model.
    su.run_model(
        SRC_DIR, tmp_dir, out_dir, [tmp_dir, out_dir],
        convert_to_original_format=True, verbose=verbose, keep_tmp=keep_tmp,
        animate_progress=True)

    # Clean up.
    su.clean_up(tmp_dir, keep_tmp)

    print("\nPredictions were generated successfully in "
          f"\n'{out_dir}'.\n")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch solution.")
    parser.add_argument("dataset_path", metavar="dataset-path", type=str,
                        help=("Path to the .tar or .csv file that contains "
                              "the data."))
    parser.add_argument("--output-dir", type=str, default="",
                        help="""Output_directory in which to save the
                              predictions.""")
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
    main(args.dataset_path, args.output_dir, verbose_flag, keep_tmp_flag)
