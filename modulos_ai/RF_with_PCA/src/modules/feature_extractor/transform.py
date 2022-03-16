# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import os
import argparse
import sys

from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.module_interfaces import argument_definitions as cmdl_args
from modulos_utils.module_interfaces import module_types as mod_types

BATCH_SIZE = 32


def main(args: argparse.Namespace) -> None:
    """
    Main function of the feature extractor script.

    Args:
        args (argparse.Namespace): Parsed arguments for the script.
    """
    # Hack as long as common_code is not a package.
    tmp_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
    sys.path.append(os.path.normpath(os.path.join(tmp_dir, ".")))
    from common_code import pca_common_code as pca

    # Read in data from file.
    dataset_converter = dc.DatasetConverter()
    input_data = dataset_converter.read_data_as_dict(
        args.input_data_file, retrieve_metadata=args.train_new)

    # Run PCA Feature Extractor.
    if args.train_new:
        fe = pca.PCAFeatureExtractor.initialize_new(
            args.config_choice_file)
        transformed_data = fe.fit_transform(
            input_data.data, input_data.metadata)
        fe.save_weights(args.weights_dir)
    else:
        fe = pca.PCAFeatureExtractor.initialize_from_weights(
            args.weights_dir)
        transformed_data = fe.transform(input_data.data, check_input=True)

    # Write output data to file.
    transformed_metadata = fe.get_transformed_metadata()
    dataset_converter.write_dataset_to_hdf5(
        transformed_data, transformed_metadata, args.output_data_file)


if __name__ == "__main__":
    cmdline_description = ("Execute the PCA feature extractor.")
    parser = cmdl_args.get_argument_parser(
        os.path.abspath(__file__), mod_types.ModuleType.FEATURE_EXTRACTOR,
        description=cmdline_description)
    args = parser.parse_args()
    main(args)
