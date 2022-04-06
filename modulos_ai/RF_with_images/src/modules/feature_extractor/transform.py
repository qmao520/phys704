# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import os
import argparse
import sys

from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.data_handling import data_handler as dh
from modulos_utils.data_handling import data_utils
from modulos_utils.module_interfaces import argument_definitions as cmdl_args
from modulos_utils.module_interfaces import module_types as mod_types


def main(args: argparse.Namespace) -> None:
    """
    Main function of the feature extractor script.

    Args:
        args (argparse.Namespace): Parsed arguments for the script.
    """
    # Hack as long as common_code is not a package.
    tmp_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
    sys.path.append(os.path.normpath(os.path.join(tmp_dir, ".")))
    from common_code import identity_common_code as identity

    # Read in data from file.
    nr_samples = dh.DatasetReader(args.input_data_file).get_n_samples()
    batch_size = data_utils.compute_batch_size(
        args.input_data_file, nr_samples)
    dataset_converter = dc.DatasetConverter()
    input_data = dataset_converter.read_data_as_generator(
        args.input_data_file, retrieve_metadata=args.train_new,
        batch_size=batch_size)

    # Run Identity Feature Extractor.
    if args.train_new:
        fe = identity.IdentityFeatureExtractor.initialize_new(
            args.config_choice_file)
        transformed_data = fe.fit_transform_generator(
            input_data.data_generator, input_data.metadata)
        fe.save_weights(args.weights_dir)
    else:
        fe = identity.IdentityFeatureExtractor.initialize_from_weights(
            args.weights_dir)
        transformed_data = fe.transform_generator(
            input_data=input_data.data_generator, check_input=True)

    # Write output data to file.
    transformed_metadata = fe.get_transformed_metadata()
    dataset_converter.write_dataset_to_hdf5_generator(
        transformed_data, transformed_metadata, args.output_data_file)


if __name__ == "__main__":
    cmdline_description = ("Execute the identity feature extractor.")
    parser = cmdl_args.get_argument_parser(
        os.path.abspath(__file__), mod_types.ModuleType.FEATURE_EXTRACTOR,
        description=cmdline_description)
    args = parser.parse_args()
    main(args)
