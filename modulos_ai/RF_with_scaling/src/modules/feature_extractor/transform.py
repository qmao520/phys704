# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import argparse
import os
import sys

from modulos_utils.convert_dataset import dataset_converter as dc
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
    from common_code import \
        t_test_feature_selection_common_code as ttfs
    from common_code import encode_categories as ec
    from common_code import scale_numbers as sn

    # Read input data from file.
    dataset_converter = dc.DatasetConverter()

    # Run T-test Feature Selection Feature Extractor.
    if args.train_new:
        input_data, label_data = \
            dataset_converter.read_input_labels_as_generator(
                args.input_data_file, args.label_data_file,
                retrieve_metadata=True, batch_size=1000)

        fe = ttfs.TTestFeatureSelector.initialize_new(
            args.config_choice_file,
            num_transformation=sn.NumberScalingTypes.STANDARDSCALING,
            cat_transformation=ec.CategoryEncoderTypes.ONEHOTENCODING)
        fe.fit_generator(input_data.data_generator, input_data.metadata,
                         label_data.data_generator, label_data.metadata)
        fe.save_weights(args.weights_dir)
    else:
        input_data = dataset_converter.read_data_as_generator(
            args.input_data_file, batch_size=1000)
        fe = ttfs.TTestFeatureSelector.initialize_from_weights(
            args.weights_dir)

    transformed_data = fe.transform_generator(
        input_data=input_data.data_generator, check_input=True)

    # Write output data to file.
    transformed_metadata = fe.get_transformed_metadata()
    dataset_converter.write_dataset_to_hdf5_generator(
        transformed_data, transformed_metadata, args.output_data_file)


if __name__ == "__main__":
    cmdline_description = ("Execute the t-test feature selection feature "
                           "extractor.")
    parser = cmdl_args.get_argument_parser(
        os.path.abspath(__file__), mod_types.ModuleType.FEATURE_EXTRACTOR,
        description=cmdline_description)
    args = parser.parse_args()
    main(args)
