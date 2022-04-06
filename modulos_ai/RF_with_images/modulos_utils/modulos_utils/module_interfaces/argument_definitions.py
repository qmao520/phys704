# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import argparse
import sys

from modulos_utils.module_interfaces import module_types as mod_types
from modulos_utils.module_interfaces import module_modes as mod_modes


class ModelArgumentDefinitionError(Exception):
    """Errors in the definition of the model command line arguments.
    """
    pass


def get_argument_parser(
        module_dir: str, module_type: mod_types.ModuleType,
        mode:  mod_modes.ModuleModes = mod_modes.ModuleModes.NONE,
        description: str = "Run the module as a script.")\
        -> argparse.ArgumentParser:
    """Get an argument parser object with the command line arguments for the
    model script of either training or validation.

    Args:
        module_dir (str): Path to the module directory.
        module_type (module_types.ModuleType): Type of the module.
        mode (str): Either `training` or `validation`.
        description (str): Description of the argument parser. This is model
            dependent, and therefore needs to be given as an input parameter
            to this function.

    Returns:
        argparse.ArgumentParser: Argument parser object.
    """
    parser = argparse.ArgumentParser(description=description)
    if module_type == mod_types.ModuleType.MODEL:
        if mode == mod_modes.ModuleModes.TRAINING:
            parser.add_argument("--input-data-file", type=str, required=True,
                                help="Path to the input dataset file.")
            parser.add_argument("--label-data-file", type=str, required=True,
                                help="Path to the label dataset file.")
            parser.add_argument("--transformed-label-data-file", type=str,
                                required=True,
                                help="Path to the transformed label dataset "
                                "file.")
            parser.add_argument("--config-choice-file", type=str,
                                required=True,
                                help="Path to the configuration file which "
                                "contains the config choice for the model.")
            parser.add_argument("--weights-dir", type=str, required=True,
                                help="Path to the folder in which to store "
                                "the model weights.")
        elif mode == mod_modes.ModuleModes.VALIDATION:
            parser.add_argument("--weights-dir", type=str,
                                help="Path to the folder in which to store "
                                "the model weights.")
            parser.add_argument("--input-data-file", type=str,
                                help="Path to input data file.")
            parser.add_argument("--output-data-file", type=str,
                                help="Output path where predictions will be "
                                "saved.")
        else:
            raise ModelArgumentDefinitionError(
                "The argument `mode` can either be `train` or `predict`.")
    elif module_type == mod_types.ModuleType.FEATURE_EXTRACTOR:
        parser.add_argument("--input-data-file", type=str, required=True,
                            help="Path to the input dataset file.")
        parser.add_argument("--output-data-file", type=str,
                            help="Output path where predictions will "
                            "be saved.")
        parser.add_argument("--weights-dir", type=str, required=True,
                            help="Path to the folder in which to store "
                            "the model weights.")
        parser.add_argument("--train-new", action="store_true",
                            help="""Set this flag if feature extractor
                            should be trained from scratch.""")
        parser.add_argument("--label-data-file", type=str,
                            required="--train-new" in sys.argv,
                            help="Path to the label dataset file.")
        parser.add_argument("--transformed-label-data-file", type=str,
                            required="--train-new" in sys.argv,
                            help="Path to the transformed label dataset "
                            "file.")
        parser.add_argument("--config-choice-file", type=str,
                            required="--train-new" in sys.argv,
                            help="Path to the configuration file which "
                            "contains the config choice for the model.")
    elif module_type == mod_types.ModuleType.OBJECTIVE:
        parser.add_argument("--config-choice-file", type=str, required=True,
                            help="Path to the configuration file which "
                            "contains the config choice for this objective.",
                            default="")
        parser.add_argument("--predictions-paths", type=str, required=True,
                            help="Path to the json file, that contains a list "
                            "of paths to the prediction dataset files of the "
                            "different splits.")
        parser.add_argument("--labels-paths", type=str, required=True,
                            help="Path to the json file that contains a list "
                            "of paths to the label dataset files of the "
                            "different split.")
        parser.add_argument("--train-labels-paths", type=str, required=False,
                            help="Path to the json file that contains a list "
                            "of paths to the train label dataset files of the "
                            "different split.")
        parser.add_argument("--output-file", type=str, required=True,
                            help="Path to json file in which to store the "
                            "final predictions.")
    elif module_type == mod_types.ModuleType.OPTIMIZER:
        parser.add_argument("--config-file", type=str, required=True,
                            help="Path to the json file containing the entire "
                            "logical plan of the workflow")
        parser.add_argument("--history-file", type=str, required=True,
                            help="Path to the history file which contains a "
                            "list of tried candidate_plans.")
        parser.add_argument("--output-data-file", type=str, required=True,
                            help="Path to the output file.")
        parser.add_argument("--weights-dir", type=str, required=True,
                            help="Path to dir that contains the optimizer "
                            "weights.")
        parser.add_argument("--params-path", type=str, required=True,
                            help="Path to the json file that contains the "
                            "parameters (config choice) of the optimizer.")
        parser.add_argument("--minimization", action="store_true",
                            help="Whether it is a minimization or a "
                            "maximization task.")
        parser.add_argument("--num-cp", type=int,
                            help="Number of tasks to generate.", default=1)
    elif module_type == mod_types.ModuleType.TRANSFORMER:
        parser.add_argument("--input-data-file", type=str,
                            help="Path to input data file.")
        parser.add_argument("--output-data-file", type=str,
                            help="Output path where predictions will be "
                            "saved.")
    else:
        raise NotImplementedError
    return parser
