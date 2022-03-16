# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import argparse
import joblib
import numpy as np
import os
import sklearn.ensemble as sk_ens
from typing import Optional

from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.module_interfaces import argument_definitions as cmdl_args
from modulos_utils.module_interfaces import module_types as mod_types
from modulos_utils.module_interfaces import module_modes as mod_modes
from modulos_utils.data_handling import data_handler
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.metadata_handling import metadata_properties as meta_prop


class RandomForestClassProbError(Exception):
    """Exception class for errors in the random forest classifier.
    """
    pass


class RandomForestClassProbModel:
    """Wrapper around sklearn random forest for classification that outputs
    probabilities.
    """

    def __init__(self):
        """Initialize object. Define which member variables the class has.
        """
        self._sklearn_model: Optional[sk_ens.RandomForestClassifier] = \
            None
        self._upload_unique_labels: Optional[np.ndarray] = None

    @classmethod
    def load_from_weights(cls, weights_dir: str) -> \
            "RandomForestClassProbModel":
        """Load trained model from saved weights.

        Args:
            weights_dir: Path to directory containing weights. Note that these
                are not only the pure sklearn weights, but all information,
                that is necessary for this class to be constructed.
        """
        obj = cls()
        obj._sklearn_model = joblib.load(
            os.path.join(weights_dir, "model.bin"))
        obj._upload_unique_labels = joblib.load(
            os.path.join(weights_dir, "upload_unique_labels.bin"))
        return obj

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run predictions on an input n x m array (n is the number of samples
        and m the number of features) and return predictions as a one
        dimensional array.

        Args:
            input_data (np.ndarray): Input dataset.

        Returns:
            np.ndarray: Predictions as a one-dimensional array.
        """
        if self._sklearn_model is None or self._upload_unique_labels is None:
            raise RandomForestClassProbError(
                "The model has not been initialized properly. Use the "
                "function 'load_from_weights'.")

        preds = self._sklearn_model.predict_proba(input_data)
        return preds


def main(args: argparse.Namespace) -> None:
    """
    Main function of the predict script.

    Args:
        args (argparse.Namespace): Parsed arguments for the script.
    """
    # Get the data.
    dataset_converter = dc.DatasetConverter()
    input_data = dataset_converter.read_data_as_matrix(args.input_data_file)

    # Load model and get predictions.
    model = RandomForestClassProbModel.load_from_weights(args.weights_dir)
    predictions = model.predict(input_data.data_matrix)

    # Write output file.
    dataset_converter.write_model_predictions(
        predictions, args.output_data_file)

    # Add label meta data to prediction hdf5.
    with data_handler.get_Dataset_writer(
        dataset_file_path=args.output_data_file,
        n_samples=dataset_converter._number_of_samples
    ) as writer:
        meta_obj = meta_prop.AllProperties()
        meta_obj.unique_values.set(model._upload_unique_labels)
        meta_handler.PropertiesWriter().write_to_ds_writer(
            meta_obj, dc.PREDICTIONS_NODE_NAME, writer)


if __name__ == "__main__":
    cmdline_description = "Run random forest prediction script."
    parser = cmdl_args.get_argument_parser(
        os.path.abspath(__file__), mod_types.ModuleType.MODEL,
        mod_modes.ModuleModes.VALIDATION, description=cmdline_description)
    args = parser.parse_args()
    main(args)
