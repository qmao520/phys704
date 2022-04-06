# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the resources of the solution rest api.
"""
import copy
import flask
import os
import yaml
from typing import Dict, List, Optional


from solution_server import exceptions


API_CONFIG_KEYS = [
    "tensor_file_extensions", "label_file_extension",
    "label_node_name", "metadata_path", "tmp_dir", "port", "host"
]
SUPPORTED_TENSOR_EXTENSIONS = [".jpg", ".npy", ".png", ".tif"]


class Response:
    """Custom response class that is a wrapper around `flask.make_response`.
    """

    CODE = "code"
    MESSAGE = "message"
    RESPONSE_CONTENT = "response"
    DATA = "data"

    def __init__(
            self, code: int, message: str,
            data: Optional[List[Dict]] = None) -> None:
        """Constructor of the custom response class.

        Args:
            code (int): Http code of the response.
            message (str): Response message.
            data (Optional[List[Dict]], optional): Data of the response,
                e.g. the predictions in a list. Defaults to None.
        """
        self.code = code
        self.message = message
        if data is None:
            self.data = []
        else:
            if not isinstance(data, list):
                raise ValueError("The response data must be a list.")
            self.data = copy.deepcopy(data)
        return None

    def to_dict(self) -> Dict:
        """Convert the response object to a json serializable dictionary.

        Returns:
            Dict: Response as a json serializable dictionary.
        """
        return {
            Response.CODE: self.code,
            Response.RESPONSE_CONTENT: {
                Response.MESSAGE: self.message, Response.CODE: self.code},
            Response.DATA: copy.deepcopy(self.data)
        }

    def make(self) -> flask.Response:
        """Convert this object to a flask response, i.e. call the flask
        function `make_response` with the fields of this response.

        Returns:
            flask.Response: Flask response.
        """
        return flask.make_response(flask.jsonify(self.to_dict()), self.code)


def wrong_url_response(e):
    """Error response for the case where the url is wrong. This overrides
    the builtin flask 404 error.

    Returns:
        flask.Response: Flask response.
    """
    return Response(code=404, message="This url does not exist.").make()


def load_config_file(config_path: str) -> Dict:
    """Load config file of the solution dir that is given as input to the
    start script.

    Args:
        config_path (str): Path to the config file of the server.

    Returns:
        Dict: Loaded config as a dictionary.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict


def check_config_file(config_path: str, src_path: str) -> None:
    """Check whether the api config file contains the necessary keys, and
    their values are allowed.

    Args:
        solution_dir (str): Path to the solution directory.
        config_path (str): Path to the api config yml.
        src_path (str): Path to the src directory.

    Raises:
        exceptions.APIConfigError: Errors for when the server is misconfigured,
            i.e. the config file is faulty.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Raise an error, if any of the required keys is missing.
    missing_keys = [k for k in API_CONFIG_KEYS if k not in config_dict]
    if missing_keys != []:
        raise exceptions.APIConfigError(
            f"The solution API config file at {config_path} has the "
            f"following missing keys: {missing_keys}.")

    # Raise an error if the api config file contains an unsupported tensor
    # file extension.
    unsupported_file_extensions = [
        ext for ext in config_dict["tensor_file_extensions"].values()
        if ext not in SUPPORTED_TENSOR_EXTENSIONS]
    if unsupported_file_extensions != []:
        raise exceptions.APIConfigError(
            f"The solution API config file at {config_path} contains the "
            "following tensor file extensions, that are not supported: "
            f"{unsupported_file_extensions}.")

    # Raise an error if the api config file contains a metadata path that
    # does not exist.
    metadata_abs_path = os.path.join(src_path, config_dict["metadata_path"])
    if not os.path.isfile(metadata_abs_path):
        raise exceptions.APIConfigError(
            "The metadata path, that is given in the solution api config "
            f"file at {config_path}, does not exist: {metadata_abs_path}")
    return None


def format_response_data(label_name: str, predictions: Dict) -> Dict:
    """Format the response data, by adding the prediction np.ndarray to
    a dictionary with the key <label_name>__predicted.

    Args:
        label_name (str): Name of the label node, as read from the api config.
        predictions (Dict): The prediction dictionary that is returned by
            the function `modulos_utils.predict.generate_predictions.predict`.
    """
    predictions_copy = copy.deepcopy(predictions)
    predictions_name = f"{label_name}__predicted"
    predictions_list = predictions_copy.pop("predictions")
    predictions_copy[predictions_name] = predictions_list
    return predictions_copy
