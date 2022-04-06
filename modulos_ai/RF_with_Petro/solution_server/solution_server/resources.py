# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the resources of the solution rest api.
"""
import jsonschema
import os
from typing import Dict, Tuple

import flask
import flask_restful
import werkzeug

from modulos_utils.predict import generate_predictions as gp
from solution_server import data_utils as du
from solution_server import server_utils as su


DATA_KEY = "data"
REQUEST_BODY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        DATA_KEY: {
            "type": "array",
            "items": {
                "type": "object"
            }
        }
    },
    "required": [DATA_KEY],
    "additionalProperties": False
}


class PredictResource(flask_restful.Resource):
    """This class contains the prediction endpoint of the solution rest api.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructor of the PredictResource class. It sets the config path
        and calls the parent constructor.

        Args:
            args: Optional unnamed arguments.
            kwargs: Optional keyword arguments.
        """
        self.src_dir = kwargs.get("src_dir")
        self.config_path = kwargs.get("config_path")
        if self.src_dir is None or self.config_path is None:
            raise ValueError(
                "The arguments `src_dir` and/or "
                "`config_path` are missing.")
        self.api_config = su.load_config_file(self.config_path)
        super(PredictResource, self).__init__()
        return None

    def _validate_request_body(self, body_as_dict: Dict) -> Tuple[bool, str]:
        """Validate the request format.

        Args:
            Tuple[bool, str]: A tuple where the first entry defines whether
                the validation was successful (True) or not (False) and the
                second entry is the error message (defaults to an empty str).
        """

        # Perform some basic json validation.
        message = ""
        status = True
        try:
            jsonschema.validate(
                instance=body_as_dict, schema=REQUEST_BODY_JSON_SCHEMA)
        except jsonschema.ValidationError as e:
            message = str(e)
            status = False
            return status, message

        # Check that exactly one sample is given. Note that this
        # is the only data check we perform. Proper checking of the data
        # content beyond this is done by the solution code that is called
        # by this API.
        if len(body_as_dict[DATA_KEY]) != 1:
            message = ("The number of samples in the input data "
                       "must be exactly one.")
            status = False
        return status, message

    def post(self):
        """Post of the solution rest api. It calls the online client and
        returns the prediction.
        """
        # Convert request body to json and validate its format and content.
        try:
            payload = flask.request.json
        except werkzeug.exceptions.BadRequest as e:
            return su.Response(code=400, message=str(e)).make()
        status, msg = self._validate_request_body(payload)
        if not status:
            return su.Response(code=400, message=msg).make()

        # Retrieve input sample.
        sample = payload[DATA_KEY][0]

        # For all tensor nodes we decode or/and convert to numpy.
        sample_dict = du.load_all_tensors(sample, self.api_config)

        prediction = gp.predict(
            sample_dict,
            os.path.join(os.path.dirname(self.config_path),
                         self.api_config["tmp_dir"]),
            self.src_dir)

        response_data_single_sample = su.format_response_data(
            self.api_config["label_node_name"], prediction)

        # HACK: Convert the prediction back to a batch of predictions. Once
        # REB-850 is done, we pass a batch to the predict function, hence
        # we don't have to convert back here.
        response_data = [response_data_single_sample]

        return su.Response(
            code=200, data=response_data,
            message="Prediction succeeded.").make()
