# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the flask app that is used to serve the solution REST
API.
"""
import flask
import flask_restful
import logging

from modulos_utils.predict import generate_predictions as gp
from solution_server import exceptions
from solution_server import server_utils as su


class SolutionRestAPI(flask_restful.Api):
    """
    A class, which overrides the function handle_error() of the flask_restful
    API class.
    """

    def handle_error(self, exception: Exception) -> flask.Response:
        """Handle all unhandled exceptions individually.

        Args:
            exception (Exception): Exception that is to be handled.
        """
        # First we log the exception with its full stack trace.
        logging.exception(exception)

        # No we formulate the response, that only contains the exception
        # message and its code.
        if isinstance(exception, gp.SampleInputError):
            return su.Response(422, message="The input sample is invalid: "
                               f"{str(exception)}").make()
        elif isinstance(exception, gp.FeatureExtractorError):
            return su.Response(500, "Error in the feature extractor: "
                               f"{str(exception)}").make()
        elif isinstance(exception, gp.ModelError):
            return su.Response(500,
                               f"Error in the model: {str(exception)}").make()
        elif isinstance(exception, exceptions.APIConfigError):
            return su.Response(500, "Error in the solution api config: "
                               f"{str(exception)}").make()
        elif isinstance(exception, exceptions.TensorInputFormatError):
            return su.Response(400, "Reading tensor(s) from the request "
                               f"failed: {str(exception)}").make()
        else:
            return su.Response(500, "Internal Server Error.").make()
