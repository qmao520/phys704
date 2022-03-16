# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the code that creates the solution app and adds the
resources and the error handling to it.
"""
import flask
import flask_cors

from solution_server import resources
from solution_server import server_utils as su
from solution_server import rest_api


def create_app(config_path: str, src_dir: str) -> flask.Flask:
    """Create a runnable flask app that serves the solution server.

    Args:
        config_path (str): Path to the solution api config yml file.
        src_dir (str): Path to the src directory that contains the modules
            (and other non-code information like e.g. the metadata)
            in a downloaded solution.

    Returns:
        flask.Flask: A flask app that can be run with app.run().
    """
    # Create and start the app.
    app = flask.Flask(__name__)
    app.register_error_handler(404, su.wrong_url_response)
    api = rest_api.SolutionRestAPI(app)
    flask_cors.CORS(app)
    api.add_resource(
        resources.PredictResource, "/predict",
        resource_class_kwargs={
            "src_dir": src_dir,
            "config_path": config_path
        }
    )
    return app
