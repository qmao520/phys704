# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains exceptions that can be used in the modules.
"""


class PredictionNANError(Exception):
    """Class for when a model predicts NaN values.
    """
    pass
