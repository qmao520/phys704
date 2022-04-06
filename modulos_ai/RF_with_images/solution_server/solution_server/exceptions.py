# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains errors for the solution server.
"""


class APIConfigError(Exception):
    """Errors in the config yaml of the solution api.
    """
    pass


class Base64DecodingError(Exception):
    """Errors in the base64 decoding of tensors.
    """
    pass


class TensorInputFormatError(Exception):
    """Error, if a tensor is given in a wrong format."""
    pass
