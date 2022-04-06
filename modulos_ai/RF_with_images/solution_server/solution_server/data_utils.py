# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the resources of the solution rest api.
"""
import base64
import io
from typing import Callable, Dict, List, Union

import numpy as np
from PIL import Image as PILImage


from solution_server import exceptions
from modulos_utils.predict import generate_predictions as gp


def _decode_base64_string(
        base64_string: str, buffer_to_array: Callable) -> np.ndarray:
    """Decode a base64encoded .npy/.png/.jpg/.tif file and return it as a
    np.ndarray.

    Args:
        base64_string (str): String that represents a base64 encoded file.
        buffer_to_array (Callable): Function that is used to convert the
            buffer of the decoded bytes to a numpy array.

    Returns:
        np.ndarray: Decoded and loaded numpy array.
    """
    try:
        tensor_bytes = base64.b64decode(base64_string.encode("utf-8"))
        buf = io.BytesIO(tensor_bytes)
        decoded_array = buffer_to_array(buf)
    except Exception as e:
        raise exceptions.TensorInputFormatError(str(e)) from e
    return decoded_array


def load_tensor_to_numpy(
        raw_node: Union[List, str], node_name: str,
        on_platform_tensor_file_ext: str,
        on_platform_shape: List[int]) -> np.ndarray:
    """This function takes the raw tensor node as it was sent by the
    request, and converts it to a numpy array (either by base64 decoding
    it and loading it with npy or PIL, or by calling .tolist()).

    Args:
        raw_node (Union[List, str]): The node, as it is retrieved from
            the request json.
        node_name (str): Name of the tensor node. This is only needed for
            error messages.
        on_platform_tensor_file_ext (str): File extension of the tensor
            as it was uploaded to the platform originally.
        on_platform_shape (List[int]): Shape of the node as it was used
            on-platform. This information should be retrieved from the
            downloaded metadata.

    Returns:
        np.ndarray: A numpy array of the tensor node.
    """
    if isinstance(raw_node, str):
        if on_platform_tensor_file_ext == ".npy":
            return _decode_base64_string(
                raw_node, lambda x: np.load(x))
        elif on_platform_tensor_file_ext in [".jpg", ".png", ".tif"]:
            image = _decode_base64_string(
                raw_node, lambda x: np.array(PILImage.open(x)))

            # We enforce that the type is float.
            image = image.astype(float)
            # HACK: To simplify schema matching, we enforce all images (.jpg,
            # .png, and .tif) to have three dimension on the platform, i.e. in
            # the dataset upload, we add an empty dimension for monochromatic
            # images: e.g. [32, 32] -> [32, 32, 1]. This causes a discrepancy
            # between the original images and the images that the model trains
            # with. To correct for this, we have to add an empty dimension
            # for monochromatic images here as well.
            # TODO (REB-997) Remove the hack in the upload and therefore the
            # need for this hack.
            image_shape = list(image.shape)
            if len(on_platform_shape) == 3 and on_platform_shape[-1] == 1 \
                    and len(image_shape) == 2:
                image = np.expand_dims(image, axis=-1)
            elif on_platform_shape != image_shape:
                raise exceptions.TensorInputFormatError(
                    f"The input tensor `{node_name}` has shape "
                    f"{image_shape}, "
                    "however the server config file specifies that it should "
                    f"have shape {on_platform_shape}.")

            return image

        else:
            raise exceptions.APIConfigError(
                f"The file extension `{on_platform_tensor_file_ext}`, "
                "that is specified (in the api config file) for feature "
                f"`{node_name}`, is not a supported tensor file "
                "extension.")
    elif isinstance(raw_node, list):
        return np.array(raw_node)
    else:
        raise exceptions.TensorInputFormatError(
            f"The data of the tensor feature `{node_name}` was given in a "
            "format that cannot be understood.")


def load_all_tensors(
        sample_dict: Dict, api_config_dict: Dict) -> Dict:
    """This function iterates over all fields in the dictionary
    `sample_dict` and checks in the api config dict, whether it is
    a tensor node and whether it is a base64 encoded string. Tensors
    that were base64 encoded, are decoded and converted to numpy arrays.
    Tensors, that are not encoded, are converted to numpy arrays directly.

    Attention:  Note that this function does not create a copy of the
        sample dictionary. The input is therefore modified!

    Args:
        sample_dict (Dict): Dictionary that represents one sample of the
            data that the user sends.
        api_config_dict (Dict): Config dict of the API.

    Returns:
        Dict: Modified sample dictionary.
    """
    for node_name, file_ext in api_config_dict[
            "tensor_file_extensions"].items():
        node_dim = api_config_dict["tensor_dimensions"][node_name]
        if node_name not in sample_dict:
            raise gp.SampleInputError(
                f"Feature `{node_name}`, which is present in the solution "
                "api config file, is missing from the input sample.")
        sample_dict[node_name] = load_tensor_to_numpy(
            sample_dict[node_name], node_name, file_ext, node_dim)
    return sample_dict
