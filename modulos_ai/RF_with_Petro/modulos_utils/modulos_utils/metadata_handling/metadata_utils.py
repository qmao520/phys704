# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
Helper functions for metadata handler and properties.
"""
import os
import joblib
from typing import Dict

from modulos_utils.metadata_handling import metadata_properties as meta_prop


class WrongVersionError(Exception):
    """ Error that is being raised if the version of an object or dictionary
    does not match the latest implementation of the AllProperties class.
    """
    pass


def check_version(meta_dict_version: float) -> None:
    """ Ensure that the version of the metadata dictionary corresponds
    to the latest version of the AllProperties class.

    Args:
        meta_dict_version (float): metadata dictionary version

    Returns:
        None

    Raises:
        WrongVersionError: if the AllProperties version and the metadata
        dictionary version do not match.
    """
    if meta_dict_version != meta_prop.MetaDataVersionProperty()._default:
        raise WrongVersionError(
            f"The metadata dictionary ({meta_dict_version}) does not "
            f"correspond to the latest version of the "
            f"AllProperties class "
            f"({meta_prop.MetaDataVersionProperty()._default})."
        )
    return None


class MetadataDumper:
    """ Save and load metadata from a binary file.

    Attributes:
        file path: path pointing to binary file where data will be saved/loaded
            from
        meta_dict: metadata dictionary, containing metadata for multiple nodes
        meta_obj: metadata object, containing metadata for a single node
        is_dictionary: bool, are we saving/loading a metadata dictionary
            or a single metadata object
    """

    def __init__(self):
        """ Init for metadata dumper class.
        """
        self.file_path = None
        self.meta_dict = None
        self.meta_obj = None
        self.is_dictionary = False

    def _check_extension(self) -> None:
        """ Check if file path has the correct extension.

        Returns:
            None.
        Raises:
            TypeError: if file path does not point to a binary file.
        """

        _, file_ext = os.path.splitext(self.file_path)
        if file_ext != ".bin":
            raise TypeError(f"Expected file path to point to '.bin' file, not"
                            f"'{file_ext}."
                            )
        return None

    def _check_file_not_exist(self) -> None:
        """ Ensure that the file that the file path is pointing to does NOT
        exist already.

        Returns:
            None.
        Raises:
            FileExistsError: if file exists already.
        """

        if os.path.exists(self.file_path):
            raise FileExistsError(f"File {self.file_path} exists already.")
        return None

    def _check_file_exist(self) -> None:
        """ Ensure that file that file path is pointing to does exist.

        Returns:
            None.
        Raises:
            FileNotFoundError: if file does not exist.
        """

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The file {self.file_path} could not be found.")
        return None

    def _check_type(self) -> None:
        """ Ensure that the metadata object/the metadata dictionary has the
        right type.

        Returns:
            None.
        Raises:
            TypeError: if metadata object is not an instance of the
                AllProperties class or if the metadata dictionary does not
                contain AllProperties objects.
        """
        if not self.is_dictionary:
            if not isinstance(self.meta_obj, meta_prop.AllProperties):
                raise TypeError("Expected input/content to be an "
                                "AllProperties object."
                                )
        else:
            for key, value in self.meta_dict.items():
                if not isinstance(value, meta_prop.AllProperties):
                    raise TypeError(f"The value of key '{key}' is not an"
                                    "AllProperties object."
                                    )
        return None

    def write_single_node(self, input: meta_prop.AllProperties,
                          file_path: str) -> None:
        """ Save metadata for single node in binary file.

        Args:
            input (meta_prop.AllProperties): metadata object
            file_path (str): file path

        Returns:
            None.
        """
        self.meta_obj = input
        self.file_path = file_path
        self.is_dictionary = False

        self._check_extension()
        self._check_file_not_exist()
        self._check_type()

        joblib.dump(self.meta_obj, self.file_path)
        return None

    def write_all_nodes(self, input: Dict[str, meta_prop.AllProperties],
                        file_path: str) -> None:
        """ Save metadata for multiple nodes in a binary file.

        Args:
            input (Dict[str, meta_prop.AllProperties]): dictionary containing
                multiple metadata objects
            file_path (str): file path

        Returns:
            None.
        """
        self.meta_dict = input
        self.file_path = file_path
        self.is_dictionary = True

        self._check_extension()
        self._check_file_not_exist()
        self._check_type()
        joblib.dump(self.meta_dict, self.file_path)
        return None

    def load_single_node(self, file_path: str) -> meta_prop.AllProperties:
        """ Load metadata for a single node from a binary file.

        Args:
            file_path (str): file path

        Returns:
            meta_prop.AllProperties: metadata object.
        """
        self.file_path = file_path
        self.is_dictionary = False

        self._check_extension()
        self._check_file_exist()

        self.meta_obj = joblib.load(self.file_path)
        self._check_type()

        return self.meta_obj

    def load_all_nodes(self, file_path: str) -> \
            Dict[str, meta_prop.AllProperties]:
        """ Load metadata for multiple nodes from a binary file.

        Args:
            file_path (str): file path.

        Returns:
            Dict[str, meta_prop.AllProperties]: dictionary containing metadata
                objects for multiple nodes.
        """
        self.file_path = file_path
        self.is_dictionary = True

        self._check_extension()
        self._check_file_exist()
        self.meta_dict = joblib.load(self.file_path)
        self._check_type()

        return self.meta_dict
