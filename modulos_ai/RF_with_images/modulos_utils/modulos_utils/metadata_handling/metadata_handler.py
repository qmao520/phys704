# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
Metadata handler including read, write, and transfer methods.
"""

from typing import Dict
import warnings

from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.data_handling import data_handler as dh


class MissingKeysError(Exception):
    """ Error that is being raised if keys are missing from the internal
    dataset metadata entry.
    """
    pass


class PropertiesReader:
    """ Read in metadata from a dataset and convert it into an object which
    allows access to its properties (e.g. mean, component name, node type,
    standard deviation, etc.)
    """

    def __init__(self):
        """ Init for Properties Reader Class.
        """
        pass

    def _check_keys(self, meta_dict: Dict) -> None:
        """ Ensure that the metadata dictionary contains all the keys required
        by the AllProperties class. Warn the user if the metadata dictionary
        contains keys which will be ignored.

        Args:
            meta_dict (Dict): metadata dictionary

        Returns:
            None

        Raises:
            MissingKeysError: if required keys are missing from the metadata
            dictionary.
        """
        missing_keys = []
        required_keys = meta_prop.AllProperties().get_keys()
        for key in required_keys:
            if key not in meta_dict:
                missing_keys.append(key)
        if missing_keys:
            raise MissingKeysError("The following keys are missing from the"
                                   f" metadata dictionary: {missing_keys}")
        # Check if the metadata contains additional keys which will be ignored.
        ignored_keys = [key for key in meta_dict.keys()
                        if key not in required_keys
                        ]
        if ignored_keys:
            warnings.warn("The following metadata keys will be ignored: "
                          f"{ignored_keys}")
        return None

    def _deserialize_data(self, meta_dict: Dict) -> meta_prop.AllProperties:
        """ Create a new AllProperties object and set its values to the
        values given in the metadata dictionary.

        Args:
            meta_dict (Dict): metadata dictionary

        Returns:
            AllProperties object: new AllProperties object
        """
        new_obj = meta_prop.AllProperties()
        for obj in new_obj.all_props:
            obj.import_from_array(meta_dict[obj.key])

        return new_obj

    def read_from_ds_reader(self, node_name: str,
                            dataset_reader: dh.DatasetReader) \
            -> meta_prop.AllProperties:
        """ Given a dataset reader object and a node name, create
        a new AllProperties object which can be updated, used to determine if
        a node is numerical, etc.

        Args:
            node_name (str): node name
            dataset_reader (dh.DatasetReader): dataset reader object

        Returns:
            AllProperties object: new AllProperties object
        """
        # Save metadata in dictionary.
        meta_dict = dataset_reader.get_metadata_of_node(node_name)

        # Ensure that the metadata dictionary contains all
        # the required keys.
        self._check_keys(meta_dict)
        # Create a new AllProperties object and set its properties.
        new_obj = self._deserialize_data(meta_dict)

        # Ensure that the meta data dictionary was created with the latest
        # version.
        meta_utils.check_version(new_obj._version.get())

        return new_obj

    def read_from_ds_path(self, node_name: str, dataset_path: str) \
            -> meta_prop.AllProperties:
        """ Given an internal dataset file and a node name, create a new
        AllProperties object which can be updated, used to determine if a
        node is numerical, etc.

        Args:
            node_name (str): node name
            dataset_path (str): path to internal dataset file

        Returns:
            AllProperties object: new AllProperties object
        """
        # Read in internal dataset file.
        dataset_reader = dh.DatasetReader(dataset_path)
        new_metadata_obj = self.read_from_ds_reader(node_name, dataset_reader)
        return new_metadata_obj


class PropertiesWriter:
    """ Given an AllProperties metadata object, save in the internal dataset
    format.
    """

    def __init__(self):
        """ Init for the Properties Writer class.
        """
        pass

    def _check_instance(self, meta_obj: meta_prop.AllProperties) -> None:
        """ Ensure that the metadata object is indeed an instance of the
        AllProperties class.

        Args:
            meta_obj (AllProperties): AllProperties object

        Returns:
            None

        Raises:
            TypeError: if the metadata object is not an instance of the
            AllProperties class.
        """

        if not isinstance(meta_obj, meta_prop.AllProperties):
            raise TypeError("The meta object is not an instance of the "
                            "AllProperties class.")
        return None

    def _serialize_data(self, meta_obj: meta_prop.AllProperties) -> Dict:
        """ Create dictionary which will be saved in internal dataset format.
        To be able to save metadata all properties have to be converted to
        arrays.

        Args:
            meta_obj (object): metadata object

        Returns:
            Dict: the keys are the properties' keys, the values are the
            properties' values converted to arrays
        """
        meta_dict = {}
        for obj in meta_obj.all_props:
            meta_dict[obj.key] = obj.export_to_array()
        return meta_dict

    def write_to_ds_writer(self, meta_obj: meta_prop.AllProperties,
                           node_name: str,
                           dataset_writer: dh.DatasetWriter) -> None:
        """ Given an AllProperties object and a dataset writer, save
        the metadata object in the internal dataset format.

        NOTE: the hdf5 file must already contain data for this node!

        Args:
            meta_obj (AllProperties): AllProperties object
            node_name (str): node name
            dataset_writer (dh.DatasetWriter): dataset writer object

        Returns:
            None
        """
        # Ensure that input object is an instance of the AllProperties class.
        self._check_instance(meta_obj)
        # Ensure that the metadata object has been generated with the
        # latest version of the AllProperties class.
        meta_utils.check_version(meta_obj._version.get())
        # Get the properties values in the form of a dictionary.
        serialized_meta_dict = self._serialize_data(meta_obj)
        # Save in internal dataset format.
        dataset_writer.add_metadata_to_node(node_name, serialized_meta_dict)
        return None

    def write_to_ds_path(self, meta_obj: meta_prop.AllProperties,
                         node_name: str, dataset_path: str) -> None:
        """ Given an AllProperties object, save it in the internal dataset
        format.

        NOTE: the hdf5 file must already contain data for this node!

        Args:
            meta_obj (AllProperties): AllProperties object
            node_name (str): node name
            dataset_path (str): path to internal dataset

        Returns:
            None
        """
        n_samples = dh.DatasetReader(dataset_path).get_n_samples()
        with dh.get_Dataset_writer(dataset_path, n_samples) as dataset_writer:
            self.write_to_ds_writer(meta_obj, node_name, dataset_writer)

        return None


def get_metadata_all_nodes(dataset_path: str) -> \
        Dict[str, meta_prop.AllProperties]:
    """ Given an internal dataset, use the metadata handler to create a
    dictionary containing an AllProperties object for each node.

    Args:
        dataset_path (str): path to internal dataset

    Returns:
        dict: with node names as keys and the corresponding AllProperties
        objects as values
    """
    all_node_names = dh.DatasetReader(dataset_path).get_node_names()
    if dh.SAMPLE_IDS in all_node_names:
        all_node_names.remove(dh.SAMPLE_IDS)

    meta_dict = {}
    for node in all_node_names:
        meta_dict[node] = PropertiesReader().read_from_ds_path(node,
                                                               dataset_path
                                                               )

    return meta_dict


def save_metadata_all_nodes(meta_dict: Dict[str, meta_prop.AllProperties],
                            dataset_path: str) -> None:
    """ Given an internal dataset, add metadata to all nodes. Note: all nodes
    given in the metadata dictionary must already exist and must be filled
    with data.

    Args:
        meta_dict (Dict): node names as keys, AllProperties objects as values
        dataset_path (str): path to internal dataset

    Returns:
        None

    Raises:
        KeyError: if metadata dictionary keys do not match internal dataset
        node names
    """
    all_node_names = dh.DatasetReader(dataset_path).get_node_names()
    if dh.SAMPLE_IDS in all_node_names:
        all_node_names.remove(dh.SAMPLE_IDS)

    if set(meta_dict.keys()) != set(all_node_names):
        raise KeyError(f"The node names given in the metadata dictionary "
                       f"and the node names of the existing dataset file "
                       f"do not match. \n"
                       f"Metadata dict: {meta_dict.keys()}\n"
                       f"Internal dataset: {all_node_names}")

    for node in meta_dict.keys():
        PropertiesWriter().write_to_ds_path(meta_dict[node],
                                            node, dataset_path
                                            )

    return None
