# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
Handles the transfer of metadata from one internal dataset to another.
"""

from typing import Dict

from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.data_handling import data_handler as dh


def transfer_single_node_obj_to_obj(old_meta_obj: meta_prop.AllProperties) \
        -> meta_prop.AllProperties:
    """ Given a metadata object for a single node, transfer it a new metadata
    object.

    NOTE: Non computable properties (e.g. component name) are copied.
    Computable properties are set to their default values.

    NOTE: the version will never be copied. The version key will be set when
    we create a new default metadata object, i.e. the version of the
    new object will always correspond to the latest version of the
    AllProperties class.

    Args:
        old_meta_obj (meta_prop.AllProperties): original metadata object

    Returns:
        meta_prop.AllProperties: new metadata object
    """

    # Create new metadata object with default values.
    new_meta_obj = meta_prop.AllProperties()

    for prop in new_meta_obj.all_props:
        # Check if property is computable and if it corresponds to the
        # version.
        if not prop.is_computable \
                and not isinstance(prop,
                                   meta_prop.MetaDataVersionProperty
                                   ):
            # Non-computable: copy to new metadata object.
            prop.set(old_meta_obj.get_values()[prop.key])

        # Copy node type and node dim from old to new dataset.
        # NOTE: this is a hack! BAS-678
        new_meta_obj.node_type.set(old_meta_obj.node_type.get())
        new_meta_obj.node_dim.set(old_meta_obj.node_dim.get())

    return new_meta_obj


class DatasetTransferer:
    """ Metadata Transferer Class for entire datasets: transfer metadata from
    one dataset to another.

    Note: during the transfer, computable metadata properties will be set to
    their default values; non-computable properties will be copied.

    Args:
        meta_dict (Dict[str, meta_prop.AllProperties]): Dictionary consisting
            of node names and the corresponding metadata objects; this
            dictionary contains the "old" metadata objects, i.e. before the
            transfer
    """

    def __init__(self, meta_dict: Dict[str, meta_prop.AllProperties]):
        """ Init for DatasetTransferer class.

        Args:
            meta_dict (Dict[str, meta_prop.AllProperties]): Dictionary
                consisting of node names and the corresponding metadata
                objects; this dictionary contains the "old" metadata
                objects, i.e. before the transfer
        """
        self.meta_dict = meta_dict

    @classmethod
    def from_ds(cls, dataset_path: str) -> "DatasetTransferer":
        """ Initialize class from a dataset.

        Args:
            dataset_path (str): path to original dataset

        Returns:
            DatasetTransferer: return initialized class
        """

        meta_dict = {}
        all_node_names = dh.DatasetReader(dataset_path).get_node_names()
        if dh.SAMPLE_IDS in all_node_names:
            all_node_names.remove(dh.SAMPLE_IDS)
        for node in all_node_names:
            meta_dict[node] = \
                meta_handler.PropertiesReader().read_from_ds_path(node,
                                                                  dataset_path
                                                                  )
        return cls(meta_dict)

    @classmethod
    def from_dict(cls, meta_dict: Dict[str, meta_prop.AllProperties]) \
            -> "DatasetTransferer":
        """ Initialize class from a metadata dictionary.

        Args:
            meta_dict (Dict[str, meta_prop.AllProperties]): metadata dictionary
                containing node names and corresponding metadata objects for
                original dataset

        Returns:
            DatasetTransferer: return initialized class
        """
        return cls(meta_dict)

    def get_dict(self) -> Dict[str, meta_prop.AllProperties]:
        """ Return dictionary containing original nodes and corresponding
        transferred metadata objects.

        Returns:
            Dict [str, meta_prop.AllProperties]: transferred metadata
                dictionary
        """

        new_meta_dict = {}
        for node, node_obj in self.meta_dict.items():
            new_meta_dict[node] = transfer_single_node_obj_to_obj(node_obj)
        return new_meta_dict

    def save_new_ds_with_ds_witer(self, ds_writer: dh.DatasetWriter) -> None:
        """ Save transferred metadata to new dataset directly using a given
        dataset writer.

        Note:
            - nodes must already exist in new dataset and these nodes
        must already contain data

        Args:
            ds_writer (dh.DatasetWriter): dataset writer object

        Returns:
            None
        """
        new_meta_dict = self.get_dict()
        for node, node_obj in new_meta_dict.items():
            # Write metadata to new hdf5 file.
            meta_handler.PropertiesWriter().write_to_ds_writer(
                node_obj, node, ds_writer
            )
        return None

    def save_new_ds(self, new_dataset_path: str) -> None:
        """ Save transferred metadata to new dataset.

        Note:
            - nodes must already exist in new dataset and these nodes
        must already contain data
            - the dataset must have been validated.

        Args:
            new_dataset_path (str): path to new dataset

        Returns:
            None
        """
        new_meta_dict = self.get_dict()
        for node, node_obj in new_meta_dict.items():

            # Write metadata to new hdf5 file.
            meta_handler.PropertiesWriter().write_to_ds_path(node_obj, node,
                                                             new_dataset_path
                                                             )
        return None


class NodeTransferer:
    """ Metadata Transferer class for single node: transfer metadata from
    a single node to another node.

    Note: during the transfer, computable metadata properties will be set to
    their default values; non-computable properties will be copied.

    Args:
        meta_obj: original metadata object
    """

    def __init__(self, meta_obj: meta_prop.AllProperties):
        """ Init for NodeTransferer class.

        Args:
            meta_obj (meta_prop.AllProperties): original metadata object
        """
        self.meta_obj = meta_obj

    @classmethod
    def from_ds(cls, node: str, dataset_path: str) -> "NodeTransferer":
        """ Initialize class from a dataset.

        Args:
            node (str): name of original node
            dataset_path (str): path to original node

        Returns:
            NodeTransferer: return initialized class
        """
        return cls(meta_handler.PropertiesReader().read_from_ds_path(
            node, dataset_path)
        )

    @classmethod
    def from_obj(cls, meta_obj: meta_prop.AllProperties) -> "NodeTransferer":
        """ Initialize class from metadata object.

        Args:
            meta_obj (meta_prop.AllProperties): original metadata object

        Returns:
            NodeTransferer: return initialized class
        """
        return cls(meta_obj)

    def get_obj(self) -> meta_prop.AllProperties:
        """ Return transfer3ed object.

        Returns:
            meta_prop.AllProperties: new, transferred metadata object
        """
        return transfer_single_node_obj_to_obj(self.meta_obj)

    def save_new_node_with_ds_witer(self, new_node: str,
                                    ds_writer: dh.DatasetWriter) -> None:
        """ Save new node to a new dataset, directly using a given
        DatasetWriter.

        Args:
            new_node (str): name of new node
            ds_writer (dh.DatasetWriter): dataset writer object

        Returns:
            None
        """
        new_meta_obj = self.get_obj()
        meta_handler.PropertiesWriter().write_to_ds_writer(
            new_meta_obj, new_node, ds_writer
        )
        return None

    def save_new_node(self, new_node: str, new_dataset_path: str) -> None:
        """ Save transferred node to a new dataset.

        Note:
            - the node must exist already and must already contain data.
            - the dataset has to be validate.

        Args:
            new_node (str): name of new node
            new_dataset_path (str): path to new dataset

        Returns:
            None
        """

        new_meta_obj = self.get_obj()
        meta_handler.PropertiesWriter().write_to_ds_path(new_meta_obj,
                                                         new_node,
                                                         new_dataset_path
                                                         )
        return None
