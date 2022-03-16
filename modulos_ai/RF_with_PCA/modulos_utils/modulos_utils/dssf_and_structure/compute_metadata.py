# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
Compute metadata for newly created internal dataset.
"""
import math
import numpy as np
import os
from typing import List, Dict, Optional

from base.statistics.dataset import basic_computation as basic_comp
from base.statistics.dataset import single_node_computation as single_comp
from base.statistics.dataset import utils as meta_utils
from modulos_utils.data_handling import data_handler as dh
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.dssf_and_structure import DSSFErrors

# Default batch size: 1 GB.
DEFAULT_BATCH_SIZE = 10**9


class NewMetadataComputer:
    """ Metadata computer class for newly created internal dataset.
    Compute metadata properties which are needed for schema inference,
    workflow creation, label transformation, etc.

    Attributes:
        _node_list (List[str]): List of nodes for which metadata is computed.
        _ds_reader (dh.DatasetReader): Datahandler Reader for original dataset.
        _is_computed (bool): Is set to True after the new metadata has been
            computed.
        _meta_dict (Dict[str, meta_prop.AllProperties]): Dictionary containing
            metadata objects for each node.
        _non_basic_properties (List[str]): All metadata properties which are
            meant to be computed.
        _upload_properties (List[str]): Properties for which we manually
            want to set "upload_" + property.
        _dssf_properties (List[str]): Properties which should have been set
            by structure code.
        dshf (dshf_handler.DSHFHandler): History file instance to get
            information about components and nodes of the dataset.
    """
    def __init__(self,
                 ds_reader: dh.DatasetReader,
                 node_list: List[str],
                 ds_folder: str,
                 dshf_path: Optional[str] = None):
        """ Init for NewMetadata Computer class.

        Args:
            ds_reader (dh.DatasetReader): Datahandler Reader for original
                dataset.
            node_list (List[str]): List of nodes for which metadata is
                computed.
            ds_folder (str): Dataset folder path for looking up the history
                file, if there is no history file path given.
            dshf_path (Optional[str]): Path to the history file to look up
                information about the nodes and components. If it is None,
                it is assumed to be in the same directory as the dataset.
                Defaults to None.
        """
        self._node_list = node_list
        self._ds_reader = ds_reader
        dshf_path = dshf_path or os.path.join(
            ds_folder, dshf_handler.DSHF_FILE_NAME)
        self.dshf = dshf_handler.DSHFHandler(dshf_path)
        all_nodes = (self._ds_reader.get_node_names() +
                     self._ds_reader.get_datetime_node_names())
        for node in self._node_list:
            if node not in all_nodes:
                raise KeyError(f"Node \"{node}\" is not part of the given"
                               "dataset.")

        self._is_computed = False
        self._meta_dict: Dict[str, meta_prop.AllProperties] = {}

        self._non_basic_properties = ["min", "max",
                                      "nr_unique_values",
                                      "unique_values_subset",
                                      "unique_values",
                                      "samples_identical", "samples_sorted",
                                      "samples_equally_spaced"
                                      ]

        self._upload_properties = ["node_type",
                                   "node_dim",
                                   "min",
                                   "max",
                                   "nr_unique_values",
                                   "unique_values",
                                   "unique_values_subset",
                                   "schema_node_type"
                                   ]
        self._dssf_properties = ["dssf_component_name",
                                 "dssf_component_type",
                                 "dssf_component_file_path"
                                 ]

    @classmethod
    def from_ds_reader(cls, ds_reader: dh.DatasetReader, ds_folder: str,
                       dshf_path: Optional[str] = None)\
            -> "NewMetadataComputer":
        """ Instantiate class using a dataset reader object.

        Args:
            ds_reader (dh.DatasetReader): Dataset Reader.
            ds_folder (str): Dataset folder path.
            dshf_path (Optional[str]): Path to the history file to look up
                information about the nodes and components. If it is None,
                it is assumed to be in the same directory as the dataset.
                Defaults to None.

        Returns:
            "NewMetadataComputer": Instantiated class.
        """
        node_list = ds_reader.get_node_names()
        if dh.SAMPLE_IDS in node_list:
            node_list.remove(dh.SAMPLE_IDS)
        return cls(ds_reader, node_list, ds_folder, dshf_path)

    @classmethod
    def from_ds_path(cls, ds_path: str,
                     dshf_path: Optional[str] = None) -> "NewMetadataComputer":
        """ Instantiate class using a dataset path.

        Args:
            ds_path (str): Dataset path.
            dshf_path (Optional[str]): Path to the history file to look up
                information about the nodes and components. If it is None,
                it is assumed to be in the same directory as the dataset.
                Defaults to None.

        Returns:
            "NewMetadataComputer": Instantiated class.
        """

        ds_reader = dh.DatasetReader(ds_path)
        dataset_folder = os.path.dirname(ds_path)
        return cls.from_ds_reader(ds_reader, dataset_folder, dshf_path)

    def _get_batch_size_node(self, node: str) -> int:
        """ Determine batch size for specific node.

        Args:
            node (str): Node name.

        Returns:
            batch_size (int): Node specific batch size.
        """
        # Determine size of a single sample. Make sure sample is an array.
        # Scalars are returned as numpy int & float.
        sample = np.atleast_1d(
            next(self._ds_reader.get_data_of_node_in_batches(node, 1))[0]
        )
        # Determine size of sample in bytes.
        sample_size = sample.nbytes

        # Determine how many samples fit into default batch size.
        batch_size_node = int(
            math.floor(DEFAULT_BATCH_SIZE/float(sample_size))
        )

        return batch_size_node

    def _set_dssf_props_node(
            self, meta_obj: meta_prop.AllProperties,
            node_name: str, comp_name: str) -> meta_prop.AllProperties:
        """ Set DSSF properties for a specific node based on the corresponding
        Component object.

        Args:
            meta_obj (meta_prop.AllProperties): Metadata object for this node.
            node_name (str): Name of the node.
            comp_name (str): Name of the component containing the node.

        Returns:
            meta_prop.AllProperties: Updated metadata object.

        Raises:
            KeyError: If method is called before metadata has been computed.
        """
        comp_stats = self.dshf.dssf_info
        dssf_path = comp_stats[comp_name]["path"]
        file_path = (dssf_path if not isinstance(dssf_path, list)
                     else dssf_path[0])
        meta_obj.dssf_component_file_path.set(file_path)
        meta_obj.dssf_component_name.set(comp_name)
        meta_obj.dssf_component_type.set(comp_stats[comp_name]["type"])

        # Additionally set the upload node name (original node name for
        # internally created nodes).
        meta_obj.upload_node_name.set(
            self.dshf.current_to_upload_name[node_name])
        return meta_obj

    def _set_upload_props_node(self, node: str,
                               meta_obj: meta_prop.AllProperties) \
            -> meta_prop.AllProperties:
        """ Manually set upload properties. For all properties given in the
        `_upload_properties` list we set `upload_` + prop to the property's
        current value.

        Args:
            node (str): Node name.
            meta_obj (meta_prop.AllProperties): Metadata object for the
                given node.

        Returns:
            meta_prop.AllProperties: New metadata object.
        """
        for prop in self._upload_properties:
            prop_obj = getattr(meta_obj, prop)
            # Not all properties are applicable to all nodes.
            # Only copy if property is not set to default.
            if not prop_obj.is_default():
                getattr(meta_obj, "upload_" + prop).set(prop_obj.get())
            # Raise an error if the property is set to default and is
            # a basic property. Basic properties are computed for all
            # nodes and should always be set.
            else:
                if prop_obj.is_basic:
                    raise ValueError(f"The basic property '{prop}' for "
                                     f"node '{node}' has not been set.")

        return meta_obj

    def _compute_node(
            self, node: str, batch_size_node: int) \
            -> meta_prop.AllProperties:
        """ Compute metadata properties for single node.

        3 main steps:
        1) Compute basic properties.
        2) Add DSSF properties.
        3) Compute non-basic properties.
        4) Set upload properties.

        Args:
            node (str): Node name.
            batch_size_node (int): Batch size for this specific node.

        Returns:
            meta_prop.AllProperties: New metadata object for the given node.
        """
        try:
            meta_obj = meta_handler.PropertiesReader().read_from_ds_reader(
                node, self._ds_reader)
        except dh.MetaDataDoesNotExistError:
            meta_obj = meta_prop.AllProperties()

        # 1) Compute basic properties (only if they are not yet computed).
        if any(getattr(meta_obj, prop).is_default()
                for prop in meta_obj.get_basic_property_names()):
            basic_computer = basic_comp.BasicPropertiesComputer.from_ds_reader(
                node, self._ds_reader, batch_size_node
            )
            basic_computer.compute()
            meta_obj = basic_computer.get_meta_obj()

        if meta_obj.categorical.is_default():
            if node in self.dshf.node_type:
                meta_obj.categorical.set(
                    self.dshf.node_type[node] in
                    ["categorical", "bool"]
                )
                if self.dshf.node_type[node] == "probability" \
                        and node in self.dshf.node_type_default \
                        and self.dshf.node_type_default[node] in [
                            "categorical", "bool"]:
                    meta_obj.categorical.set(True)

        # NOTE: this is a hack. At the moment we still use the old node types
        # in parallel with the new node types. This will be removed.
        if not meta_obj.categorical.is_default() \
                and meta_obj.categorical.get() \
                and "_cat" not in meta_obj.node_type.get():
            meta_obj.node_type.set(
                meta_obj.node_type.get() + "_cat")
        if node in self._ds_reader.get_datetime_node_names():
            meta_obj.node_type.set("datetime")

        # Find corresponding DSSF component name.
        comp_name = self.dshf.get_component_name(node)
        # 2) Add DSSF properties.
        meta_obj = self._set_dssf_props_node(meta_obj, node, comp_name)

        # 3) Compute non-basic properties that have not been computed yet.
        uncomputed_props = [prop for prop in self._non_basic_properties
                            if getattr(meta_obj, prop).is_default()]
        relevant_stats_all = meta_utils.convert_props_list_stats_list(
            uncomputed_props)
        relevant_stats_applicable = [
            stat for stat in relevant_stats_all
            if meta_utils.is_stats_class_applicable(meta_obj, stat)]
        if len(relevant_stats_applicable) > 0:
            non_basic_computer = single_comp.SingleNodeComputer(
                relevant_stats_applicable, meta_obj, node, self._ds_reader,
                batch_size_node, uncomputed_props
            )
            non_basic_computer.compute()
            meta_obj = non_basic_computer.get_meta_obj()

        # 4) Set upload properties.
        meta_obj = self._set_upload_props_node(node, meta_obj)

        return meta_obj

    def compute(self, global_batch_size: int = -1) -> None:
        """ Compute metadata properties for all nodes in `_node_list`.

        Note: if `batch_size` is given, the same batch size will be applied to
            all nodes. If `batch_size` is set to its default value, the batch
            size for each node will be estimated individually.
        Note: Metadata can only be computed for datasets that have been
            validated.

        Args:
            dssf_comp_dict (Dict[str, DSSFComponent.ComponentClass]):
                Dictionary containing the names of the DSSF components as
                keys and the corresponding component objects as values.
            global_batch_size (int): Batch size for all nodes.
                Defaults to -1, i.e. batch size will be determined individually
                for each node.

        Returns:
            None.
        """
        for node in self._node_list:
            batch_size_node = self._get_batch_size_node(node) \
                if global_batch_size == -1 else global_batch_size
            new_meta_obj = self._compute_node(node, batch_size_node)
            self._meta_dict[node] = new_meta_obj

        self._is_computed = True

        return None

    def compute_basic(self, global_batch_size: int = -1) -> None:
        """ Compute basic metadata properties for all nodes in `_node_list`.

        Note: if `batch_size` is given, the same batch size will be applied to
            all nodes. If `batch_size` is set to its default value, the batch
            size for each node will be estimated individually.
        Note: Metadata can only be computed for datasets that have been
            validated.

        Args:
            global_batch_size (int): Batch size for all nodes.
                Defaults to -1, i.e. batch size will be determined individually
                for each node.

        Returns:
            None.
        """
        for node in self._node_list:
            batch_size_node = self._get_batch_size_node(node) \
                if global_batch_size == -1 else global_batch_size
            basic_computer = basic_comp.BasicPropertiesComputer.from_ds_reader(
                node, self._ds_reader, batch_size_node)
            basic_computer.compute()
            meta_obj = basic_computer.get_meta_obj()
            self._meta_dict[node] = meta_obj

        self._is_computed = True

        return None

    def get_meta_dict(self) -> Dict[str, meta_prop.AllProperties]:
        """ Return dictionary with newly computed metadata for all nodes.
        If the metadata has not been computed yet, the dictionary is empty.

        Returns:
            Dict[str, meta_prop.AllProperties]: Metadata dictionary.
        """
        return self._meta_dict

    def save(self, output_ds_path: str) -> None:
        """ Save newly computed metadata.

        Note: Assumes that the node names of the original and the output
        dataset are identical. Also assumes that data has been added to the
        nodes of the output dataset.
        Both of these conditions are fulfilled if the new metadata is saved
        to the original dataset.

        Args:
            output_ds_path (str): Path to output dataset.

        Returns:
            None.

        Raises:
            ValueError: If the new metadata has not been computed yet, i.e.
                if `save` is called before `compute`.
            NodesMissingError: If the nodes of the output dataset do not
                match the node names of the original input dataset.
        """
        if not self._is_computed:
            raise ValueError("The new metadata has not been computed yet. "
                             "Please call `compute` first.")
        output_nodes = dh.DatasetReader(output_ds_path).get_node_names() +\
            dh.DatasetReader(output_ds_path).get_datetime_node_names()
        if not all([n in output_nodes for n in self._node_list]):
            raise DSSFErrors.NodesMissingError(
                "The following nodes are not part of the output dataset: "
                f"{[n for n in self._node_list if n not in output_nodes]}")

        with dh.get_Dataset_writer(
                output_ds_path, self._ds_reader.get_n_samples()) as writer:
            for node in self._node_list:
                meta_handler.PropertiesWriter().write_to_ds_writer(
                    self._meta_dict[node], node, writer
                )
        return None


def set_schema_node_type_all_nodes(
        ds_path: str, dshf_path: str) -> None:
    """Set the schema type (from the dshf) for all nodes in a given metadata
    dictionary.

    Args:
        hdf5_path (str): Path to the dataset hdf5 file.
        dshf_path (dshf_handler.DSHFHandler): Path to the dshf to read the node
            types from.
    """
    # Read in the metadata and the dshf.
    metadata_dict = meta_handler.get_metadata_all_nodes(ds_path)
    dshf = dshf_handler.DSHFHandler(dshf_path)

    # Set the node type for each node, by looking it up in the dshf.
    for node_name, meta_obj in metadata_dict.items():
        meta_obj.schema_node_type.set(dshf.node_type[node_name])

    # Write the modified metadata into the dataset file.
    n_samples = dh.DatasetReader(ds_path).get_n_samples()
    with dh.get_Dataset_writer(ds_path, n_samples) as writer:
        for node_name, meta_obj in metadata_dict.items():
            meta_handler.PropertiesWriter().write_to_ds_writer(
                meta_obj, node_name, writer)
    return None
