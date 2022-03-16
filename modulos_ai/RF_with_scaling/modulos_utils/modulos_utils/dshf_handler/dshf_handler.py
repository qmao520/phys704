# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Contains class which generates and reads out the dataset history file."""
from contextlib import contextmanager
import copy
from enum import Enum
import json
import os
from typing import Dict, List, Optional, Union


DSHF_FILE_NAME = "dataset_history_file.json"
VERSION = 0.1
VERSION_KEY = "__version__"
STATES_KEY = "history_states"


class MissingNodeException(Exception):
    """Raised if a node for which changes should be made is not in the dshf."""
    pass


class EntryTypes(Enum):
    """Enum for the different entry types for adding to the history of the
    dataset."""
    dssf = "DSSF saved"
    nodes_added = "nodes added"
    nodes_generated = "nodes generated"
    nodes_removed = "nodes removed"
    heuristics = "heuristics saved"
    user_input = "user input saved"


class DSHFKeys:
    """Class defining the keys of the history file."""
    current_nodes = "current_nodes"
    current_to_upload_name = "current_to_upload_name"
    dataset_path = "dataset_path"
    entry_type = "entry_type"
    description = "description"
    removed_upload_nodes = "removed_upload_nodes"
    generated_nodes = "generated_nodes"
    node_types_probable = "node_types_probable"
    node_types_possible = "node_types_possible"
    node_type_default = "node_type_default"
    node_type = "node_type"
    node_type_dssf = "node_type_dssf"
    dssf_info = "dssf_info"
    nodes_to_component = "nodes_to_component"
    not_categorical = "not_categorical"
    not_datetime = "not_datetime"
    datetime_format = "datetime_format"


class DSHFAlreadyOpened(Exception):
    """Exception raised if another dshf handler instanced is initiated with
    write access to the dshf."""
    pass


class DSHFVersionError(Exception):
    """Error raised if the dshf has the wrong or no version tag.
    """
    pass


class DSHFEntryError(Exception):
    """Raised if the entry is in the wrong form."""
    pass


@contextmanager
def get_dshf_writer(dshf_path: str):
    """This context manager has to be used in order to append to the history
    file. This is mandatory to control that only one instance is writing to
    the same history file at the same time.

    Args:
        dshf_path (str): Path to the dshf.

    Raises:
        DSHFAlreadyOpened: Raised if there is already instance opened to write
            on this history file.

    Yields:
        DSHFHandler: dshf handler object to get and add information.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryFile() as tf:
        ...     with get_dshf_writer(tf) as dshf:
        ...         dshf.add_to_dshf({"comp1": ["node1"]}, EntryTypes.nodes)
        ...         dshf.get_component_to_nodes()
        {"comp1": ["node1"]}
    """
    if dshf_path in DSHFHandler.open_write_instances:
        raise DSHFAlreadyOpened(
            "There is already an open DSHF writer for this dataset. "
            "There is only one dshf writer per dataset allowed at the "
            "same time!"
        )
    else:
        DSHFHandler.open_write_instances.append(dshf_path)
    dshf_handler = DSHFHandler(dshf_path)
    dshf_handler._write_access = True

    try:
        yield dshf_handler
    finally:
        dshf_handler._write_to_json()
        DSHFHandler.open_write_instances.remove(dshf_path)


class DSHFHandler:
    """Handler class for reading and writing to the DSHF file.

    For writing to the dshf, the contextmanager function `get_dshf_writer` has
    to be used.

    Attributes:
        dshf_path (str): Path of this DSHF file.
        version (str): Version of the history file.
        entry_type (str): Name of the last entry Type.
        description (str): Name of the last step done in the dataset
            preparation.
        dssf_info (Dict[str, Dict]): Saves the components with their
            source files and type.
        dssf_sample_id_col (Dict[str, str]): Saves the column name of the
            sample id column of a table component.
        dataset_path (str): Path to hdf5 file that is described by this
            history file.
        current_nodes (List[str]): The list of all the nodes int the current
            state of the dataset. This is potentially different from the list
            of nodes at a different time step in the history.
        node_type_dssf (Dict[str, str]): The DSSF node type for each node
            whose type was enforced in the DSSF.
        node_type (Dict[str, str]): For each node, contains the
            definitive node type (i.e. after user confirmation or after
            generation of the node if it is a generated node).
        node_types_possible (Dict[str, List[str]]): For each node,
            contains the list of possible node types (according to the
            heuristics). See heuristics_computer.py for more info on
            heuristics.
        node_types_probable (Dict[str, List[str]]): For each node,
            contains the list of probable node types (according to the
            heuristics). See heuristics_computer.py for more info on
            heuristics.
        node_type_default (Dict[str, str]): For each node, contains the
            node type picked as default by the heuristics.
        not_categorical (List[str]): List of nodes that are enforced not
            to be categorical.
        not_datetime (List[str]): List of nodes that are enforced not
            to be datetime.
        datetime_format (Dict[str, str]): The datetime format string of the
            datetime nodes.
        current_to_upload_name (Dict[str, str]): Dictionary to map each
            node to its ancestor node at the state of the upload. Example:
            We have a node that is uploaded with the name "some_node", and we
            generate features "some_node_0", "some_node_1" from it. In the
            state before feature generation, the dictionary
            `self.current_to_upload_name` is {"some_node": "some_node"}, as
            the node maps to itself. After generation of the features, the
            dictionary is {"some_node": "some_node",
                           "some_node_0":"some_node",
                           "some_node_1": "some_node"}.
        removed_upload_nodes (List[str]): List of uploaded nodes,
            that were removed. Note that only the nodes that were present after
            the upload, are included in the list if deleted. If a generated
            node is removed, it is not written into this list.
        self.generated_nodes (List[str]): List of generated nodes. If a
            generated node is later removed, it will be also removed from this
            list. Therefore in the solution we can just read this list to
            know which nodes we have to generate.
        history_states (List[Dict]): List of history states. Each state is a
            dictionary like e.g. the following {"current_nodes": [],
            "node_types_possible" {}, ... }.
    """
    open_write_instances: List[str] = []

    def __init__(self, dshf_path: str):
        """Initialize the Handler and loads the file if it exists. If not, it
        fills the value with defaults. Per default the handler has no write
        access, this is only granted if the instance is created in the context
        manager."""
        self.dshf_path = dshf_path
        self.version = VERSION
        self.entry_type = ""
        self.description = ""
        self.dataset_path = ""
        self.current_nodes: List[str] = []
        self.current_to_upload_name: Dict[str, str] = {}
        self.dssf_info: Dict[str, Dict] = {}
        self.node_type_dssf: Dict[str, str] = {}
        self.node_type: Dict[str, str] = {}
        self.node_types_possible: Dict[str, List[str]] = {}
        self.node_types_probable: Dict[str, List[str]] = {}
        self.node_type_default: Dict[str, str] = {}
        self.removed_upload_nodes: List[str] = []
        self.generated_nodes: List[str] = []
        self.history_states: List[Dict] = []
        self.not_categorical: List[str] = []
        self.not_datetime: List[str] = []
        self.datetime_format: Dict[str, str] = {}
        if os.path.exists(self.dshf_path):
            self._load_file()
        self._write_access = False
        return None

    @staticmethod
    def _read_json(dshf_path: str) -> Dict:
        """Read the DSHF and return it as a dictionary.

        Args:
            dshf_path (str): Path to the dshf json file.

        Returns:
            Dict: The DSHF in dictionary form.
        """
        with open(dshf_path, "r") as json_file:
            dshf_json = json.load(json_file)
        DSHFHandler._check_version(dshf_json)
        return dshf_json

    @staticmethod
    def _check_version(dshf_json: Dict) -> None:
        """Check that the dshf has the current version tag.
        At the moment, there is no conversion of other versions implemented.

        Args:
            dshf_json (Dict): Json dictionary of the dshf.

        Raises:
            DSHFVersionError: Raised if either the version tag is missing or
                the version is not valid.

        Returns:
            None
        """
        if VERSION_KEY not in dshf_json:
            raise DSHFVersionError("Missing version tag!")
        version = dshf_json[VERSION_KEY]
        if version != VERSION:
            raise DSHFVersionError("Not a valid version!")
        return None

    def _load_file(self) -> None:
        """Prepare the DSHF handler and load the history and the current
        status of the dataset of the dshf file."""
        history_json = self._read_json(self.dshf_path)
        self.version = history_json[VERSION_KEY]
        self.history_states = history_json[STATES_KEY]
        current_state = copy.deepcopy(self.history_states[-1])
        self.dssf_info = current_state[DSHFKeys.dssf_info]
        self.dataset_path = current_state[DSHFKeys.dataset_path]
        self.description = current_state[DSHFKeys.description]
        self.entry_type = current_state[DSHFKeys.entry_type]
        self.node_type_dssf = current_state[DSHFKeys.node_type_dssf]
        self.node_type = current_state[DSHFKeys.node_type]
        self.node_types_probable = current_state[
            DSHFKeys.node_types_probable]
        self.node_type_default = current_state[
            DSHFKeys.node_type_default]
        self.node_types_possible = current_state[DSHFKeys.node_types_possible]
        self.current_nodes = current_state[DSHFKeys.current_nodes]
        self.current_to_upload_name = current_state[
            DSHFKeys.current_to_upload_name]
        self.removed_upload_nodes = current_state[
            DSHFKeys.removed_upload_nodes]
        self.generated_nodes = current_state[DSHFKeys.generated_nodes]
        self.not_categorical = current_state[DSHFKeys.not_categorical]
        self.not_datetime = current_state[DSHFKeys.not_datetime]
        self.datetime_format = current_state[DSHFKeys.datetime_format]
        return None

    def _add_dssf(self, entry: List) -> None:
        """Update the dshf with a dssf.

        Args:
            entry (List): DSSF entry containing the DSSF information.
        """
        new_dssf_info = {}
        node_type_dssf: Dict[str, str] = {}
        for comp in entry:
            comp_name = comp["name"]
            comp_path = comp["path"]
            comp_type = comp["type"]
            comp_sample_id_col: Optional[str] = None
            comp_columns_to_include: Optional[List[str]] = None
            if "optional_info" in comp:
                comp_sample_id_col = comp["optional_info"].get(
                    "sample_id_column", None)
                comp_columns_to_include = comp["optional_info"].get(
                    "columns_to_include", None)
            new_dssf_info[comp_name] = {
                "type": comp_type,
                "path": comp_path
            }
            if comp_sample_id_col is not None:
                new_dssf_info[comp_name]["sample_id_column"] = \
                    comp_sample_id_col
            if comp_columns_to_include is not None:
                new_dssf_info[comp_name]["columns_to_include"] = \
                    comp_columns_to_include
            if "optional_info" in comp \
                    and "feature_type" in comp["optional_info"]:
                if comp_type == "table":
                    if comp["optional_info"]["feature_type"] is not None:
                        self._update_dict(
                            node_type_dssf,
                            comp["optional_info"]["feature_type"]
                        )
                    if "not_categorical" in comp["optional_info"]:
                        self.not_categorical = list(set(
                            self.not_categorical + comp[
                                "optional_info"]["not_categorical"]))
                    if "not_datetime" in comp["optional_info"]:
                        self.not_datetime = list(set(
                            self.not_datetime + comp[
                                "optional_info"]["not_datetime"]))
                    if "datetime_format" in comp["optional_info"]:
                        self.datetime_format.update(
                            comp["optional_info"]["datetime_format"])
                else:
                    if comp["optional_info"]["feature_type"] is not None:
                        node_type_dssf[comp_name] = comp[
                            "optional_info"]["feature_type"]
                    if "not_categorical" in comp["optional_info"] \
                            and comp["optional_info"]["not_categorical"]:
                        self.not_categorical.append(comp["name"])
                    if "not_datetime" in comp["optional_info"] \
                            and comp["optional_info"]["not_datetime"]:
                        self.not_categorical.append(comp["name"])
                    if "datetime_format" in comp["optional_info"] \
                            and comp["optional_info"]["datetime_format"]:
                        self.datetime_format[comp_name] = \
                            comp["optional_info"]["datetime_format"]

        self._update_dict(self.dssf_info, new_dssf_info)
        self._update_dict(self.node_type_dssf, node_type_dssf)
        return None

    def add_to_dshf(self, entry: Union[List, Dict], entry_type: EntryTypes,
                    description: str,
                    dataset_path: Optional[str] = None) -> None:
        """Add one (or more) entries to the json file.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.
            entry_type (DSHFKeys): The type of the entry.
            description (str): The name of the step that is currently run
                on the dataset.
            dataset_path (Optional[str]): Path to the h5_file of the dataset,
                if there is a new one.. Defaults to None.

        Examples:
            Adding a dssf to the dshf:
            >>> with get_dshf_writer(".") as dshf:
            ...     dshf.add_to_dshf({"dssf": [
            ...         {"name": "images", "path": "img_{id}.jpg",
            ...          "type": "num"},
            ...         {"name": "labels", "path": "labels.csv",
            ...          "type": "table",
            ...          "optional_info": {"dssf_sample_id_col": "id"}}]},
            ...         EntryTypes.dssf)

            Adding some nodes list:
            >>> with get_dshf_writer(".") as dshf:
            ...     dshf.add_to_dshf(
            ...         {"images": ["images"], "labels": ["label"]},
            ...         EntryTypes.nodes)

            Adding some heuristics / user input (same format):
            >>> with get_dshf_writer(".") as dshf:
            ...     dshf.add_to_dshf(
            ...         {"dssf_sample_id_column": {"labels": "id"},
            ...          "categorical": {"images": False, "labels": True}},
            ...         EntryTypes.heuristics)
        """
        if not self._write_to_json:
            raise Exception("Not allowed!")

        if entry_type == EntryTypes.dssf:
            self._add_to_dshf_dssf(entry)

        elif entry_type == EntryTypes.nodes_added:
            self._add_to_dshf_nodes_added(entry)

        elif entry_type == EntryTypes.nodes_removed:
            self._add_to_dshf_nodes_removed(entry)

        elif entry_type == EntryTypes.nodes_generated:
            self._add_to_dshf_nodes_generated(entry)

        elif entry_type == EntryTypes.heuristics:
            self._add_to_dshf_heuristics(entry)

        elif entry_type == EntryTypes.user_input:
            self._add_to_dshf_user_input(entry)

        # Update the path and step and then add it to the history.
        self.entry_type = entry_type.value
        self.dataset_path = dataset_path or self.dataset_path
        self.description = description
        self.history_states.append(self.get_status())
        return None

    def _add_to_dshf_dssf(self, entry: Union[List, Dict]) -> None:
        """Make a dshf entry of the entry type 'dssf'.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.

        Raises:
            DSHFEntryError: Error, if the input does not have the type
                expected.
        """
        if not isinstance(entry, list):
            raise DSHFEntryError("The entry for a dssf has to be a list, "
                                 f"not a {type(entry)}.")
        self._add_dssf(entry)

    def _add_to_dshf_nodes_added(self, entry: Union[List, Dict]) -> None:
        """Make a dshf entry of the entry type 'nodes added'.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.

        Raises:
            DSHFEntryError: Error, if the input does not have the type
                expected.
        """
        if not isinstance(entry, dict):
            raise DSHFEntryError("The entry for nodes has to be a "
                                 f"dictionary, not a {type(entry)}.")
        new_nodes = entry[DSHFKeys.current_nodes]
        added_nodes = [n for n in new_nodes if n not in self.current_nodes]
        self.current_nodes += added_nodes
        node_to_component = entry[DSHFKeys.nodes_to_component]
        for n, c in node_to_component.items():
            if n not in added_nodes:
                continue
            self.dssf_info[c].setdefault("nodes", []).append(n)
        self._update_dict(
            self.current_to_upload_name, {k: k for k in added_nodes}
        )
        return None

    def _add_to_dshf_nodes_removed(self, entry: Union[List, Dict]) -> None:
        """Make a dshf entry of the entry type 'nodes removed'.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.

        Raises:
            DSHFEntryError: Error, if the input does not have the type
                expected.
        """
        if not isinstance(entry, dict):
            raise DSHFEntryError("The entry for removed nodes has to be a "
                                 f"dictionary, not a {type(entry)}.")
        self._remove_nodes(entry[DSHFKeys.removed_upload_nodes])
        return None

    def _add_to_dshf_nodes_generated(self, entry: Union[List, Dict]) -> None:
        """Make a dshf entry of the entry type 'nodes generated'.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.

        Raises:
            DSHFEntryError: Error, if the input does not have the type
                expected.
        """
        if not isinstance(entry, dict):
            raise DSHFEntryError("The entry for generated nodes has to be "
                                 f"a dictionary, not a {type(entry)}.")
        new_nodes = entry[DSHFKeys.current_nodes]
        added_nodes = [n for n in new_nodes if n not in self.current_nodes]
        original_nodes = {
            k: v for k, v in entry[DSHFKeys.current_to_upload_name].items()
            if k in added_nodes}
        self.generated_nodes += added_nodes
        self.current_nodes += added_nodes
        if set(added_nodes) != set(entry[DSHFKeys.node_type].keys()):
            raise DSHFEntryError(
                "For the DSHF entry of adding generated nodes, the node "
                "type must be given for each node that is added.")
        dt_formats = entry[DSHFKeys.datetime_format]
        self._update_dict(
            self.node_type, entry[DSHFKeys.node_type]
        )
        self._update_dict(
            self.current_to_upload_name, original_nodes
        )
        self._update_dict(self.datetime_format, dt_formats)
        return None

    def _add_to_dshf_heuristics(self, entry: Union[List, Dict]) -> None:
        """Make a dshf entry of the entry type 'heuristics'.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.

        Raises:
            DSHFEntryError: Error, if the input does not have the type
                expected.
        """
        if not isinstance(entry, dict):
            raise DSHFEntryError("The entry for heuristics has to be a "
                                 f"dictionary, not a {type(entry)}.")
        self._update_dict(
            self.node_types_probable,
            entry[DSHFKeys.node_types_probable]
        )
        self._update_dict(
            self.node_types_possible,
            entry[DSHFKeys.node_types_possible]
        )
        self._update_dict(
            self.node_type_default,
            entry[DSHFKeys.node_type_default]
        )
        return None

    def _add_to_dshf_user_input(self, entry: Union[List, Dict]) -> None:
        """Make a dshf entry of the entry type 'user input'.

        Args:
            entry (Union[List, Dict]): All entries in entry are added to the
                json file.

        Raises:
            DSHFEntryError: Error, if the input does not have the type
                expected.
        """
        if not isinstance(entry, dict):
            raise DSHFEntryError("The entry for user input has to be a "
                                 f"dictionary, not a {type(entry)}.")
        self._update_dict(
            self.node_type, entry[DSHFKeys.node_type]
        )
        return None

    def _remove_nodes(self, nodes: List[str]) -> None:
        """Remove nodes from either datetime or node entries and update all
        other entries accordingly.

        Args:
            nodes (List[str]): Name of the nodes to be removed.
        """
        for node in nodes:
            if node not in self.current_nodes:
                raise ValueError("Node to be removed does not exist: "
                                 f"'{node}'")
            parent_name = self.current_to_upload_name[node]
            self.current_nodes.remove(node)
            self.node_type_dssf.pop(node, None)
            self.node_types_probable.pop(node, None)
            self.node_types_possible.pop(node, None)
            self.node_type_default.pop(node, None)
            self.node_type.pop(node, None)
            self.current_to_upload_name.pop(node, None)
            if node in self.generated_nodes:
                self.generated_nodes.remove(node)
            if parent_name not in self.current_to_upload_name.values():
                self.removed_upload_nodes.append(parent_name)
        return None

    def get_component_name(self, node: str) -> str:
        """Get the dssf component name of a node's original upload node.

        Args:
            node (str): Node whose parents component name is retrieved.

        Returns
            str: Component name.
        """
        original_node = self.current_to_upload_name[node]
        for comp_name, comp_info in self.dssf_info.items():
            if original_node in comp_info["nodes"]:
                return comp_name
        raise ValueError(
            f"The given node '{node}' is not known at this state of the "
            "history. Either it was removed, or it was never added in the "
            "first place.")

    def get_status(self) -> Dict:
        """Get the full status of the last stage in the history.

        Returns:
            Dict: The dictionary that is written in the dshf as the last entry.
        """
        return copy.deepcopy({
            DSHFKeys.entry_type: self.entry_type,
            DSHFKeys.dataset_path: self.dataset_path,
            DSHFKeys.description: self.description,
            DSHFKeys.dssf_info: self.dssf_info,
            DSHFKeys.current_nodes: self.current_nodes,
            DSHFKeys.node_type_dssf: self.node_type_dssf,
            DSHFKeys.node_types_probable: self.node_types_probable,
            DSHFKeys.node_type: self.node_type,
            DSHFKeys.node_types_possible: self.node_types_possible,
            DSHFKeys.node_type_default: self.node_type_default,
            DSHFKeys.removed_upload_nodes: self.removed_upload_nodes,
            DSHFKeys.generated_nodes: self.generated_nodes,
            DSHFKeys.current_to_upload_name: self.current_to_upload_name,
            DSHFKeys.not_categorical: self.not_categorical,
            DSHFKeys.not_datetime: self.not_datetime,
            DSHFKeys.datetime_format: self.datetime_format
            })

    def _write_to_json(self):
        """Write the updated history back to the dshf file.
        """
        # Overwrite the history with the updated one.
        with open(self.dshf_path, "w") as outfile:
            dic = {
                VERSION_KEY: self.version,
                STATES_KEY: self.history_states
            }
            json.dump(dic, outfile, indent=4)

    @staticmethod
    def _update_dict(target: Dict, entry: Dict) -> None:
        """Update the target with the dictionary items from entry
        (recursively).

        Args:
            target (Dict): dictionary to update
            entry (Dict): dictionary to update from
        """
        for key, value in entry.items():
            if isinstance(value, dict):
                target.setdefault(key, {})
                DSHFHandler._update_dict(target[key], value)
            else:
                target.update({key: value})
        return None

    def get_updated_node_names(
            self, node_names_historical: List[str],
            history_index: int = None) -> List[str]:
        """Get the list of current node names from a list of node names in the
        history of a dataset. For example, given the node name "date", it will
        return all the generated nodes ["date__year", "date__hour", ...], if
        this function is called after the generation of the datetime features.
        If history_index is not given, the history is searched for the given
        node names.

        Args:
            node_names_historical (List[str]): List of node names at a past
                point in the history of a dataset.
            history_index (int): Index of past state in the history file.
            dshf (dshf_handler.DSHFHandler): Loaded DSHF file.

        Returns:
            List[str]: List of all names at the latest history state, i.e.
                with all the generated nodes included, and the removed nodes
                removed.
        """
        node_names_to_update = [n for n in node_names_historical
                                if n not in self.current_nodes]
        if len(node_names_to_update) == 0:
            return node_names_historical

        if history_index is None:
            for index in range(len(self.history_states)):
                if all(n in self.history_states[index][DSHFKeys.current_nodes]
                       for n in node_names_to_update):
                    history_index = index
        if history_index is None:
            raise ValueError(
                "There is no point in the dataset history, where "
                "all all of the given node names were present.")
        historical_to_original = self.history_states[history_index][
            DSHFKeys.current_to_upload_name]
        latest_to_original = self.history_states[-1][
            DSHFKeys.current_to_upload_name]

        node_names = []
        for n in node_names_historical:
            if n in self.current_nodes:
                node_names.append(n)
            else:
                upload_name = historical_to_original[n]
                child_nodes_now = [
                    k for k, v in latest_to_original.items()
                    if v == upload_name]
                node_names += child_nodes_now
        return node_names
