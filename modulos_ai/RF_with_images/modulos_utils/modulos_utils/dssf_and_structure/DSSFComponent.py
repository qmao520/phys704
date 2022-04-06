# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""DSSFComponent is a representation of a component in the DSSF.

It is used to determine the default values for the missing keys, validate the
optional keys and read and save the data to the internal dataset.
"""
from abc import ABC, abstractmethod
import glob
import itertools
import os
from typing import List, Dict, Any, Iterator, Optional

import numpy as np

from modulos_utils.dssf_and_structure import read_data
from modulos_utils.data_handling import data_handler as dh
from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_and_structure import structure_logging as struc_log
from modulos_utils.datetime_utils import datetime_utils
from modulos_utils.sliding_window import utils as sw_utils


ID_PLACEHOLDER = "{id}"

# These two constants are defined in `base/base/utils.constants.py` and are
# therefore a duplicate.
# TODO: Fix this duplication issue in PRB-178.
COLUMN_LIMIT = 500
# Note that this batch size is ignored for images, as a bug fix. They have
# their own default batch size of 100. This needs to be reworked: BAS-826
BATCH_SIZE_GB = 10**9

SUPPORTED_NODE_TYPES = [
    "bool", "datetime", "probability", "categorical", "numerical",
    "text", "probability_distribution"]


def compute_component_batch_size(
        batch_size_gb: int, path_list: List[str], n_samples_tot: int) -> int:
    """Compute the batch size (in number of samples) to load a dataset
    component consisting of all the files in path_list, such that on batch has
    the size batch_size_gb GB.

    Args:
        batch_size_gb (int): Batch size in gigabytes.
        path_list (List[str]): List of paths that belong to the component.
        n_samples_tot (int): Total number of samples of the dataset.

    Returns:
        int: Batch size (number of samples in the batch):
    """
    n_batches_float = os.path.getsize(path_list[0]) * len(path_list) \
        / float(BATCH_SIZE_GB)
    batch_sample_count = int(n_samples_tot / n_batches_float)
    return batch_sample_count


class ComponentClass(ABC):
    """
    This is the abstract base class for classes reading in the data.
    """
    defaults: Dict[str, Optional[str]] = {}
    forbidden_names = [dh.SAMPLE_IDS, dh.GENERATED_SAMPLE_ID_NAME]
    forbidden_name_parts = [sw_utils.GENERATED_NODES_SUFFIX]
    forbidden_char_in_names = ["/"]

    def __init__(
            self, comp: Dict[str, Any], dataset_dir_path: str, batch_size: int,
            logging_purpose: struc_log.LoggingPurpose) -> None:
        self.name: str = comp["name"]
        self.path: Any = comp["path"]
        self.type: str = comp["type"]
        self.dataset_dir_path: str = dataset_dir_path
        self.batch_size: int = batch_size
        self.opt_info: Dict[str, Any] = (comp["optional_info"]
                                         if "optional_info" in comp else {})
        self.sample_ids: List[str] = []
        self.data_generator: Iterator[dict]
        self.allowed_file_types: List[str] = []
        self.ignored_keys: List[str] = []
        self.wrong_type_keys: List[str] = []
        self._set_defaults_and_check_()
        self.logging_purpose = logging_purpose
        self.component_logger = struc_log.ComponentLoggingManager(
            self.name, logging_purpose)
        self.node_names: List[str] = []
        self.datetime_subfeatures: Dict[str, List] = {}
        self.dtypes: Dict[str, str] = {}
        return None

    def _set_defaults_and_check_(self) -> None:
        """Checks if there are keys with wrong values and replaces them with
        default values, checks if there are undefined keys and deletes them
        and at last set default values for all missing keys.
        """
        for info in list(self.opt_info):
            if info in self.defaults:
                if not self._check_type_opt_info(info):
                    self.wrong_type_keys.append(info)
                    self.opt_info[info] = self.defaults[info]
            else:
                self.opt_info.pop(info)
                self.ignored_keys.append(info)
        for info in self.defaults:
            if info not in self.opt_info:
                self.opt_info[info] = self.defaults[info]
        return None

    @abstractmethod
    def _check_type_opt_info(self, info: str) -> bool:
        """Used to test that the value of the optional key is
        correct. This function is implemented in the child classes.

        Args:
            info (dict): the optional keys
        """
        pass

    @abstractmethod
    def _check_enforced_feature_type_info(self) -> None:
        """Raise an exception if the given feature type information is not
        valid.
        """
        pass

    def get_sample_ids(self) -> List[str]:
        """Return the sample ids.

        If the data generator is not initialized, it will do that and find the
        ids in the initialization.

        Returns:
            List[str]: List of samples ids as strings.
        """
        if self.sample_ids == []:
            self.prepare_data_for_reading()
        return self.sample_ids

    @abstractmethod
    def get_nr_samples(self) -> int:
        """Get the number of samples of the dataset.

        Returns:
            int: Number of samples as an int.
        """
        pass

    @abstractmethod
    def prepare_data_for_reading(self) -> int:
        """Prepare the generator to read the data.

        Check if the data is correct, prepare the batches and generator and
        get the sample ids.

        Returns:
            int: number of samples
        """
        pass

    @abstractmethod
    def save_component(self, h5_writer: dh.DatasetWriter,
                       shuffled_positions: List[int],
                       nodes_to_be_saved: Optional[List[str]]) -> None:
        """Use the generator to read all data and save it in the given sequence.

        The list `shuffled_positions` describes the way the data is shuffled.
        The `n`th entry is the index of the input data sample which is saved at
        position `n` in the internal dataset.

        Args:
            h5_writer (dh.DatasetWriter): h5 datahandler writer instance
            shuffled_positions (List[int]): sequence to save the data
            nodes_to_be_saved (Optional[List[str]]): None if all data should be
                saved, else the names of the nodes that should be saved
        """
        pass

    def _check_file_extension(self) -> None:
        """Check if the file extension is allowed for the specific component
        type.

        Raises:
            DSSFErrors.WrongExtension: Raised if the file extension is not
                allowed.
            TypeError: Raised if path value has the wrong type.
        """
        if isinstance(self.path, str):
            ext = os.path.splitext(self.path)[1]
            if ext not in self.allowed_file_types:
                raise DSSFErrors.WrongExtensionError(
                    f"The file extension '{ext}' of component '{self.name}' "
                    "can not be used for component type "
                    f"{self.__class__.__name__}.\n")
            return None
        elif isinstance(self.path, list):
            for path in self.path:
                ext = os.path.splitext(path)[1]
                if ext not in self.allowed_file_types:
                    raise DSSFErrors.WrongExtensionError(
                        f"The file extension '{ext}' of component "
                        f"'{self.name}' can not be used for "
                        "component type "
                        f"{self.__class__.__name__}.\n")
        else:
            raise TypeError("The path argument in the dssf has to be a string "
                            f" or a list. '{self.path}' is neither.")
        return None

    def _get_sample_ids_from_paths(self, paths: List[str]) -> List[str]:
        """Return the list of samples ids from a list of paths.

        Args:
            paths (List[str]): List of absolute paths containing the
                `ID_PLACEHOLDER`

        Returns:
            List[str]: sample ids as a list of strings
        """
        file_fix = os.path.join(self.dataset_dir_path,
                                self.path).split(ID_PLACEHOLDER)
        sample_ids: List[str] = []
        for path in paths:
            sample_ids.append(
                path.replace(file_fix[0], "", 1).replace(file_fix[1], ""))
        return sample_ids

    def _check_for_forbidden_names(self, names_to_check: List[str]) -> None:
        """Check whether any used names are in the list of forbidden
        column / component names and raise an exception if yes.
        """
        for c in names_to_check:
            if c in ComponentClass.forbidden_names:
                raise DSSFErrors.ColumnHeaderError(
                    f"`{c}` is not allowed as a column name. The following "
                    "names are reserved for internal use and can therefore "
                    "not be used as column names: "
                    f"{', '.join(ComponentClass.forbidden_names)}.")
            for part in ComponentClass.forbidden_name_parts:
                if part in c:
                    raise DSSFErrors.ColumnHeaderError(
                        f"Column names that contain the string {part}, are "
                        "not allowed, however there is a column named "
                        f"{c}.")
            for char in ComponentClass.forbidden_char_in_names:
                if char in c:
                    raise DSSFErrors.ColumnHeaderError(
                        f"Names that contain the character '{char}' "
                        "are not allowed as component or feature names. "
                        "Please replace the character.")
        return None


class TableComponent(ComponentClass):
    """
    This is the abstract base class for all table data component classes.
    """
    defaults: Dict[str, Any] = {"sample_id_column": None,
                                "columns_to_include": None,
                                "not_categorical": [],
                                "not_datetime": [],
                                "feature_type": {},
                                "datetime_format": {}}

    def __init__(
            self, comp: Dict[str, Any], dataset_dir_path: str, batch_size: int,
            logging_purpose: struc_log.LoggingPurpose) -> None:
        super().__init__(comp, dataset_dir_path, batch_size, logging_purpose)
        self.allowed_file_types: List[str] = [".csv"]
        self.columns: List[str] = []
        self.dtypes: Dict[str, str] = {}
        return None

    def _check_enforced_column_names(
            self, names: List[str], field: str) -> None:
        """Check whether column names used in optional info to enforce types,
        are names of existing columns.

        Args:
            names (List[str]): List of column names to check.
            field (str): Field which the check is performed for.
        """
        cols_enforced_invalid = [col for col in names
                                 if col not in self.columns]
        if cols_enforced_invalid != []:
            msg = (f"The following entries of '{field}' are invalid column "
                   f"names: {', '.join(cols_enforced_invalid)}")
            raise DSSFErrors.DSSFNodesMissing(msg)
        return None

    def _check_type_opt_info(self, info: str) -> bool:
        """Used to test that the value of the optional key is valid.

        Args:
            info (str): name of optional key to test

        Returns:
            bool: true if it is valid
        """
        if info == "sample_id_column":
            return (self.opt_info[info] is None or
                    (isinstance(self.opt_info[info], (str, int))))
        elif info == "columns_to_include":
            return (self.opt_info[info] is None or
                    all(isinstance(elem, (str, int))
                        for elem in self.opt_info[info]))
        elif info in ["not_categorical", "not_datetime"]:
            return (isinstance(self.opt_info[info], list) and
                    all(isinstance(elem, str)
                        for elem in self.opt_info[info]))
        elif info == "feature_type":
            return (isinstance(self.opt_info[info], dict))
        elif info == "datetime_format":
            return (isinstance(self.opt_info[info], dict))
        return False

    def _check_enforced_feature_type_info(self) -> None:
        """Raise an exception if the given feature type information is not
        valid.
        """
        cols_enforced = list(self.opt_info["feature_type"].keys())
        self._check_enforced_column_names(cols_enforced, "feature_type")
        invalid_feature_types = [
            v for v in self.opt_info["feature_type"].values()
            if v not in SUPPORTED_NODE_TYPES]
        if invalid_feature_types != []:
            msg = ("The following entries of 'feature_type' are "
                   "invalid feature types: "
                   f"{', '.join(invalid_feature_types)}")
            raise DSSFErrors.DSSFNodeTypeNotSupported(msg)
        return None

    def _check_enforced_datetime_format_info(self) -> None:
        """Raise an exception if the given datetime format information is not
        valid.
        """
        cols_enforced = list(self.opt_info["datetime_format"].keys())
        self._check_enforced_column_names(cols_enforced, "datetime_format")
        invalid_feature_types = [
            v for v in self.opt_info["datetime_format"].values()
            if not isinstance(v, str)]
        if invalid_feature_types != []:
            msg = ("The following entries of 'datetime_format' are "
                   "invalid feature types: "
                   f"{', '.join(invalid_feature_types)}")
            raise DSSFErrors.DSSFNodeTypeNotSupported(msg)
        # Enforce the feature_type `datetime` if a datetime format is given.
        if "feature_type" not in self.opt_info:
            self.opt_info["feature_type"] = {
                col: "datetime" for col in cols_enforced}
            return None
        for col in cols_enforced:
            if col not in self.opt_info["feature_type"]:
                self.opt_info["feature_type"][col] = "datetime"
            elif self.opt_info["feature_type"][col] != "datetime":
                msg = (f"Feature {col} has a `datetime_format` specified in "
                       "the dssf, but its `feature_type` is set to "
                       f"{self.opt_info['feature_type'][col]}. It has to be "
                       "set to 'datetime'.")
                raise DSSFErrors.DSSFNodeTypeNotSupported(msg)
        return None

    def save_component(self, h5_writer: dh.DatasetWriter,
                       shuffled_positions: List[int],
                       nodes_to_be_saved: Optional[List[str]]) -> None:
        """Use the generator to read all data and save it in the given sequence.

        The list `shuffled_positions` describes the way the data is shuffled.
        The `n`th entry is the index of the input data sample which is saved at
        position `n` in the internal dataset.

        Args:
            h5_writer (dh.DatasetWriter): h5 datahandler writer instance
            shuffled_positions (List[int]): sequence to save the data
            nodes_to_be_saved (Optional[List[str]]): None if all data should be
                saved, else the names of the nodes that should be saved
        """
        if self.sample_ids == []:
            self.prepare_data_for_reading()
        # Check if the data has to be shuffled.
        is_shuffled = (
            shuffled_positions != list(range(len(shuffled_positions))))
        if is_shuffled:
            shuffling_dict = {x: n for n, x in enumerate(shuffled_positions)}
        # Check which columns are to be saved and adapt self.columns.
        self._prepare_columns_to_save(nodes_to_be_saved)
        if self.columns == []:
            self.component_logger.log_warning(
                f"No nodes are saved for table '{self.name}' "
                "as there are none of the in the list of nodes"
                " to be saved!")
            return None
        self.node_names.extend(self.columns)

        counter = 0
        n_samples = len(self.sample_ids)
        n_columns = len(self.columns)
        n_cells_total = n_samples * n_columns
        n_cells_filled = 0
        self.component_logger.log_name()
        for batch in self.data_generator:
            for col in self.columns:
                n = counter
                # If the data has to be shuffled, it has to be packed into a
                # dictionary with the new position as key.
                if is_shuffled:
                    col_dict: Dict[int, Any] = {}
                    for value in batch[col]:
                        col_dict[shuffling_dict[n]] = value
                        n += 1
                    h5_writer.add_dict_to_node(col_dict, col)
                    n_cells_filled += len(list(col_dict.keys()))
                # If the data is not shuffled, then it can just be appended in
                # the hdf5.
                else:
                    values = np.array(batch[col])
                    h5_writer.add_data_to_node(col, values)
                    n += len(values)
                    n_cells_filled += len(values)
                self.component_logger.log_percentage(
                    float(n_cells_filled) / n_cells_total * 100)
            counter = n
        self.dtypes = {key: value if key not in self.datetime_subfeatures
                       else "datetime" for key, value in self.dtypes.items()}
        return None

    def _prepare_columns_to_save(
            self, nodes_to_be_saved: Optional[List[str]]) -> None:
        """Filter out the columns given in the DSSF or the DSSFSaver.

        Args:
            nodes_to_be_saved (Optional[List[str]]): None if all data should be
                saved based on the DSSFSaver, else the names of the nodes that
                should be saved.

        Raises:
            DSSFErrors.DSSFNodesMissing: Raised if a node name given by the
                user is not in the table.
        """
        # List of nodes given by the user in the DSSF.
        cols_to_include_dssf: Optional[List[str]] = \
            self.opt_info["columns_to_include"]
        if cols_to_include_dssf is not None:
            invalid_col_names = [col for col in cols_to_include_dssf
                                 if col not in self.columns]
            if invalid_col_names != []:
                msg = ("The following entries of 'columns_to_include' are "
                       f"missing: {', '.join(invalid_col_names)}")
                raise DSSFErrors.DSSFNodesMissing(msg)
        self._check_enforced_feature_type_info()
        self._check_enforced_datetime_format_info()
        self._check_enforced_column_names(
            self.opt_info["not_categorical"], "not_categorical")
        self._check_enforced_column_names(
            self.opt_info["not_datetime"], "not_datetime")
        # Add to the list given by the DSSFSaver (for example to ignore the
        # label in the predictions).
        if (nodes_to_be_saved is not None or
                cols_to_include_dssf is not None):
            if nodes_to_be_saved is None:
                nodes_to_be_saved = cols_to_include_dssf
            elif cols_to_include_dssf is None:
                pass
            else:
                nodes_to_be_saved = list(
                    set(nodes_to_be_saved).intersection(cols_to_include_dssf))
        if nodes_to_be_saved is not None:
            self.columns = [i for i in self.columns if i in nodes_to_be_saved]
        # Pop sample_id column (we don't save them here).
        if self.opt_info["sample_id_column"] is not None:
            self.columns = [i for i in self.columns
                            if i != self.opt_info["sample_id_column"]]
        return None


class SingleNodeComponent(ComponentClass):
    """
    This class represent component with a single node (like images, numpy files
    or texts). Every sample has its own file.
    """
    defaults = {
        "not_categorical": None, "not_datetime": None, "feature_type": None,
        "datetime_format": None}

    def __init__(
            self, comp: Dict[str, Any], dataset_dir_path: str, batch_size: int,
            logging_purpose: struc_log.LoggingPurpose) -> None:
        super().__init__(comp, dataset_dir_path, batch_size, logging_purpose)
        self.paths: List[str] = []
        self.allowed_file_types = (
            [".txt", ".jpg", ".npy", ".tif", ".png"] if comp["type"] == "num"
            else [".txt", ".npy"])

    def get_nr_samples(self) -> int:
        """Get the number of samples of the dataset.

        Returns:
            int: Number of samples as an int.
        """
        self._check_file_extension()
        abs_path = os.path.join(self.dataset_dir_path, self.path)
        abs_path = abs_path.replace(ID_PLACEHOLDER, "*")
        path_generator = glob.iglob(abs_path)
        nr_paths = 0
        for _ in path_generator:
            nr_paths += 1
        if nr_paths == 0:
            raise FileNotFoundError("There are no files matching the "
                                    f"description '{self.path}' "
                                    f"for component '{self.name}'!")
        return nr_paths

    def _check_type_opt_info(self, info: str) -> bool:
        """Used to test that the value of the optional key is valid.

        Args:
            info (str): name of optional key to test

        Returns:
            bool: true if it is valid
        """
        if info in ["not_categorical", "not_datetime"]:
            return (isinstance(self.opt_info[info], bool) or
                    self.opt_info[info] is None)
        if info == "feature_type":
            return (isinstance(self.opt_info[info], str) or
                    self.opt_info[info] is None)
        if info == "datetime_format":
            return (isinstance(self.opt_info[info], str) or
                    self.opt_info[info] is None)
        return False

    def _check_enforced_feature_type_info(self) -> None:
        """Raise an exception if the given feature type information is not
        valid.
        """
        feature_type = self.opt_info["feature_type"]
        if feature_type is None:
            return None
        if feature_type not in SUPPORTED_NODE_TYPES:
            msg = (f"The feature type entry '{feature_type}' is not a "
                   "valid feature type.")
            raise DSSFErrors.DSSFNodeTypeNotSupported(msg)
        return None

    def prepare_data_for_reading(self) -> int:
        """Prepare the generator to read the data.

        Check if the data is correct and get the sample ids and paths.

        Raises:
            FileNotFoundError: Raised if no files were found for the path
                given by the user.

        Returns:
            int: number of samples
        """
        self._check_file_extension()
        abs_path = os.path.join(self.dataset_dir_path, self.path)
        abs_path = abs_path.replace(ID_PLACEHOLDER, "*")
        # The following glob.glob should be replaced by iglob, once the sample
        # ids are read as generators and the number of samples limit is
        # increased: REB-957.
        self.paths = glob.glob(abs_path)
        if len(self.paths) == 0:
            raise FileNotFoundError("There are no files matching the "
                                    f"description '{self.path}' "
                                    f"for component '{self.name}'!")
        self._check_for_forbidden_names([self.name])
        self.sample_ids = self._get_sample_ids_from_paths(self.paths)
        if self.opt_info is not None \
                and self.opt_info.get("feature_type", "") == "datetime":
            self.datetime_subfeatures = {
                self.name: datetime_utils.get_datetime_node_names(self.name)
            }
        else:
            self.datetime_subfeatures = {}
        return len(self.sample_ids)

    def save_component(self, h5_writer: dh.DatasetWriter,
                       shuffled_positions: List[int],
                       nodes_to_be_saved: Optional[List[str]]) -> None:
        """Use the generator to read all data and save it in the given sequence.

        The list `shuffled_positions` describes the way the data is shuffled.
        The `n`th entry is the index of the input data sample which is saved at
        position `n` in the internal dataset.

        Args:
            h5_writer (dh.DatasetWriter): h5 datahandler writer instance
            shuffled_positions (List[int]): sequence to save the data
            nodes_to_be_saved (Optional[List[str]]): None if all data should be
                saved, else the names of the nodes that should be saved

        Raises:
            DSSFErrors.WrongTypeError: The data has another type as specified
                in the dssf.
            DSSFErrors.DataShapeMissmatchError: Not all data has the same
                shape.
        """
        self._check_enforced_feature_type_info()
        if self.sample_ids == []:
            self.prepare_data_for_reading()
        if self.batch_size == -1:
            self.batch_size = 100
        counter = 0
        if (nodes_to_be_saved is not None and
                self.name not in nodes_to_be_saved):
            self.component_logger.log_warning(
                f"Node {self.name} not saved as it is not in the "
                "list of nodes to be saved.")
            return None

        self.component_logger.log_name()
        self.node_names.append(self.name)

        # Iterate over all batches.
        for i in range((len(shuffled_positions) + self.batch_size - 1)
                       // self.batch_size):
            data_batch: List[np.ndarray] = []

            if i == 0:
                path = self.paths[shuffled_positions[0]]
                reader = read_data.read_data(path)
                shape = reader.get_shape()

            end = (counter + self.batch_size
                   if counter + self.batch_size <= len(shuffled_positions)
                   else len(shuffled_positions))
            for n in range(counter, end):
                # Get the sample which is saved at position n in the internal
                # dataset.
                path = self.paths[shuffled_positions[n]]
                reader = read_data.read_data(path)
                shape_n = reader.get_shape()
                if shape_n != shape:
                    raise DSSFErrors.DataShapeMissmatchError(
                        "Shapes of data within one feature must be the same!")
                data = reader.get_data()
                data_batch.append(data)

            counter = end
            # Write the batch to the internal dataset.
            h5_writer.add_data_to_node(self.name, np.array(data_batch))
            self.component_logger.log_percentage(
                float(counter) / len(shuffled_positions) * 100)
        self.dtypes = {self.name: self.type}
        return None


class TableComponentOneFilePerSample(TableComponent):
    """This class class represents tables with one file per sample.
    """
    def __init__(
            self, comp: Dict[str, Any], dataset_dir_path: str, batch_size: int,
            logging_purpose: struc_log.LoggingPurpose) -> None:
        super().__init__(comp, dataset_dir_path, batch_size, logging_purpose)
        self.batch_size = batch_size
        self.paths: List[str] = []
        return None

    def get_nr_samples(self) -> int:
        """Get the number of samples of the dataset.

        Returns:
            int: Number of samples as an int.
        """
        self._check_file_extension()
        abs_path = os.path.join(self.dataset_dir_path, self.path)
        abs_path = abs_path.replace(ID_PLACEHOLDER, "*")
        path_generator = glob.iglob(abs_path)
        nr_paths = 0
        for _ in path_generator:
            nr_paths += 1
        if nr_paths == 0:
            raise FileNotFoundError("There are no files matching the "
                                    f"description '{self.path}' "
                                    f"for component '{self.name}'!")
        return nr_paths

    def prepare_data_for_reading(self) -> int:
        """Prepare the generator to read the data.

        Check if the data is correct and get the sample ids and paths.

        Raises:
            DSSFErrors.ColumnOverflowException: Raised if there are more
                columns than the upper limit defined in COLUMN_LIMIT.
            FileNotFoundError: Raised if no files were found for the path
                given by the user.


        Returns:
            int: number of samples
        """
        self._check_file_extension()
        abs_path = os.path.join(self.dataset_dir_path, self.path)
        abs_path = abs_path.replace(ID_PLACEHOLDER, "*")
        # The following glob.glob should be replaced by iglob, once the sample
        # ids are read as generators and the number of samples limit is
        # increased: REB-957.
        self.paths = glob.glob(abs_path)
        if len(self.paths) == 0:
            raise FileNotFoundError("There are no files matching the "
                                    f"description '{self.path}' "
                                    f"for component '{self.name}'!")
        self.sample_ids = self._get_sample_ids_from_paths(self.paths)
        reader = read_data.read_data(self.paths[0])
        self.columns = reader.columns
        self._check_for_forbidden_names(self.columns + [self.name])
        if len(self.columns) > COLUMN_LIMIT:
            raise DSSFErrors.ColumnOverflowException(
                "The upper limit for number of columns in a csv for "
                f"AutoML is '{COLUMN_LIMIT}'. This .csv has "
                f"'{len(self.columns)}' columns!")
        self.dtypes = reader.get_dtypes()

        return len(self.sample_ids)

    def save_component(self, h5_writer: dh.DatasetWriter,
                       shuffled_positions: List[int],
                       nodes_to_be_saved: Optional[List[str]]) -> None:
        """Use the generator to read all data and save it in the given sequence.

        The list `shuffled_positions` describes the way the data is shuffled.
        The `n`th entry is the index of the input data sample which is saved at
        position `n` in the internal dataset.

        Args:
            h5_writer (dh.DatasetWriter): h5 datahandler writer instance
            shuffled_positions (List[int]): sequence to save the data
            nodes_to_be_saved (Optional[List[str]]): None if all data should be
                saved, else the names of the nodes that should be saved

        Raises:
            DSSFErrors.DataShapeError: Raised if the csv has more than one
                sample.
        """
        if self.columns == []:
            self.prepare_data_for_reading()
        if self.batch_size == -1:
            self.batch_size = compute_component_batch_size(
                BATCH_SIZE_GB, self.paths, len(self.sample_ids))
        self._prepare_columns_to_save(nodes_to_be_saved)
        if self.columns == []:
            self.component_logger.log_warning(
                f"No nodes are saved for table '{self.name}' "
                "as there are none of the in the list of nodes"
                " to be saved!")
            return None
        self.node_names.extend(self.columns)

        counter = 0
        self.component_logger.log_name()
        for i in range((len(shuffled_positions) + self.batch_size - 1)
                       // self.batch_size):
            data_batch: Dict[str, List] = {i: [] for i in self.columns}
            end = (counter + self.batch_size
                   if counter + self.batch_size <= len(self.sample_ids)
                   else len(self.sample_ids))
            for n in range(counter, end):
                path = self.paths[shuffled_positions[n]]
                reader = read_data.read_data(path)
                if reader.get_shape()[0] != 1:
                    raise DSSFErrors.DataShapeError(
                        "CSV is only allowed to have one line. This csv has "
                        f"{reader.get_shape()[0]}!")
                data = reader.get_data()
                for col in self.columns:
                    data_batch[col].extend(data[col])
            for col in self.columns:
                h5_writer.add_data_to_node(col,
                                           np.array(data_batch[col]))
            counter = end
            self.component_logger.log_percentage(
                counter / len(shuffled_positions) * 100)
        self.dtypes = {key: value if key not in self.datetime_subfeatures
                       else "datetime" for key, value in self.dtypes.items()}
        return None


class TableComponentSingleFile(TableComponent):
    """This class reads and processes table data from a single file.
    """
    def __init__(
            self, comp: Dict[str, Any], dataset_dir_path: str, batch_size: int,
            logging_purpose: struc_log.LoggingPurpose) -> None:
        super().__init__(comp, dataset_dir_path, batch_size, logging_purpose)
        return None

    def get_nr_samples(self) -> int:
        """Get the number of samples of the dataset.

        Returns:
            int: Number of samples as an int.
        """
        self._check_file_extension()
        path = os.path.join(self.dataset_dir_path, self.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"There is no file '"
                                    f"{self.path}' "
                                    f"for component '{self.name}'!")
        reader = read_data.read_data(path)
        shape = reader.get_shape()
        return shape[0]

    def prepare_data_for_reading(self) -> int:
        """Prepare the generator to read the data.

        Check if the data is correct and prepare the batches and generator.

        Raises:
            DSSFErrors.ColumnOverflowException: Raised if there are more
                columns than the upper limit defined in COLUMN_LIMIT.
            FileNotFoundError: The csv file containing the table is missing.
            DSSFErrors.SampleIDError: No unique sample_ids.
            DSSFErrors.DSSFOptionalKeyError: Sample id column not found in
                the saved columns of the table.

        Returns:
            int: number of samples
        """
        self._check_file_extension()
        path = os.path.join(self.dataset_dir_path, self.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"There is no file '"
                                    f"{self.path}' "
                                    f"for component '{self.name}'!")
        reader = read_data.read_data(path)
        self.columns: List[str] = reader.columns
        self._check_for_forbidden_names(self.columns + [self.name])
        if len(self.columns) > COLUMN_LIMIT:
            raise DSSFErrors.ColumnOverflowException(
                "The upper limit for number of columns in a csv for "
                f"AutoML is '{COLUMN_LIMIT}'. This component has "
                f"'{len(self.columns)}' columns!")
        self.dtypes = reader.get_dtypes()
        sample_id_column = self.opt_info["sample_id_column"]

        if sample_id_column in self.columns:
            self.sample_ids = list(map(str, reader.get_data_single_column(
                sample_id_column)))
            if len(self.sample_ids) != len(set(self.sample_ids)):
                msg = (f"The sample_ids in column '{sample_id_column}' are not"
                       " unique!")
                self.component_logger.log_error(msg)
                raise DSSFErrors.SampleIDError(msg)
        else:
            if sample_id_column is not None:
                msg = (f"Sample_id column '{sample_id_column}' not found in "
                       "table!")
                self.component_logger.log_error(msg)
                raise DSSFErrors.DSSFOptionalKeyError(msg)

            self.sample_ids = [str(i) for i in range(reader.len)]
        if self.batch_size == -1:
            self.batch_size = compute_component_batch_size(
                BATCH_SIZE_GB, [path], len(self.sample_ids))
        self.data_generator = reader.get_data_in_batches(self.batch_size)
        return reader.len


class TableComponentMultiFile(TableComponent):
    """This class reads and processes table data from more than one file.
    """
    def __init__(
            self, comp: Dict[str, Any], dataset_dir_path: str, batch_size: int,
            logging_purpose: struc_log.LoggingPurpose) -> None:
        super().__init__(comp, dataset_dir_path, batch_size, logging_purpose)
        return None

    def get_nr_samples(self) -> int:
        """Get the number of samples of the dataset.

        Returns:
            int: Number of samples as an int.
        """
        self._check_file_extension()
        nr_samples = 0
        for rel_path in self.path:
            abs_path = os.path.join(self.dataset_dir_path, rel_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"There is no file '"
                                        f"{rel_path}' "
                                        f"for component '{self.name}'!")
            reader = read_data.read_data(abs_path)
            nr_samples += reader.get_shape()[0]
        return nr_samples

    def prepare_data_for_reading(self) -> int:
        """Prepare the generator to read the data.

        Check if the data is correct and prepare the batches and generator.

        Raises:
            DSSFErrors.ColumnOverflowException: Raised if there are more
                columns than the upper limit defined in COLUMN_LIMIT.
            FileNotFoundError: The csv file containing the table is missing.
            DSSFErrors.SampleIDError: No unique sample_ids.
            DSSFErrors.DSSFOptionalKeyError: Sample id column not found in
                the saved columns of the table.
            DSSFErrors.ColumnHeaderError: Raised if the csv do not have the
                same header.

        Returns:
            int: number of samples
        """
        self._check_file_extension()
        paths = [os.path.join(self.dataset_dir_path, p) for p in self.path]
        for abs_path, rel_path in zip(paths, self.path):
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"There is no file '"
                                        f"{rel_path}' "
                                        f"for component '{self.name}'!")
        readers = [read_data.read_data(p) for p in paths]
        self.columns: List[str] = readers[0].columns
        self._check_for_forbidden_names(self.columns + [self.name])
        if len(self.columns) > COLUMN_LIMIT:
            raise DSSFErrors.ColumnOverflowException(
                "The upper limit for number of columns in a csv for "
                f"AutoML is '{COLUMN_LIMIT}'. This component has "
                f"'{len(self.columns)}' columns!")
        self.dtypes = readers[0].get_dtypes()
        for r in readers:
            if r.columns != self.columns:
                raise DSSFErrors.ColumnHeaderError(
                    "Split csv need the same header, but "
                    "there are two different ones: "
                    f"'{self.columns}' != '{r.columns}'!")
        sample_id_column = self.opt_info["sample_id_column"]

        if sample_id_column in self.columns:
            self.sample_ids = [str(i) for r in readers
                               for i in r.get_data_single_column(
                                   sample_id_column)]
            if len(self.sample_ids) != len(set(self.sample_ids)):
                msg = (f"The sample_ids in column '{sample_id_column}' are not"
                       " unique!")
                self.component_logger.log_error(msg)
                raise DSSFErrors.SampleIDError(msg)
        else:
            if sample_id_column is not None:
                msg = (f"Sample_id column '{sample_id_column}' not found in "
                       "table!")
                self.component_logger.log_error(msg)
                raise DSSFErrors.DSSFOptionalKeyError(msg)
            self.sample_ids = [
                    str(i) for i in range(sum([r.len for r in readers]))
                ]
        if self.batch_size == -1:
            self.batch_size = compute_component_batch_size(
                BATCH_SIZE_GB, paths, len(self.sample_ids))
        self.data_generator = itertools.chain(
            *[r.get_data_in_batches(self.batch_size) for r in readers])
        return sum([r.len for r in readers])
