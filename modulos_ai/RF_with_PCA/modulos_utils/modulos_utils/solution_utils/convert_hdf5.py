# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
from datetime import datetime
import os
import pandas as pd
import numpy as np
import slugify
from typing import List, Dict, Optional
from PIL import Image
from enum import Enum

from modulos_utils.data_handling import data_handler as dh
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.solution_utils import utils


DictOfArrays = Dict[str, np.ndarray]
BATCH_SIZE = 1000


class PredictionConversionError(Exception):
    """Exception for when predictions have been converted already.
    """
    pass


class CellFileType(Enum):
    """Enum class for file types supported by the CellSaver class.
    """
    IMAGE = "image"
    NPY = "numpy"
    TXT = "txt"


class RowFileType(Enum):
    """Enum class for file types supported by the RowSaver class.
    """
    CSV = "csv"


class WholeTableFileType(Enum):
    """Enum class for file types supported by the WholeTableSaver class.
    """
    CSV = "csv"


class CellSaver():
    """Class to save each cell of the dataset table in a separate file, i.e.
    each file represents one sample of one node.
    """

    CELL_FILETYPE_TRANSLATOR = {
        ".jpg": CellFileType.IMAGE,
        ".png": CellFileType.IMAGE,
        ".tif": CellFileType.IMAGE,
        ".npy": CellFileType.NPY,
        ".txt": CellFileType.TXT
        }

    def __init__(self, file_ext: str, is_datetime=False):
        """Initialize CellSaver with a certain file extension.

        Args:
            file_ext (str): File extension.
            is_datetime (bool, optional): True if label is a datetime stamp.
                Defaults to False.
        """
        self._file_ext = file_ext
        self._type = CellSaver.CELL_FILETYPE_TRANSLATOR[file_ext]
        self.is_datetime = is_datetime

    @classmethod
    def create(cls, file_ext: str, is_datetime=False):
        """Create CellSaver object and return Exception if file format is
        not supported.

        Args:
            file_ext (str): File extension.
            is_datetime (bool, optional): True if label is a datetime stamp.
                Defaults to False.

        Raises:
            TypeError: Type error if the file extension is not supported by the
                class.

        Returns:
            CellSaver: Initialized CellSaver object.
        """
        if file_ext not in cls.CELL_FILETYPE_TRANSLATOR:
            raise TypeError(f"File extension {file_ext} is currently not "
                            "supported for cell-wise saving.")
        return cls(file_ext=file_ext, is_datetime=is_datetime)

    def _generate_file_path(self, output_dir: str, sample_id: str,
                            node_name: str, single_node: bool) -> str:
        """Generate a file path for a cell of a dataset.

        Args:
            output_dir (str): Output dir.
            sample_id (str): The sample id of the cell.
            node_name (str): The node name of the cell.
            single_node (bool): Flag for whether this is the only node we save.
        """
        if single_node:
            return os.path.join(output_dir, sample_id + self._file_ext)
        else:
            # Slugify the node name to make sure the path handling does
            # run into issues with white spaces, slashes etc.
            node_name_slugified = slugify.slugify(
                node_name.replace("-", "__minus__"),
                separator="_", lowercase=False,
                regex_pattern=r"[^a-zA-Z0-9_]+").replace("__minus__", "-")
            return os.path.join(
                output_dir, node_name_slugified, sample_id + self._file_ext
                )

    def save_batch(self, batch: DictOfArrays, output_dir: str,
                   single_node: bool, sample_id_keyword: str) -> None:
        """Save a batch of samples to files where each node for each sample
        is saved in a separate file.

        Args:
            batch (DictOfArrays): Dictionary containing an array of sample ids
                and a arrays of values for each node.
            output_dir (str): Directory where the output images are written to.
            single_node (bool): Flag for whether this is the only node we save.
            sample_id_keyword (str): Keyword for sample ids.
        """
        # Get node name from batch.
        node_names = list(batch.keys())
        node_names.remove(sample_id_keyword)
        # Test assumptions about the data.
        for node_name in node_names:
            example_tensor = np.array(batch[node_name][0])
            if self._type == CellFileType.IMAGE \
                    and len(example_tensor.shape) < 2 \
                    and len(example_tensor.shape) > 3:
                raise ValueError(
                    "Images must have 2 spatial dimensions and at most one "
                    "channel dimension."
                    )
            if self._type == CellFileType.TXT and example_tensor.shape != ():
                raise ValueError(
                    "Data that is saved in .txt files must be scalars."
                    )
        # Iterate over samples and nodes and save all the cells in separate
        # files, grouped in node folders.
        sample_ids = batch[sample_id_keyword]
        for node_name, node in batch.items():
            if node_name == sample_id_keyword:
                continue
            for sample_index, cell in enumerate(batch[node_name]):
                file_path = self._generate_file_path(
                    output_dir, sample_ids[sample_index], node_name,
                    single_node
                    )
                if sample_index == 0:
                    dir_name = os.path.dirname(file_path)
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                if self._type == CellFileType.IMAGE:
                    im_arr = np.array(cell)
                    im_arr[im_arr > 255] = 255.
                    im_arr[im_arr < 0] = 0.
                    if im_arr.shape[-1] == 1:
                        im_arr = np.squeeze(im_arr, -1)
                    im = Image.fromarray(np.array(im_arr, dtype=np.uint8))
                    im.save(file_path)
                elif self._type == CellFileType.NPY:
                    if self.is_datetime:
                        cell = datetime.fromtimestamp(
                            np.rint(cell)).isoformat()
                    np.save(file_path, np.array(cell))
                elif self._type == CellFileType.TXT:
                    if self.is_datetime:
                        cell = datetime.fromtimestamp(
                            np.rint(cell)).isoformat()
                    with open(file_path, "w") as f:
                        f.write(str(cell))
        return None


class RowSaver():
    """Class to save each row of the dataset table in a separate file, i.e.
    each file represents one sample and all nodes. At the moment we only allow
    one dimensional nodes (i.e. each cell in the dataset is a scalar) and csv
    as a file extension, resulting in csv files with one row and as many
    columns as there are nodes.
    """

    ROW_FILETYPE_TRANSLATOR = {
        ".csv": RowFileType.CSV
        }

    def __init__(self, file_ext: str, is_datetime=False):
        """Initialize RowSaver with a certain file extension.

        Args:
            file_ext (str): File extension.
            is_datetime (bool, optional): True if label is a datetime stamp.
                Defaults to False.
        """
        self._file_ext = file_ext
        self._type = RowSaver.ROW_FILETYPE_TRANSLATOR[file_ext]
        self.is_datetime = is_datetime

    @classmethod
    def create(cls, file_ext: str, is_datetime=False):
        """Create RowSaver object and return Exception if file format is
        not supported.

        Args:
            file_ext (str): File extension.
            is_datetime (bool, optional): True if label is a datetime stamp.
                Defaults to False.

        Raises:
            TypeError: Type error if the file extension is not supported by the
                class.

        Returns:
            RowSaver: Initialized RowSaver object.
        """
        if file_ext not in cls.ROW_FILETYPE_TRANSLATOR:
            raise TypeError(f"File extension {file_ext} is currently not "
                            "supported for row-wise saving.")
        return cls(file_ext=file_ext, is_datetime=is_datetime)

    def save_batch(self, batch: DictOfArrays, output_dir: str,
                   sample_id_keyword: str) -> None:
        """Save a batch of rows to the format of self._file_ext.

        Args:
            batch (DictOfArrays): Dictionary containing an array of sample ids
                and arrays of scalars. The number of arrays is arbitrary but
                all arrays are assumed to have the same length.
            output_dir (str): Directory where the output images are written to.
            sample_id_keyword (str): Sample id keyword.
        """
        # Get node name from batch.
        node_names = list(batch.keys())
        node_names.remove(sample_id_keyword)
        # Test assumptions of one-dimensional nodes.
        for node_name in node_names:
            if len(batch[node_name].shape) > 2 \
                    or (len(batch[node_name].shape) == 2
                        and batch[node_name].shape[1] != 1):
                raise TypeError("Only one dimensional nodes can be saved "
                                "in one single file.")

        # Convert nested lists to scalars.
        batch_scalars = {}
        for key, value in batch.items():
            batch_scalars[key] = value.reshape(-1)

        # For each sample, create a pandas dataframe from all the nodes and
        # and save it in its own csv file.
        sample_ids = batch_scalars[sample_id_keyword]
        for batch_index, sample_id in enumerate(sample_ids):
            row_dict = {sample_id_keyword: [sample_id]}
            for node_name in node_names:
                row_dict[node_name] = [batch_scalars[node_name][batch_index]]
            filename = os.path.join(output_dir, sample_id + self._file_ext)
            df = pd.DataFrame(row_dict)
            df.set_index(sample_id_keyword, inplace=True)
            if self.is_datetime:
                df[dc.PREDICTIONS_NODE_NAME] = pd.to_datetime(
                    df[dc.PREDICTIONS_NODE_NAME], unit="s")
            df.to_csv(filename)
        return None


class WholeTableSaver():
    """Class to save the whole dataset in one file. This is currently only
    supported, if all the nodes are one-dimensional and the file format is csv.
    """

    WholeTable_FILETYPE_TRANSLATOR = {
        ".csv": WholeTableFileType.CSV
        }

    def __init__(self, file_ext: str, file_name: str = None,
                 is_datetime=False):
        """Initialize WholeTableSaver with a certain file extension.

        Args:
            file_ext (str): File extension.
            file_name (str): Optional file name. If it is None, the filename
                will be generated by appending the node names to each others
                (separated by an underscore).
            is_datetime (bool, optional): True if label is a datetime stamp.
                Defaults to False.
        """
        self._file_ext: str = file_ext
        self._type: WholeTableFileType = \
            WholeTableSaver.WholeTable_FILETYPE_TRANSLATOR[file_ext]
        self._file_path: Optional[str] = None
        self._file_name = file_name
        self.is_datetime = is_datetime

    @classmethod
    def create(cls, file_ext: str, file_name: str = None, is_datetime=False):
        """Create WholeTableSaver object and return Exception if file format is
        not supported.

        Args:
            file_ext (str): File extension.
            file_name (str): Optional file name. If it is None, the filename
                will be generated by appending the node names to each others
                (separated by an underscore).
            is_datetime (bool, optional): True if label is a datetime stamp.
                Defaults to False.

        Raises:
            TypeError: Type error if the file extension is not supported by the
                class.

        Returns:
            WholeTableSaver: Initialized WholeTableSaver object.
        """
        if file_ext not in cls.WholeTable_FILETYPE_TRANSLATOR:
            raise TypeError(f"File extension {file_ext} is currently not "
                            "supported for saving the whole dataset in one "
                            "file.")
        return cls(file_ext=file_ext, file_name=file_name,
                   is_datetime=is_datetime)

    def save_batch(self, batch: DictOfArrays, output_dir: str,
                   sample_id_keyword: str) -> None:
        """Save batches of rows to one file which has the file extension
        self._file_ext.

        Args:
            batch (DictOfArrays): Dictionary containing an array of sample ids
                and arrays of scalars. The number of arrays is arbitrary but
                all arrays are assumed to have the same length.
            output_dir (str): Directory where the output images are written to.
            sample_id_keyword (str): Sample id keyword.
        """
        # Get node name from batch.
        node_names = list(batch.keys())
        node_names.remove(sample_id_keyword)
        # Test assumptions of one-dimensional nodes.
        for node_name in node_names:
            if len(batch[node_name].shape) > 2 \
                    or (len(batch[node_name].shape) == 2
                        and batch[node_name].shape[1] != 1):
                raise TypeError("Only one dimensional nodes can be saved "
                                "in one single file.")

        # Convert nested arrays to scalars.
        batch_scalars = {}
        for key, value in batch.items():
            batch_scalars[key] = value.reshape(-1)

        # Create pandas data frame.
        df = pd.DataFrame.from_dict(batch_scalars)
        df.set_index(sample_id_keyword, inplace=True)

        # Convert to datetime string, if it is a datetime stamp.
        if self.is_datetime:
            df[dc.PREDICTIONS_NODE_NAME] = pd.to_datetime(
                df[dc.PREDICTIONS_NODE_NAME], unit="s")

        # If file does not exist, we create it. Otherwise we append to it.
        if self._file_path is None:
            if self._file_name is None:
                node_names_string = "__".join(node_names)
                if self._type == WholeTableFileType.CSV:
                    self._file_path = os.path.join(
                        output_dir, f"{node_names_string}.csv")
                else:
                    raise TypeError(f"File extension {self._file_ext} is not "
                                    "supported by WholeTableSaver class.")
            else:
                self._file_path = os.path.join(output_dir, self._file_name)
            df.to_csv(self._file_path)
        else:
            df.to_csv(self._file_path, mode="a", header=False)
        return None


def save_predictions_to_label_format(
        hdf5_path: str, label_metadata: Dict[str, meta_prop.AllProperties],
        output_dir: str, dshf: dshf_handler.DSHFHandler
) -> None:
    """Convert predictions from hdf5 back to original format, that was uploaded
    to the auto-ml system.

    Args:
        hdf5_path (str): Path to hdf5 prediction file.
        label_metadata (dict): Label metadata dictionary. We need this so that
            we know which format we have to convert back to.
        output_dir (str): Directory where output files are written to.
        dshf (dshf_handler.DSHFHandler): History file of the training data.

    Raises:
        PredictionConversionError: Exception to avoid overwriting of the
            output if the output_dir already exists.
    """
    label_node_name = list(label_metadata.keys())[0]
    pred_name_list = [dc.PREDICTIONS_NODE_NAME]
    node_dims = {pred_name_list[0]: label_metadata[
        label_node_name].node_dim.get()}
    external_sample_id_name = utils.get_sample_id_column_name(dshf=dshf)
    # Create output dir if it does not exist (otherwise raise an Exception).
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        raise PredictionConversionError("Predictions have been converted "
                                        "already. Nothing is overwritten. ")

    # Check if the label is a datetime stamp and if yes, give this information
    # to the file saver.
    is_datetime = label_node_name in dshf.generated_nodes

    component_example_path = label_metadata[label_node_name]. \
        dssf_component_file_path.get()

    ext = os.path.splitext(component_example_path)[1]

    data_generator = dh.DatasetReader(hdf5_path).get_data_in_batches(
            batch_size=BATCH_SIZE)
    if label_metadata[label_node_name].is_upload_single_table():
        whole_table_saver = WholeTableSaver.create(
            ext, is_datetime=is_datetime)
        for batch in data_generator:
            whole_table_saver.save_batch(
                format_data_batch(batch, node_dims, pred_name_list,
                                  external_sample_id_name),
                output_dir, external_sample_id_name)
    elif label_metadata[label_node_name].is_upload_sample_wise_table():
        row_wise_saver = RowSaver.create(ext, is_datetime)
        for batch in data_generator:
            row_wise_saver.save_batch(
                format_data_batch(batch, node_dims, pred_name_list,
                                  external_sample_id_name),
                output_dir,  external_sample_id_name)
    else:
        cell_wise_saver = CellSaver.create(ext, is_datetime=is_datetime)
        for batch in data_generator:
            single_node_flag = len(list(batch.keys())) == 2
            cell_wise_saver.save_batch(
                format_data_batch(batch, node_dims, pred_name_list,
                                  external_sample_id_name),
                output_dir, single_node_flag, external_sample_id_name)
    return None


def slice_data_batch(
        data_batch: DictOfArrays, node_names: List[str],
        sample_id_keyword: str) -> DictOfArrays:
    """Get subset of data batch including sample ids.

    Args:
        data_batch (DictOfArrays): Data batch.
        node_names (List[str]): Nodes to select.
        sample_id_keyword (str): Keyword used for sample ids.

    Returns:
        DictOfArrays: Dictionary with node names as keys and node data as
            value.
    """
    sub_batch = {n: data_batch[n] for n in node_names}
    sub_batch[sample_id_keyword] = data_batch[sample_id_keyword]
    return sub_batch


def format_data_batch(
        data_batch: Dict[str, np.ndarray], node_dims: Dict[str, List[int]],
        node_names: List[str], external_sample_ids_key: str,
        internal_sample_ids_key: str = dh.SAMPLE_IDS
        ) -> DictOfArrays:
    """Remove empty dimensions for scalar nodes. The reason that we remove
    empty dimensions for scalar nodes, is that we don't want nested arrays,
    when we save them in their original (For example in a csv, we want the
    columns to be flat arrays). However we have the conventions for the modules
    that they always output a two dimensional tensor for each node, where the
    first dimension is the number of samples and the second dimension is the
    dimension of the node (i.e. 1 for scalars).

    Args:
        data_batch (Dict[str, np.ndarray]): Batch of data (node names as keys).
        node_dims (Dict[str, List[int]]): Dictionary with node names as keys
            and node dim lists as values.
        node_names (List[str]): List of node names to include in the formatted
            batch.
        sample_ids_key (str = dh.SAMPLE_IDS): Sample id key in the batch
            dictionary. The default is the internal dataset sample id keyword.
            But it will be removed as of BAS-595 (see comment below).

    Returns:
        DictOfArrays: Batch of data, where node data are numpy arrays
            and have empty dimensions removed for scalars.
    """
    # Note that the default value for sample_ids_key will be removed in
    # BAS-595. This function should always be called with a sample id key,
    # because either we are saving a generated dataset with an arbitrary
    # sample id column name, or we are converting a dataset back to its
    # original format, where there was a sample id column name defined in the
    # DSSF (and if not, we don't save any sample ids).
    new_batch = {}
    for key in node_names:
        value = data_batch[key]
        if node_dims[key] == [1]:
            new_batch[key] = np.atleast_1d(np.squeeze(value))
        else:
            new_batch[key] = value
    new_batch[external_sample_ids_key] = np.atleast_1d(
        np.squeeze(data_batch[internal_sample_ids_key]))
    return new_batch


def save_nodes_in_original_format(
        hdf5_path: str, node_names_to_save: List, output_dir: str,
        dshf_path: str,
        sample_ids_to_save: List[int] = None,
        metadata: Dict[str, meta_prop.AllProperties] = None) -> Dict[str, str]:
    """Convert nodes from an hdf5 back to original format.

    Args:
        hdf5_path (str): Path to hdf5 prediction file.
        node_names_to_save (list): List of node names to convert.
        output_dir (str): Directory where output files are written to.
        dshf_path (str): Path to the dshf of the dataset.
        sample_ids_to_save (List[int]): Sample ids to save. The default is
            that all samples are saved.
        metadata (Dict[str, meta_prop.AllProperties] = None): Metadata of the
            dataset. If not given, it is read in from the hdf5 file.

    Returns:
        Dict[str, str]: For each tensor node name, the name of the folder
            that contains the output files.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    data_reader = dh.DatasetReader(hdf5_path)
    if metadata is None:
        metadata = meta_handler.get_metadata_all_nodes(hdf5_path)
    node_dims = {n: metadata[n].node_dim.get() for n in node_names_to_save}

    dshf = dshf_handler.DSHFHandler(dshf_path)
    external_sample_id_name = utils.get_sample_id_column_name(dshf=dshf)

    cell_wise_nodes: Dict[str, List[str]] = {}
    cell_wise_savers: Dict[str, CellSaver] = {}
    single_file_nodes: Dict[str, List[str]] = {}
    single_file_savers: Dict[str, WholeTableSaver] = {}
    row_wise_nodes: Dict[str, List[str]] = {}
    row_wise_savers: Dict[str, RowSaver] = {}
    for node_name in node_names_to_save:
        exmpl_path = metadata[node_name].dssf_component_file_path.get()
        ext = os.path.splitext(exmpl_path)[1]
        if metadata[node_name].is_upload_single_table():
            single_file_nodes.setdefault(ext, []).append(node_name)
            if ext not in single_file_savers:
                single_file_savers[ext] = WholeTableSaver.create(ext)
        elif metadata[node_name].is_upload_sample_wise_table():
            row_wise_nodes.setdefault(ext, []).append(node_name)
            if ext not in row_wise_savers:
                row_wise_savers[ext] = RowSaver.create(ext)
        else:
            cell_wise_nodes.setdefault(ext, []).append(node_name)
            if ext not in cell_wise_savers:
                cell_wise_savers[ext] = CellSaver.create(ext)

    data_generator = data_reader.get_data_in_batches(batch_size=BATCH_SIZE)
    tensor_node_dirnames = {}
    for batch in data_generator:
        batch_sample_ids = batch[dh.SAMPLE_IDS]
        if sample_ids_to_save is not None:
            if any(s_id in sample_ids_to_save
                   for s_id in batch_sample_ids):
                slice_indices = [i for i, s_id in enumerate(batch_sample_ids)
                                 if s_id in sample_ids_to_save]
                for node_name, node_data in batch.items():
                    batch[node_name] = node_data[slice_indices]
            else:
                continue
        batch_formatted = format_data_batch(
            batch, node_dims, node_names_to_save,
            external_sample_id_name)
        for file_ext, node_names in single_file_nodes.items():
            single_file_savers[file_ext].save_batch(
                slice_data_batch(batch_formatted, node_names,
                                 external_sample_id_name),
                output_dir, external_sample_id_name)
        for file_ext, node_names in row_wise_nodes.items():
            row_wise_savers[file_ext].save_batch(
                slice_data_batch(batch_formatted, node_names,
                                 external_sample_id_name),
                output_dir, external_sample_id_name)
        for file_ext, node_names in cell_wise_nodes.items():
            cell_wise_savers[file_ext].save_batch(
                slice_data_batch(batch_formatted, node_names,
                                 external_sample_id_name),
                output_dir, False, external_sample_id_name)
            for n in node_names:
                rel_path = cell_wise_savers[file_ext]._generate_file_path(
                    output_dir, batch_sample_ids[0], n, False)
                tensor_node_dirnames[n] = os.path.basename(
                    os.path.dirname(rel_path))
    return tensor_node_dirnames
