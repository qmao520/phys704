# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Jupyter Notebook utility functions."""
import re
import os
import pathlib
from typing import Optional, Dict, Any, List

import pandas as pd

from modulos_utils.data_handling import data_handler as dh
from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.solution_utils import utils
from PIL import Image


ALLOWED_IMGS_EXTENSIONS = [".tif", ".jpg", ".png"]


class JupyterDisplayerError(Exception):
    pass


class JupyterDisplayer():
    """Helper class to display prediction results in a jupyter notebook.

    Attr:
        internal_dataset_path (str): Path to the internal dataset (hdf5).
        tmp_data_path (str): Path to tmp data.
        predictions_hdf5_path (str): Path to the hdf5 prediction data.
        prediction_path (str): Path to the file predictions.
        base_dir (str): Path to the base directory of solution folder code.
        n_samples (int): Number of samples to show.

        prediction_df (Optional[pd.DataFrame]): DataFrame of the predictions.
        input_df (Optional[pd.DataFrame]): DataFrame of the input data.
    """

    def __init__(self, base_dir: str, n_samples: int):
        """Init method of the JupyterDisplayer class.

        Args:
            base_dir (str): Path to the base directory of solution folder code.
            n_samples (int): Number of samples to show.
        """

        self.base_dir = base_dir
        self.tmp_data_path = os.path.join(self.base_dir,
                                          "output_batch_client/tmp_data_dir")
        self.predictions_hdf5_path = \
            os.path.join(self.tmp_data_path, "predictions.hdf5")
        self.internal_dataset_path = \
            os.path.join(self.tmp_data_path,
                         "check_my_dataset_output/temp.hdf5")
        self.dshf_path = os.path.join(
            self.tmp_data_path, dshf_handler.DSHF_FILE_NAME)
        self.dataset_history_path = os.path.join(
            self.base_dir, "src/dataset_history",
            dshf_handler.DSHF_FILE_NAME)
        self.prediction_path = os.path.join(self.base_dir,
                                            "output_batch_client/predictions")

        self.n_samples = n_samples

        self.prediction_df: Optional[pd.DataFrame] = None
        self.input_df: Optional[pd.DataFrame] = None
        self._merged_df: Optional[pd.DataFrame] = None
        self.image_node_names: Optional[List[str]] = None

        self.thumbnail_size = (50, 50)
        self.img_display_width = 50

    @classmethod
    def construct(cls,
                  base_dir: str, n_samples: int = 10) -> "JupyterDisplayer":
        """Constructor of the Jupyter Displayer. Infers the prediction
        dataframe.

        Args:
            base_dir (str): Path to the base directory of solution folder code.
            n_samples (int): Number of samples to show.

        Returns:
            (JupyterDisplayer): Returns a fully initialized jupyter displayer.
        """

        displayer = cls(base_dir=base_dir, n_samples=n_samples)
        displayer._infer_input_df()
        displayer._infer_prediction_df()

        return displayer

    def show(self) -> pd.DataFrame:
        """Display the combined DataFrame (predictions and input) in the
        Jupyter Notebook.

        Returns:
            pd.DataFrame: Combined DataFrame (predictions and input)
        """
        self._merge_df()
        self._find_image_nodes()
        self._create_thumbnails()
        self._replace_thumbnail_path()
        if self._merged_df is not None:
            dshf = dshf_handler.DSHFHandler(self.dataset_history_path)
            external_sample_id_name = utils.get_sample_id_column_name(dshf)
            self._merged_df = \
                self._merged_df.rename(
                    columns={dh.SAMPLE_IDS: external_sample_id_name})
            _merged_df = self._merged_df.style.apply(
                lambda x: ["background: #6daa9c"
                           if x.name == "predictions" else ""
                           for i in x])
            return _merged_df.hide_index().render()
        else:
            raise JupyterDisplayerError(
                "The internal dataset and the prediction have not been merged"
                "yet.")

    def _merge_df(self):
        """Merge Dataframes."""
        _merged_df = pd.merge(self.input_df, self.prediction_df, how="inner",
                              on=dh.SAMPLE_IDS)

        self._merged_df = self._reorder_dataframe(_merged_df)
        return None

    def _find_image_nodes(self):
        """Find all image nodes of the prediction and input data frames."""

        self.image_node_names = []
        for node_name, value in dict(self._merged_df.iloc[0]).items():
            filename, file_ext = os.path.splitext(str(value))
            if isinstance(value, str) and file_ext in ALLOWED_IMGS_EXTENSIONS:
                self.image_node_names.append(node_name)
        return None

    def _replace_thumbnail_path(self):
        """Replace images_paths in merged prediction and input data frame with
           HTML tags.
        """
        for node_name in self.image_node_names:
            self._merged_df[node_name] = self.thumbnail_html_tags[node_name]

    def _create_thumbnails(self):
        """Convert Images to thumbnails and save relative path."""
        self.thumbnail_html_tags = {}
        for node_name in self.image_node_names:
            html_img_tags = []
            for rel_img_path in self._merged_df[node_name].values:
                if "predictions" not in rel_img_path:
                    rel_img_path = os.path.join(
                        "output_batch_client/tmp_data_dir", rel_img_path)
                img_path = os.path.join(self.base_dir, rel_img_path)
                image = Image.open(img_path)

                image.thumbnail(self.thumbnail_size)
                rel_filepath, file_ext = os.path.splitext(rel_img_path)
                rel_filepath = rel_filepath + "_thumbnail.png"
                image.save(os.path.join(self.base_dir, rel_filepath))
                html_img_tags.append(f"<img src='{rel_filepath}'"
                                     f"width='{self.img_display_width}'/>")
            self.thumbnail_html_tags[node_name] = html_img_tags

    def _reorder_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rearrange elements of the DataFrame such that the sample ids are in
        the first column and predictions are in the second.

        Args:
            df (pd.DataFrame): DataFrame to rearrange.
        Returns:
            pd.DataFrame: rearrange DataFrame.
        """
        column_names = df.columns.tolist()
        elements_dict = {dh.SAMPLE_IDS: 0, "predictions": 1}

        for key, value in elements_dict.items():
            tmp_el = column_names[value]
            tmp_idx = column_names.index(key)
            column_names[value] = key
            column_names[tmp_idx] = tmp_el
        return df[column_names]

    def _infer_prediction_df(self) -> None:
        """Infer the prediction DataFrame, from the path to the folder of the
        predictions.
        """
        label_metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(self.base_dir, "src/metadata/label_metadata.bin")
        )
        label_node_name = list(label_metadata.keys())[0]
        if label_metadata[label_node_name].is_scalar():

            dataset_reader = dh.DatasetReader(self.predictions_hdf5_path)
            predictions_dict = dataset_reader\
                .get_data_all_as_one_tensor()
            predictions_dict["predictions"] = predictions_dict["data"]\
                .reshape(-1)
            if (label_metadata[
                    label_node_name].upload_node_name.get() != label_node_name
                    and "unixtime" in label_node_name):
                predictions_dict["predictions"] = pd.to_datetime(
                    predictions_dict["predictions"], unit="s")
            predictions_dict.pop("data")
            prediction_df = pd.DataFrame(predictions_dict)

        else:
            predictions_dict = {dh.SAMPLE_IDS: [],
                                "predictions": []}
            for file_path in pathlib.Path(self.prediction_path).rglob("*.*"):
                file_name_ext = os.path.basename(file_path)
                file_name = os.path.splitext(file_name_ext)[0]
                rel_filepath = os.path.join(
                    "output_batch_client/",
                    str(file_path).split("output_batch_client/")[1])
                predictions_dict["predictions"].append(str(rel_filepath))
                predictions_dict[dh.SAMPLE_IDS].append(file_name)

                prediction_df = pd.DataFrame(predictions_dict)

        prediction_df = prediction_df.astype({dh.SAMPLE_IDS: str})
        self.prediction_df = prediction_df

    def _infer_input_df(self) -> None:
        """Infer preview input dataset DataFrame."""
        input_metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(self.base_dir, "src/metadata/input_metadata.bin")
        )
        data_reader = dh.DatasetReader(self.internal_dataset_path)
        node_names = data_reader.get_upload_node_names()
        if dh.SAMPLE_IDS in node_names:
            node_names.remove(dh.SAMPLE_IDS)
        sample_ids = data_reader.get_sample_ids_all()[
            :self.n_samples].tolist()
        df_dict: Dict[str, Any] = {dh.SAMPLE_IDS: sample_ids}
        for node in node_names:
            if node not in input_metadata or input_metadata[node].is_scalar():
                node_data = next(data_reader.get_data_of_node_in_batches(
                    node, batch=self.n_samples))[:, 0].tolist()
            elif (len(input_metadata[node].node_dim.get()) == 1 and
                    input_metadata[node].node_dim.get()[0] < 5):
                node_data = next(data_reader.get_data_of_node_in_batches(
                    node, batch=self.n_samples)).tolist()
            else:
                node_data = self._get_paths_for_ids(node, sample_ids)
            df_dict[node] = node_data
        input_df = pd.DataFrame(df_dict)
        input_df = input_df.astype({dh.SAMPLE_IDS: str})
        self.input_df = input_df

    def _get_paths_for_ids(
            self, node: str, sample_ids: List[str]) -> List[str]:
        """Return the paths to the original file for some samples
        defined by their sample_id of node `node`.

        Args:
            sample_ids (List[str]): IDs of the samples to retrieve.
            node (str): Node for which to retrieve the paths.

        Returns:
            List[str]: List of paths.
        """
        dshf = dshf_handler.DSHFHandler(self.dshf_path)
        comp_name = dshf.get_component_name(node)
        encoded_file_path = dshf.dssf_info[comp_name]["path"]
        file_paths = []
        for sample in sample_ids:
            file_paths.append(
                re.sub(r"\{(.*?)\}", sample, encoded_file_path))
        return file_paths
