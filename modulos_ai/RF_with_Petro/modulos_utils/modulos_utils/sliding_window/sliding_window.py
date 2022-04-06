# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Create additional features by window sliding."""
import os
from typing import Dict, List, Optional

import numpy as np

from modulos_utils.data_handling import data_handler as dh
from modulos_utils.metadata_handling import metadata_handler as meta_handler
from modulos_utils.data_handling import data_utils
from modulos_utils.sliding_window import utils as sw_utils


class WindowSliderError(Exception):
    """Exception for errors in the WindowSlider."""
    pass


class WindowSlider:
    """Window Slider class.

    Attributes:
        source_path (str): Path to original dataset.
        output_path (str): Path to new dataset with slided window dataset.
        orig_ds_reader (dh.DatasetReader): Dataset Handler Reader for
            original dataset.
        batch_size (int): Batch size for the dataset.
        feature_time_points (Dict[str, List[int]]): The config parameters that
            define the shifted nodes that are generated for each feature.
    """

    def __init__(
            self, source_path: str, output_path: str,
            time_steps_config: Dict[str, List[int]],
            forecast_step: Optional[int]):
        """ Init for Window slider class.

        Args:
            source_path (str): Path to original dataset.
            output_path (str): Path to the new dataset with the slided window.
            time_steps_config (Dict[str, List[int]]): This argument specifies
                the generated nodes for each original node in the dataset. It
                is a dictionary with the node names as keys and integer lists
                as values. The integers specify the time shift relative to
                time step `t`.
                Example: feature_time_points = {"temperature": [-5, -1, 0, 3]}
                means that this function outputs the following nodes:
                `temperature_t-5`, `temperature_t-1`, `temperature_t`,
                `temperature_t+3`.
            forecast_step (Optional[int]): The forecast step of the
                time series workflow. If it is not given,
                the window sliding is assumed to be executed off-platform. (In
                this case we only generate input nodes, so the forecast step
                is not needed.)

        Raises:
            FileNotFoundError: Raised if original dataset does not exist.
        """
        self.output_path = output_path
        self.source_path = source_path
        if not os.path.isfile(self.source_path):
            raise FileNotFoundError(
                f"File {self.source_path} does not exist.")

        self.orig_ds_reader = dh.DatasetReader(self.source_path)
        self.batch_size = data_utils.compute_batch_size(
            self.source_path, self.orig_ds_reader.get_n_samples())
        self.nr_samples: int
        self.first_sample_index: int
        self.feature_time_points = time_steps_config
        self.forecast_step = forecast_step
        return None

    def _check_input_parameters(
            self, new_nr_samples: int, first_used_sample_index: int) -> None:
        """Perform some checks for the function `slide_window` and raise
        errors if necessary.

        Args:
            new_nr_samples (int): The number of samples of the output dataset.
            first_used_sample_index (int): The index of the first sample, that
                can be used (see doc string of the function `slide_window` for
                more details.)
        """
        if self.batch_size < first_used_sample_index:
            raise WindowSliderError(
                "The batch size that the window sliding was called with, is "
                "too small. The batch size needs to bet >= the maximum lag "
                "absolute value in the argument `feature_time_points`.")
        return None

    def slide_window(self) -> None:
        """Apply the sliding window technique to the dataset generate new
        nodes. We can either generated lagged nodes or add values from a
        future time step.

        The following example should explain the two options:
        Let's say we have a dataset with two nodes: ['temperature',
        'humidity']. This function then has the following options:
        1) generate lagged nodes: For a sample at time step `t`, we can
           add the new node `temperature_t-1` containing the value of
           `temperature` at time step `t-1`. If we do this for each sample
           (i.e. for each time step), the generated node is just a shifted
           version of the original node `temperature`.
        2) generate nodes by adding values from a future step: We can also
           add values from the future. For example we could (for each time step
           `t`) add the node `temperature_t+1` containing the value of
           `temperature` at time step `t+1`.  If we do this for each sample
           (i.e. for each time step), the generated node is just a shifted
           version of the original node `temperature`.

        We see that both options consist of shifting the original nodes. Once,
        we shift it from the past to the present, and in the other option, we
        shift it from the future to the present.
        """
        new_nr_samples, first_used_sample_index = \
            sw_utils.compute_new_nr_samples(
                self.orig_ds_reader.get_n_samples(), self.feature_time_points)
        self._check_input_parameters(new_nr_samples, first_used_sample_index)
        self.nr_samples = new_nr_samples
        self.first_sample_index = first_used_sample_index

        with dh.get_Dataset_writer(self.output_path, new_nr_samples) as writer:
            for node in self.orig_ds_reader.get_node_names():
                if node in self.feature_time_points:
                    self._slide_window_for_node(
                        node, self.feature_time_points[node], writer)
                else:
                    self._copy_node_in_new_dataset(node, writer)
            # This is necessary for keeping the separate datetime group in the
            # dataset. If this group is removed, we can delete that here
            # (REB-228).
            for dt_node in self.orig_ds_reader.get_datetime_node_names():
                self._copy_node_in_new_dataset(
                    dt_node, writer, is_datetime=True)

        # Check that the produced dataset is valid.
        reader = dh.DatasetReader(self.output_path)
        val_info = reader.get_validation_info()
        if "Dataset valid. Everything looks good." not in val_info:
            raise WindowSliderError(
                "Unknown error in the sliding window. "
                "The created hdf5 file is not valid with the following "
                f"validation info entry: \n{val_info}")
        return None

    def _copy_node_in_new_dataset(
            self, node: str, writer: dh.DatasetWriter,
            is_datetime=False) -> None:
        """Copy the node `node` into the new dataset. We have to cut off
        sample at the beginning and the end to account for past and future
        sliding.

        Args:
            node (str): Node to copy into the new dataset.
            writer (dh.DatasetWriter): Dataset writer.
            is_datetime (bool): If node is a datetime node. Defaults to False.
        """
        # This is necessary for keeping the separate datetime group in the
        # dataset. If this group is removed, we can delete that here
        # (REB-228).
        if is_datetime:
            data = self.orig_ds_reader.get_datetime_data_of_node(node)
            data = data[self.first_sample_index:]
            if len(data) > self.nr_samples:
                data = data[:self.nr_samples]
            writer.add_datetime(node, data)
        else:
            nr_samples_already_added = 0
            for n, batch in enumerate(
                    self.orig_ds_reader.get_data_of_node_in_batches(
                        node, self.batch_size)):
                if n == 0:
                    # Cut off the first `shift` samples, as we can not use
                    # it. This will then shift the whole dataset, as
                    # `add_data_to_node` is just appending it at the end.
                    batch = batch[self.first_sample_index:]
                if nr_samples_already_added + len(batch) > self.nr_samples:
                    cut_off = self.nr_samples - nr_samples_already_added
                    batch = batch[:cut_off]
                nr_samples_already_added += len(batch)
                writer.add_data_to_node(node, batch)
                # We can stop early, if all samples, that we need, are already
                # in the new dataset.
                if nr_samples_already_added == self.nr_samples:
                    break
        # Copy metadata
        try:
            meta_obj = meta_handler.PropertiesReader().\
                read_from_ds_reader(
                node, self.orig_ds_reader)
            # Add metadata.
            meta_handler.PropertiesWriter().write_to_ds_writer(
                meta_obj, node, writer)
        except dh.MetaDataDoesNotExistError:
            pass
        return None

    def _shift_node_batch_wise(
            self, original_node_name: str, shift: int,
            writer: dh.DatasetWriter, new_node_name: str) -> None:
        """Perform a shift of a fixed length to a single node and write the
        result the the output dataset using the dataset writer `writer` given
        as an input argument to this function. Note that this function
        performs only one single shift. The function _slide_window_for_node
        calls this function for different shifts, until it has generated all
        new nodes.

        Args:
            original_node_name (str): Name of the node.
            shift (int): Size of the shift to perform. The generated not is
                the one where the original node is shifted to time step
                t+-shift (the sign is determined by the shift).
            writer (dh.DatasetWriter): Dataset writer.
            new_node_name (str): Name of the new node that is generated by
                by function.
        """
        nr_samples_already_added = 0
        for n, batch in enumerate(
                self.orig_ds_reader.get_data_of_node_in_batches(
                    original_node_name, self.batch_size)):
            # Cut off the samples with index < `first_sample_index` samples,
            # as we can not use them. This will then automatically shift the
            # whole dataset, as `add_data_to_node` is just appending to what
            # has already been added to the node.
            if n == 0:
                batch = batch[self.first_sample_index + shift:]
            # We also need to cut off samples at the end. We can maximally have
            # `self.nr_samples` samples.
            if nr_samples_already_added + len(batch) > self.nr_samples:
                cut_off = self.nr_samples - nr_samples_already_added
                batch = batch[:cut_off]
            nr_samples_already_added += len(batch)
            writer.add_data_to_node(new_node_name, batch)
            # We can stop early, if all samples, that we need, are already in
            # the new dataset.
            if nr_samples_already_added == self.nr_samples:
                break

        # Copy metadata
        try:
            meta_obj = meta_handler.PropertiesReader().read_from_ds_reader(
                original_node_name, self.orig_ds_reader)
            meta_handler.PropertiesWriter().write_to_ds_writer(
                meta_obj, new_node_name, writer)
        except dh.MetaDataDoesNotExistError:
            pass
        return None

    def _slide_window_for_node(
            self, original_node_name: str, shifts: List[int],
            writer: dh.DatasetWriter) -> None:
        """Create new nodes, that are shifted versions of the original node
        with the name `original_node_name`. For each entry in the list
        `shifts`, we create a new node and the entry in the list specifies,
        by how many time steps the new node is shifted. For example if the
        argument `shifts` is equal to the list [-2, 4], it means that we
        create the two new nodes `t-2` and `t+4`.

        Args:
            original_node_name (str): Node to create the shifted new nodes for.
            shifts (List[int]): List of integers, that define by how much
                the nodes are shifted. For each integer n in the list, the
                node `original_name_t+-n` is generated, where the sign is
                determined be the integer. If an integer entry is negative,
                we shift the node from the past to the future, i.e. we add
                values from the past (t-1, t-2, etc.). If an integer entry is
                positive, we shift the node from the future to the past, i.e.
                we add values from the future (t+1, t+2, etc.)
            writer (dh.DatasetWriter): Dataset writer.
        """
        for shift in shifts:
            if shift == 0:
                self._copy_node_in_new_dataset(
                    original_node_name, writer, is_datetime=False)
                continue
            shift_string = f"+{shift}" if shift > 0 else str(shift)
            new_node_name = (f"{original_node_name}"
                             f"{sw_utils.GENERATED_NODES_SUFFIX}"
                             f"{shift_string}")
            self._shift_node_batch_wise(
                original_node_name, shift, writer, new_node_name)
        return None

    def adapt_label_for_mimo(self) -> None:
        """
        After the naive window sliding, that just generates new nodes based
        on the config dictionary, this function checks whether the
        configuration belongs to a MIMO (Multiple Input Multiple Output)
        problem, i.e.  whether we forecast multiple time step as one single
        vector. If yes, it merges the individual generated nodes into one
        vector per sample, where the different components belong to the
        different time steps.

        Example:
            Let's say we have a univariate forecasting problem with
            forecast_step = 1 and forecast_horizon=2, and the target node is
            `count`. The function `adapt_label_for_mimo` then transforms the
            dataset

            count  |  count__shifted_t+1  |  count__shifted_t+2
            -----------------------------------------
            0         1                      7
            1         7                      13
            7         13                     15
            13        15                     20

            into

            count  |  count__mimo_label
            ---------------------------
            1         (1, 7)
            7         (7, 13))
            13        (13, 15)
            15        (15, 20)
        """
        # Get all the generated nodes that need to be merged into the label
        # vector.
        if self.forecast_step is None:
            raise ValueError("The variable `forecast_step` can not be "
                             "None.")
        mimo_label_comps = sw_utils.get_mimo_label_components(
            self.feature_time_points, self.forecast_step
        )
        if len(mimo_label_comps) == 0:
            # If the task is not a multi-step prediction task, there is no
            # MIMO label to be created.
            return None
        target_name = sw_utils.get_original_node_name(mimo_label_comps[0])
        # We need to open the dataset that was created by the sliding function
        # and replace the individual nodes by the merged vector node.
        reader = dh.DatasetReader(self.output_path)
        n_samples = reader.get_n_samples()
        batch_size = data_utils.compute_batch_size(
            self.output_path, n_samples)
        old_label_gen = [reader.get_data_of_node_in_batches(
            label, batch=batch_size) for label in mimo_label_comps]
        mimo_label_name = sw_utils.get_mimo_vector_node_name(target_name)
        with dh.get_Dataset_writer(self.output_path, n_samples) as writer:
            for batch in zip(*old_label_gen):
                # Now we merge the individual nodes in each batch.
                batch_as_array = np.array(batch)
                new_label_data = np.transpose(batch_as_array, (
                    1, 0, *list(range(2, len(batch_as_array.shape)))))
                writer.add_data_to_node(mimo_label_name, new_label_data)
            # Now we copy the metadata from one of the generated nodes (
            # it is the same for all generated nodes per construction).
            meta_obj = \
                meta_handler.PropertiesReader().read_from_ds_reader(
                    mimo_label_comps[0], reader)
            new_shape = [len(mimo_label_comps), *meta_obj.node_dim.get()]
            meta_obj.node_dim.set(new_shape)
            meta_handler.PropertiesWriter().write_to_ds_writer(
                meta_obj, mimo_label_name, writer)
            # Now we iterate over all individual nodes, that were merged, and
            # delete them.
            for label in mimo_label_comps:
                writer.remove_node(label)
        return None


def run(
        input_path: str, output_path: str,
        config_dict: Dict[str, List[int]],
        forecast_step: Optional[int] = None) -> None:
    """This is the function that runs the window sliding module on an input
    dataset with an input config dict.

    Args:
        input_path (str): Path to the input hdf5 file.
        output_path (str): Path to the file that is generated by the window
            sliding module.
        config_dict (Dict): Config dictionary that contains the configuration
            parameters for the window sliding module.
        forecast_step (Optional[int]): The forecast step of the
            time series workflow. If it is not given,
            the window sliding is assumed to be executed off-platform. (In
            the off-platform case we only generate input nodes, so the
            forecast step is not needed.)
    """
    slider = WindowSlider(
        input_path, output_path, config_dict, forecast_step)
    slider.slide_window()
    # The following function checks whether the forecasting problem is a
    # multi-step prediction problem and if yes, it creates a MIMO label vector
    # and by merging the time steps t+forecast_step, t+forecast_step+1, ...,
    # t+forecast_step+forecast_horizon into one vector and deleting the
    # individual nodes.
    if forecast_step is not None:
        slider.adapt_label_for_mimo()
