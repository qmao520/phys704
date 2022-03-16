# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This class is responsible for loading the dataset and saving it in the
internal format. It will use the Interpreter and Component classes as helper.
"""
import numpy as np
import os
import random
from typing import List, Optional, Dict

from modulos_utils.dssf_and_structure import DSSFInterpreter
from modulos_utils.dssf_and_structure import DSSFComponent
from modulos_utils.data_handling import data_handler as dh
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.dssf_and_structure import structure_logging as struc_log

MINIMUM_NUMBER_OF_SAMPLES_IN_DS = 10
# The following constant is the maximum number of samples a dataset can have.
# We had to introduce this limit because we read the sample ids into RAM
# completely, instead of using generators. We picked the limit such that the
# maximal RAM usage during the dataset upload did not exceed 11 GB.
# TODO: Remove this limit and read the sample ids with generated everywhere
# they are used: REB-957.
MAXIMUM_NUMBER_OF_SAMPLES_IN_DS = int(1e7)


class DSSFSaver():
    """Save the uploaded dataset to the internal format.
    """

    def __init__(
            self, dssf_path: str, dataset_dir_path: str,
            internal_dataset_path: str, batch_size: int = -1,
            nodes_to_be_saved: Optional[List] = None,
            to_be_shuffled: bool = True, logging_purpose:
            struc_log.LoggingPurpose = struc_log.LoggingPurpose.INTERNAL,
            min_number_of_samples: Optional[int] = None,
            dshf_path: Optional[str] = None) \
            -> None:
        self.dssf_path = dssf_path
        self.dataset_dir_path = dataset_dir_path
        self.internal_dataset_path = internal_dataset_path
        self.batch_size = batch_size
        self.nodes_to_be_saved = nodes_to_be_saved
        self._sample_ids: List[str] = []
        self.components: Dict[str, DSSFComponent.ComponentClass] = {}
        if not min_number_of_samples:
            self.min_number_of_samples = MINIMUM_NUMBER_OF_SAMPLES_IN_DS
        else:
            self.min_number_of_samples = min_number_of_samples
        # shuffling map
        self.random_positions: List[int] = []
        # reordering map
        self.reordering: Dict[str, List] = {}
        self.to_be_shuffled = to_be_shuffled
        self.logging_purpose = logging_purpose
        self.dshf_path = dshf_path or os.path.join(os.path.dirname(
            self.internal_dataset_path), dshf_handler.DSHF_FILE_NAME)
        return None

    def initialize(self, seed: int = 42) -> None:
        """initialize Prepares all the resources needed by the saver:
                      seeds RNG, creates the shuffling map, the reordering map
                      and creates the dh5 writer.

        Args:
            seed (int, optional): Seed for shuffling. Defaults to 42.

        Raises:
            SampleIDError: Raised if sample ids of different components are not
                the same.

        Returns:
            None
        """
        interpreter = DSSFInterpreter.DSSFileInterpreter(self.dssf_path)
        (ignored_keys, wrong_keys) = interpreter.initialize_components(
            self.dataset_dir_path, self.batch_size,
            self.logging_purpose)[-1]
        with dshf_handler.get_dshf_writer(self.dshf_path) as dshf:
            dshf.add_to_dshf(
                interpreter.dssf, dshf_handler.EntryTypes.dssf,
                "Read DSSF and initialize Saver.",
                dataset_path=self.internal_dataset_path)
        keys_with_error: List[List] = []
        for comp_name in ignored_keys:
            for key in ignored_keys[comp_name]:
                keys_with_error.append(["ignored", comp_name, key])
        for comp_name in wrong_keys:
            for key in wrong_keys[comp_name]:
                keys_with_error.append(["wrong", comp_name, key])
        if keys_with_error:
            msg_wrongkeys = "\n".join(
                f"Optional key {err[2]} of component {err[1]} has an invalid "
                "value." for err in keys_with_error if err[0] == "wrong")
            msg_ignoredkeys = "\n".join(
                f"Optional key {err[2]} of component {err[1]} is not a valid "
                "key." for err in keys_with_error if err[0] == "ignored")
            raise DSSFErrors.DSSFOptionalKeyError(
                "Error(s) in the optional info of the DSSF:\n"
                + "\n".join((msg_ignoredkeys, msg_wrongkeys)))

        self.components = interpreter.get_components()
        self.check_nr_samples_and_raise()
        self.reordering = self._compute_reordering_and_sample_ids()

        self.random_positions = list(range(len(self._sample_ids)))
        if self.to_be_shuffled:
            random.seed(seed)
            random.shuffle(self.random_positions)

        self._add_sample_ids()

        return None

    def check_nr_samples_and_raise(self) -> None:
        """Check that the number of samples are below the defined maximum
        of the platform and raise if they are not.
        """
        nr_samples = list(self.components.values())[0].get_nr_samples()
        if nr_samples > MAXIMUM_NUMBER_OF_SAMPLES_IN_DS:
            raise DSSFErrors.NRSamplesOverflowError(
                f"The dataset has {nr_samples} samples but the maximum "
                "number of samples allowed by the platform is "
                f"{MAXIMUM_NUMBER_OF_SAMPLES_IN_DS}.")
        return None

    def save_component(self, component_name: str) -> None:
        """Save and convert a component.

        Needs to be called after initialized.

        Args:
            component_name (str): Name of the component to be saved.

        Returns:
            None.
        """
        shuffling = ([self.reordering[component_name][i]
                      for i in self.random_positions]
                     if component_name in self.reordering
                     else self.random_positions)
        with dh.get_Dataset_writer(self.internal_dataset_path,
                                   len(self._sample_ids)) as d5_writer:
            self.components[component_name].save_component(
                d5_writer, shuffling, self.nodes_to_be_saved)
            with dshf_handler.get_dshf_writer(self.dshf_path) as dshf_writer:
                node_names = self.components[component_name].node_names
                nodes_to_comp = {n: component_name for n in node_names}
                dshf_writer.add_to_dshf(
                    {dshf_handler.DSHFKeys.current_nodes: node_names,
                     dshf_handler.DSHFKeys.nodes_to_component:
                         nodes_to_comp},
                    dshf_handler.EntryTypes.nodes_added,
                    f"Saving component {component_name}.")
        return None

    def finalize(self) -> None:
        """Finalize validation of converted components.

        Needs to be called after initialized and save_component.

        Raises:
            SaveError: Raised if no nodes are saved.
            SaveError: Raised if not all nodes specified by nodes_to_be_saved
                were saved.

        Returns:
            None
        """
        self._check_post_saving()
        return None

    def get_components_names(self) -> List[str]:
        """Get the components names of this dataset.

        Returns:
            List[str]: list of component names.
        """
        return list(self.components.keys())

    def main(self, seed=42) -> None:
        """Save a dataset with a dssf into the internal dataset format.

        Args:
            seed (int, optional): Seed for shuffling. Defaults to 42.

        Raises:
            SampleIDError: Raised if sample ids of different components are not
                the same.
            SaveError: Raised if no nodes are saved.
            SaveError: Raised if not all nodes specified by nodes_to_be_saved
                were saved.
        """
        self.initialize(seed)
        for name in self.components:
            self.save_component(name)
        self.finalize()
        return None

    def _check_post_saving(self) -> None:
        """Checks that the dataset has not too many features and that
        everything is saved as mandated.

        Raises:
            DSSFErrors.ColumnOverflowException: too many features
            DSSFErrors.NodesMissingError: No node is saved
            DSSFErrors.NodesMissingError: Not all nodes in 'nodes_to_be_saved'
                are saved

        Returns:
            None
        """
        d5_reader = dh.DatasetReader(self.internal_dataset_path)
        n_features = d5_reader.get_data_info()["n_features"]
        if n_features > DSSFComponent.COLUMN_LIMIT:
            raise DSSFErrors.ColumnOverflowException(
                f"There is the upper limit of '{DSSFComponent.COLUMN_LIMIT}' "
                "for the number of features. This dataset has "
                f"'{n_features}'")
        nodes = d5_reader.get_node_names()
        if dh.SAMPLE_IDS in nodes:
            nodes.remove(dh.SAMPLE_IDS)
        if len(nodes) == 0:
            raise DSSFErrors.NodesMissingError("No nodes were saved.",
                                               self.nodes_to_be_saved)
        elif (self.nodes_to_be_saved is not None and
              not all([any(i in node for node in nodes)
                       for i in self.nodes_to_be_saved])):
            error = [i for i in self.nodes_to_be_saved if i not in nodes]
            raise DSSFErrors.NodesMissingError(
                f"Node(s) '{error}' "
                "is/are not in the dataset and could "
                "therefore not be saved into the internal "
                "dataset.", error)
        return None

    def _compute_reordering_and_sample_ids(self) -> Dict[str, List]:
        """Get the sample ids for all components. Use the first one as
        reference for the others. Compute the reordering map for the other
        components. Return this map.

        Raises:
            SampleIDError: raised if sample ids are not equal for individual
                components

        Returns:
            Dict[str, List]: reordering map
        """
        reordering: Dict[str, List] = {}
        # Make sure, that a table is used as basis for the order of sample_ids.
        component_names = list(self.components.keys())
        for name in self.components:
            if self.components[name].type == "table":
                component_names.remove(name)
                component_names.insert(0, name)
                break
        for name in component_names:
            sample_ids = self.components[name].get_sample_ids()
            if not self._sample_ids:
                if self.to_be_shuffled:
                    self._sample_ids = sorted(sample_ids)
                else:
                    self._sample_ids = sample_ids
            if sample_ids != self._sample_ids:
                if set(sample_ids) == set(self._sample_ids):
                    sample_ids_indexing_dict = {
                        x: n for n, x in enumerate(sample_ids)}
                    reordering[name] = [sample_ids_indexing_dict[i]
                                        for i in self._sample_ids]
                else:
                    if len(sample_ids) != len(self._sample_ids):
                        raise DSSFErrors.MissingSampleError(
                            "The number of samples is not the same for the "
                            f"components '{name}': ({len(sample_ids)}) and "
                            f"'{component_names[0]}': "
                            f"({len(self._sample_ids)})!")
                    else:
                        raise DSSFErrors.SampleIDError(
                            "Sample IDs are not the same!")
        return reordering

    def _add_sample_ids(self) -> None:
        """Write the sample ids.

        Returns:
            None
        """
        if len(self._sample_ids) < self.min_number_of_samples:
            raise DSSFErrors.NotEnoughSamplesError(
                            "Not enough samples in dataset.")
        with dh.get_Dataset_writer(self.internal_dataset_path,
                                   len(self._sample_ids)) as d5_writer:
            d5_writer.add_sample_ids(
                np.array([self._sample_ids[i] for i in self.random_positions]))
        return None
