# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This Feature Extractor preprocesses tables by standard scaling and encoding
of categorical values and then selects the top x% features by use of t-test
feature selection. For classification tasks it performs a t-test for each
feature and each label class c for the following two populations:
-  all samples
-  only samples belonging to a class c.

For regression tasks, it bins the labels into 100 classes and then proceeds in
the same way as for classification tasks.
"""
import copy
import joblib
import json
import math
import numpy as np
import os
from typing import Dict, List, Generator, Optional

from modulos_utils.metadata_handling import metadata_transferer as meta_trans
from modulos_utils.metadata_handling import metadata_utils as meta_utils
from modulos_utils.metadata_handling import metadata_properties as meta_prop
from modulos_utils.convert_dataset import dataset_converter as dc
from modulos_utils.convert_dataset import dataset_return_object as d_obj
from modulos_utils.module_interfaces import feature_extractor as fe_interface
from . import encode_categories as ec
from . import scale_numbers as sn
from . import helpers as fe_helpers

DictOfArrays = Dict[str, np.ndarray]
GenOfDicts = Generator[dict, None, None]
DictOfMetadata = Dict[str, meta_prop.AllProperties]

N_BINS = 20


class TTestFeatureSelector(fe_interface.IFeatureExtractor):
    """Feature Extractor that ranks table columns according to a t-statistics.
    """

    def __init__(self) -> None:
        """Initialize object of the class.
        """
        self._num_transformation: Optional[sn.NumberScalingTypes] = None
        self._cat_transformation: Optional[ec.CategoryEncoderTypes] = None
        self._node_list: Optional[List[str]] = None
        self._metadata: Optional[dict] = None
        self._encoders: Optional[dict] = None
        self._scalers: Optional[dict] = None
        self._transformed_metadata: Optional[dict] = None
        self._weights_loaded: bool = False
        # Note that we select all nodes, if this fraction times the number of
        # selectable nodes is less than 5.
        self._n_features_fraction: Optional[float] = None
        self._selected_nodes: Optional[np.ndarray] = None

    @staticmethod
    def initialize_new(
            config_choice_path: str, num_transformation: sn.NumberScalingTypes,
            cat_transformation: ec.CategoryEncoderTypes
            ) -> "TTestFeatureSelector":
        """Initialize a new (untrained) feature extractor from a config choice
        file.

        Args:
            config_choice_path (str): Path to config choice file.
            num_transformation (sn.NumberScalingTypes): Which transformation
                to apply to numerical nodes.
            cat_transformation (ec.CategoryEncoderTypes): Which transformation
                to apply to categorical nodes.

        Returns:
            TTestFeatureSelector: An initialized object of this
                class.
        """
        result_obj = TTestFeatureSelector()
        result_obj._num_transformation = num_transformation
        result_obj._cat_transformation = cat_transformation
        with open(config_choice_path, "r") as f:
            config_choice = json.load(f)
        result_obj._n_features_fraction = \
            float(config_choice["n_features_fraction"])
        return result_obj

    @staticmethod
    def initialize_from_weights(weights_folder: str) \
            -> "TTestFeatureSelector":
        """Load a trained feature extractor from weights. These weights are
        generalized weights meaning anything that is saved in the
        training phase and used in the prediction phase. (Therefore the term
        weights is not used in a strict sense as in the parameters of a neural
        network that are optimized during training.) The weights contain all
        the information necessary to reconstruct the Feature Extractor object.

        Args:
            weights_folder (str): Path to folder containing weights.

        Returns:
            TTestFeatureSelector: An initialized object of this
                class.
        """
        # Check whether weights path exits.
        if not os.path.isdir(weights_folder):
            raise fe_helpers.TTEstFeatureSelectionError(
                f"Directory {weights_folder} does not exist.")
        result_obj = TTestFeatureSelector()
        metadata = meta_utils.MetadataDumper().load_all_nodes(
            os.path.join(weights_folder, "input_metadata.bin")
        )
        result_obj._metadata = metadata
        result_obj._node_list = list(metadata.keys())
        result_obj._encoders = joblib.load(
            os.path.join(weights_folder, "encoders.bin")
            )
        result_obj._scalers = joblib.load(
            os.path.join(weights_folder, "scalers.bin"))
        result_obj._num_transformation = joblib.load(
            os.path.join(weights_folder, "num_transformation.bin")
        )
        result_obj._cat_transformation = joblib.load(
            os.path.join(weights_folder, "cat_transformation.bin")
        )
        result_obj._selected_nodes = joblib.load(
            os.path.join(weights_folder, "selected_nodes.bin")
        )
        result_obj._weights_loaded = True
        return result_obj

    def save_weights(self, weights_folder: str) -> None:
        """Save feature extractor weights.

        Args:
            weights_folder (str): Path to folder where weights should be saved.
        """
        if self._metadata is None or self._scalers is None or \
                self._encoders is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Generalized weights of this feature extractor cannot be "
                "saved because the feature extractor has not been trained "
                "yet.")
        if not os.path.isdir(weights_folder):
            os.makedirs(weights_folder)
        meta_utils.MetadataDumper().write_all_nodes(
            self._metadata, os.path.join(weights_folder, "input_metadata.bin")
        )
        joblib.dump(self._scalers, os.path.join(weights_folder, "scalers.bin"))
        joblib.dump(self._encoders, os.path.join(weights_folder,
                                                 "encoders.bin"))
        joblib.dump(self._num_transformation,
                    os.path.join(weights_folder, "num_transformation.bin"))
        joblib.dump(self._cat_transformation,
                    os.path.join(weights_folder, "cat_transformation.bin"))
        joblib.dump(self._selected_nodes,
                    os.path.join(weights_folder, "selected_nodes.bin"))

    def _select_top_k_features(
            self, node_names: List,
            inclass_means: Dict[str, Dict[str, float]],
            overall_means: Dict[str, float],
            unique_label_categories: np.ndarray,
            n_samples_classwise: Dict[str, int], n_samples: int,
            S_i_squared: List) -> None:
        """Compute a rank for each node by maximizing the t-statistics over
        the different classes. Then select the top k nodes, where
        k = self._n_features_fraction * len(self._nodes_list). The algorithm
        is described in the following paper:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5054219/.

        Args:
            node_names (List): All node names that are given as an input
                to t-test feature selection (i.e. the output nodes of the
                table prep step.)
            inclass_means (Dict[str, Dict[str, float]]): Nested dictionary,
                where for each node n and class c it contains the means of the
                node n computed by averaging over samples belonging to class c.
            overall_means (Dict[str, float]): Means averaged over all samples
                for each node.
            unique_label_categories (np.ndarray): Unique categories of labels.
            n_samples_classwise (Dict[str, int]): The number of samples
                belonging to each label class.
            n_samples (int): The total number of samples.
            S_i_squared (List): [description]

        Returns:
            [type]: [description]
        """
        if self._n_features_fraction is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TTestFeatureSelector object was not constructed "
                "properly. Use the function 'initialize_from_weights' or "
                "'initialize_new' to construct it.")
        # Compute feature rankings. For each node we define the rank as the
        # maximum of the t-statistics over all classes.
        feature_rankings = []
        for ni, nn in enumerate(node_names):
            mean_diffs = {c: np.abs(inclass_means[c][nn] - overall_means[nn])
                          for c in unique_label_categories}
            t_scores = []
            for c in unique_label_categories:
                if n_samples_classwise[c] == 0:
                    M_c = math.sqrt(1. / n_samples)
                else:
                    M_c = math.sqrt(1. / n_samples_classwise[c] +
                                    1. / n_samples)
                if S_i_squared[ni] == 0:
                    t_scores.append(0)
                else:
                    t_scores.append(mean_diffs[c] / S_i_squared[ni] / M_c)
            feature_rankings.append(np.max(t_scores))

        top_k = np.array(feature_rankings).argsort()[
            -int(self._n_features_fraction * len(node_names)):]
        self._selected_nodes = np.sort(np.array(node_names)[top_k])
        return None

    def _fit_table_prep_part(
            self, input_data: DictOfArrays, metadata: DictOfMetadata,
            online: bool = False) -> None:
        """Train scalers for numerical nodes and encoders for categorical
        nodes.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.
            online (bool = False): Whether the column transformators are
                trained in an online fashion
        """
        if self._num_transformation is None \
                or self._cat_transformation is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TTestFeatureSelector object was not constructed "
                "properly. ")

        # This function is used both for online training and "all-in-once"
        # training. To not overwrite stuff, when it is called multiple times
        # (while iterating over batches), we set the metadata and the nodes
        # list only the first time.
        if self._metadata is None and self._node_list is None:
            self._metadata = copy.deepcopy(metadata)
            self._node_list = list(self._metadata.keys())
            self._encoders = {}
            self._scalers = {}
        if self._node_list is None or self._metadata is None or \
                self._encoders is None or self._scalers is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Training the column transformators failed.")

        # Train categorical nodes on the metadata (They don't need to see the
        # actual data.) and numerical nodes on the data.
        for node_name in self._node_list.copy():
            node_meta = self._metadata[node_name]
            if node_meta.is_categorical():
                # We only train the encoders once.
                if node_name not in self._encoders:
                    col_trans_cat = fe_helpers.ColumnTransformatorCategorical(
                        node_name, self._encoders, self._cat_transformation)
                    col_trans_cat.train_transformator(np.array([]), metadata)
            elif node_meta.is_numerical() and not node_meta.is_categorical():
                col_trans_num = fe_helpers.ColumnTransformatorNumerical(
                    node_name, self._scalers, self._num_transformation)
                if online:
                    col_trans_num.train_transformator_online(
                        input_data[node_name], metadata)
                else:
                    col_trans_num.train_transformator(
                        input_data[node_name], metadata)
            else:
                # We want to ignore string columns.
                if node_name in self._node_list:
                    self._node_list.remove(node_name)
                continue
        return None

    def fit(self, input_data: DictOfArrays, metadata: DictOfMetadata,
            label_data: DictOfArrays, label_metadata: DictOfMetadata) \
            -> "TTestFeatureSelector":
        """Train a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.
            label_data (DictOfArrays): A dictionary with the label node name
                as the only key and the labels as value.
            label_metadata (DictOfMetadata): Dictionary containing metadata of
                the labels.

        Returns:
            IFeatureExtractor: The class itself.
        """
        if self._n_features_fraction is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TTestFeatureSelector object was not constructed "
                "properly. Use the function 'initialize_from_weights' or "
                "'initialize_new' to construct it.")

        self._fit_table_prep_part(input_data, metadata)
        if self._node_list is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Fitting encoders and scalers failed for an unknown reason.")

        # Apply trained encoders and scalers and get purely numerical nodes.
        tr_dataset = self._transform_pre_selection(input_data)
        # Get the names of the new names after preprocessing the table with
        # scaling and encoding.
        new_node_names: List = []
        for subnodes in self._get_node_names_table_prep().values():
            new_node_names += subnodes

        # If the number of remaining nodes (after deleting all str nodes), is
        # less than 5./self._n_features_fraction, we select all nodes.
        if self._node_list is not None \
                and self._n_features_fraction * len(new_node_names) < 5:
            self._n_features_fraction = min(5. / len(new_node_names), 1.0)

        # Retrieve the unique label categories from the label metadata. This
        # will be needed to group the samples according to their classification
        # labels.
        label_meta = list(label_metadata.values())[0]
        label_name = list(label_metadata.keys())[0]
        unique_labels = label_meta.upload_unique_values.get()
        # If this Feature Extractor is used in a regression setting
        # we bin the numerical labels into a fixed number of categories.
        if not label_meta.is_upload_categorical():
            label_data = fe_helpers.categorize_regression_label(
                label_data, N_BINS, float(label_meta.upload_min.get()),
                float(label_meta.upload_max.get()), label_name)
            unique_labels = list(range(N_BINS - 1))
        n_classes = len(unique_labels)

        # Get number of samples per label class.
        n_samples_classwise = fe_helpers.get_n_samples_classwise(
            label_data, unique_labels, label_name)
        n_samples = sum(n_samples_classwise.values())

        # For each node, get the in-class mean (for all classes) and the
        # overall mean.
        inclass_means, overall_means = fe_helpers.get_means(
            new_node_names, unique_labels, tr_dataset, label_data,
            n_samples, n_samples_classwise, label_name)

        # Compute for each node and each class the sum of squares of all
        # samples belonging to this class.
        S_i_squared = fe_helpers.get_sum_of_squares_classwise(
            new_node_names, tr_dataset, label_data, unique_labels,
            inclass_means, n_classes, n_samples, label_name)

        # Compute a rank for each node by maximizing the t-statistics with
        # respect to the different classes. Then select the top k ranks, where
        # k = self._n_features_fraction * len(self._nodes_list).
        self._select_top_k_features(
            new_node_names, inclass_means, overall_means, unique_labels,
            n_samples_classwise, n_samples, S_i_squared)
        return self

    def fit_generator(
            self, input_data: d_obj.DatasetGenerator, metadata: DictOfMetadata,
            label_data: d_obj.DatasetGenerator,
            label_metadata: DictOfMetadata) \
            -> "TTestFeatureSelector":
        """Train the feature extractor with a generator over batches as input.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of data. The keys of the dictionaries are
                the node names.
            metadata (DictOfMetadata): Dictionary containing metadata.
            label_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of labels.
            label_metadata (DictOfMetadata): Dictionary containing metadata of
                the labels.

        Returns:
            The class itself.
        """
        if self._n_features_fraction is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TTestFeatureSelector object was not constructed "
                "properly. Use the function 'initialize_from_weights' or "
                "'initialize_new' to construct it.")

        # Iterate over batches and train encoders and scalers in an online
        # fashion (The encoders will only be trained the first time since they)
        # only use metadata and not data.
        for batch in input_data:
            self._fit_table_prep_part(batch, metadata, online=True)

        # Apply trained encoders and scalers and get purely numerical nodes.
        tr_dataset = self._transform_pre_selection_generator(input_data)

        # Get the names of the new names after preprocessing the table with
        # scaling and encoding.
        new_node_names: List = []
        for subnodes in self._get_node_names_table_prep().values():
            new_node_names += subnodes

        # If the number of remaining nodes (after deleting all str nodes), is
        # less than 5./self._n_features_fraction, we select all nodes.
        if self._node_list is not None \
                and self._n_features_fraction * len(new_node_names) < 5:
            self._n_features_fraction = min(5. / len(new_node_names), 1.0)

        # Retrieve the unique label categories from the label metadata. This
        # will be needed to group the samples according to their classification
        # labels.
        label_meta = list(label_metadata.values())[0]
        label_name = list(label_metadata.keys())[0]
        unique_labels = label_meta.upload_unique_values.get()
        # If this Feature Extractor is used in a regression setting
        # we bin the numerical labels into a fixed number of categories.
        if not label_meta.is_upload_categorical():
            label_data_prep = fe_helpers.categorize_regression_label_generator(
                label_data, N_BINS, float(label_meta.upload_min.get()),
                float(label_meta.upload_max.get()), label_name)
            unique_labels = list(range(N_BINS - 1))
        else:
            label_data_prep = label_data
        n_classes = len(unique_labels)

        # Get number of samples per label class.
        n_samples_classwise = \
            fe_helpers.get_n_samples_classwise_from_generator(
                label_data_prep, unique_labels, label_name)
        n_samples = sum(n_samples_classwise.values())

        # For each node, get the in-class mean (for all classes) and the
        # overall mean. Note in case of regression labels we have to redefine
        # the generator of the categorized labels, so that the generator is
        # reset.
        if not label_meta.is_categorical():
            label_data_prep = fe_helpers.categorize_regression_label_generator(
                label_data, N_BINS, float(label_meta.upload_min.get()),
                float(label_meta.upload_max.get()), label_name)
        inclass_means, overall_means = fe_helpers.get_means_from_generator(
            new_node_names, unique_labels, tr_dataset, label_data_prep,
            n_samples, n_samples_classwise, label_name)

        # Compute for each node and each class the sum of squares of all
        # samples belonging to this class. Note that we have to redefine
        # transformed data generator (otherwise generator is not reset). The
        # same for the generator of the categorized labels.
        tr_dataset = self._transform_pre_selection_generator(input_data)
        if not label_meta.is_categorical():
            label_data_prep = fe_helpers.categorize_regression_label_generator(
                label_data, N_BINS, float(label_meta.upload_min.get()),
                float(label_meta.upload_max.get()), label_name)

        S_i_squared = fe_helpers.get_sum_of_squares_classwise_from_generator(
            new_node_names, tr_dataset, label_data_prep, unique_labels,
            inclass_means, n_classes, n_samples, label_name)

        # Compute a rank for each node by maximizing the t-statistics with
        # respect to the different classes. Then select the top k ranks, where
        # k = self._n_features_fraction * len(self._nodes_list).
        self._select_top_k_features(
            new_node_names, inclass_means, overall_means, unique_labels,
            n_samples_classwise, n_samples, S_i_squared)
        return self

    def fit_transform(
            self, input_data: DictOfArrays, metadata: DictOfMetadata,
            label_data: DictOfArrays, label_metadata: DictOfMetadata) \
            -> DictOfArrays:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            metadata (DictOfMetadata): Dictionary containing metadata.
            label_data (DictOfArrays): A dictionary with the label node name
                as the only key and the labels as value.
            label_metadata (DictOfMetadata): Dictionary containing metadata of
                the labels.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        self.fit(input_data, metadata, label_data, label_metadata)
        return self.transform(input_data)

    def fit_transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            metadata: DictOfMetadata,
            label_data: d_obj.DatasetGenerator,
            label_metadata: DictOfMetadata) \
            -> GenOfDicts:
        """Train and apply a feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                that returns batches of data. The keys of the dictionaries are
                the node names.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            GenOfDicts: Transformed samples, batched as a
            generator.
        """
        self.fit_generator(input_data, metadata, label_data, label_metadata)
        return self.transform_generator(input_data)

    def _transform_pre_selection_generator(
            self, input_generator: d_obj.DatasetGenerator,
            check_input: bool = False) -> GenOfDicts:
        """Apply only table prep part to the dataset, to get purely numerical
        nodes.

        Args:
            input_generator (d_obj.DatasetGenerator): A dictionary with node
                names as keys and node data for all samples as values.
            check_input (bool): Whether or not to check the input data.

        Returns:
            GenOfDicts: Transformed samples, all at once in a
                dictionary of lists.

        Raises:
            fe_helpers.TTEstFeatureSelectionError: If sample ids are not
                in the batches.
        """
        for batch in input_generator:
            if dc.SAMPLE_ID_KEYWORD not in batch:
                raise fe_helpers.TTEstFeatureSelectionError(
                    "Input generator must be a generator over dictionaries, "
                    "where every dictionary must contain the sample ids with "
                    f"the key {dc.SAMPLE_ID_KEYWORD}.")
            batch_copy = copy.deepcopy(batch)
            sample_ids = batch_copy.pop(dc.SAMPLE_ID_KEYWORD)
            output_batch = self._transform_pre_selection(
                batch_copy, check_input=check_input)
            output_batch[dc.SAMPLE_ID_KEYWORD] = sample_ids
            yield output_batch

    def _transform_pre_selection(
            self, input_data: DictOfArrays, check_input: bool = False) \
            -> DictOfArrays:
        """Apply only table prep part to the dataset, to get purely numerical
        nodes.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            check_input (bool): Whether to check the input data.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        if self._num_transformation is None \
                or self._cat_transformation is None \
                or self._node_list is None or self._metadata is None \
                or self._encoders is None or self._scalers is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TablePrepFeatureExtractor object has not been trained yet.")

        # If input check flag is True, we perform checks and raise user
        # friendly exceptions, if the input type is wrong.
        if check_input:
            fe_helpers.check_data_types(input_data)

        # Loop over all nodes and transform that according to their data
        # type.
        output_data: DictOfArrays = {}
        for node_name in self._node_list:
            node_data = input_data[node_name]
            node_metadata = self._metadata[node_name]
            column_transformator: fe_helpers.ColumnTransformator
            if node_metadata.is_categorical():
                column_transformator = \
                    fe_helpers.ColumnTransformatorCategorical(
                        node_name, self._encoders,
                        self._cat_transformation)
            elif node_metadata.is_numerical() and not \
                    node_metadata.is_categorical():
                column_transformator = \
                    fe_helpers.ColumnTransformatorNumerical(
                        node_name, self._scalers, self._num_transformation)
            else:
                continue

            # Apply transformator.
            new_col = column_transformator.apply_trained_transformator(
                node_data)
            new_subnode_names = column_transformator.get_new_node_names()
            if len(new_subnode_names) > 1:
                for new_node_index in range(len(new_subnode_names)):
                    subnode_name = new_subnode_names[new_node_index]
                    output_data[subnode_name] = \
                        np.array([new_col[i][new_node_index]
                                  for i in range(len(new_col))])
            else:
                output_data[new_subnode_names[0]] = new_col
        return output_data

    def transform(self, input_data: DictOfArrays, check_input: bool = False) \
            -> DictOfArrays:
        """Apply a trained feature extractor with a list of samples as input
        and output.

        Args:
            input_data (DictOfArrays): A dictionary with node names
                as keys and node data for all samples as values.
            check_input (bool): Whether or not to check whether input data
                type/format.

        Returns:
            DictOfArrays: Transformed samples, all at once in a dictionary of
                lists.
        """
        if self._selected_nodes is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TablePrepFeatureExtractor object has not been trained yet.")
        self._compute_transformed_metadata()

        transformed_dataset = self._transform_pre_selection(
            input_data, check_input=check_input)
        new_node_names = list(transformed_dataset.keys())
        for nn in new_node_names:
            if nn not in self._selected_nodes:
                del transformed_dataset[nn]
        return transformed_dataset

    def _get_node_names_table_prep(self) -> Dict[str, List]:
        if self._node_list is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Internal error in TTestFeatureSelection class. Node list "
                "is not set.")
        if self._encoders is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Internal error in TTestFeatureSelection class. Encoders dict "
                "is not set.")
        if self._scalers is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Internal error in TTestFeatureSelection class. Scalers dict "
                "is not set.")
        new_node_names_dict = {}
        for n in self._node_list:
            if n in self._encoders and isinstance(self._encoders[n],
                                                  ec.OneHotEncoder):
                new_node_names = [n + f"_{i}" for i in range(
                    self._encoders[n].get_n_unique_categories())]
            else:
                new_node_names = [n]
            new_node_names_dict[n] = new_node_names
        return new_node_names_dict

    def _compute_transformed_metadata(self) -> None:
        """Compute transformed metadata and save it in member variable.

        Raises:
            fe_helpers.TTEstFeatureSelectionError: Error if object has not been
                initialized properly.
        """
        if self._node_list is None or self._scalers is None \
                or self._encoders is None or self._metadata is None \
                or self._selected_nodes is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TablePrepFeatureExtractor object has not been trained yet.")
        if self._transformed_metadata is not None:
            return None
        self._transformed_metadata = {}
        new_node_names_dict = self._get_node_names_table_prep()
        for old_node_name, new_node_names in new_node_names_dict.items():
            new_node_meta = meta_trans.NodeTransferer.from_obj(
                    self._metadata[old_node_name]).get_obj()
            new_node_meta.node_type.set("num")
            for nn in new_node_names:
                self._transformed_metadata[nn] = new_node_meta
        for nnn in list(self._transformed_metadata.keys()).copy():
            if nnn not in self._selected_nodes:
                del self._transformed_metadata[nnn]
        return None

    def transform_generator(
            self, input_data: d_obj.DatasetGenerator,
            check_input: bool = False) -> GenOfDicts:
        """Apply a trained feature extractor with a list of samples as input
        and output.

        Args:
            input_data (d_obj.DatasetGenerator): A generator of dictionaries
                where the keys are the node names and the values are the
                batched node data as lists.
            metadata (DictOfMetadata): Dictionary containing metadata.

        Returns:
            GenOfDicts: Transformed samples, batched as a
            generator.
        """
        if self._selected_nodes is None:
            raise fe_helpers.TTEstFeatureSelectionError(
                "TablePrepFeatureExtractor object has not been trained yet.")
        self._compute_transformed_metadata()
        nodes_transformed_all = self._transform_pre_selection_generator(
            input_data, check_input=check_input)
        return fe_helpers.slice_nodes_from_generator(
            nodes_transformed_all, self._selected_nodes)

    def get_transformed_metadata(self) -> DictOfMetadata:
        """Get transformed metadata after training the Feature Extractor.

        Returns:
            DictOfMetadata: Transformed metadata.

        Raises:
            fe_helpers.TTEstFeatureSelectionError: Error, if transformed
                metadata has not been computed yet.
        """
        if not self._transformed_metadata:
            raise fe_helpers.TTEstFeatureSelectionError(
                "Metadata of transformed data can not be retrieved because "
                "this feature extractor has not been run yet."
                )
        return self._transformed_metadata
