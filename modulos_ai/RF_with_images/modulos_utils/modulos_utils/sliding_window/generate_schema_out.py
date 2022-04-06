# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Compute the schema out of the node generation of window sliding."""
import copy
from typing import Dict, List

from modulos_utils.sliding_window import utils as sw_utils


def compute_output_schema(
        ds_schema_input: Dict,
        feature_time_points: Dict[str, List[int]],
        label_time_points: List[int]) -> Dict:
    """This function computes the output schema given the input dataset schema
    and a specific  config choice.
    Args:
        ds_schema_input (Dict): Schema of dataset input selection.
        feature_time_points (Dict[str, List[int]]): This argument specifies
            the generated nodes for each original node in the dataset. It
            is a dictionary with the node names as keys and integer lists
            as values. The integers specify the time shift relative to
            time step `t`.
            Example: feature_time_points = {"temperature": [-5, -1, 0, 3]}
            means that this function outputs the following nodes:
            `temperature_t-5`, `temperature_t-1`, `temperature_t`,
            `temperature_t+3`.
        label_time_points (List[int]): Time points of the generated nodes
            that belong to the label.

    Return:
        Dict: Output schema of node generation of window sliding.
    """
    # Raise an error, if the resulting number of samples is smaller than the
    # global minimum of the platform.
    # Reduce number of samples due to shifting.
    nr_samples_original = ds_schema_input["nr_samples"]
    if isinstance(nr_samples_original, int):
        nr_samples_original_int = nr_samples_original
    elif isinstance(nr_samples_original, dict):
        if nr_samples_original["min"] == "inf" \
                or nr_samples_original["max"] == "inf" \
                or nr_samples_original["min"] != nr_samples_original["max"]:
            raise ValueError(
                "The number of samples need to be finite and fixed, i.e. "
                "min = max != inf.")
        nr_samples_original_int = nr_samples_original["min"]
    else:
        raise ValueError(
            "The number of samples need to be finite and fixed, i.e. "
            "min = max != inf.")
    nr_samples_out, _ = sw_utils.compute_new_nr_samples(
        nr_samples_original_int, feature_time_points)

    schema_out = copy.deepcopy(ds_schema_input)
    # Add new nodes to schema. Note that we add the t+1, t+2 etc. nodes as
    # individual nodes here. We will merge them into one node at the end of
    # this function.
    for node_name, time_points in feature_time_points.items():

        # We delete the original node and replace it by the generated nodes.
        node_dict = schema_out["nodes"].pop(node_name)
        nr_instances = node_dict["nr_instances"]
        nr_inst_not_one: bool = (
            # Either nr_instances is a int != 1,
            (isinstance(nr_instances, int) and nr_instances != 1)
            # or nr_instances is a string (which would be "arbitrary"),
            or isinstance(nr_instances, str)
            # or nr_instances is a dict with min != 1 or max max != 1.
            or (isinstance(nr_instances, dict)
                and (nr_instances["min"] != 1 or nr_instances["max"] != 1))
        )
        if nr_inst_not_one:
            raise ValueError("The nr_instances of all nodes must be 1!")

        for time_point in time_points:
            new_node = copy.deepcopy(node_dict)
            new_node_name = sw_utils.get_generated_node_name(
                node_name, time_point)
            schema_out["nodes"][new_node_name] = new_node
    schema_out["nr_samples"] = nr_samples_out

    # If the forecast horizon is bigger than 1, the labels are gathered
    # into one multidimensional target node.
    mimo_label_components = sw_utils.get_mimo_label_components(
        feature_time_points, sorted(label_time_points)[0])
    if mimo_label_components != []:
        node_copy = copy.deepcopy(
            schema_out["nodes"][mimo_label_components[0]])
        for sub_node in mimo_label_components:
            schema_out["nodes"].pop(sub_node)
        mimo_label_node_name = sw_utils.get_mimo_vector_node_name(
            sw_utils.get_original_node_name(sub_node))
        # The dimension changes from [1] to [H, 1] for a single value with H
        # the forecast horizon or from [100, 100, 3] to [H, 100, 100, 3] for an
        # image.
        node_copy["dim"].insert(0, len(mimo_label_components))
        schema_out["nodes"][mimo_label_node_name] = node_copy

    # TODO: Data range could possibly change, as some samples are removed.
    # Investigate if that is a problem in REB-415.
    return schema_out
