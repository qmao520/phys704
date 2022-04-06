# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the functions to convert or edit timestamps."""
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import logging
import warnings

import numpy as np
import pandas as pd

from modulos_utils.datetime_utils import datetime_guesser as dt_guesser
from modulos_utils.dshf_handler import dshf_handler
from modulos_utils.dssf_and_structure import DSSFErrors
from modulos_utils.metadata_handling import metadata_properties as meta_prop


TIME_FEATURES_CAT: Dict[str, List[int]] = {
    "year": [], "day": list(range(1, 32)), "hour": list(range(24)),
    "month": list(range(1, 13)), "weekday": list(range(7)), "unixtime": []}
TIME_FEATURES_VALUES = {
    "year": lambda dt: dt.year, "day": lambda dt: dt.day,
    "hour": lambda dt: dt.hour, "month": lambda dt: dt.month,
    "weekday": lambda dt: dt.weekday,
    "unixtime": lambda dt: dt.view(np.int64) / 10**9}


class DATETIME_FORMAT(Enum):
    DAYFIRST = "dayfirst"
    MONTHFIRST = "monthfirst"
    FAILED = "failed"
    UNKNOWN = "unknown"


def get_possible_datetime_features(
        node_name: str, node_generator: Generator[np.ndarray, None, None],
        metadata_obj: meta_prop.AllProperties,
        dshf: Optional[dshf_handler.DSHFHandler] = None) -> List[str]:
    """Check if a node can be used to generate datetime features. If yes,
    return the list of them, else return an empty list.

    Args:
        node_name (str): Name of the node.
        metadata_obj (meta_prop.AllProperties): Metadata object of the
            node in question.
        node_generator (Generator[np.ndarray, None, None]): Generator of the
            actual data of the node. The generator iterates over numpy arrays.
        metadata_obj (meta_prob.AllProperties): Metadata object to check if the
            type is correct.
        dshf (dshf_handler.DSHFHandler): Dataset history object to get the
            datetime format if it is already known. Defaults to None.

    Returns:
        List[str]: List of the possible datetime features.
    """
    # Check first if it is numerical, if yes return [].\
    if metadata_obj.is_numerical():
        return []
    # Check first if the dshf is given and the format for the node is saved in
    # it. If it is correct, we can already finish early.
    if dshf and node_name in dshf.datetime_format:
        dt_format = dshf.datetime_format[node_name]
        for batch in node_generator:
            df = pd.DataFrame({node_name: batch.flatten()})
            try:
                df[node_name] = pd.to_datetime(df[node_name], format=dt_format)
            except ValueError:
                msg = (f"The given datetime format string (`{dt_format}`) is "
                       "not correct.")
                _, format = check_and_infer_format(df, node_name)
                if format != "":
                    msg += f" The format `{format}` would be valid."
                raise ValueError(msg)
        return get_datetime_node_names(node_name)

    # Iterate over all batches and break out of the loop as soon as a
    # batch cannot be parsed by pandas' datetime parsers.
    for index, batch in enumerate(node_generator):
        if index == 0:
            # Check first three elements, so that we are able to fail fast in
            # case of large batch sizes.
            first_values = batch[:3]
            df_first_values = pd.DataFrame({"dt": first_values.flatten()})
            try:
                parse_datetime_in_dataframe_column(df_first_values, "dt")
            except DSSFErrors.DSSFDateTimeNotReadable:
                return []
        try:
            df = pd.DataFrame({"dt": batch.flatten()})
            parse_datetime_in_dataframe_column(df, "dt")
        except DSSFErrors.DSSFDateTimeNotReadable:
            return []
    return get_datetime_node_names(node_name)


def parse_datetime_in_dataframe_column(
        df: pd.DataFrame, column_name: Union[int, str]) -> str:
    """Parse a column of a pandas.DataFrame to datetime and replace it with it.
    If the function fails, the thrown exception has to be caught by the
    function caller.

    Args:
        df (pd.DataFrame): DataFrame with a parsable datetime strings.
        column_name (Union[int, str]): Name or index of the column to parse.

    Raises:
        DSSFErrors.DSSFDateTimeNotReadable: Raised if the parsed date is out
            of range or the format is not known.

    Returns:
        str: The datetime format string.

    """
    try:
        if df[column_name].dtype != np.number:
            # Check if the datetime format is consistent with dayfirst True or
            # False and set it. See also
            # https://github.com/pandas-dev/pandas/issues/12585
            dayfirst, inferred_format = check_and_infer_format(
                df, column_name)
            # We set dayfirst to True, if it the resulting parsing is
            # consistent over the full list. Else we set it to false as this is
            # the default of pandas.
            is_dayfirst = (
                dayfirst.value == DATETIME_FORMAT.DAYFIRST.value)
        else:
            # If the datetime is a number (unixtime), the dayfirst argument
            # does not matter.
            inferred_format = ""
            is_dayfirst = True
        if inferred_format != "":
            df[column_name] = pd.to_datetime(
                df[column_name], format=inferred_format)
        else:
            df[column_name] = pd.to_datetime(
                df[column_name], infer_datetime_format=True,
                dayfirst=is_dayfirst)
    except Exception as err:
        raise DSSFErrors.DSSFDateTimeNotReadable(
            f"Could not parse the datetime column! {err}")
    return inferred_format


def check_and_infer_format(
        df: pd.DataFrame, column_name: Union[int, str]) -> \
            Tuple[DATETIME_FORMAT, str]:
    """Check all strings of the datetime column and infer whether using
    dayfirst equal True or False results in consistent parsing.

    For the case ["12-05-2020", "13-05-2020"], the guessed format for the first
    would be '%m/%d/%Y', the second '%d/%m/%Y' using `dayfirst=False`. If you
    use `dayfirst=True`, both are evaluated to '%d/%m/%Y'.
    Therefore this function will check if all the guessed formats are the same,
    if not it will try again with the keyword `dayfirst=True`.

    Args:
        df (pd.DataFrame): The dataframe to test.
        column_name (Union[int, str]): The name of the column to test.

    Returns:
        Tuple[DATETIME_FORMAT, str]: Returns `dayfirst` and the inferred format
            if using dayfirst=True results in consistent format.
            `monthfirst` and the format if dayfirst=False is consistent.
            If neither dayfirst True or False resulted in consistent format,
            then `ambiguous` and an empty string is returned.
            If there were errors while using
            the parsing function, we return `undetermined`.
    """
    format_guesser = dt_guesser.guess_datetime_format
    try:
        series = df[column_name]
        # Check first if using dayfirst = False results in a consistent
        # format.
        is_consistent, format = _is_datetime_format_consistent(
            format_guesser, series, dayfirst=False)
        if is_consistent:
            return DATETIME_FORMAT.MONTHFIRST, format

        # Check then if using dayfirst = True results in a consistent
        # format.
        is_consistent, format = _is_datetime_format_consistent(
            format_guesser, series, dayfirst=True)
        if is_consistent:
            return DATETIME_FORMAT.DAYFIRST, format

        # In case that none of the cases above result in a consistent datetime
        #  format, we log a warning and return "unknown".
        logging.warning(f"The datetime format of column `{column_name}` is not"
                        " consistent across all the samples.")
    except Exception as err:
        msg = ("There were issues in trying to guess the format of the "
               "datetime string: ")
        warnings.warn(msg + str(err))
        return DATETIME_FORMAT.FAILED, ""
    return DATETIME_FORMAT.UNKNOWN, ""


def _is_datetime_format_consistent(
        format_guesser: Callable, series: pd.Series, dayfirst: bool) -> \
            Tuple[bool, str]:
    """Check if the datetime format in a pandas series is consistent.

    Args:
        format_guesser (Callable): Function to return the format of a datetime
            string.
        series (pd.Series): Pandas series to check.
        dayfirst (bool): If dayfirst is assumed as bias in guessing the format.

    Raises:
        ValueError: Raised if no format was found for the first value.

    Returns:
        Tuple[bool, str]: True and the formatstring if it is consistent, false
        and an empty string otherwise.
    """
    first_format = format_guesser(series[0], dayfirst=dayfirst)
    if first_format is None:
        raise ValueError(
            "The datetime format could not be read by the parser.")
    for val in series[1:]:
        format = format_guesser(val, dayfirst=dayfirst)
        if format != first_format:
            return False, ""
    return True, first_format


def compute_features_from_timestamp(
        timestamps: pd.DatetimeIndex) -> Dict[str, List[int]]:
    """Compute the features defined in TIME_FEATURES_VALUES.

    Args:
        timestamps (pd.DatetimeIndex): The timestamps.

    Returns:
        Dict[str, List[int]]: Newly created features.
    """
    features: Dict[str, List[int]] = {}
    for feature, get_values in TIME_FEATURES_VALUES.items():
        features[feature] = list(get_values(timestamps))
    return features


def get_datetime_node_names(datetime_name: str) -> List[str]:
    """Return a list with all the names of the generated subfeature
    of the datetime node.

    Args:
        datetime_name (str): Name of the datetime node.

    Returns:
        List[str]: List of the names of the subfeatures of the datetime node.
    """
    return [get_datetime_node_name_for_feature(
        datetime_name, f) for f in TIME_FEATURES_VALUES]


def get_datetime_node_name_for_feature(datetime_name: str,
                                       feature: str) -> str:
    """Return a string with the name of the generated subfeature
    of the datetime node.

    Args:
        datetime_name (str): Name of the datetime node.
        feature (str): Name of the feature.

    Returns:
        str: Name of the subfeature of the datetime node.
    """
    return f"{datetime_name}__{feature}"


def split_datetime_feature_name(datetime_feature: str) -> List[str]:
    """Return the original datetime node name as well as the feature name.

    Args:
        datetime_feature (str): Name of the datetime feature node.

    Returns:
        List[str]: [Original node name, feature name]
    """
    split = datetime_feature.split("__")
    original_node = "".join(split[:-1])
    feature = split[-1]
    if feature in TIME_FEATURES_CAT:
        return [original_node, feature]
    else:
        raise Exception("The given node is not a datetime feature!")
