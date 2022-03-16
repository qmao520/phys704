# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# THIS FILE IS DUPLICATED IN ALL TABLE PREP FEATURE EXTRACTORS and the
# T-TEST FEATURE EXTRACTORS. SO IF YOU CHANGE SOMETHING HERE, YOU HAVE TO COPY
# PASTE TO EVERY MEMBER OF THE TABLE PREP FAMILY!
"""This file contains classes to scale numerical columns of tables.
"""
from enum import Enum, unique
import sklearn.preprocessing as skprep
import numpy as np
from typing import Any


@unique
class NumberScalingTypes(Enum):
    """Enum class for different encoder classes for categories.
    """
    STANDARDSCALING = "standard_scaling"


class StandardScaler:
    """
    Class that performs standard scaling of numerical vectors/scalars.
    """

    def __init__(self) -> None:
        """
        Initialize class.
        """
        self._scaler = skprep.StandardScaler()

    def __eq__(self, other: Any) -> bool:
        """Implement equality operator.

        Args:
            other (Any): Any object to compare with.

        Returns:
            bool: True if the object is equal to the instance of this class.
        """
        mean_equal = False
        scale_equal = False
        if not isinstance(other, StandardScaler):
            return False
        if not hasattr(self._scaler, "mean_") and not \
                hasattr(other._scaler, "mean_"):
            mean_equal = True
        if hasattr(self._scaler, "mean_") and hasattr(other._scaler, "mean_"):
            mean_equal = other._scaler.mean_ == self._scaler.mean_
        if not hasattr(self._scaler, "scale_") and not \
                hasattr(other._scaler, "scale_"):
            scale_equal = True
        if hasattr(self._scaler, "scale_") and \
                hasattr(other._scaler, "scale_"):
            scale_equal = other._scaler.scale_ == self._scaler.scale_
        return mean_equal and scale_equal

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit a standard scaler to a set of data.

        Args:
            X (np.ndarray): Input list.

        Returns:
            StandardScaler: Class itself.
        """
        self._scaler.fit(X.astype(float))
        return self

    def partial_fit(self, X: np.ndarray) \
            -> "StandardScaler":
        """Partially fit a standard scaler to a set of data. (Used for online
        training.)

        Args:
            X (np.ndarray): Input list.

        Returns:
            StandardScaler: Class itself.
        """
        self._scaler.partial_fit(X.astype(float))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standard scaling to a list of input values.

        Args:
            X (np.ndarray): Input list.
        """
        result_exp = self._scaler.transform(X.astype(float))
        return np.reshape(result_exp, -1)


NumberScalingPicker = {
    NumberScalingTypes.STANDARDSCALING: StandardScaler,
}
