# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
# THIS FILE IS DUPLICATED IN ALL TABLE PREP FEATURE EXTRACTORS AND
# IMG_IDENTITY_CAT_INTENC. SO IF YOU CHANGE SOMETHING HERE, YOU HAVE TO COPY
# PASTE TO ALL THESE FILES!
"""This file contains classes to encode categorical values.
"""

from enum import Enum, unique
import numpy as np
from typing import Dict, Type, Any, Optional
from abc import ABC, abstractmethod


@unique
class CategoryEncoderTypes(Enum):
    """Enum class for different encoder classes for categories.
    """
    ONEHOTENCODING = "onehot_encoding"
    INTEGERENCODING = "integer_encoding"


class CategoryEncoderTypeError(Exception):
    """
    Exception for when encoder type is not defined in the CategoryEncoderType
    Enum.
    """
    pass


class IntegerEncoderError(Exception):
    """
    General errors in the IntegerEncoding class.
    """
    pass


class OneHotEncoderError(Exception):
    """
    Errors in the Onehot Encoding Class.
    """
    pass


class Encoder(ABC):
    """Base class for encoders.
    """

    @abstractmethod
    def __init__(self):
        """Initialize class.
        """
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Implement equality operator.

        Args:
            other (Any): Any object to compare with.

        Returns:
            bool: True if the object is equal to the instance of this class.
        """
        pass

    def fit(self, X) -> "Encoder":
        """Generic fit method. Mypy will check that the input type matches one
        of the overloads. Note that due to the overloads, this function must
        not have a type hint for its arguments.
        """
        return self._fit(X.astype(str))

    @abstractmethod
    def _fit(self, X: np.ndarray) -> "Encoder":
        """Private fit function that accepts only strings.
        """
        pass

    def transform(self, X) -> np.ndarray:
        """Generic transform method. Mypy will check that the input type
        matches one of the overloads. Note that due to the overloads, this
        function must not have a type hint for its arguments.
        """
        X_strings = X.astype(str)
        return self._transform(X_strings)

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Private transform function that accepts only strings.

        Args:
            X (EncoderInput): Input data as a list of strings.

        Returns:
            np.ndarray: Encoded values as a list of ints.
        """
        pass

    @abstractmethod
    def get_n_unique_categories(self) -> int:
        """Get number of unique categories.

        Returns:
            int: Number of categories.
        """
        pass


class IntegerEncoder(Encoder):
    """
    Class that performs integer encoding on categories, using only
    standard python libraries.
    """

    def __init__(self) -> None:
        """Initialize class.
        """
        self._lookup: dict = {}
        self._n_categories: Optional[int] = None

    def __eq__(self, other: Any) -> bool:
        """Implement equality operator.

        Args:
            other (Any): Any object to compare with.

        Returns:
            bool: True if the object is equal to the instance of this class.
        """
        if not isinstance(other, IntegerEncoder):
            return False
        if other._lookup != self._lookup:
            return False
        if other._n_categories != self._n_categories:
            return False
        return True

    def _fit(self, X: np.ndarray) -> "IntegerEncoder":
        """Fit integer encoder to a list of values.

        Args:
            X (np.ndarray): Input data as a list of strings.

        Returns:
            IntegerEncoder: The class itself.
        """
        unique_sorted = np.sort(np.unique(X.reshape(-1).astype(str)))
        n_categories = len(unique_sorted)

        # Create look up (as a dictionary).
        self._lookup = dict(
            zip(unique_sorted, list(range(n_categories)))
            )
        self._n_categories = n_categories
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform an input vector by integer encoding.

        Args:
            X: Input data. Mypy checks that the type matches with one of the
                overloads defined in the base class.

        Returns:
            np.ndarray: Integer encoded values as a list.
        """
        if self._lookup == {}:
            raise IntegerEncoderError("Transform can only be called after "
                                      "training.")
        return np.array([self._lookup[s] for s in X.reshape(-1).astype(str)])

    def get_n_unique_categories(self) -> int:
        """Get number of unique categories.

        Returns:
            int: Number of categories.
        """
        if self._n_categories is None:
            raise IntegerEncoderError(
                "get_number_of_unique_categories can only be called after "
                "training.")
        return self._n_categories


class OneHotEncoder(Encoder):
    """
    Class that performs onehot encoding on categories using only
    standard python libraries.
    """

    def __init__(self) -> None:
        """Initialize class.
        """
        self._lookup: dict = {}
        self._n_categories: Optional[int] = None

    def __eq__(self, other: Any) -> bool:
        """Implement equality operator.

        Args:
            other (Any): Any object to compare with.

        Returns:
            bool: True if the object is equal to the instance of this class.
        """
        if not isinstance(other, OneHotEncoder):
            return False
        if other._lookup != self._lookup:
            return False
        if other._n_categories != self._n_categories:
            return False
        return True

    def _fit(self, X: np.ndarray) -> "OneHotEncoder":
        """Fit onehot encoder to a list of values.

        Args:
            X (np.ndarray): Input data as a list of strings.

        Returns:
            IntegerEncoder: The class itself.
        """
        unique_sorted = np.sort(np.unique(X.reshape(-1).astype(str)))
        n_categories = len(unique_sorted)

        # Create look up (as a dictionary).
        self._lookup = dict(zip(unique_sorted,
                                np.identity(n_categories)))
        self._n_categories = n_categories
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform an input vector by onehot encoding.

        Args:
            X: Input data. Mypy checks that the type matches with one of the
                overloads defined in the base class.

        Returns:
            np.ndarray: Onehot encoded values as a list.
        """
        if self._lookup == {}:
            raise OneHotEncoderError("Transform can only be called after "
                                     "training.")
        return np.array([self._lookup[s] for s in X.reshape(-1).astype(str)])

    def get_n_unique_categories(self) -> int:
        """Get number of unique categories.

        Returns:
            int: Number of categories.
        """
        if self._n_categories is None:
            raise IntegerEncoderError(
                "get_number_of_unique_categories can only be called after "
                "training.")
        return self._n_categories


CategoryEncoderPicker: Dict[CategoryEncoderTypes, Type[Encoder]] = {
    CategoryEncoderTypes.ONEHOTENCODING: OneHotEncoder,
    CategoryEncoderTypes.INTEGERENCODING: IntegerEncoder
}
