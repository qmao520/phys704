# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""
Collection of metadata properties. Summarized in AllProperties.
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

ID_PLACEHOLDER = "{id}"


def is_nan(value: Any) -> bool:
    """ Test if input is NaN or not.

    Args:
        value (Any): input value

    Returns:
        bool: input is NaN yes/no
    """
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


class PropertyBaseClass(ABC):
    """ Base class for all metadata properties.

    Attributes:
        _default: Default value of the property, set to NaN in the base class
            implementation.
        description: A property's description.
        is_computable: Flag. Is this property computable yes or no. Has an
            impact on the metadata transferer.
        key: Unique key.
        _value: Value of the property, set to NaN in base class implementation.
        is_basic: Flag. Is this a basic property, i.e. one which determines
            the basic properties of a node? Has an impact on the metadata
            computation.
    """
    # Define key as a class variable to be able to retrieve it without
    # initializing.
    key = ""

    def __init__(self, description="", is_computable=False, is_basic=False,
                 _default: Union[float, int, str, List] = float("NaN")):
        """ Init for the Property Base Class.
        """
        self.description = description
        self.is_computable = is_computable
        self.is_basic = is_basic
        self._default = _default
        self._value = self._default

    def __eq__(self, other) -> bool:
        """ Overwrite equal to be able to compare two Property objects.

        Args:
            other (PropertyBaseClass): second PropertyBaseClass object.

        Returns:
            bool: equal yes/no
        """
        # Protect against other being a different class.
        if self.__class__ != other.__class__:
            return False
        is_equal = [
            self.description == other.description,
            self.is_computable == other.is_computable,
            self.is_basic == other.is_basic,
            self.key == other.key,
            self.is_same_default(other._default),
            self.is_same_value(other.get()),
        ]
        if all(is_equal):
            return True
        return False

    def _check_array_input(self, arr: np.ndarray) -> None:
        """ Check for the 'set_from_array' method. Ensure that the input is
        indeed an array and that it has at least one entry.

        Args:
            arr (array): input array

        Returns:
            None

        Raises:
            TypeError: if input is not an array
            ValueError: if input array is empty
        """

        if not isinstance(arr, np.ndarray):
            raise TypeError("Expected input to be an array, and not "
                            f"{type(arr)}."
                            )
        if arr.size == 0:
            raise ValueError("Expected input array to have at least one "
                             f"entry. Array contains {arr.size} entries.")
        return None

    def is_default(self) -> bool:
        """ Check if value is equal to default.

        Returns:
            bool: value is equal to default yes/no
        """
        return self.is_same_default(self._value)

    def is_same_default(self, new_value: Any) -> bool:
        """ Check if new value is equal to the current default.

        Args:
            new_value (Any): new value

        Returns:
            bool: new value is equal to current default
        """
        if is_nan(self._default) and is_nan(new_value):
            return True
        return self._default == new_value

    def is_same_value(self, new_value: Any) -> bool:
        """ Check if new value is equal to current value.

        Args:
            new_value (Any): new value

        Returns:
            bool: new value is equal to current value
        """
        if is_nan(self._value) and is_nan(new_value):
            return True
        return self._value == new_value

    def get(self) -> Any:
        """ Return current value.

        Returns:
            float, List, str: return current value
        """
        return self._value

    def set(self, new_value: Any) -> None:
        """ Set new value. Only allowed if new value has same type as
        current value. If the current value is NaN, the new value can have a
        different type.

        Args:
            new_value (Any): new value

        Returns:
            None

        Raises:
            TypeError: if new value does not have the same type as the current
            value
        """
        if not is_nan(self._value) and \
                not isinstance(new_value, type(self._value)):
            raise TypeError(f"The type of the new value ({type(new_value)}) "
                            "does not match the type of the "
                            f"old value ({type(self._value)})."
                            )
        self._value = new_value
        return None

    @abstractmethod
    def export_to_array(self) -> np.ndarray:
        """ Return current value as an array. This method is needed to store
        metadata in the internal dataset format.

        Returns:
            array: array containing current value.
        """
        pass

    @abstractmethod
    def import_from_array(self, arr: np.ndarray) -> None:
        """ Set new value using an array as input. This method is needed to
        retrieve values from the internal dataset format.

        Args:
            arr (np.ndarray): array to retrieve value from

        Returns:
            None
        """
        pass


class MetaDataVersionProperty(PropertyBaseClass):
    """
    Property Class for the metadata version.
    """
    key = "_version"

    def __init__(self):
        """ Init for the Version Property Class.
        """
        super().__init__(
            description=("This is the current version of the metadata ",
                         "property implementation."), _default=0.1)

    def _set(self, new_value: float) -> None:
        """ Private set function for the metadata version.

        Args:
            new_value (float): new version number

        Returns:
            None
        """
        if not isinstance(new_value, float):
            raise TypeError("The type of the metadata version is float, not"
                            f" {type(new_value)}.")
        self._value = new_value

    def set(self, new_value: Any) -> None:
        """ Overwrite set function in parent class. The metadata is read only
        and cannot be changed by the user. The new value will be ignored.

        Args:
            new_value (Any): new value

        Returns:
            None
        """
        warnings.warn("The metadata version cannot be set. New value will "
                      "be ignored.")
        pass

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self._set(float(arr[0]))
        return None


class NumberSamplesProperty(PropertyBaseClass):
    """
    Property Class for a dataset's number of samples.
    """
    key = "nr_samples"

    def __init__(self):
        """ Init for the Number of Samples Property Class.
        """
        super().__init__(
            description="This is the number of samples in this dataset.",
            is_computable=True, is_basic=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(int(arr[0]))
        return None


class DSSFComponentNameProperty(PropertyBaseClass):
    """
    Property Class for a node's original DSSF component name.
    """
    key = "dssf_component_name"

    def __init__(self):
        """ Init for the Component Name Property Class.
        """
        super().__init__(
            description="This is the node's original DSSF component name.",
            _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class UploadNodeNameProperty(PropertyBaseClass):
    """
    Property Class for a node's original upload node name.
    """
    key = "upload_node_name"

    def __init__(self):
        """ Init for the Node Name Property Class.
        """
        super().__init__(
            description="This is the node's original upload node name.",
            _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class DSSFComponentTypeProperty(PropertyBaseClass):
    """
    Property Class for a node's original DSSF component type.
    """
    key = "dssf_component_type"

    def __init__(self):
        """ Init for the Component Type Property Class.
        """
        super().__init__(
            description="This is the node's original DSSF component type.",
            _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class DSSFComponentFilePathProperty(PropertyBaseClass):
    """
    Property Class for a node's original file path as defined in the DSSF.
    """
    key = "dssf_component_file_path"

    def __init__(self):
        """ Init for the Component File Path Property Class.
        """
        super().__init__(
            description=("This is the node's original file "
                         "path as defined in the DSSF."), _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class NodeTypeProperty(PropertyBaseClass):
    """
    Property Class for a node's type.
    """
    key = "node_type"

    def __init__(self):
        """ Init for the Node Type Property Class.
        """
        super().__init__(
            description="This is the type of this node.",
            is_computable=True, is_basic=True, _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class UploadNodeTypeProperty(PropertyBaseClass):
    """
    Property Class for a node's type upon upload.

    NOTE: in contrast to NodeTypeProperty, this property is
    not computable, i.e. the value will be copied if metadata is transferred.
    """
    key = "upload_node_type"

    def __init__(self):
        """ Init for the Upload Node Type Property Class.
        """
        super().__init__(
            description="This is the type of this node upon upload.",
            _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class NodeDimProperty(PropertyBaseClass):
    """
    Property Class for a node's dimension.
    """
    key = "node_dim"

    def __init__(self):
        """ Init for the Node Dim Property Class.
        """
        super().__init__(
            description=("This is the node's dimension for a single "
                         "sample."), is_computable=True, is_basic=True)

    def export_to_array(self) -> np.ndarray:
        if is_nan(self._value):
            return np.array([self._value])
        else:
            return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class UploadNodeDimProperty(PropertyBaseClass):
    """
    Property Class for a node's dimension.

    NOTE: in contrast to NodeTypeProperty, this property is
    not computable, i.e. the value will be copied if metadata is transferred.
    """
    key = "upload_node_dim"

    def __init__(self):
        """ Init for the Upload Node Dim Property Class.
        """
        super().__init__(
            description=("This is the node's dimension for a single "
                         "sample upon upload."))

    def export_to_array(self) -> np.ndarray:
        if is_nan(self._value):
            return np.array([self._value])
        else:
            return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class MeanProperty(PropertyBaseClass):
    """
    Property Class for a node's mean.
    """
    key = "mean"

    def __init__(self):
        """ Init for the Mean Property Class.
        """
        super().__init__(
            description="This is the mean of the node.",
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class IdenticalProperty(PropertyBaseClass):
    """Property class for saving if a node consists of identical entries."""
    key = "samples_identical"

    class States():
        not_set = "not set"
        identical = "all samples identical"
        not_identical = "not all samples identical"

    def __init__(self):
        """ Init for the Identical Property Class.
        """
        super().__init__(
            description=("This shows if the node consists of identical "
                         "entries."),
            _default=IdenticalProperty.States.not_set,
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None

    def set(self, new_value: str) -> None:
        """ Set new value. It can be either 'all samples identical',
        'not all samples identical' or 'not set'.

        Args:
            new_value (str): new value

        Returns:
            None

        Raises:
            TypeError: if new value does not have the same type as the current
            value.
        """
        if new_value not in (self.States.identical, self.States.not_identical,
                             self.States.not_set):
            raise TypeError(f"The new value ({new_value}) is not valid for the"
                            f" '{self.key}' property. It can be set to either "
                            f"'{self.States.identical}', "
                            f"'{self.States.not_identical}' or "
                            f"'{self.States.not_set}'."
                            )
        self._value = new_value
        return None


class SortedProperty(PropertyBaseClass):
    """Property class for saving if a node is sorted or not."""
    key = "samples_sorted"

    class States():
        ASCENDING = "ascending"
        DESCENDING = "descending"
        IDENTICAL = "identical"
        UNSORTED = "unsorted"
        NOT_SET = "not_set"

    def __init__(self):
        """ Init for the SortedProperty class.
        """
        super().__init__(
            description=("This property describes if the node is sorted."),
            _default=SortedProperty.States.NOT_SET,
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None

    def set(self, new_value: str) -> None:
        """ Set new value.

        Args:
            new_value (str): new value

        Raises:
            TypeError: if new value does not have the same type as the current
            value.
        """
        if new_value not in (self.States.ASCENDING, self.States.DESCENDING,
                             self.States.UNSORTED, self.States.NOT_SET,
                             self.States.IDENTICAL):
            raise TypeError(f"The new value ({new_value}) is not valid for the"
                            f" '{self.key}' property. It can be set to either "
                            f"'{self.States.ASCENDING}', "
                            f"'{self.States.DESCENDING}' or "
                            f"'{self.States.UNSORTED}', or "
                            f"'{self.States.IDENTICAL}' or "
                            f"'{self.States.NOT_SET}'."
                            )
        self._value = new_value
        return None


class EquallySpacedProperty(PropertyBaseClass):
    """Property class for saving if a node is equally spaced or not. Note
    that only sorted nodes can be equally spaced. Unsorted nodes always are
    unequally spaced.."""
    key = "samples_equally_spaced"

    def __init__(self):
        """ Init for the EquallySpacedProperty class.
        """
        super().__init__(
            description=("This property describes if the node is equally "
                         "spaced."),
            _default=float("NaN"), is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if np.isnan(arr[0]):
            self.set(arr[0])
        elif isinstance(arr[0], np.bool_):
            self.set(bool(arr[0]))
        return None


class VarianceProperty(PropertyBaseClass):
    """
    Property Class for a node's variance.
    """
    key = "variance"

    def __init__(self):
        """ Init for the Variance Property Class.
        """
        super().__init__(
            description="This is the variance of the node.",
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class StandardDeviationProperty(PropertyBaseClass):
    """
    Property Class for a node's mean.
    """
    key = "standard_deviation"

    def __init__(self):
        """ Init for the Standard Deviation Property Class.
        """
        super().__init__(
            description="This is the standard deviation of the node.",
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class MinValueProperty(PropertyBaseClass):
    """
    Property Class for a node's minimum value.
    """
    key = "min"

    def __init__(self):
        """ Init for the Min Value Property Class.
        """
        super().__init__(
            description="This is the node's minimum value.",
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class UploadMinValueProperty(PropertyBaseClass):
    """
    Property Class for a node's minimum value upon upload.

    NOTE: in contrast to MinValueProperty, this property is
    not computable, i.e. the value will be copied if metadata is transferred.
    """
    key = "upload_min"

    def __init__(self):
        """ Init for the Upload Min Value Property Class.
        """
        super().__init__(
            description="This is the node's minimum value upon upload.")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class MaxValueProperty(PropertyBaseClass):
    """
    Property Class for a node's maximum value.
    """
    key = "max"

    def __init__(self):
        """ Init for the Max Value Property Class.
        """
        super().__init__(
            description="This is the node's maximum value.",
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class UploadMaxValueProperty(PropertyBaseClass):
    """
    Property Class for a node's maximum value upon upload.

    NOTE: in contrast to MinValueProperty, this property is
    not computable, i.e. the value will be copied if metadata is transferred.
    """
    key = "upload_max"

    def __init__(self):
        """ Init for the Upload Max Value Property Class.
        """
        super().__init__(
            description="This is the node's maximum value upon upload.")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(float(arr[0]))
        return None


class CategoricalProperty(PropertyBaseClass):
    """
    Property Class for a node's categorical property.
    """
    key = "categorical"

    def __init__(self):
        """ Init for th Categorical Property Class.
        """
        super().__init__(
            description=("This property indicates if the node is "
                         "categorical or not."),
            is_computable=True, is_basic=False)
        # mypy has some problems with the type.
        self._value: float

    def export_to_array(self) -> np.ndarray:
        if is_nan(self._value):
            return np.array([self._value])
        else:
            return np.array([int(self._value)])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(bool(arr[0]))
        return None


class NumberUniqueValuesProperty(PropertyBaseClass):
    """
    Property Class for a node's number of unique values.
    """
    key = "nr_unique_values"

    def __init__(self):
        """ Init for the Number of Unique Values Property Class.
        """
        super().__init__(
            description=("This is the node's number of unique "
                         "values."), is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(int(arr[0]))
        return None


class UploadNumberUniqueValuesProperty(PropertyBaseClass):
    """
    Property Class for a node's number of unique values upon upload.

    NOTE: in contrast to NumberUniqueValuesProperty, this property is
    not computable, i.e. the value will be copied if metadata is transferred.
    """
    key = "upload_nr_unique_values"

    def __init__(self):
        """ Init for the Upload Number of Values Property Class.
        """
        super().__init__(
            description=("This is the node's number of unique "
                         "values upon upload."))

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(int(arr[0]))
        return None


class UniqueValuesProperty(PropertyBaseClass):
    """
    Property Class for a node's unique values.
    """
    key = "unique_values"

    def __init__(self):
        """ Init for the Unique Values Property Class.
        """
        super().__init__(
            description=("These are the node's unique values."),
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class UniqueValuesSubsetProperty(PropertyBaseClass):
    """
    Property Class for the randomly sampled subset of a node's unique values.
    Note that we sample without replacement from the whole list of unique
    values. The maximum size of this list is defined in the statistics class
    `unique_values_summary` and it should be kept small (5 - 10 samples).
    """
    key = "unique_values_subset"

    def __init__(self):
        """ Init for the Unique Values Random Subset Property Class.
        """
        super().__init__(
            description=("These are a randomly sampled subset of a node's "
                         "unique values."),
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class UploadUniqueValuesSubsetProperty(PropertyBaseClass):
    """
    Property Class for the randomly sampled subset of a node's unique values
    upon upload.
    Note that we sample without replacement from the whole list of unique
    values. The maximum size of this list is defined in the statistics class
    `unique_values_summary` and it should be kept small (5 - 10 samples).
    """
    key = "upload_unique_values_subset"

    def __init__(self):
        """ Init for the Unique Values Random Subset Property Class.
        """
        super().__init__(
            description=("These are a randomly sampled subset of a node's "
                         "unique values upon upload."))

    def export_to_array(self) -> np.ndarray:
        return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class UploadUniqueValuesProperty(PropertyBaseClass):
    """
    Property Class for a node's unique values upon upload.

    NOTE: in contrast to UniqueValuesProperty, this property is
    not computable, i.e. the value will be copied if metadata is transferred.
    """
    key = "upload_unique_values"

    def __init__(self):
        """ Init for the Upload Unique Values Property Class.
        """
        super().__init__(
            description=("These are the node's unique values "
                         "upon upload."))

    def export_to_array(self) -> np.ndarray:
        return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class UniqueValuesCounterProperty(PropertyBaseClass):
    """
    Property Class for a node's unique values counter.
    """
    key = "unique_values_counter"

    def __init__(self):
        """ Init for the Unique Values Counter Property Class.
        """
        super().__init__(
            description=("This is the absolute number of times a unique "
                         "value appears within the node."),
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class UniqueValuesFrequencyProperty(PropertyBaseClass):
    """
    Property Class for a node's unique values frequency.
    """
    key = "unique_values_frequency"

    def __init__(self):
        """ Init for the Unique Values Frequency Property Class.
        """
        super().__init__(
            description=("This is the fraction of times a unique "
                         "value appears within the node."),
            is_computable=True)

    def export_to_array(self) -> np.ndarray:
        return np.array(self._value)

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        if arr.size == 1 and is_nan(arr[0]):
            self.set(arr[0])
        else:
            self.set(arr.tolist())
        return None


class SchemaNodeTypeProperty(PropertyBaseClass):
    """
    Property Class for a node's schema type.
    """
    key = "schema_node_type"

    def __init__(self):
        """ Init for the Schema Node Type Property Class.
        """
        super().__init__(
            description="This is the schema type of this node.",
            is_computable=True, is_basic=False, _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class UploadSchemaNodeTypeProperty(PropertyBaseClass):
    """
    Property Class for a node's schema type upon upload.
    """
    key = "upload_schema_node_type"

    def __init__(self):
        """ Init for the Upload Schema Node Type Property Class.
        """
        super().__init__(
            description="This is the schema type of this node upon upload.",
            is_computable=False, is_basic=False, _default="")

    def export_to_array(self) -> np.ndarray:
        return np.array([self._value])

    def import_from_array(self, arr: np.ndarray) -> None:
        self._check_array_input(arr)
        self.set(str(arr[0]))
        return None


class AllProperties:
    """
    All Properties Class. Combine all properties into one big object.

    Attributes:

        all_props: a list containing all property objects
        mean: Mean Property object
        dssf_component_name: Component Name object
        node_type: Node type object
    """
    def __init__(self):
        """ Init for AllProperties class.
        """
        # Initialize all children of the PropertyBaseClass.
        # NOTE: we set these attributes explicitly on purpose. Alternatively,
        # we could iterate over all children of the PropertyBaseClass.
        # However, this leads to unresolved attribute reference warnings.

        # By convention: the attribute names have to correspond
        # to the property keys.
        self._version = MetaDataVersionProperty()
        # Basic properties.
        self.nr_samples = NumberSamplesProperty()
        self.node_type = NodeTypeProperty()
        self.node_dim = NodeDimProperty()
        self.categorical = CategoricalProperty()
        self.schema_node_type = SchemaNodeTypeProperty()
        # DSSF properties.
        self.dssf_component_name = DSSFComponentNameProperty()
        self.dssf_component_type = DSSFComponentTypeProperty()
        self.dssf_component_file_path = DSSFComponentFilePathProperty()
        # Upload properties.
        self.upload_node_type = UploadNodeTypeProperty()
        self.upload_node_dim = UploadNodeDimProperty()
        self.upload_node_name = UploadNodeNameProperty()
        self.upload_min = UploadMinValueProperty()
        self.upload_max = UploadMaxValueProperty()
        self.upload_nr_unique_values = UploadNumberUniqueValuesProperty()
        self.upload_unique_values = UploadUniqueValuesProperty()
        self.upload_unique_values_subset = \
            UploadUniqueValuesSubsetProperty()
        self.upload_schema_node_type = UploadSchemaNodeTypeProperty()
        # General properties.
        self.mean = MeanProperty()
        self.variance = VarianceProperty()
        self.standard_deviation = StandardDeviationProperty()
        self.samples_identical = IdenticalProperty()
        self.samples_sorted = SortedProperty()
        self.samples_equally_spaced = EquallySpacedProperty()
        self.min = MinValueProperty()
        self.max = MaxValueProperty()
        self.nr_unique_values = NumberUniqueValuesProperty()
        self.unique_values = UniqueValuesProperty()
        self.unique_values_subset = \
            UniqueValuesSubsetProperty()
        self.unique_values_counter = UniqueValuesCounterProperty()
        self.unique_values_frequency = UniqueValuesFrequencyProperty()

        self.all_props = [value for attr_name, value in self.__dict__.items()
                          if isinstance(value, PropertyBaseClass) and
                          value.key == attr_name
                          ]
        self._check_input()

    def _check_input(self):
        """ Check
            - if there are any children of the PropertyBaseClass
            which are not part of the `all_props` list because the attribute
            name does not match the property key.
            - if there are any children of the PropertyBaseClass which are not
            initialized here.

        Note: Type hints are purposefully disabled for this class. While the
        `not_init` check works perfectly fine, the type hints seem to be
        unhappy: `Cannot instantiate abstract class 'PropertyBaseClass'
        with abstract attributes 'export_to_array' and 'import_from_array'`
        Suggestions welcome!


        Returns:
            None.

        Raises:
            KeyError: If properties are not part of the `all_props` list
                because they have the wrong attribute name.
            Warning: If not all children of the PropertyBaseClass are
                initialised in the AllProperties class.
        """
        not_prop_key = [attr_name for attr_name, value in self.__dict__.items()
                        if isinstance(value, PropertyBaseClass) and
                        value.key != attr_name
                        ]
        if not_prop_key:
            raise KeyError("The following properties are not part of the "
                           "all properties list because their attribute "
                           "names do not match their property keys: "
                           f"{not_prop_key}")

        not_init = [Child.key for Child in self.all_props
                    if not hasattr(Child, "_value")]
        if not_init:
            warnings.warn("The following properties are not initialised "
                          f"in the AllProperties class: {not_init}")
        return None

    def __eq__(self, other) -> bool:
        """ Overwrite equal to be able to compare two AllProperties objects.
        Ensure that the attributes of all properties, are equal.

        Args:
            other (AllProperties): second AllProperties object.

        Returns:
            bool: equal yes/no
        """
        # Protect against other being a different class.
        if self.__class__ != other.__class__:
            return False

        # Iterate over all properties.
        for obj in self.all_props:
            try:
                other_obj = other.get_obj_for_key(obj.key)
            except KeyError as e:
                print(e)
                return False
            if not obj == other_obj:
                return False
        return True

    def is_numerical(self) -> bool:
        """ Check if node is numerical.

        Returns:
            bool: node is numerical yes/no
        """
        return "num" in self.node_type.get()

    def is_string(self) -> bool:
        """ Check if node is a string.

        Returns:
            bool: Node is string yes/no.
        """
        return "str" in self.node_type.get()

    def is_upload_numerical(self) -> bool:
        """ Check if node was numerical upon upload.

        Returns:
            bool: node was numerical upon upload yes/no
        """
        return "num" in self.upload_node_type.get()

    def is_upload_string(self) -> bool:
        """ Check if node was a string upon upload.

        Returns:
            bool: Node was string upon upload yes/no.
        """
        return "str" in self.upload_node_type.get()

    def is_categorical(self) -> bool:
        """ Check if node is categorical.

        Returns:
            bool: node is categorical yes/no
        """
        return "_cat" in self.node_type.get()

    def is_upload_categorical(self) -> bool:
        """ Check if node was categorical upon upload.

        Returns:
            bool: node was categorical upon upload yes/no
        """
        return "_cat" in self.upload_node_type.get()

    def is_scalar(self) -> bool:
        """ Check if node contains scalars.
        Conditions:
            - node_dim = [1]

        Returns:
            bool: node contains scalars yes/no
        """
        if self.node_dim.is_default():
            raise ValueError("The node dimension is set to its default value. "
                             "It cannot be determined if it is a scalar.")
        node_dim = self.node_dim.get()
        if not isinstance(node_dim, list):
            raise TypeError("The node dimension property was expected to "
                            "be a list.")

        return len(node_dim) == 1 \
            and node_dim[0] == 1

    def is_upload_scalar(self) -> bool:
        """ Check if original upload node contained scalars.
        Conditions:
            - node_dim = [1]

        Returns:
            bool: upload node contained scalars yes/no
        """
        if self.upload_node_dim.is_default():
            raise ValueError("The upload node dimension is set to its "
                             "default value. It cannot be determined "
                             "if it is a scalar.")
        upload_node_dim = self.upload_node_dim.get()
        if not isinstance(upload_node_dim, list):
            raise TypeError("The upload node dimension property was "
                            "expected to be a list.")
        return len(upload_node_dim) == 1 and upload_node_dim[0] == 1

    def is_tensor(self) -> bool:
        """ Check if node contains tensors.
        Conditions:
            - numerical
            - node_dim format: either e.g. [1, 2, 3] or [4]

        Returns:
            bool: node contains tensors yes/no
        """
        if self.node_dim.is_default():
            raise ValueError("The node dimension is set to its default value. "
                             "It cannot be determined if it is a tensor.")
        node_dim = self.node_dim.get()
        if not isinstance(node_dim, list):
            raise TypeError("The node dimension property was expected to "
                            "be a list.")
        is_more_dim = len(node_dim) > 1
        is_not_scalar = len(node_dim) == 1 and node_dim[0] > 1
        return self.is_numerical() and (is_more_dim or is_not_scalar)

    def is_upload_tensor(self) -> bool:
        """ Check if original upload node contained tensors.
        Conditions:
            - numerical
            - node_dim format: either e.g. [1, 2, 3] or [4]

        Returns:
            bool: upload node contained tensors yes/no
        """
        if self.upload_node_dim.is_default():
            raise ValueError("The upload node dimension is set to its "
                             "default value. It cannot be determined "
                             "if it is a tensor.")
        upload_node_dim = self.upload_node_dim.get()
        if not isinstance(upload_node_dim, list):
            raise TypeError("The upload node dimension property was "
                            "expected to be a list.")
        is_more_dim = len(upload_node_dim) > 1
        is_not_scalar = len(upload_node_dim) == 1 and upload_node_dim[0] > 1
        return self.is_upload_numerical() and (is_more_dim or is_not_scalar)

    def is_upload_single_table(self) -> bool:
        """ Check if node originates from a single table or not.

        NOTE: Table Component Multi Files are treated the same as single
        tables. I.e. if the samples are split among different tables and the
        path entry in the DSSF contains a list, this function will return True.

        NOTE: Table Components Multi Files (multiple samples per
        table, list of tables in DSSF) is not the same as Table Component
        one file per sample (id placeholder in file path).

        Returns:
            bool: node originates from a single file yes/no
        Raises:
            TypeError: if DSSF component type is not understood
        """
        if self.dssf_component_type.get() == "table" and \
                ID_PLACEHOLDER not in self.dssf_component_file_path.get():
            # Table component single file or table component multi file.
            return True
        elif self.dssf_component_type.get() in [
                "table", "num", "str", "datetime"] and \
                ID_PLACEHOLDER in self.dssf_component_file_path.get():
            # Single node components and Table components with
            # one file per sample.
            return False
        else:
            raise TypeError("The DSSF component type is not understood.")

    def is_upload_sample_wise_table(self):
        """ Check if node originates from a sample wise table or not, i.e.
        were the samples provided as single tables, each containing a single
        row?

        Returns:
            bool: node originates from a sample wise table yes/no
        """
        if self.dssf_component_type.get() == "table" and ID_PLACEHOLDER in \
                self.dssf_component_file_path.get():
            return True
        return False

    def get_obj_for_key(self, key: str) -> PropertyBaseClass:
        """ Given a key, return the corresponding property object.

        Args:
            key (str): key, has to correspond to one of the property keys

        Returns:
            PropertyBaseClass: property object matching the key

        Raises:
            KeyError: if key does not match any of the property keys.
        """

        if key not in self.__dict__.keys():
            raise KeyError(f"Key '{key}' is not defined. Please choose "
                           f"from {self.get_keys()}.")
        return getattr(self, key)

    def get_values(self) -> Dict:
        """ Return dictionary containing property keys and corresponding
        values.

        Returns:
            Dict: value dictionary
        """
        value_dict = {}
        for obj in self.all_props:
            value_dict[obj.key] = obj.get()
        return value_dict

    def get_defaults(self) -> Dict:
        """ Return dictionary containing property keys and corresponding
        default values.

        Returns:
            Dict: default dictionary
        """
        default_dict = {}
        for obj in self.all_props:
            default_dict[obj.key] = obj._default
        return default_dict

    def get_basic_property_names(self) -> List[str]:
        """ Return list with all basic property names.

        Returns:
            List[str]: List of basic property names.
        """
        return [obj.key for obj in self.all_props if obj.is_basic]

    def get_keys(self) -> List:
        """ Return list with all available property keys.

        Returns:
            List: property keys
        """
        return [obj.key for obj in self.all_props]
