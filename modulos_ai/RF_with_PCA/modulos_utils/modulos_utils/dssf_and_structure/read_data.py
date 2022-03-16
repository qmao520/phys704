# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Union, Iterator, Dict, Any


class DataBaseClass(ABC):
    """
    Data Base class.
    """
    @abstractmethod
    def get_dtypes(self) -> Union[Dict[str, str], str]:
        """ Get a dictionary of data types with node names as keys for a csv
        file. Else it is a single string with the type.

        Returns:
            Union[Dict[str, str], str]: of data types.
        """
        pass

    @abstractmethod
    def get_shape(self) -> List[int]:
        """ Get shape of data.

        Returns:
            List: data shape.
        """
        pass

    @abstractmethod
    def get_data(self) -> Union[Dict, np.ndarray, List, float, str]:
        """Get all data of file.

        Returns:
            Union[np.ndarray, List, float, str]: all data
        """


class CSVDataType(DataBaseClass):
    """Read csv data:
            * read the data column wise for sample ids or batchwise for
            yielding
    """

    def __init__(self, file_path: str) -> None:
        ext = os.path.splitext(file_path)[1]
        if ext != '.csv':
            raise TypeError("CSVDataType class can only be used for "
                            "csv tables, input file type = %s" %
                            ext)
        self.file_path = file_path

        df = pd.read_csv(file_path, nrows=3)
        self.orig_columns: List[str] = list(df.columns)
        self.columns: List[str] = list(df.columns)
        self.dtypes = {}
        for col in self.columns:
            df = pd.read_csv(self.file_path, usecols=[col])
            self.dtypes[col] = \
                ("num" if np.issubdtype(df.dtypes[col], np.number) else "str")
        df = pd.read_csv(self.file_path, usecols=[0])
        self.len = len(df)
        return None

    def get_data_single_column(self, column_name: str) -> List[
            Union[float, int, str]]:
        """This reads the data of one column and returns it.

        Args:
            column_name (Union[str, int]): name or column number

        Returns:
            List[Union[float, int, str]]: data of the column
        """
        if self.dtypes[column_name] == "num":
            df = pd.read_csv(self.file_path, usecols=[column_name])
        else:
            df = pd.read_csv(self.file_path, usecols=[column_name], dtype=str)
        return list(df[column_name].values)

    def get_data_in_batches(self, batch_size: int) -> Iterator[
            Dict[str, np.ndarray]]:
        """It returns the data in the csv batchwise as a generator.

        Args:
            batch_size (int): size of a batch

        Raises:
            ValueError: raised if there are missing values

        Yields:
            Iterator[Dict[str, np.ndarray]]: data in batches
        """
        for i in range(0, self.len, batch_size):
            column_dict: Dict[str, np.ndarray] = {}
            df = pd.read_csv(self.file_path, header=0, skiprows=i,
                             nrows=batch_size, names=self.orig_columns)
            df = df.replace(r"^\s*$", np.nan, regex=True)

            # Replace Excel `#DIV/0!` infinite values with np.inf.
            df = df.replace("#DIV/0!", np.inf)
            if df.isnull().values.any():
                raise ValueError("Dataset contains missing values.")
            if df.isin([np.inf, -np.inf]).values.sum() > 0:
                raise ValueError("Dataset contains infinite values.")
            for x in self.columns:
                if self.dtypes[x] == "num":
                    column_dict[x] = df[x].values
                else:
                    column_dict[x] = df[x].values.astype(str)
            yield column_dict

    def get_shape(self) -> List[int]:
        """It returns the number of samples and features as a list.

        Returns:
            List[int]: shape of the csv
        """
        return [self.len, len(self.columns)]

    def get_dtypes(self) -> Dict[str, str]:
        """Returns the dtypes of the csv.

        Returns:
            Dict[str]: dtypes of all columns
        """
        return self.dtypes

    def get_data(self) -> Dict[str, List[Union[str, float, int]]]:
        """It returns the whole csv as a dictionary with the column names as
        keys and features in lists as values.

        Raises:
            ValueError: raised if data contains missing values

        Returns:
            Dict[str, List[Union[str, float, int]]]: all data of csv
        """
        df = pd.read_csv(self.file_path, header=0)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Replace Excel `#DIV/0!` infinite values with np.inf.
        df = df.replace("#DIV/0!", np.inf)
        if df.isnull().values.any():
            raise ValueError("Dataset contains missing values.")
        if df.isin([np.inf, -np.inf]).values.sum() > 0:
            raise ValueError("Dataset contains infinite values.")
        return df.to_dict("list")


class NPYDataType(DataBaseClass):
    """Read numpy files and returns them.
    """

    def __init__(self, file_path: str) -> None:
        ext = os.path.splitext(file_path)[1]
        if ext != '.npy':
            raise TypeError("NPYDataType class can only be used for "
                            "npy files, input file type = %s" % ext)

        self.data = np.load(file_path)
        if (np.issubdtype(self.data.dtype, np.number)):
            if np.isinf(self.data).any():
                raise ValueError("Dataset contains infinite values.")
            if np.isnan(self.data).any():
                raise ValueError("Dataset contains missing values.")
            if np.issubdtype(self.data.dtype, np.float128):
                raise TypeError(
                    "128 bit numpy floats are currently not supported. "
                    "Please use 64 bit floats.")
            if np.issubdtype(self.data.dtype, np.unsignedinteger):
                self.data = self.data.astype(np.uint64)
            elif np.issubdtype(self.data.dtype, np.integer):
                self.data = self.data.astype(np.int64)
            else:
                self.data = self.data.astype(np.float64)
            self.dtypes = "num"
        elif np.issubdtype(self.data.dtype, np.datetime64):
            if np.isnat(self.data).any():
                raise ValueError("Dataset contains missing dates.")
            self.dtypes = "str"
            self.data = np.datetime_as_string(self.data)
        else:
            self.dtypes = "str"
            self.data = self.data.astype(str)
        return None

    def get_dtypes(self) -> str:
        """It returns the type of the data.

        Returns:
            str: The type of the data.
        """
        return self.dtypes

    def get_shape(self) -> List[int]:
        """It returns the shape of the numpy data.

        Returns:
            List[int]: shape
        """
        return list(self.data.shape)

    def get_data(self) -> np.ndarray:
        """It returns the numpy ndarray data.

        Returns:
            np.ndarray: data
        """
        return self.data


class ImageDataType(DataBaseClass):
    """Read Images and return them.
    """

    def __init__(self, file_path: str) -> None:
        ext = os.path.splitext(file_path)[1]
        if ext not in [".jpg", ".tif", ".png"]:
            raise TypeError("ImageDataType class can only be used for "
                            ".jpg, .tif, and .png files,"
                            " input file type = %s" % ext)

        data = np.asarray(Image.open(file_path)).astype(float)
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=-1)
        self.data = data
        # The data should always be numerical (enforced by astype(float)).
        self.dtype = "num"
        return None

    def get_dtypes(self) -> str:
        """It returns the type of the data.

        Returns:
            List[str]:
        """
        return self.dtype

    def get_shape(self) -> List:
        """It returns the shape of the image data.

        Returns:
            List[int]: shape
        """
        return list(self.data.shape)

    def get_data(self) -> np.ndarray:
        return self.data


class TXTDataType(DataBaseClass):
    """Read and return text file.
    """
    def __init__(self, file_path: str):
        ext = os.path.splitext(file_path)[1]
        if ext != '.txt':
            raise TypeError("TXTDataType class can only be used for "
                            ".txt files, input file type = %s" % ext)

        with open(file_path, "r") as f:
            data = f.read()
        # Determine if string that was read in is numeric or not.
        self.data: Union[str, float] = data
        # NOTE: the ValueError will not be raised if data is equal to
        # "NaN", "nan", "inf", "-iNF".
        # NOTE: this also assumes that the txt file only contains one entry.
        try:
            self.data = float(data)
            self.dtype = "num"
        except ValueError:
            self.data = str(data)
            self.dtype = "str"
        if isinstance(self.data, float) and np.isnan(self.data):
            raise ValueError("Txt file contains NaN values.")
        if isinstance(self.data, float) and np.isinf(self.data):
            raise ValueError("Txt file contains inf values.")
        # Check for excel infinite values.
        if isinstance(self.data, str) and self.data == "#DIV/0!":
            raise ValueError("Txt file contains inf values.")

    def get_dtypes(self) -> str:
        """It returns the type of the data.

        Returns:
            str: type of the text data
        """
        return self.dtype

    def get_shape(self) -> List[int]:
        """It returns the shape of the text data (currently fixed to [1]).

        Returns:
            List[int]: shape
        """
        return [1]

    def get_data(self) -> Union[str, float]:
        """It returns the entry of the text file.

        Returns:
            Union[str, float]: data
        """
        return self.data


def read_data(file_path: str) -> Any:
    """It returns a Data class instance to read files.

    Args:
        file_path (str): path to file to read

    Raises:
        TypeError: raised if file type is not supported

    Returns:
        DataBaseClass: data class instance for reading files
    """

    ext = os.path.splitext(file_path)[1]

    if ext == ".csv":
        return CSVDataType(file_path)
    elif ext == ".npy":
        return NPYDataType(file_path)
    elif ext in [".jpg", ".tif", ".png"]:
        return ImageDataType(file_path)
    elif ext == ".txt":
        return TXTDataType(file_path)
    else:
        raise TypeError("File format %s is currently not supported." % ext)
