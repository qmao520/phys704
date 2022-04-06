# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Custom exceptions for the dataset converter Class.
"""


class DatasetOutputWriteError(Exception):
    """Exception for the case where the output dataset is not valid.
    """
    pass


class DataFileInexistentError(Exception):
    """Exception for the case where the output dataset is not valid.
    """
    pass


class SampleIdsInconsistentError(Exception):
    """Exception for the case where the output dataset is not valid.
    """
    pass


class MODOutputWriteError(Exception):
    """Exception for the case where the output dataset is not valid.
    """
    pass


class DatasetMetadataError(Exception):
    """Exception for the case where input metadata is empty in the hdf5.
    """
    pass


class DatasetConverterInternalError(Exception):
    """Exception for internal errors of the DatasetConverter Class.
    """
    pass
