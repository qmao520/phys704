# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""All Errors and Exceptions used by the dssf and structure code."""


# Errors used in the DSSFComponent class.
class ColumnHeaderError(Exception):
    """This error is thrown if the a splitted csv does not have the save
    header.
    """
    pass


class DataShapeError(Exception):
    """This error is thrown if the shape of the data is not allowed.
    """
    pass


class MetadataException(Exception):
    """Raised if the metadata is not correctly filled before saving.
    """
    pass


class WrongExtensionError(Exception):
    """Raised if the extension of the path is not allowed by the type."""
    pass


class WrongTypeError(Exception):
    """Raised if the type of data is not the same as the one in the dssf."""
    pass


class DataShapeMissmatchError(Exception):
    """Raised if the data of a node has not the same shape"""
    pass


class NodesMissingError(Exception):
    """Raised if some or all expected nodes are missing from a dataset (for
    post saving checks of checks of the tar file in the batch client solution.)
    """
    def __init__(self, msg: str, missing_nodes=[]):
        super().__init__(self, msg, missing_nodes)
        self.missing_nodes = missing_nodes
        self.msg = msg

    def __str__(self):
        return self.msg


class ColumnOverflowException(Exception):
    """Raised if the data in the csv has to many columns."""
    pass


class NRSamplesOverflowError(Exception):
    """Raised if the number of samples is too big."""
    pass


# Errors used in the DSSFSaver class.
class SampleIDError(Exception):
    """SampleIDs are not a match.
    """
    pass


class SaveError(Exception):
    """Raised if a node in the list of nodes to save is not saved or no node is
    saved at all.
    """
    pass


class MissingSampleError(Exception):
    """Raised if there are missing samples in a node."""
    pass


class DatasetTooBigException(Exception):
    """Raised if there are more features than allowed."""


# Errors from the DSSFInterpreter.
class DSSFFormatError(Exception):
    """Thrown if the json can not be read in."""
    pass


class DSSFOldFormatError(Exception):
    """Thrown if the DSSF is in the old format.
    """
    pass


class DSSFEntryError(Exception):
    """
    Thrown if there are missing keys or the provided values are
    not compatible.
    """
    pass


class DSSFNodesMissing(Exception):
    """ Thrown if nodes defined in the DSSF are missing.
    """
    pass


class DSSFNodeTypeNotSupported(Exception):
    """Thrown if an unknown node type is enforced in the DSSF.
    """
    pass


class DSSFVersionError(Exception):
    """
    Thrown if the version is incorrect or the version keyword is not provided.
    """
    pass


class DSSFOptionalKeyError(Exception):
    """Thrown if the given optional keys are wrong ore have invalid values."""
    pass


class NotEnoughSamplesError(Exception):
    """Error is thrown if the dataset has not enough samples."""
    pass


class DSSFDateTimeNotReadable(Exception):
    """
    Thrown if the DateTime stamp could not be read in.
    """
    pass


class DSSFKeywordNotValidInVersion(Exception):
    """Error is thrown if a keyword is not valid in a specific dssf version."""
    pass
