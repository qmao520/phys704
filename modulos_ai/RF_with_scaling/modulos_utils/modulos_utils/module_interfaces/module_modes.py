# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
from enum import Enum, unique


@unique
class ModuleModes(Enum):
    """Enum class for modes of models, feature extractors and optimizers.
    """
    TRAINING = "training"
    VALIDATION = "validation"
    NONE = None
