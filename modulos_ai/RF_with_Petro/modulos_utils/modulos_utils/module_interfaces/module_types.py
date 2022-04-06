# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
from enum import Enum, unique


@unique
class ModuleType(Enum):
    """Enum class for module type.
    """
    FEATURE_EXTRACTOR = "feature_extractors"
    MODEL = "models"
    OPTIMIZER = "optimizers"
    OBJECTIVE = "objectives"
    SPLIT_STRATEGY = "split_strategies"
    ASSESSOR = "assessors"
    DATASET_STATS = "dataset_stats"
    TRANSFORMER = "transformer"
