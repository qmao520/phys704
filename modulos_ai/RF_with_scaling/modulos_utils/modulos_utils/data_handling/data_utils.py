# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""Helper functions for data handling."""
import os

import numpy as np


ONE_GB = 10**9


def is_modulos_numerical(dtype: np.dtype) -> bool:
    """Check if a numpy array has an accepted numerical
    type for the Modulos AutoML platform.

    Args:
        dtype (np.dtype): Dtype of the numpy array.

    Returns:
        bool: True if it is accepted.
    """
    return (np.issubdtype(dtype, np.integer) or
            np.issubdtype(dtype, np.floating))


def is_modulos_string(dtype: np.dtype) -> bool:
    """Check if a numpy array has an accepted string type for the
    Modulos AutoML platform.

    Args:
        dtype (np.dtype): Dtype fo the numpy array.

    Returns:
        bool: True if it is accepted.
    """
    return dtype.type is np.str_


def compute_batch_size(hdf5_file: str, n_samples_tot: int,
                       batch_size_gb: float = 1.) -> int:
    """Compute the number of batches by dividing the size of the hdf5 file by
    the size of one batch. We then get the number samples by dividing the total
    number of samples by the number of batches.

    Args:
        hdf5_file (str): Path to the hdf5 file of the dataset.
        n_samples_tot (int): Total number of samples in dataset.
        batch_size_gb (float): Size of one batch in GB. Defaults to 1.

    Returns:
        int: Number of samples in one batch of size `batch_size_gb` GB.
    """
    n_batches_float = os.path.getsize(hdf5_file) \
        / (float(batch_size_gb) * ONE_GB)
    # There has to be atleast 1 batch.
    n_batches_float = n_batches_float if n_batches_float > 1. else 1.
    batch_sample_count = int(n_samples_tot / n_batches_float)
    return batch_sample_count
