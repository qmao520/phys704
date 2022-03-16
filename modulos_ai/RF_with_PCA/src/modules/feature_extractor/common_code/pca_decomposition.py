# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
"""This file contains the class to perform PCA to tables.
"""
import sklearn.decomposition as skdecomp
import numpy as np
from typing import Dict, Optional, Any


class PCAError(Exception):
    """Error in the PCA FE.
    """
    pass


class PCA_FE:
    """
    Class that performs PCA of numerical vectors/scalars.
    """

    def __init__(self, n_components: int) -> None:
        """
        Initialize class.
        """
        self._pca = skdecomp.PCA(n_components=n_components)
        self._node_names: Optional[np.ndarray] = None

    def __eq__(self, other: Any) -> bool:
        """Implement equality operator.

        Args:
            other (Any): Any object to compare with.

        Returns:
            bool: True if the object is equal to the instance of this class.
        """
        if not isinstance(other, PCA_FE):
            return False
        if self._node_names is None and other._node_names is not None \
                or self._node_names is not None and other._node_names is None:
            return False
        elif self._node_names is not None and other._node_names is not None \
                and (self._node_names != other._node_names).any():
            return False
        if self._pca.n_components != other._pca.n_components:
            return False
        # After fitting _pca has the attribute 'components_' which saves its
        # weights.
        if hasattr(self._pca, "components_"):
            if hasattr(other._pca, "components_"):
                if not np.array_equal(self._pca.components_,
                                      other._pca.components_):
                    return False
            else:
                return False
        else:
            if hasattr(other._pca, "components_"):
                return False
        return True

    def fit(self, X: Dict[str, np.ndarray]) -> "PCA_FE":
        """Fit PCA to a set of data.

        Args:
            X (Union[np.ndarray, np.ndarray]): Input list.

        Returns:
            PCA_FE: Class itself.
        """
        self._node_names = np.array(list(X.keys()))
        data = np.array([X[node_name] for node_name in self._node_names],
                        dtype=float).T
        self._pca.fit(data)
        return self

    def transform(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply PCA transformation to a list of input values.

        Args:
            X (Union[np.ndarray, np.ndarray]): Input list.

        Raises:
            PCAError: Raised if PCA is not fitted yet or the nodes in X have
                not the expected names.
        """
        node_names = np.array(list(X.keys()))
        if self._node_names is None:
            raise PCAError("PCA is not fitted yet! Prediction failed!")
        if set(self._node_names) != set(node_names):
            raise PCAError("Wrong nodes in the prediction data!")
        data = np.array([X[node_name] for node_name in self._node_names],
                        dtype=float).T
        result_exp = self._pca.transform(data)
        return {f"pca{n}": component for n, component in
                enumerate(result_exp.T)}
