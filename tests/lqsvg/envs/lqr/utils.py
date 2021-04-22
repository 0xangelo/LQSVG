from __future__ import annotations

import numpy as np


def sort_eigfactors(
    eigval: np.ndarray, eigvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    idxs = np.argsort(eigval, axis=-1)
    return np.take_along_axis(eigval, idxs, axis=-1), np.take_along_axis(
        eigvec, idxs[..., np.newaxis], axis=-1
    )


def scalar_to_matrix(arr: np.ndarray) -> np.ndarray:
    return arr[..., np.newaxis, np.newaxis]


def vector_to_matrix(arr: np.ndarray) -> np.ndarray:
    """In column form."""
    return arr[..., np.newaxis]
