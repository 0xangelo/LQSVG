# pylint:disable=missing-module-docstring
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.random._generator import Generator


RNG = Union[int, Generator, None]


def make_spd_matrix(
    n_dim: int, *, sample_shape: tuple[int, ...] = (), rng: RNG = None
) -> np.ndarray:
    """Generate a random symmetric, positive-definite matrix.

    Mirrors the `make_spd_matrix` function in sklearn with additional support
    for multiple samples (via `sample_shape`) and using numpy's `Generator`
    class instead of RandomState.

    Args:
        n_dim: The matrix dimension.
        sample_shape: Sizes of the sample dimensions.
        rng: Determines random number generation for dataset creation. Pass an
            int for reproducible output across multiple function calls.

    Returns:
        An array of shape `sample_shape` + [n_dim, n_dim] containing the random
        symmetric, positive-definite matrix (possibly batched).
    """
    # pylint:disable=invalid-name
    generator = np.random.default_rng(rng)

    A = generator.random(size=sample_shape + (n_dim, n_dim))
    axes = list(range(len(sample_shape) + 2))
    # The code inside `transpose` simply inverts the order of the last two axes
    U, _, V = np.linalg.svd(A.transpose(axes[:-2] + axes[-2:][::-1]) @ A)
    X = U @ (1.0 + np.eye(n_dim) * generator.random(sample_shape + (1, n_dim))) @ V

    return X


def np_expand(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Expand a numpy array by broadcasting it to the desired shape.

    Mirrors the behavior of torch.Tensor.expand.
    """
    return np.broadcast_to(arr, shape)
