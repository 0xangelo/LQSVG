# pylint:disable=missing-module-docstring
from __future__ import annotations

from typing import Union

import numpy as np
from numpy.random import Generator

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
    U, _, V = np.linalg.svd(A.transpose(axes[:-2] + [axes[-1], axes[-2]]) @ A)
    X = U @ (1.0 + np.eye(n_dim) * generator.random(sample_shape + (1, n_dim))) @ V

    return X


def np_expand(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Expand a numpy array by broadcasting it to the desired shape.

    Mirrors the behavior of torch.Tensor.expand.
    """
    return np.broadcast_to(arr, shape)


def random_unit_vector(
    size: int, sample_shape: tuple[int, ...] = (), eps: float = 1e-4, rng: RNG = None
) -> np.ndarray:
    """Vector uniformly distributed on the unit sphere.

    Args:
        size: size of the vector
        sample_shape: shape of the sample, prepended to the vector shape.
            Useful for sampling batches of vectors.
        eps: minimum norm of the random normal vector to avoid dividing by
            very small numbers. The function will resample random normal
            vectors until all have a norm larger than this value
        rng: random number generator parameter

    Returns:
        Vector uniformly distributed on the unit sphere.
    """
    rng = np.random.default_rng(rng)
    vec = rng.normal(size=sample_shape + (size,))
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    while np.any(norm < eps):
        vec = rng.normal(size=sample_shape + (size,))
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / norm


def random_unit_col_matrix(
    n_row: int,
    n_col: int,
    sample_shape: tuple[int, ...] = (),
    eps: float = 1e-4,
    rng: RNG = None,
) -> np.ndarray:
    """Matrix with column vectors uniformly distributed on the unit sphere.

    Args:
        n_row: number of rows
        n_col: number of columns
        sample_shape: shape of the sample, prepended to the vector shape.
            Useful for sampling batches of vectors.
        eps: tolerance parameters for `func:random_unit_vector`
        rng: random number generator parameter

    Returns:
        Matrix with column vectors uniformly distributed on the unit sphere.
    """
    sample_shape = sample_shape + (n_col,)
    tranposed = random_unit_vector(n_row, sample_shape=sample_shape, eps=eps, rng=rng)
    return np.swapaxes(tranposed, -2, -1)
