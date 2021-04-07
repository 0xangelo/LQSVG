# pylint:disable=unsubscriptable-object,invalid-name
from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import pytest
from scipy.stats import norm as normal
from scipy.stats import ortho_group

from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_linsdynamics
from lqsvg.envs.lqr.utils import (
    ctrb,
    random_mat_with_eigval_range,
    random_matrix_from_eigs,
    wrap_sample_shape_to_size,
)

from .utils import standard_fixture


@pytest.fixture(params=(0, 2))
def dim(request) -> int:
    return request.param


# noinspection PyUnresolvedReferences
@pytest.fixture
def sampler(dim: int) -> callable[[int], np.ndarray]:
    call = normal.rvs if dim == 0 else partial(ortho_group.rvs, dim=3)

    def _sample(size: int) -> np.ndarray:
        return call(size=size)

    return _sample


# noinspection PyUnresolvedReferences
def test_wrap_sample_shape_to_size(sampler: callable[[int], np.ndarray], dim: int):
    wrapped = wrap_sample_shape_to_size(sampler, dim)

    def prefix(arr: np.ndarray) -> tuple[int, ...]:
        return arr.shape[:-dim] if dim else arr.shape

    sample_shape = ()
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape

    sample_shape = (1,)
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape

    sample_shape = (2,)
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape

    sample_shape = (2, 1)
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape


vec_dim = standard_fixture((2, 3, 4), "VecDim")
batch_shape = standard_fixture([(), (1,), (2,), (2, 1)], "BatchShape")


@pytest.fixture()
def eigvals(vec_dim: int, batch_shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = batch_shape + (vec_dim,)
    return rng.uniform(low=-1.0, high=1.0, size=shape)


def test_random_matrix_from_eigs(eigvals: np.ndarray, seed: int):
    mat, eigvecs = random_matrix_from_eigs(eigvals, rng=seed)
    check_mat_eigdecomp(mat, eigvals, eigvecs)


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


def check_mat_eigdecomp(mat: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    assert eigvecs.shape == eigvals.shape + eigvals.shape[-1:]

    for idx in range(eigvals.shape[-1]):
        mat_prod = mat @ vector_to_matrix(eigvecs[..., idx])
        eig_prod = scalar_to_matrix(eigvals[..., idx]) * vector_to_matrix(
            eigvecs[..., idx]
        )
        assert np.allclose(mat_prod, eig_prod)

    eigvals, eigvecs = sort_eigfactors(eigvals, eigvecs)
    _vals, _vecs = np.linalg.eig(mat)
    _vals, _vecs = sort_eigfactors(_vals, _vecs)
    assert np.allclose(eigvals, _vals)
    abs_cossim = np.abs(
        np.sum(eigvecs * _vecs, axis=-1)
        / (np.linalg.norm(eigvecs, axis=-1) * np.linalg.norm(_vecs, axis=-1))
    )
    assert np.allclose(abs_cossim, 1.0)


mat_dim = standard_fixture((2, 3, 4), "MatDim")
eigval_range = standard_fixture([(0, 1), (0.5, 1.5)], "EigvalRange")
n_batch = standard_fixture((None, 1, 4), "NBatch")


# noinspection PyArgumentList
def test_random_mat_with_eigval_range(
    mat_dim: int,
    eigval_range: tuple[float, float],
    horizon: int,
    n_batch: Optional[int],
    seed: int,
):
    mat = random_mat_with_eigval_range(
        mat_dim, eigval_range=eigval_range, horizon=horizon, n_batch=n_batch, rng=seed
    )

    assert mat.size("C") == mat_dim
    assert mat.size("R") == mat_dim
    assert mat.size("H") == horizon
    assert not n_batch or mat.size("B") == n_batch

    eigvals, _ = np.linalg.eig(mat.numpy())
    low, high = eigval_range
    assert np.all(np.abs(eigvals) >= low)
    assert np.all(np.abs(eigvals) <= high)


@pytest.fixture
def dynamics(n_state: int, n_ctrl: int, horizon: int) -> lqr.LinSDynamics:
    return make_linsdynamics(n_state, n_ctrl, horizon)


def test_ctrb(dynamics: lqr.LinSDynamics, n_state: int, n_ctrl: int):
    C = ctrb(dynamics)
    assert isinstance(C, np.ndarray)
    assert C.shape == (n_state, n_state * n_ctrl)
