# pylint:disable=unsubscriptable-object,invalid-name
from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import pytest
from scipy.stats import norm as normal
from scipy.stats import ortho_group

from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.utils import (
    ctrb,
    minimal_sample_shape,
    random_mat_with_eigval_range,
    random_matrix_from_eigs,
    random_unit_col_matrix,
    random_unit_vector,
    wrap_sample_shape_to_size,
)
from lqsvg.np_util import RNG
from lqsvg.testing.fixture import standard_fixture

from .utils import scalar_to_matrix, sort_eigfactors, vector_to_matrix

mat_dim = standard_fixture((2, 3, 4), "MatDim")
eigval_range = standard_fixture([(0, 1), (0.5, 1.5)], "EigvalRange")
n_batch = standard_fixture((None, 1, 4), "NBatch")
vec_dim = standard_fixture((2, 3, 4), "VecDim")
batch_shape = standard_fixture([(), (1,), (2,), (2, 1)], "BatchShape")
n_row = standard_fixture((2, 3, 4), "Rows")
n_col = standard_fixture((2, 3, 4), "Columns")


@pytest.fixture(params=(0, 2))
def dim(request) -> int:
    return request.param


@pytest.fixture
def sampler(dim: int) -> Callable[[int], np.ndarray]:
    call = normal.rvs if dim == 0 else partial(ortho_group.rvs, dim=3)

    def _sample(size: int) -> np.ndarray:
        return call(size=size)

    return _sample


def test_wrap_sample_shape_to_size(sampler: Callable[[int], np.ndarray], dim: int):
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


@pytest.fixture
def rng(seed: int) -> RNG:
    return np.random.default_rng(seed)


@pytest.fixture
def eigvals(vec_dim: int, batch_shape: tuple[int, ...], rng: RNG) -> np.ndarray:
    shape = batch_shape + (vec_dim,)
    return rng.uniform(low=-1.0, high=1.0, size=shape)


def test_random_matrix_from_eigs(eigvals: np.ndarray, rng: RNG):
    mat, eigvecs = random_matrix_from_eigs(eigvals, rng=rng)
    check_mat_eigdecomp(mat, eigvals, eigvecs)


def check_mat_eigdecomp(mat: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray):
    assert mat.shape == eigvecs.shape == eigvals.shape + eigvals.shape[-1:]

    for idx in range(eigvals.shape[-1]):
        mat_prod = mat @ vector_to_matrix(eigvecs[..., idx])
        eig_prod = scalar_to_matrix(eigvals[..., idx]) * vector_to_matrix(
            eigvecs[..., idx]
        )
        assert np.allclose(mat_prod, eig_prod)

    eigvals, eigvecs = sort_eigfactors(eigvals, eigvecs)
    _vals, _vecs = np.linalg.eig(mat)
    _vals, _vecs = sort_eigfactors(_vals, _vecs)
    assert np.allclose(eigvals, _vals), "Not all eigvals match"
    abs_cossim = np.abs(
        np.sum(eigvecs * _vecs, axis=-1)
        / (np.linalg.norm(eigvecs, axis=-1) * np.linalg.norm(_vecs, axis=-1))
    )
    assert np.allclose(abs_cossim, 1.0), "Cosine similarity should be 1 or -1"


@pytest.fixture()
def sample_shape(horizon: int, stationary: bool, n_batch: int) -> tuple[int, ...]:
    return minimal_sample_shape(horizon=horizon, stationary=stationary, n_batch=n_batch)


def test_random_mat_with_eigval_range(
    mat_dim: int,
    eigval_range: tuple[float, float],
    sample_shape: tuple[int, ...],
    rng: RNG,
):
    mat, eigval, eigvec = random_mat_with_eigval_range(
        mat_dim, eigval_range=eigval_range, sample_shape=sample_shape, rng=rng
    )

    assert mat.shape[-2:] == (mat_dim, mat_dim)
    assert mat.shape[:-2] == sample_shape

    check_mat_eigdecomp(mat, eigval, eigvec)
    low, high = eigval_range
    assert np.all(np.abs(eigval) >= low)
    assert np.all(np.abs(eigval) <= high)


@pytest.fixture
def dynamics(n_state: int, n_ctrl: int, horizon: int) -> lqr.LinSDynamics:
    dynamics = make_lindynamics(n_state, n_ctrl, horizon, stationary=True)
    return make_linsdynamics(dynamics, stationary=True)


def test_ctrb(dynamics: lqr.LinSDynamics, n_state: int, n_ctrl: int):
    C = ctrb(dynamics)
    assert isinstance(C, np.ndarray)
    assert C.shape == (n_state, n_state * n_ctrl)


def test_random_unit_vector(vec_dim: int, batch_shape: tuple[int, ...], seed: int):
    vector = random_unit_vector(vec_dim, sample_shape=batch_shape, rng=seed)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == batch_shape + (vec_dim,)
    assert np.isfinite(vector).all()
    assert np.all(np.isclose(np.linalg.norm(vector, axis=-1), 1.0))


def test_random_unit_col_matrix(
    n_row: int, n_col: int, batch_shape: tuple[int, ...], seed: int
):
    matrix = random_unit_col_matrix(n_row, n_col, sample_shape=batch_shape, rng=seed)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == batch_shape + (n_row, n_col)
    assert np.isfinite(matrix).all()
    assert np.all(np.isclose(np.linalg.norm(matrix, axis=-2), 1.0))
