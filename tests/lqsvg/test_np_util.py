from __future__ import annotations

import numpy as np
import torch

from lqsvg.np_util import make_spd_matrix, random_unit_col_matrix, random_unit_vector
from lqsvg.testing.fixture import standard_fixture

seed = standard_fixture((42, 69, 37), "Seed")
batch_shape = standard_fixture([(), (1,), (2,), (2, 1)], "BatchShape")
vec_dim = standard_fixture((2, 3, 4), "VecDim")
n_row = standard_fixture((2, 3, 4), "Rows")
n_col = standard_fixture((2, 3, 4), "Columns")


def test_spd_matrix(n_row: int, batch_shape: tuple[int, ...], seed: int):
    A = make_spd_matrix(n_row, sample_shape=batch_shape, rng=seed)

    assert A.shape == batch_shape + (n_row, n_row)
    B = torch.as_tensor(A)
    assert torch.allclose(B, B.transpose(-2, -1))
    eigval, _ = torch.linalg.eigh(B)
    assert eigval.ge(0).all()


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
