from __future__ import annotations

import pytest
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.random import make_spd_matrix

n_row = standard_fixture((1, 2, 4), "Rows")


@pytest.fixture
def spdm(n_row: int, seed: int) -> Tensor:
    return nt.matrix(make_spd_matrix(n_row, sample_shape=(), rng=seed))


def test_cholesky(spdm: Tensor):
    scale_tril = nt.cholesky(spdm)

    assert scale_tril.shape == spdm.shape
    assert scale_tril.names == spdm.names
    assert scale_tril.dtype == spdm.dtype
    assert (nt.diagonal(scale_tril) >= 0).all()
    assert nt.allclose(scale_tril @ nt.transpose(scale_tril), spdm)
