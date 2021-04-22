from typing import Union

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.utils import (
    assemble_cholesky,
    default_generator_seed,
    default_rng,
    disassemble_cholesky,
    make_spd_matrix,
    softplusinv,
)

n_dim = standard_fixture((1, 2, 4), "NDim")
beta = standard_fixture((1.0, 0.2, 0.5), "Beta")


def test_softplusinv(n_dim: int, beta: float):
    vec = nt.vector(torch.randn(n_dim))
    softplus = nt.softplus(vec, beta=beta)

    assert (softplus > 0).all()
    assert nt.allclose(softplusinv(softplus, beta=beta), vec, atol=1e-6)


@pytest.fixture
def spdm(n_dim: int) -> Tensor:
    return nt.matrix(make_spd_matrix(n_dim, sample_shape=()))


def test_disassemble_cholesky(spdm: Tensor):
    ltril, pre_diag = disassemble_cholesky(spdm)

    assert torch.is_tensor(ltril) and torch.is_tensor(pre_diag)
    assert torch.isfinite(ltril).all() and torch.isfinite(pre_diag).all()
    assert ltril.shape == spdm.shape
    assert ltril.dtype == spdm.dtype
    assert ltril.names == spdm.names

    assert pre_diag.shape == ltril.shape[:-1]
    assert pre_diag.dtype == spdm.dtype


def test_assemble_cholesky(spdm: Tensor, n_dim: int):
    ltril, pre_diag = disassemble_cholesky(spdm)
    ltril.requires_grad_(True)
    pre_diag.requires_grad_(True)
    restored = assemble_cholesky(ltril, pre_diag)

    assert restored.shape == spdm.shape
    assert restored.names == spdm.names
    assert restored.dtype == spdm.dtype
    assert (nt.diagonal(restored) >= 0).all()
    assert nt.allclose(restored, nt.cholesky(spdm))

    restored.sum().backward()
    assert torch.allclose(torch.triu(ltril.grad, diagonal=0), torch.zeros([]))
    tril_idxs = torch.tril_indices(n_dim, n_dim, offset=-1)
    assert not torch.isclose(
        ltril.grad[..., tril_idxs[0], tril_idxs[1]], torch.zeros([])
    ).any()
    assert not torch.isclose(pre_diag.grad, torch.zeros([])).any()


class TestTorchRNG:
    @pytest.fixture(params=(None, 123, torch.Generator()))
    def rng(self, request):
        return request.param

    def test_default_rng(self, rng: Union[None, int, torch.Generator]):
        rng = default_rng(rng)
        assert isinstance(rng, torch.Generator)

    def test_default_generator_seed(self):
        random = torch.randn(10)
        with default_generator_seed(42):
            first = torch.randn(10)
        with default_generator_seed(42):
            second = torch.randn(10)

        assert not any(torch.allclose(t, random) for t in (first, second))
        assert torch.allclose(first, second)
