import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.torch.utils import (
    assemble_cholesky,
    default_generator_seed,
    default_rng,
    disassemble_cholesky,
    make_spd_matrix,
    softplusinv,
)


@pytest.fixture(params=(1, 2, 4), ids=lambda x: f"NDim:{x}")
def n_dim(request):
    return request.param


@pytest.fixture(params=(1.0, 0.2, 0.5), ids=lambda x: f"Beta:{x}")
def beta(request) -> float:
    return request.param


def test_softplusinv(n_dim: int, beta: float):
    vec = nt.vector(torch.randn(n_dim))
    softplus = nt.softplus(vec, beta=beta)

    assert (softplus > 0).all()
    assert nt.allclose(softplusinv(softplus, beta=beta), vec, atol=1e-6)


@pytest.fixture
def covariance(n_dim: int) -> Tensor:
    return nt.matrix(make_spd_matrix(n_dim, sample_shape=()))


def test_disassemble_cholesky(covariance: Tensor):
    ltril, pre_diag = disassemble_cholesky(covariance)

    assert ltril.shape == covariance.shape
    assert ltril.dtype == covariance.dtype
    assert ltril.names == covariance.names

    assert pre_diag.shape == ltril.shape[:-1]
    assert pre_diag.dtype == covariance.dtype


def test_assemble_cholesky(covariance: Tensor):
    scale_tril = nt.cholesky(covariance)

    assert scale_tril.shape == covariance.shape
    assert scale_tril.names == covariance.names
    assert scale_tril.dtype == covariance.dtype
    assert (nt.diagonal(scale_tril) >= 0).all()
    assert nt.allclose(scale_tril @ nt.transpose(scale_tril), covariance)

    ltril, pre_diag = disassemble_cholesky(covariance)
    restored = assemble_cholesky(ltril, pre_diag)

    assert restored.shape == scale_tril.shape
    assert restored.names == scale_tril.names
    assert restored.dtype == scale_tril.dtype
    assert (nt.diagonal(restored) >= 0).all()
    assert nt.allclose(restored, scale_tril)


@pytest.fixture(params=(None, 123, torch.Generator()))
def seed(request):
    return request.param


def test_default_rng(seed):
    rng = default_rng(seed)
    assert isinstance(rng, torch.Generator)


def test_default_generator_seed():
    random = torch.randn(10)
    with default_generator_seed(42):
        first = torch.randn(10)
    with default_generator_seed(42):
        second = torch.randn(10)

    assert not any(torch.allclose(t, random) for t in (first, second))
    assert torch.allclose(first, second)
