from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.random import default_generator_seed, default_rng, make_spd_matrix
from lqsvg.torch.utils import (
    assemble_cholesky,
    disassemble_cholesky,
    softplusinv,
    tensors_to_vector,
    vector_to_tensors,
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


tensor_shapes = standard_fixture(
    [[(1,), (2,)], [(), (1,), (1, 1)], [(2, 4), (2,)]], "TensorShapes"
)


@pytest.fixture
def tensors(tensor_shapes: list[tuple[int, ...]]) -> list[Tensor]:
    unnamed = [torch.rand(s) for s in tensor_shapes]
    return [
        nt.scalar(t) if t.dim() == 0 else nt.vector(t) if t.dim() == 1 else nt.matrix(t)
        for t in unnamed
    ]


@pytest.fixture
def vector(tensor_shapes: list[tuple[int, ...]]) -> Tensor:
    return nt.vector(torch.rand(sum(int(np.prod(s)) for s in tensor_shapes)))


def test_tensors_to_vector(tensors: Iterable[Tensor]):
    vector = tensors_to_vector(tensors)
    assert torch.is_tensor(vector)
    assert vector.names == ("R",)
    assert vector.numel() == sum(t.numel() for t in tensors)
    offset = 0
    for tensor in tensors:
        assert torch.allclose(
            nt.unnamed(vector[offset : offset + tensor.numel()]).view_as(tensor),
            nt.unnamed(tensor),
        )
        offset += tensor.numel()


def test_vector_to_tensors(vector: Tensor, tensors: Iterable[Tensor]):
    tensors_ = vector_to_tensors(vector, tensors)
    assert all(list(torch.is_tensor(t) for t in tensors_))
    zipped = list(zip(tensors_, tensors))
    assert all(list(t_.names == t.names for t_, t in zipped))
    assert all(list(t_.shape == t.shape for t_, t in zipped))
