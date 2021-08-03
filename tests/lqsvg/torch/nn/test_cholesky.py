from __future__ import annotations

import pytest
import torch
from torch import nn

import lqsvg.torch.named as nt
from lqsvg.torch.nn.cholesky import CholeskyFactor, SPDMatrix
from lqsvg.torch.utils import make_spd_matrix


class TestCholeskyFactor:
    @pytest.fixture(params=(2, 8), ids=lambda x: f"Size:{x}")
    def size(self, request) -> int:
        return request.param

    @pytest.fixture(params=[(), (1,), (2,), (2, 2)], ids=lambda x: f"SampleShape:{x}")
    def sample_shape(self, request) -> tuple[int, ...]:
        return request.param

    @pytest.fixture
    def shape(self, size: int, sample_shape: tuple[int, ...]) -> tuple[int, ...]:
        return sample_shape + (size, size)

    @pytest.fixture
    def module(self, shape: tuple[int, ...]) -> CholeskyFactor:
        return CholeskyFactor(shape)

    @pytest.mark.parametrize("shape", [(), (1,), (2, 1), (20, 3, 2)])
    def test_wrong_shapes(self, shape: tuple[int, ...]):
        with pytest.raises(AssertionError):
            CholeskyFactor(shape)

    def test_init(self, module: CholeskyFactor, shape: tuple[int, ...]):
        assert hasattr(module, "beta")
        assert hasattr(module, "ltril")
        assert hasattr(module, "pre_diag")

        assert isinstance(module.ltril, nn.Parameter)
        assert isinstance(module.pre_diag, nn.Parameter)
        assert module.ltril.shape == shape
        assert module.pre_diag.shape == shape[:-1]

        cholesky = module()
        assert nt.allclose(cholesky, nt.matrix(torch.eye(shape[-1])))

    def test_call(self, module: CholeskyFactor, size: int):
        L = module()

        assert torch.is_tensor(L)
        assert torch.isfinite(L).all()

        L.sum().backward()
        assert nt.allclose(torch.triu(module.ltril.grad, diagonal=0), torch.zeros([]))
        tril_idxs = torch.tril_indices(size, size, offset=-1)
        assert not torch.isclose(
            module.ltril.grad[..., tril_idxs[0], tril_idxs[1]], torch.zeros([])
        ).any()
        assert not torch.isclose(module.pre_diag.grad, torch.zeros([])).any()

    @pytest.mark.parametrize("use_sample_shape", (True, False))
    def test_factorize_(
        self,
        module: CholeskyFactor,
        size: int,
        sample_shape: tuple[int, ...],
        use_sample_shape: bool,
        seed: int,
    ):
        # pylint:disable=too-many-arguments
        sample_shape = sample_shape if use_sample_shape else ()
        A = make_spd_matrix(size, sample_shape=sample_shape, rng=seed)
        module.factorize_(nt.matrix(A))
        L = nt.unnamed(module())
        C = nt.cholesky(A)
        C, L = torch.broadcast_tensors(C, L)
        isclose = torch.isclose(C, L, atol=1e-6)
        assert isclose.all(), (C[~isclose].tolist(), L[~isclose].tolist())


class TestSPDMatrix(TestCholeskyFactor):
    @pytest.fixture
    def module(self, shape: tuple[int, ...]) -> SPDMatrix:
        return SPDMatrix(shape)

    @pytest.mark.parametrize("use_sample_shape", (True, False))
    def test_factorize_(
        self,
        module: SPDMatrix,
        size: int,
        sample_shape: tuple[int, ...],
        use_sample_shape: bool,
        seed: int,
    ):
        # pylint:disable=too-many-arguments
        sample_shape = sample_shape if use_sample_shape else ()
        A = make_spd_matrix(size, sample_shape=sample_shape, rng=seed)
        module.factorize_(nt.matrix(A))
        B = nt.unnamed(module())
        A, B = torch.broadcast_tensors(A, B)
        isclose = torch.isclose(A, B, atol=1e-6)
        assert isclose.all(), (A[~isclose].tolist(), B[~isclose].tolist())
