from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr import pack_obs, unpack_obs
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.nn.cholesky import CholeskyFactor
from lqsvg.torch.nn.dynamics.gauss import InitStateModule

dim = standard_fixture((2, 4, 8), "Dim")


@pytest.fixture
def module(dim: int) -> InitStateModule:
    return InitStateModule(dim)


@pytest.fixture
def state(dim: int, batch_shape: tuple[int, ...]) -> Tensor:
    return nt.vector(torch.randn(batch_shape + (dim,)))


@pytest.fixture
def obs(
    state: Tensor, batch_shape: tuple[int, ...], batch_names: tuple[str, ...]
) -> Tensor:
    time = nt.vector(torch.zeros(batch_shape + (1,)).int())
    return pack_obs(state, time).refine_names(*batch_names, ...).requires_grad_(True)


@pytest.fixture
def init(dim: int) -> lqr.GaussInit:
    return lqr.GaussInit(nt.vector(torch.randn(dim)), nt.matrix(torch.eye(dim)))


class TestInitStateDynamics:
    def test_constructor(self, dim: int):
        module = InitStateModule(dim)
        params = list(module.parameters())

        assert len(params) == 3
        assert hasattr(module, "loc")
        assert hasattr(module, "scale_tril")
        assert isinstance(module.loc, nn.Parameter)
        assert isinstance(module.scale_tril, CholeskyFactor)

    @pytest.mark.parametrize("sample_shape", [(), (1,), (2,), (2, 2)])
    def test_rsample(self, module: InitStateModule, sample_shape: tuple[int, ...]):
        obs, _ = module.rsample(sample_shape)

        assert obs.shape == sample_shape + (module.n_state + 1,)
        assert obs.dtype == torch.float32
        assert obs.names[-1] == "R"

        obs.sum().backward()
        params = (module.loc, module.scale_tril.ltril, module.scale_tril.pre_diag)
        assert all(list(p.grad is not None for p in params))
        assert not any(list(torch.allclose(p.grad, torch.zeros([])) for p in params))

    def test_log_prob(self, module: InitStateModule, obs: Tensor):
        log_prob = module.log_prob(obs)

        assert log_prob.shape == obs.shape[:-1]
        assert log_prob.dtype == torch.float32
        _, time = unpack_obs(obs)
        assert log_prob.names == nt.vector_to_scalar(time).names

        assert log_prob.grad_fn is not None
        log_prob.sum().backward()
        assert obs.grad is not None
        assert not nt.allclose(obs.grad, torch.zeros_like(obs.grad))
        grads = list(p.grad for p in module.parameters())
        assert all(list(g is not None for g in grads))
        assert all(list(not torch.allclose(g, torch.zeros_like(g)) for g in grads))

    def test_standard_form(self, module: InitStateModule):
        mu, sigma = module.standard_form()
        (mu.sum() + sigma.sum()).backward()

        loc, scale_tril = module.loc, module.scale_tril
        assert torch.allclose(loc.grad, torch.ones([]))
        assert torch.isfinite(scale_tril.ltril.grad).all()
        assert not torch.allclose(scale_tril.ltril.grad, torch.zeros([]))
        assert torch.isfinite(scale_tril.pre_diag.grad).all()
        assert not torch.allclose(scale_tril.pre_diag.grad, torch.zeros([]))

    def test_from_existing(self, init: lqr.GaussInit):
        module = InitStateModule.from_existing(init)
        assert all(nt.allclose(a, b) for a, b in zip(init, module.standard_form()))
