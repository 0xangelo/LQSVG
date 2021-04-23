# pylint:disable=unsubscriptable-object
from __future__ import annotations

from typing import Type

import pytest
import torch
import torch.nn as nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr import pack_obs
from lqsvg.envs.lqr.modules.dynamics.gauss import InitStateDynamics
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.modules import CholeskyFactor

batch_shape = standard_fixture([(), (1,), (2,), (2, 2)], "BatchShape")
dim = standard_fixture((2, 4, 8), "Dim")


@pytest.fixture
def module_cls() -> Type[InitStateDynamics]:
    return InitStateDynamics


@pytest.fixture
def module(module_cls: Type[InitStateDynamics], dim: int) -> InitStateDynamics:
    return module_cls(dim)


@pytest.fixture
def state(dim: int, batch_shape: tuple[int, ...]) -> Tensor:
    return nt.vector(torch.randn(batch_shape + (dim,)))


@pytest.fixture
def obs(state: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    time = nt.vector(torch.zeros(batch_shape + (1,)).int())
    return pack_obs(state, time).requires_grad_(True)


@pytest.fixture
def optim(module: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(module.parameters(), lr=1e-3)


@pytest.fixture
def init(dim: int) -> lqr.GaussInit:
    return lqr.GaussInit(nt.vector(torch.randn(dim)), nt.matrix(torch.eye(dim)))


class TestInitStateDynamics:
    def test_init(self, module_cls: Type[InitStateDynamics], dim: int):
        module = module_cls(dim)
        params = list(module.parameters())

        assert len(params) == 3
        assert hasattr(module, "loc")
        assert hasattr(module, "scale_tril")
        assert isinstance(module.loc, nn.Parameter)
        assert isinstance(module.scale_tril, CholeskyFactor)

    def test_log_prob(
        self, module: InitStateDynamics, optim: torch.optim.Optimizer, obs: Tensor
    ):
        optim.zero_grad()
        with torch.enable_grad():
            logp = module.log_prob(obs)
            loss = -logp.mean()

        assert logp.shape == obs.shape[:-1]
        assert logp.dtype == torch.float32
        assert loss.shape == ()

        loss.backward()
        params = (module.loc, module.scale_tril.ltril, module.scale_tril.pre_diag)
        assert all(list(p.grad is not None for p in params))

        old = [p.clone() for p in params]
        optim.step()
        new = params
        equals = [torch.allclose(x, y) for x, y in zip(old, new)]
        assert not any(equals)

    def test_standard_form(self, module: InitStateDynamics):
        mu, sigma = module.standard_form()
        (mu.sum() + sigma.sum()).backward()

        loc, scale_tril = module.loc, module.scale_tril
        assert torch.allclose(loc.grad, torch.ones([]))
        assert torch.isfinite(scale_tril.ltril.grad).all()
        assert not torch.allclose(scale_tril.ltril.grad, torch.zeros([]))
        assert torch.isfinite(scale_tril.pre_diag.grad).all()
        assert not torch.allclose(scale_tril.pre_diag.grad, torch.zeros([]))

    def test_from_existing(self, init: lqr.GaussInit):
        module = InitStateDynamics.from_existing(init)
        assert all(nt.allclose(a, b) for a, b in zip(init, module.standard_form()))
