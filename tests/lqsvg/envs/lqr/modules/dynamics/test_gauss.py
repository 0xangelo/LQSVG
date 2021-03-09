# pylint:disable=unsubscriptable-object
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.modules.dynamics.gauss import InitStateDynamics


@pytest.fixture(params=range(4), ids=lambda x: f"Dim:{x+1}")
def dim(request) -> int:
    return request.param + 1


@pytest.fixture
def init(dim: int) -> lqr.GaussInit:
    return lqr.GaussInit(nt.vector(torch.randn(dim)), nt.matrix(torch.eye(dim)))


@pytest.fixture
def module_fn() -> callable[[lqr.GaussInit], InitStateDynamics]:
    return InitStateDynamics


def test_init(
    module_fn: callable[[lqr.GaussInit], InitStateDynamics], init: lqr.GaussInit
):
    module = module_fn(init)
    params = list(module.parameters())

    assert len(params) == 3
    assert hasattr(module, "loc")
    assert hasattr(module, "ltril")
    assert hasattr(module, "pre_diag")
    assert isinstance(module.loc, nn.Parameter)
    assert isinstance(module.ltril, nn.Parameter)
    assert isinstance(module.pre_diag, nn.Parameter)


@pytest.fixture
def module(
    module_fn: callable[[lqr.GaussInit], InitStateDynamics], init: lqr.GaussInit
) -> InitStateDynamics:
    return module_fn(init)


@pytest.fixture
def state(dim: int) -> Tensor:
    return torch.cat((torch.randn(dim), torch.zeros(1)), dim=-1)


@pytest.fixture
def optim(module: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(module.parameters(), lr=1e-3)


def test_log_prob(
    module: InitStateDynamics, optim: torch.optim.Optimizer, state: Tensor
):
    optim.zero_grad()
    with torch.enable_grad():
        logp = module.log_prob(state)
        loss = -logp.mean()

    assert logp.shape == state.shape[:-1]
    assert logp.dtype == torch.float32
    assert loss.shape == ()

    loss.backward()
    assert module.loc.grad is not None
    assert module.ltril.grad is not None
    assert module.pre_diag.grad is not None

    old = [x.clone() for x in (module.loc, module.ltril, module.pre_diag)]
    optim.step()
    new = (module.loc, module.ltril, module.pre_diag)
    equals = [torch.allclose(x, y) for x, y in zip(old, new)]
    assert not any(equals)
