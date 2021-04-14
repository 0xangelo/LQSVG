import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.policy.modules import TVLinearPolicy


@pytest.fixture
def module(n_state: int, n_ctrl: int, horizon: int) -> TVLinearPolicy:
    return TVLinearPolicy(n_state, n_ctrl, horizon)


def test_normal_call(module: TVLinearPolicy, obs: Tensor, n_ctrl: int):
    act = module(obs)

    assert torch.is_tensor(act)
    assert torch.isfinite(act).all()
    assert act.names == obs.names
    assert act.size("R") == n_ctrl

    act.sum().backward()
    assert obs.grad is not None
    assert not torch.allclose(obs.grad, torch.zeros_like(obs.grad))
    assert torch.isfinite(obs.grad).all()
    grads = [p.grad for p in module.parameters()]
    assert all(list(g is not None for g in grads))
    assert all(list(not torch.allclose(g, torch.zeros_like(g)) for g in grads))
    assert all(list(torch.isfinite(g).all() for g in grads))


def test_terminal_call(module: TVLinearPolicy, last_obs: Tensor, n_ctrl: int):
    act = module(last_obs)

    assert nt.allclose(act, torch.zeros_like(act))
    assert torch.is_tensor(act)
    assert torch.isfinite(act).all()
    assert act.names == last_obs.names
    assert act.size("R") == n_ctrl

    act.sum().backward()
    assert last_obs.grad is not None
    assert torch.allclose(last_obs.grad, torch.zeros_like(last_obs.grad))
    grads = [p.grad for p in module.parameters()]
    assert all(list(g is not None for g in grads))
    assert all(list(torch.allclose(g, torch.zeros_like(g)) for g in grads))
