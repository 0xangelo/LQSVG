# pylint:disable=invalid-name
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.utils import stationary_dynamics_factors
from lqsvg.policy.modules import TVLinearFeedback, TVLinearPolicy
from lqsvg.policy.modules.policy import stabilizing_gain


@pytest.fixture
def dynamics(n_state: int, n_ctrl: int, horizon: int, seed: int) -> lqr.LinSDynamics:
    dyn = make_lindynamics(
        n_state,
        n_ctrl,
        horizon,
        stationary=True,
        passive_eigval_range=(0.5, 1.5),
        controllable=True,
        bias=False,
        rng=seed,
    )
    return make_linsdynamics(dyn, stationary=True, rng=seed)


# noinspection PyArgumentList
def test_stabilizing_gain(dynamics: lqr.LinSDynamics, n_state: int, n_ctrl: int):
    gain = stabilizing_gain(dynamics)
    assert torch.is_tensor(gain)
    assert torch.isfinite(gain).all()
    assert gain.size("R") == n_ctrl
    assert gain.size("C") == n_state

    A, B = (x.numpy() for x in stationary_dynamics_factors(dynamics))
    K = gain.numpy()
    eigval, _ = np.linalg.eig(A + B @ K)
    assert np.all(np.abs(eigval) < 1.0)


class TestTVLinearPolicy:
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int) -> TVLinearPolicy:
        return TVLinearPolicy(n_state, n_ctrl, horizon)

    def test_normal_call(self, module: TVLinearPolicy, obs: Tensor, n_ctrl: int):
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

    def test_terminal_call(self, module: TVLinearPolicy, last_obs: Tensor, n_ctrl: int):
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

    def test_mixed_call(self, module: TVLinearPolicy, mix_obs: Tensor, n_ctrl: int):
        act = module(mix_obs)

        assert torch.is_tensor(act)
        assert torch.isfinite(act).all()
        assert act.names == mix_obs.names
        assert act.size("R") == n_ctrl

        act.sum().backward()
        assert mix_obs.grad is not None
        assert torch.isfinite(mix_obs.grad).all()
        grads = [p.grad for p in module.parameters()]
        assert all(list(g is not None for g in grads))
        assert all(list(torch.isfinite(g).all() for g in grads))


class TestTVLinearFeedBack:
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int) -> TVLinearFeedback:
        return TVLinearFeedback(n_state, n_ctrl, horizon)

    @pytest.fixture
    def linear(self, n_state: int, n_ctrl: int, horizon: int) -> lqr.Linear:
        K, k = torch.randn(horizon, n_ctrl, n_state), torch.randn(horizon, n_ctrl)
        K = nt.horizon(nt.matrix(K))
        k = nt.horizon(nt.vector(k))
        return K, k

    def test_from_existing(self, linear: lqr.Linear):
        module = TVLinearFeedback.from_existing(linear)
        params = module.gains()
        assert all(list(nt.allclose(p, l) for p, l in zip(params, linear)))

    def test_detach_linear(self, module: TVLinearFeedback, linear: lqr.Linear):
        before = tuple(x.clone() for x in linear)
        module.copy(linear)
        for par in module.parameters():
            par.data.add_(1.0)

        assert all(list(nt.allclose(b, a) for b, a in zip(before, linear)))

    def test_gains(self, module: TVLinearFeedback):
        for par in module.parameters():
            par.grad = torch.randn_like(par)

        K, k = module.gains()
        assert nt.allclose(K, module.K)
        assert nt.allclose(k, module.k)
        assert K.grad is not None
        assert k.grad is not None
        assert nt.allclose(K.grad, module.K.grad)
        assert nt.allclose(k.grad, module.k.grad)

    def test_call(
        self,
        module: TVLinearFeedback,
        obs: Tensor,
        batch_shape: tuple[int, ...],
        n_ctrl: int,
    ):
        act = module(obs)
        assert torch.is_tensor(act)
        assert torch.isfinite(act).all()
        assert act.size("R") == n_ctrl
        assert tuple(s for s, n in zip(act.shape, act.names) if n != "R") == batch_shape

        assert act.grad_fn is not None
        act.sum().backward()
        assert obs.grad is not None
        assert not torch.allclose(obs.grad, torch.zeros_like(obs.grad))
