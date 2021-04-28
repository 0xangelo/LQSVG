# pylint:disable=invalid-name
from __future__ import annotations

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.np_util import RNG
from lqsvg.policy.modules import TVLinearFeedback, TVLinearPolicy
from lqsvg.testing.fixture import standard_fixture

frozen = standard_fixture((True, False), "Frozen")


@pytest.fixture
def dynamics(n_state: int, n_ctrl: int, horizon: int, rng: RNG) -> lqr.LinSDynamics:
    return make_linsdynamics(
        make_lindynamics(
            n_state,
            n_ctrl,
            horizon,
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            rng=rng,
        ),
        stationary=True,
        rng=rng,
    )


class TestTVLinearFeedBack:
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int) -> TVLinearFeedback:
        return TVLinearFeedback(n_state, n_ctrl, horizon)

    @pytest.fixture
    def linear(self, n_state: int, n_ctrl: int, horizon: int) -> lqr.Linear:
        K, k = torch.randn(horizon, n_ctrl, n_state), torch.randn(horizon, n_ctrl)
        K, k = nt.horizon(nt.matrix(K), nt.vector(k))
        return K, k

    def test_from_existing(self, linear: lqr.Linear):
        module = TVLinearFeedback.from_existing(linear)
        params = module.gains()
        assert all(list(nt.allclose(p, l) for p, l in zip(params, linear)))

    def test_detach_linear(self, module: TVLinearFeedback, linear: lqr.Linear):
        before = tuple(x.clone() for x in linear)
        module.copy_(linear)
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
        frozen: bool,
        batch_shape: tuple[int, ...],
        n_ctrl: int,
    ):
        # pylint:disable=too-many-arguments
        act = module(obs, frozen=frozen)
        assert torch.is_tensor(act)
        assert torch.isfinite(act).all()
        assert act.size("R") == n_ctrl
        assert tuple(s for s, n in zip(act.shape, act.names) if n != "R") == batch_shape

        assert act.grad_fn is not None
        module.zero_grad(set_to_none=True)
        act.sum().backward()
        assert obs.grad is not None
        assert not torch.allclose(obs.grad, torch.zeros(()))
        assert frozen == all(list(p.grad is None for p in module.parameters()))


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
        assert not torch.allclose(obs.grad, torch.zeros(()))
        assert torch.isfinite(obs.grad).all()
        grads = [p.grad for p in module.parameters()]
        assert all(list(g is not None for g in grads))
        assert all(list(not torch.allclose(g, torch.zeros(())) for g in grads))
        assert all(list(torch.isfinite(g).all() for g in grads))

    def test_terminal_call(self, module: TVLinearPolicy, last_obs: Tensor, n_ctrl: int):
        act = module(last_obs)

        assert nt.allclose(act, torch.zeros(()))
        assert torch.is_tensor(act)
        assert torch.isfinite(act).all()
        assert act.names == last_obs.names
        assert act.size("R") == n_ctrl

        act.sum().backward()
        assert last_obs.grad is not None
        assert torch.allclose(last_obs.grad, torch.zeros(()))
        grads = [p.grad for p in module.parameters()]
        assert all(list(g is not None for g in grads))
        assert all(list(torch.allclose(g, torch.zeros(())) for g in grads))

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

    def test_frozen(self, module: TVLinearPolicy, obs: Tensor, n_ctrl: int):
        act = module.frozen(obs)

        assert torch.is_tensor(act)
        assert torch.isfinite(act).all()
        assert act.names == obs.names
        # noinspection PyArgumentList
        assert act.size("R") == n_ctrl

        module.zero_grad(set_to_none=True)
        act.sum().backward()
        assert obs.grad is not None
        assert not torch.allclose(obs.grad, torch.zeros(()))
        assert torch.isfinite(obs.grad).all()
        grads = [p.grad for p in module.parameters()]
        assert all(list(g is None for g in grads))

    def test_standard_form(self, module: TVLinearPolicy):
        K, k = module.standard_form()

        (K.sum() + k.sum()).backward()
        assert torch.allclose(module.K.grad, torch.ones_like(module.K.grad))
        assert torch.allclose(module.k.grad, torch.ones_like(module.k.grad))

    def test_stabilize_(
        self, module: TVLinearPolicy, dynamics: lqr.LinSDynamics, seed: int
    ):
        module.stabilize_(dynamics, rng=seed)
        K, k = module.K.clone(), module.k.clone()

        module.stabilize_(dynamics, rng=seed)
        assert torch.allclose(module.K, K)
        assert torch.allclose(module.k, k)
