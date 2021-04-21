from __future__ import annotations

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.modules.reward.quadratic import QuadraticReward
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs
from lqsvg.testing.fixture import standard_fixture

batch_shape = standard_fixture([(), (1,), (4,), (2, 2)], "BatchShape")


class TestQuadraticReward:
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int) -> QuadraticReward:
        return QuadraticReward(n_state + n_ctrl, horizon)

    @pytest.fixture
    def state(self, n_state: int, batch_shape: tuple[int, ...]) -> Tensor:
        return nt.vector(torch.randn(batch_shape + (n_state,)))

    @pytest.fixture
    def obs(self, state: Tensor, horizon: int, batch_shape: tuple[int, ...]) -> Tensor:
        time = torch.randint(low=0, high=horizon, size=batch_shape + (1,))
        return pack_obs(state, nt.vector(time)).requires_grad_(True)

    @pytest.fixture
    def act(self, n_ctrl: int, batch_shape) -> Tensor:
        return nt.vector(torch.randn(batch_shape + (n_ctrl,))).requires_grad_(True)

    def test_init(
        self, module: QuadraticReward, n_state: int, n_ctrl: int, horizon: int
    ):
        assert module.n_tau == n_state + n_ctrl
        assert module.horizon == horizon

    def test_call(self, module: QuadraticReward, obs: Tensor, act: Tensor):
        val = module(obs, act)
        assert torch.is_tensor(val)
        assert torch.isfinite(val).all()

        val.sum().backward()
        assert obs.grad is not None and act.grad is not None

        s_grad, t_grad = unpack_obs(nt.vector(obs.grad))
        assert not nt.allclose(s_grad, torch.zeros_like(s_grad))
        assert torch.isfinite(s_grad).all()
        assert nt.allclose(t_grad, torch.zeros_like(t_grad))

        assert not nt.allclose(act.grad, torch.zeros_like(act))
        assert torch.isfinite(act.grad).all()

    @pytest.fixture
    def last_obs(
        self, state: Tensor, horizon: int, batch_shape: tuple[int, ...]
    ) -> Tensor:
        time = torch.full(batch_shape + (1,), fill_value=horizon, dtype=torch.int)
        return pack_obs(state, nt.vector(time)).requires_grad_(True)

    def test_terminal(self, module, last_obs: Tensor, act: Tensor):
        val = module(last_obs, act)
        assert torch.is_tensor(val)
        assert nt.allclose(val, torch.zeros([]))

        val.sum().backward()
        assert last_obs.grad is not None and act.grad is not None

        assert nt.allclose(last_obs.grad, torch.zeros([]))
        assert nt.allclose(act.grad, torch.zeros([]))

    def test_standard_form(
        self, module: QuadraticReward, n_state: int, n_ctrl: int, horizon: int
    ):
        cost = module.standard_form()
        assert isinstance(cost, lqr.QuadCost)

        n_tau, horizon_ = lqr.dims_from_cost(cost)
        assert n_tau == n_state + n_ctrl
        assert horizon_ == horizon

        (cost.C.sum() + cost.c.sum()).backward()
        for p in module.parameters():
            assert p.grad is not None
            assert nt.allclose(p.grad, torch.ones_like(p))
