from __future__ import annotations

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.torch.nn.reward import QuadraticReward


class TestQuadraticReward:
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int) -> QuadraticReward:
        return QuadraticReward(n_state + n_ctrl, horizon)

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

        s_grad, t_grad = lqr.unpack_obs(nt.vector(obs.grad))
        assert not nt.allclose(s_grad, torch.zeros_like(s_grad))
        assert torch.isfinite(s_grad).all()
        assert nt.allclose(t_grad, torch.zeros_like(t_grad))

        assert not nt.allclose(act.grad, torch.zeros_like(act))
        assert torch.isfinite(act.grad).all()

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
