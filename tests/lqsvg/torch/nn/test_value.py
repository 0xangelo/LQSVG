# pylint:disable=invalid-name
from __future__ import annotations

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import Linear, LinSDynamics, QuadCost, Quadratic
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics, make_quadcost
from lqsvg.torch.nn.value import QuadQValue, QuadraticMixin, QuadVValue
from lqsvg.torch.random import random_normal_vector, random_spd_matrix


def check_quadratic_parameters(module: QuadraticMixin, quadratic: Quadratic):
    quad, linear, const = quadratic
    assert nt.allclose(module.quad, quad)
    assert nt.allclose(module.linear, linear)
    assert nt.allclose(module.const, const)


@pytest.fixture
def policy(n_state: int, n_ctrl: int, horizon: int) -> Linear:
    K = torch.rand((horizon, n_ctrl, n_state))
    k = torch.rand((horizon, n_ctrl))
    K, k = nt.horizon(nt.matrix(K), nt.vector(k))
    return K, k


@pytest.fixture
def dynamics(n_state: int, n_ctrl: int, horizon: int, seed: int) -> LinSDynamics:
    lin = make_lindynamics(n_state, n_ctrl, horizon, rng=seed)
    dyn = make_linsdynamics(lin, rng=seed)
    return dyn


@pytest.fixture
def cost(n_state: int, n_ctrl: int, horizon: int, seed: int) -> QuadCost:
    return make_quadcost(n_state, n_ctrl, horizon, linear=False, rng=seed)


# noinspection PyMethodMayBeStatic
class TestQuadVValue:
    @pytest.fixture
    def vvalue(self, n_state: int, horizon: int) -> QuadVValue:
        return QuadVValue(n_state, horizon)

    def check_val_backprop(self, vvalue: QuadVValue, obs: Tensor):
        assert obs.grad is None

        val = vvalue(obs)
        assert torch.is_tensor(val)
        assert val.shape == obs.shape[:-1]
        assert val.dtype == obs.dtype
        assert torch.isfinite(val).all()

        vvalue.zero_grad()
        val.mean().backward()
        assert obs.grad is not None
        assert not nt.allclose(obs.grad, torch.zeros_like(obs))

    def test_call(
        self,
        vvalue: QuadVValue,
        obs: Tensor,
        last_obs: Tensor,
        n_state: int,
        horizon: int,
    ):
        # pylint:disable=too-many-arguments
        assert vvalue.n_state == n_state
        assert vvalue.horizon == horizon

        self.check_val_backprop(vvalue, obs)
        self.check_val_backprop(vvalue, last_obs)

    def test_standard_form(self, vvalue: QuadVValue):
        V, v, c = vvalue.standard_form()

        (V.sum() + v.sum() + c.sum()).backward()
        for p in vvalue.parameters():
            assert torch.allclose(torch.ones_like(p), p.grad)

    @pytest.fixture()
    def params(self, n_state: int, horizon: int, seed: int) -> Quadratic:
        V = random_spd_matrix(size=n_state, horizon=horizon + 1, rng=seed)
        v = random_normal_vector(size=n_state, horizon=horizon + 1, rng=seed)
        c = random_normal_vector(size=1, horizon=horizon + 1, rng=seed).squeeze("R")
        return V, v, c

    def test_copy_(self, vvalue: QuadVValue, params: Quadratic):
        old_params = tuple(x.clone() for x in vvalue.standard_form())
        before = [p.clone() for p in vvalue.parameters()]
        vvalue.copy_(params)
        after = [p.clone() for p in vvalue.parameters()]

        allclose_parameters = [torch.allclose(b, a) for b, a in zip(before, after)]
        allclose_quadratics = [nt.allclose(a, b) for a, b in zip(params, old_params)]
        assert all(allclose_parameters) == all(allclose_quadratics)

    def test_from_existing(self, params: Quadratic, obs: Tensor):
        for par in params:
            par.requires_grad_(True)

        module = QuadVValue.from_existing(params)
        check_quadratic_parameters(module, params)

        val = module(obs)
        val.sum().backward()
        for par in params:
            assert par.grad is None

    def test_from_policy(
        self, policy: Linear, dynamics: LinSDynamics, cost: QuadCost, obs: Tensor
    ):
        module = QuadVValue.from_policy(policy, dynamics, cost)
        assert isinstance(module, QuadVValue)
        val = module(obs)
        assert val.le(0).all()


class TestQuadQValue:
    @pytest.fixture
    def qvalue(self, n_state: int, n_ctrl: int, horizon: int) -> QuadQValue:
        return QuadQValue(n_state + n_ctrl, horizon)

    def test_call(
        self,
        qvalue: QuadQValue,
        obs: Tensor,
        act: Tensor,
        n_state: int,
        n_ctrl: int,
        horizon: int,
    ):
        # pylint:disable=too-many-arguments
        assert qvalue.n_tau == n_state + n_ctrl
        assert qvalue.horizon == horizon

        val = qvalue(obs, act)
        assert torch.is_tensor(val)
        assert val.shape == obs.shape[:-1] == act.shape[:-1]
        assert val.dtype == obs.dtype == act.dtype
        assert torch.isfinite(val).all()

        val.mean().backward()
        assert obs.grad is not None
        assert not nt.allclose(obs.grad, torch.zeros_like(obs))
        assert act.grad is not None
        assert not nt.allclose(act.grad, torch.zeros_like(act))

    def test_terminal_value(self, qvalue: QuadQValue, last_obs: Tensor, act: Tensor):
        val = qvalue(last_obs, act)
        assert torch.is_tensor(val)
        assert val.shape == last_obs.shape[:-1] == act.shape[:-1]
        assert val.dtype == last_obs.dtype == act.dtype
        assert nt.allclose(val, torch.zeros_like(val))

        val.mean().backward()
        assert last_obs.grad is not None and act.grad is not None
        assert torch.allclose(last_obs.grad, torch.zeros_like(last_obs.grad))
        assert torch.allclose(act.grad, torch.zeros_like(act.grad))

    def test_standard_form(self, qvalue: QuadQValue):
        Q, q, c = qvalue.standard_form()

        (Q.sum() + q.sum() + c.sum()).backward()
        for p in qvalue.parameters():
            assert torch.allclose(torch.ones_like(p), p.grad)

    @pytest.fixture()
    def params(self, n_state: int, n_ctrl: int, horizon: int, seed: int) -> Quadratic:
        n_tau = n_state + n_ctrl
        Q = random_spd_matrix(size=n_tau, horizon=horizon, rng=seed)
        q = random_normal_vector(size=n_tau, horizon=horizon, rng=seed)
        c = random_normal_vector(size=1, horizon=horizon, rng=seed).squeeze("R")
        return Q, q, c

    def test_copy_(self, qvalue: QuadQValue, params: Quadratic):
        old_params = tuple(x.clone() for x in qvalue.standard_form())
        before = [p.clone() for p in qvalue.parameters()]
        qvalue.copy_(params)
        after = [p.clone() for p in qvalue.parameters()]

        allclose_parameters = [torch.allclose(b, a) for b, a in zip(before, after)]
        allclose_quadratics = [nt.allclose(a, b) for a, b in zip(params, old_params)]
        assert all(allclose_parameters) == all(allclose_quadratics)

    def test_from_existing(self, params: Quadratic, obs: Tensor, act: Tensor):
        for par in params:
            par.requires_grad_(True)

        module = QuadQValue.from_existing(params)
        check_quadratic_parameters(module, params)

        val = module(obs, act)
        val.sum().backward()
        for par in params:
            assert par.grad is None

    def test_from_policy(
        self,
        policy: Linear,
        dynamics: LinSDynamics,
        cost: QuadCost,
        obs: Tensor,
        act: Tensor,
    ):
        # pylint:disable=too-many-arguments
        module = QuadQValue.from_policy(policy, dynamics, cost)
        assert isinstance(module, QuadQValue)
        val = module(obs, act)
        assert val.le(0).all()
