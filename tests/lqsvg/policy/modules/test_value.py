# pylint:disable=invalid-name
from __future__ import annotations

from typing import Union

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import Quadratic
from lqsvg.envs.lqr.utils import random_normal_vector, random_spd_matrix
from lqsvg.policy.modules.value import QuadQValue, QuadVValue


def check_quadratic_parameters(
    module: Union[QuadVValue, QuadQValue], quadratic: Quadratic
):
    quad, linear, const = quadratic
    assert nt.allclose(module.quad, quad)
    assert nt.allclose(module.linear, linear)
    assert nt.allclose(module.const, const)


# noinspection PyMethodMayBeStatic
class TestQuadVValue:
    @pytest.fixture()
    def params(self, n_state: int, horizon: int, seed: int) -> Quadratic:
        V = random_spd_matrix(size=n_state, horizon=horizon + 1, rng=seed)
        v = random_normal_vector(size=n_state, horizon=horizon + 1, rng=seed)
        c = random_normal_vector(size=1, horizon=horizon + 1, rng=seed).squeeze("R")
        return V, v, c

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
        params: Quadratic,
        obs: Tensor,
        last_obs: Tensor,
        n_state: int,
        horizon: int,
    ):
        # pylint:disable=too-many-arguments
        vvalue = QuadVValue(params)
        assert vvalue.n_state == n_state
        assert vvalue.horizon == horizon
        check_quadratic_parameters(vvalue, params)

        self.check_val_backprop(vvalue, obs)
        self.check_val_backprop(vvalue, last_obs)

    @pytest.fixture()
    def other_params(self, n_state: int, horizon: int, seed: int) -> Quadratic:
        seed = seed + 1
        V = random_spd_matrix(size=n_state, horizon=horizon + 1, rng=seed)
        v = random_normal_vector(size=n_state, horizon=horizon + 1, rng=seed)
        c = random_normal_vector(size=1, horizon=horizon + 1, rng=seed).squeeze("R")
        return V, v, c

    def test_update(self, params: Quadratic, other_params: Quadratic):
        vvalue = QuadVValue(params)
        before = [p.clone() for p in vvalue.parameters()]
        vvalue.update(other_params)
        after = [p.clone() for p in vvalue.parameters()]

        allclose_parameters = [torch.allclose(b, a) for b, a in zip(before, after)]
        allclose_inputs = [nt.allclose(a, b) for a, b in zip(params, other_params)]
        assert all(allclose_parameters) == all(allclose_inputs)


class TestQuadQValue:
    @pytest.fixture()
    def params(self, n_state: int, n_ctrl: int, horizon: int, seed: int) -> Quadratic:
        n_tau = n_state + n_ctrl
        Q = random_spd_matrix(size=n_tau, horizon=horizon, rng=seed)
        q = random_normal_vector(size=n_tau, horizon=horizon, rng=seed)
        c = random_normal_vector(size=1, horizon=horizon, rng=seed).squeeze("R")
        return Q, q, c

    def test_call(
        self,
        params: Quadratic,
        obs: Tensor,
        act: Tensor,
        n_state: int,
        n_ctrl: int,
        horizon: int,
    ):
        # pylint:disable=too-many-arguments
        qvalue = QuadQValue(params)
        assert qvalue.n_tau == n_state + n_ctrl
        assert qvalue.horizon == horizon
        check_quadratic_parameters(qvalue, params)

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

    @pytest.fixture()
    def other_params(
        self, n_state: int, n_ctrl: int, horizon: int, seed: int
    ) -> Quadratic:
        seed = seed + 1
        n_tau = n_state + n_ctrl
        Q = random_spd_matrix(size=n_tau, horizon=horizon, rng=seed)
        q = random_normal_vector(size=n_tau, horizon=horizon, rng=seed)
        c = random_normal_vector(size=1, horizon=horizon, rng=seed).squeeze("R")
        return Q, q, c

    def test_update(self, params: Quadratic, other_params: Quadratic):
        qvalue = QuadQValue(params)
        before = [p.clone() for p in qvalue.parameters()]
        qvalue.update(other_params)
        after = [p.clone() for p in qvalue.parameters()]

        allclose_parameters = [torch.allclose(b, a) for b, a in zip(before, after)]
        allclose_inputs = [nt.allclose(a, b) for a, b in zip(params, other_params)]
        assert all(allclose_parameters) == all(allclose_inputs)