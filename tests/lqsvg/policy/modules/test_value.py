# pylint:disable=invalid-name
from __future__ import annotations

from typing import Union

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import Quadratic
from lqsvg.envs.lqr.utils import pack_obs, random_normal_vector, random_spd_matrix
from lqsvg.policy.modules.value import QuadQValue, QuadVValue
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.utils import default_generator_seed

batch_shape = standard_fixture([(), (1,), (4,)], "BatchShape")


@pytest.fixture()
def vvalue_params(n_state: int, horizon: int, seed: int) -> Quadratic:
    V = random_spd_matrix(size=n_state, horizon=horizon + 1, rng=seed)
    v = random_normal_vector(size=n_state, horizon=horizon + 1, rng=seed)
    c = random_normal_vector(size=1, horizon=horizon + 1, rng=seed).squeeze("R")
    return V, v, c


@pytest.fixture()
def qvalue_params(n_state: int, n_ctrl: int, horizon: int, seed: int) -> Quadratic:
    n_tau = n_state + n_ctrl
    Q = random_spd_matrix(size=n_tau, horizon=horizon, rng=seed)
    q = random_normal_vector(size=n_tau, horizon=horizon, rng=seed)
    c = random_normal_vector(size=1, horizon=horizon, rng=seed).squeeze("R")
    return Q, q, c


@pytest.fixture()
def obs(n_state: int, horizon: int, batch_shape: tuple[int, ...], seed: int) -> Tensor:
    with default_generator_seed(seed):
        state = nt.vector(torch.randn(batch_shape + (n_state,)))
        time = nt.vector(
            torch.randint_like(nt.unnamed(state[..., :1]), low=0, high=horizon)
        )
        # noinspection PyTypeChecker
        return pack_obs(state, time)


@pytest.fixture()
def last_obs(
    n_state: int, horizon: int, batch_shape: tuple[int, ...], seed: int
) -> Tensor:
    with default_generator_seed(seed):
        state = nt.vector(torch.randn(batch_shape + (n_state,)))
        time = nt.vector(torch.full_like(state[..., :1], fill_value=horizon))
        # noinspection PyTypeChecker
        return pack_obs(state, time)


@pytest.fixture()
def act(n_state: int, n_ctrl: int, batch_shape: tuple[int, ...], seed: int) -> Tensor:
    del n_state
    with default_generator_seed(seed):
        return nt.vector(torch.randn(batch_shape + (n_ctrl,)))


def check_state_val_and_backprop(vvalue: QuadVValue, obs: Tensor):
    obs.requires_grad_(True)
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


def check_quadratic_parameters(
    module: Union[QuadVValue, QuadQValue], quadratic: Quadratic
):
    quad, linear, const = quadratic
    assert nt.allclose(module.quad, quad)
    assert nt.allclose(module.linear, linear)
    assert nt.allclose(module.const, const)


def test_quadvvalue(
    vvalue_params: Quadratic,
    obs: Tensor,
    last_obs: Tensor,
    n_state: int,
    horizon: int,
):
    vvalue = QuadVValue(vvalue_params)
    assert vvalue.n_state == n_state
    assert vvalue.horizon == horizon
    check_quadratic_parameters(vvalue, vvalue_params)

    check_state_val_and_backprop(vvalue, obs)
    check_state_val_and_backprop(vvalue, last_obs)


def test_quadqvalue(
    qvalue_params: Quadratic,
    obs: Tensor,
    act: Tensor,
    n_state: int,
    n_ctrl: int,
    horizon: int,
):
    # pylint:disable=too-many-arguments
    qvalue = QuadQValue(qvalue_params)
    assert qvalue.n_tau == n_state + n_ctrl
    assert qvalue.horizon == horizon
    check_quadratic_parameters(qvalue, qvalue_params)

    obs.requires_grad_(True)
    act.requires_grad_(True)
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
def vvalue_other(n_state: int, horizon: int, seed: int) -> Quadratic:
    seed = seed + 1
    V = random_spd_matrix(size=n_state, horizon=horizon + 1, rng=seed)
    v = random_normal_vector(size=n_state, horizon=horizon + 1, rng=seed)
    c = random_normal_vector(size=1, horizon=horizon + 1, rng=seed).squeeze("R")
    return V, v, c


@pytest.fixture()
def qvalue_other(n_state: int, n_ctrl: int, horizon: int, seed: int) -> Quadratic:
    seed = seed + 1
    n_tau = n_state + n_ctrl
    Q = random_spd_matrix(size=n_tau, horizon=horizon, rng=seed)
    q = random_normal_vector(size=n_tau, horizon=horizon, rng=seed)
    c = random_normal_vector(size=1, horizon=horizon, rng=seed).squeeze("R")
    return Q, q, c


def test_vvalue_update(vvalue_params: Quadratic, vvalue_other: Quadratic):
    vvalue = QuadVValue(vvalue_params)
    before = [p.clone() for p in vvalue.parameters()]
    vvalue.update(vvalue_other)
    after = [p.clone() for p in vvalue.parameters()]

    allclose_parameters = [torch.allclose(b, a) for b, a in zip(before, after)]
    allclose_inputs = [nt.allclose(a, b) for a, b in zip(vvalue_params, vvalue_other)]
    assert all(allclose_parameters) == all(allclose_inputs)


def test_qvalue_update(qvalue_params: Quadratic, qvalue_other: Quadratic):
    qvalue = QuadQValue(qvalue_params)
    before = [p.clone() for p in qvalue.parameters()]
    qvalue.update(qvalue_other)
    after = [p.clone() for p in qvalue.parameters()]

    allclose_parameters = [torch.allclose(b, a) for b, a in zip(before, after)]
    allclose_inputs = [nt.allclose(a, b) for a, b in zip(qvalue_params, qvalue_other)]
    assert all(allclose_parameters) == all(allclose_inputs)
