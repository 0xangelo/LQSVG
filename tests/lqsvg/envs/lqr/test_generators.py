# pylint:disable=too-many-arguments,invalid-name,unsubscriptable-object
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Type, Union

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.generators import (
    LQGGenerator,
    box_ddp_random_lqr,
    make_lindynamics,
    make_linsdynamics,
    make_lqr,
    make_lqr_linear_navigation,
    make_quadcost,
    stack_lqs,
)
from lqsvg.envs.lqr.types import LinDynamics, LinSDynamics, QuadCost

from .utils import standard_fixture

GeneratorFn = Callable[[], LQGGenerator]

passive_eigval_range = standard_fixture(
    [None, (0.0, 1.0), (0.5, 1.5)], "PassiveEigvals"
)
transition_bias = standard_fixture((True, False), "TransBias")
sample_covariance = standard_fixture((True, False), "SampleCov")
cost_linear = standard_fixture((True, False), "CostLinear")
cost_cross = standard_fixture((True, False), "CostCross")


@pytest.fixture
def generator_cls() -> Type[LQGGenerator]:
    return LQGGenerator


# noinspection PyArgumentList
@pytest.fixture
def generator_fn(
    generator_cls: Type[LQGGenerator],
    n_state: int,
    n_ctrl: int,
    horizon: int,
    seed: int,
) -> GeneratorFn:
    return partial(generator_cls, n_state, n_ctrl, horizon, seed=seed)


# Test LQGGenerator interface ==========================================
def test_generator_init(
    generator_fn: GeneratorFn,
    n_state: int,
    n_ctrl: int,
    horizon: int,
    seed: int,
):
    generator = generator_fn()
    assert generator.n_state == n_state
    assert generator.n_ctrl == n_ctrl
    assert generator.horizon == horizon
    assert generator.seed == seed


n_batch = standard_fixture((None, 1, 4), "NBatch")


def check_generated_dynamics(
    dynamics: Union[LinDynamics, LinSDynamics], generator: LQGGenerator
):
    check_dynamics(
        dynamics,
        generator.n_state,
        generator.n_ctrl,
        generator.horizon,
        stationary=generator.stationary,
        transition_bias=generator.transition_bias,
        sample_covariance=generator.rand_trans_cov,
    )


def check_generated_cost(cost: QuadCost, generator: LQGGenerator):
    check_cost(
        cost,
        generator.n_state,
        generator.n_ctrl,
        generator.horizon,
        stationary=generator.stationary,
        linear=generator.cost_linear,
    )


def test_generator_defaults(generator_fn: GeneratorFn, n_batch: Optional[int]):
    generator = generator_fn()
    dynamics, cost, init = generator(n_batch=n_batch)

    tensors = [t for c in (dynamics, cost, init) for t in c]
    if n_batch is None:
        assert all("B" not in t.names for t in tensors)
    else:
        assert all("B" in t.names for t in tensors)
        assert all(t.size("B") == n_batch for t in tensors)

    check_generated_dynamics(dynamics, generator)
    check_generated_cost(cost, generator)


def test_generator_batch_call(generator_fn: GeneratorFn, n_batch: Optional[int]):
    generator = generator_fn()
    dynamics, cost, init = generator(n_batch=n_batch)

    tensors = [t for c in (dynamics, cost, init) for t in c]
    if n_batch is None:
        assert all("B" not in t.names for t in tensors)
    else:
        assert all("B" in t.names for t in tensors)
        assert all(t.size("B") == n_batch for t in tensors)

    check_generated_dynamics(dynamics, generator)
    check_generated_cost(cost, generator)


def assert_all_tensor(*tensors: Tensor):
    is_tensor = list(map(torch.is_tensor, tensors))
    assert all(is_tensor)


# noinspection PyArgumentList
def assert_row_size(tensor: Tensor, size: int):
    assert tensor.size("R") == size


# noinspection PyArgumentList
def assert_col_size(tensor: Tensor, size: int):
    assert tensor.size("C") == size


# noinspection PyArgumentList
def assert_horizon_len(tensor: Tensor, length: int):
    assert tensor.size("H") == length


def check_dynamics(
    dynamics: Union[LinDynamics, LinSDynamics],
    n_state: int,
    n_ctrl: int,
    horizon: int,
    stationary: bool,
    transition_bias: bool,
    sample_covariance: Optional[bool] = None,
):
    assert_all_tensor(*dynamics)

    if isinstance(dynamics, LinDynamics):
        (F, f), W = dynamics, None
    else:
        F, f, W = dynamics

    assert_horizon_len(F, horizon)
    assert_horizon_len(f, horizon)
    assert_row_size(F, n_state)
    assert_row_size(f, n_state)
    assert_col_size(F, n_state + n_ctrl)
    if W is not None:
        assert_horizon_len(W, horizon)
        assert_row_size(W, n_state)
        assert_col_size(W, n_state)
        eigval_W, _ = torch.symeig(nt.unnamed(W))
        assert eigval_W.ge(0).all()

    if not transition_bias:
        assert nt.allclose(torch.zeros_like(f), f)

    if horizon > 1:
        assert stationary == nt.allclose(F, F.select("H", 0))
        assert not transition_bias or stationary == nt.allclose(f, f.select("H", 0))
        assert (
            W is None
            or not sample_covariance
            or stationary == nt.allclose(W, W.select("H", 0))
        )


def check_cost(
    cost: QuadCost,
    n_state: int,
    n_ctrl: int,
    horizon: int,
    stationary: bool,
    linear: bool,
):
    assert_all_tensor(*cost)
    n_tau = n_state + n_ctrl

    C, c = cost
    assert_horizon_len(C, horizon)
    assert_horizon_len(c, horizon)
    assert_row_size(C, n_tau)
    assert_row_size(c, n_tau)
    assert_col_size(C, n_tau)

    eigval, _ = torch.symeig(nt.unnamed(C))
    assert eigval.ge(0).all()
    assert linear or nt.allclose(c, torch.zeros_like(c))

    if horizon > 1:
        assert stationary == nt.allclose(C, C.select("H", 0))
        assert not linear or stationary == nt.allclose(c, c.select("H", 0))


def test_stationary(generator_fn: GeneratorFn, stationary: bool):
    # noinspection PyArgumentList
    generator = generator_fn(stationary=stationary)
    dynamics, cost, _ = generator()

    check_generated_dynamics(dynamics, generator)
    check_generated_cost(cost, generator)


def test_rand_trans_cov(generator_fn: GeneratorFn, sample_covariance: bool):
    # noinspection PyArgumentList
    generator = generator_fn(rand_trans_cov=sample_covariance)
    dynamics, _, _ = generator()

    check_generated_dynamics(dynamics, generator)


def test_passive_eigval_range(
    generator_fn: GeneratorFn, passive_eigval_range: tuple[float, float]
):
    # noinspection PyArgumentList
    generator = generator_fn(passive_eigval_range=passive_eigval_range)
    dynamics, _, _ = generator()
    check_generated_dynamics(dynamics, generator)


def test_transition_bias(
    generator_fn: GeneratorFn,
    transition_bias: bool,
):
    # noinspection PyArgumentList
    generator = generator_fn(transition_bias=transition_bias)
    dynamics, _, _ = generator()
    check_generated_dynamics(dynamics, generator)


def test_cost_linear(
    generator_fn: GeneratorFn,
    cost_linear: bool,
):
    # noinspection PyArgumentList
    generator = generator_fn(cost_linear=cost_linear)
    _, cost, _ = generator()
    check_generated_cost(cost, generator)


@pytest.mark.slow
def test_make_linsdynamics(
    n_state: int,
    n_ctrl: int,
    horizon: int,
    stationary: bool,
    sample_covariance: bool,
    seed: int,
    passive_eigval_range: tuple[float, float],
    transition_bias: bool,
):
    dynamics = make_lindynamics(
        n_state,
        n_ctrl,
        horizon,
        stationary=stationary,
        passive_eigval_range=passive_eigval_range,
        bias=transition_bias,
        rng=seed,
    )
    dynamics = make_linsdynamics(
        dynamics,
        n_state,
        horizon,
        stationary=stationary,
        sample_covariance=sample_covariance,
        rng=seed,
    )
    check_dynamics(
        dynamics,
        n_state,
        n_ctrl,
        horizon,
        stationary,
        transition_bias=transition_bias,
        sample_covariance=sample_covariance,
    )


@pytest.mark.slow
def test_make_quadcost(
    n_state: int,
    n_ctrl: int,
    horizon: int,
    stationary: bool,
    seed: int,
    cost_linear: bool,
):
    cost = make_quadcost(
        n_state,
        n_ctrl,
        horizon,
        stationary=stationary,
        rng=seed,
        linear=cost_linear,
    )
    check_cost(
        cost, n_state, n_ctrl, horizon, stationary=stationary, linear=cost_linear
    )


@pytest.mark.slow
def test_make_lqr(
    n_state: int,
    n_ctrl: int,
    horizon: int,
    stationary: bool,
    seed: int,
    passive_eigval_range: Optional[tuple[float, float]],
    transition_bias: bool,
    cost_linear: bool,
):
    # pylint:disable=invalid-name,too-many-arguments
    dynamics, cost = make_lqr(
        state_size=n_state,
        ctrl_size=n_ctrl,
        horizon=horizon,
        stationary=stationary,
        rng=seed,
        passive_eigval_range=passive_eigval_range,
        transition_bias=transition_bias,
        cost_linear=cost_linear,
    )
    check_dynamics(
        dynamics,
        n_state,
        n_ctrl,
        horizon,
        stationary=stationary,
        transition_bias=transition_bias,
    )
    check_cost(cost, n_state, n_ctrl, horizon, stationary, cost_linear)


def test_stack_lqs(n_state, n_ctrl, horizon, seed):
    system = make_lqr(n_state, n_ctrl, horizon, rng=seed)
    dynamics, cost = stack_lqs(system)
    assert isinstance(dynamics, LinDynamics)
    assert isinstance(cost, QuadCost)

    mat_names = tuple("H B R C".split())
    vec_names = tuple("H B R".split())

    assert dynamics.F.names == mat_names
    assert dynamics.f.names == vec_names

    assert cost.C.names == mat_names
    assert cost.c.names == vec_names
    assert all(x.size("B") == 1 for y in (dynamics, cost) for x in y)


###############################################################################
# Box-DDP environment
###############################################################################


@pytest.fixture
def timestep():
    return 0.01


@pytest.fixture
def ctrl_coeff():
    return 0.1


def test_box_ddp_random_lqr(timestep, ctrl_coeff, horizon, seed):
    dynamics, cost, _ = box_ddp_random_lqr(timestep, ctrl_coeff, horizon, rng=seed)
    n_state = dynamics.F.shape[-2]
    n_ctrl = dynamics.F.shape[-1] - n_state
    check_dynamics(
        dynamics, n_state, n_ctrl, horizon, stationary=True, transition_bias=False
    )
    check_cost(cost, n_state, n_ctrl, horizon, stationary=True, linear=False)


###############################################################################
# Linear Navigation
###############################################################################


@pytest.fixture
def goal():
    return 0.5, 1.0


@pytest.fixture
def beta(ctrl_coeff):
    return ctrl_coeff


def test_make_lqr_linear_navigation(goal, beta, horizon):
    dynamics, cost, _ = make_lqr_linear_navigation(goal, beta, horizon)
    check_dynamics(dynamics, 2, 2, horizon, stationary=True, transition_bias=False)
    check_cost(cost, 2, 2, horizon, stationary=True, linear=True)
