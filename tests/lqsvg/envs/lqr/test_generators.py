from typing import Union

import pytest
import torch
from torch import Tensor

from lqsvg.envs.lqr.generators import box_ddp_random_lqr
from lqsvg.envs.lqr.generators import make_lqg
from lqsvg.envs.lqr.generators import make_lqr
from lqsvg.envs.lqr.generators import make_lqr_linear_navigation
from lqsvg.envs.lqr.generators import stack_lqs
from lqsvg.envs.lqr.types import LinDynamics
from lqsvg.envs.lqr.types import LinSDynamics
from lqsvg.envs.lqr.types import QuadCost

from .utils import standard_fixture


stationary = standard_fixture((True, False), "Stationary")


@pytest.fixture
def timestep():
    return 0.01


@pytest.fixture
def ctrl_coeff():
    return 0.1


def named_allclose(tensora: Tensor, tensorb: Tensor) -> bool:
    return torch.allclose(tensora.rename(None), tensorb.rename(None))


def check_system(
    dynamics: Union[LinDynamics, LinSDynamics],
    cost: QuadCost,
    horizon: int,
    n_state: int,
    n_ctrl: int,
    stationary: bool = False,
):
    # pylint:disable=invalid-name,too-many-arguments
    tau_size = n_state + n_ctrl

    assert all([torch.is_tensor(t) for x in (dynamics, cost) for t in x])

    if isinstance(dynamics, LinDynamics):
        F, f = dynamics
        W = None
    else:
        F, f, W = dynamics
    assert F.shape == (horizon, n_state, tau_size)
    assert f.shape == (horizon, n_state)
    if W is not None:
        assert W.shape == (horizon, n_state, n_state)
        eigval_W, _ = torch.symeig(W.rename(None))
        assert eigval_W.ge(0).all()

    C, c = cost
    assert C.shape == (horizon, tau_size, tau_size)
    assert c.shape == (horizon, tau_size)
    eigval, _ = torch.symeig(C.rename(None))
    assert eigval.ge(0).all()

    if horizon > 1:
        assert stationary == named_allclose(F[0], F[-1])
        assert stationary == named_allclose(f[0], f[-1])
        assert W is None or stationary == named_allclose(W[0], W[-1])
        assert stationary == named_allclose(C[0], C[-1])
        assert stationary == named_allclose(c[0], c[-1])


def test_box_ddp_random_lqr(timestep, ctrl_coeff, horizon, seed):
    dynamics, cost, _ = box_ddp_random_lqr(
        timestep, ctrl_coeff, horizon, np_random=seed
    )
    n_state = dynamics.F.shape[-2]
    n_ctrl = dynamics.F.shape[-1] - n_state
    check_system(dynamics, cost, horizon, n_state, n_ctrl, stationary=True)


def test_make_lqr(n_state, n_ctrl, horizon, stationary, seed):
    dynamics, cost = make_lqr(
        state_size=n_state,
        ctrl_size=n_ctrl,
        horizon=horizon,
        stationary=stationary,
        np_random=seed,
    )
    check_system(dynamics, cost, horizon, n_state, n_ctrl, stationary=stationary)


def test_make_lqg(n_state, n_ctrl, horizon, stationary, seed):
    dynamics, cost = make_lqg(
        state_size=n_state,
        ctrl_size=n_ctrl,
        horizon=horizon,
        stationary=stationary,
        np_random=seed,
    )
    check_system(dynamics, cost, horizon, n_state, n_ctrl, stationary)


@pytest.fixture
def goal():
    return 0.5, 1.0


@pytest.fixture
def beta(ctrl_coeff):
    return ctrl_coeff


def test_make_lqr_linear_navigation(goal, beta, horizon):
    dynamics, cost, _ = make_lqr_linear_navigation(goal, beta, horizon)
    check_system(dynamics, cost, horizon, 2, 2, stationary=True)


def test_stack_lqs(n_state, n_ctrl, horizon, seed):
    system = make_lqr(n_state, n_ctrl, horizon, np_random=seed)
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


# Batched LQs =================================================================
n_batch = standard_fixture((None, 1, 10), "Batch")


def test_batched_lqgs(n_state, n_ctrl, horizon, stationary, n_batch):
    dynamics, cost = make_lqg(
        state_size=n_state,
        ctrl_size=n_ctrl,
        horizon=horizon,
        stationary=stationary,
        n_batch=n_batch,
    )
    assert isinstance(dynamics, LinSDynamics)
    assert isinstance(cost, QuadCost)

    mat_names = tuple("H B R C".split()) if n_batch else tuple("H R C".split())
    vec_names = tuple("H B R".split()) if n_batch else tuple("H R".split())

    assert dynamics.F.names == mat_names
    assert dynamics.f.names == vec_names
    assert dynamics.W.names == mat_names

    assert cost.C.names == mat_names
    assert cost.c.names == vec_names

    n_tau = n_state + n_ctrl
    sample_shape = (horizon,) + ((n_batch,) if n_batch else ())

    assert dynamics.F.shape == sample_shape + (n_state, n_tau)
    assert dynamics.f.shape == sample_shape + (n_state,)
    assert dynamics.W.shape == sample_shape + (n_state, n_state)

    assert cost.C.shape == sample_shape + (n_tau, n_tau)
    assert cost.c.shape == sample_shape + (n_tau,)
