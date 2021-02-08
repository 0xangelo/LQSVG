"""Random LQR problem generators."""
# pylint:disable=invalid-name
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.np_util import make_spd_matrix
from lqsvg.torch.utils import as_float_tensor

from .named import refine_lqr
from .types import AnyDynamics
from .types import Box
from .types import LinDynamics
from .types import LinSDynamics
from .types import QuadCost
from .utils import np_expand_horizon


def stack_lqs(*systems: Tuple[AnyDynamics, QuadCost]) -> Tuple[AnyDynamics, QuadCost]:
    """Stack several linear quadratic problems into a batched representation.

    Returns dynamics and costs with an additional batch dimension.
    """
    dyns_costs: Tuple[List[AnyDynamics], List[QuadCost]] = zip(*systems)
    dyns, costs = dyns_costs

    def stack_batch(tensors: List[Tensor]) -> Tensor:
        return torch.cat([t.align_to("H", "B", ...) for t in tensors], dim="B")

    if isinstance(dyns[0], LinSDynamics):
        Fs, fs, Ws = zip(*dyns)
        dynamics = LinSDynamics(F=stack_batch(Fs), f=stack_batch(fs), W=stack_batch(Ws))
    else:
        Fs, fs = zip(*dyns)
        dynamics = LinDynamics(F=stack_batch(Fs), f=stack_batch(fs))

    Cs, cs = zip(*costs)
    cost = QuadCost(C=stack_batch(Cs), c=stack_batch(cs))

    return dynamics, cost


def box_ddp_random_lqr(
    timestep: float,
    ctrl_coeff: float,
    horizon: int,
    np_random: Optional[Union[Generator, int]] = None,
) -> Tuple[LinDynamics, QuadCost, Box]:
    # pylint:disable=line-too-long
    """Generate a random, control-limited LQR as described in the Box-DDP paper.

    Taken from `Control-limited differential dynamic programming`_.

    .. _`Control-limited differential dynamic programming`: https://doi.org/10.1109/ICRA.2014.6907001
    """
    # pylint:enable=line-too-long
    assert 0 < timestep < 1

    np_random = np.random.default_rng(np_random)
    state_size = np_random.integers(10, 100, endpoint=True)
    ctrl_size = np_random.integers(1, state_size // 2, endpoint=True)

    dynamics = _box_ddp_random_dynamics(state_size, ctrl_size, timestep, horizon)
    cost = _box_ddp_random_cost(state_size, ctrl_size, timestep, ctrl_coeff, horizon)
    dynamics, cost = refine_lqr(dynamics, cost)
    bounds: Box = tuple(
        map(as_float_tensor, (s * np.ones_like(ctrl_size) for s in (-1, 1)))
    )
    return dynamics, cost, bounds


def _box_ddp_random_dynamics(
    state_size: int, ctrl_size: int, timestep: float, horizon: int
) -> LinDynamics:
    F_x = torch.eye(state_size) + timestep * torch.randn(state_size, state_size)
    F_u = torch.randn(state_size, ctrl_size)
    F = torch.cat([F_x, F_u], dim=-1)
    F = F.expand(horizon, *F.shape)
    f = torch.zeros(horizon, state_size)
    return LinDynamics(F, f)


def _box_ddp_random_cost(
    state_size: int, ctrl_size: int, timestep: float, ctrl_coeff: float, horizon: int
) -> QuadCost:
    dim = state_size + ctrl_size
    C = torch.zeros(horizon, dim, dim)

    C_xx = torch.eye(state_size, state_size) * timestep
    C_uu = torch.eye(ctrl_size, ctrl_size) * timestep * ctrl_coeff
    C[..., :state_size, :state_size] = C_xx
    C[..., state_size:, state_size:] = C_uu

    c = torch.zeros(horizon, dim)
    return QuadCost(C, c)


def make_lindynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    np_random: Optional[Union[Generator, int]] = None,
) -> LinDynamics:
    # pylint:disable=missing-function-docstring
    n_tau = state_size + ctrl_size
    np_random = np.random.default_rng(np_random)

    if stationary:
        F = np_expand_horizon(np_random.normal(size=(state_size, n_tau)), horizon)
        f = np_expand_horizon(np_random.normal(size=(state_size,)), horizon)
    else:
        F = np_random.normal(size=(horizon, state_size, n_tau))
        f = np_random.normal(size=(horizon, state_size))

    F, f = map(as_float_tensor, (F, f))
    F = nt.horizon(nt.matrix(F))
    f = nt.horizon(nt.vector(f))
    return LinDynamics(F, f)


def make_linsdynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    np_random: Optional[Union[Generator, int]] = None,
) -> LinSDynamics:
    # pylint:disable=missing-function-docstring
    np_random = np.random.default_rng(np_random)
    F, f = make_lindynamics(
        state_size, ctrl_size, horizon, stationary=stationary, np_random=np_random
    )

    sample_shape = () if stationary else (horizon,)
    W = make_spd_matrix(state_size, sample_shape=sample_shape, rng=np_random)
    W = as_float_tensor(W).expand(horizon, state_size, state_size)
    W = nt.horizon(nt.matrix(W))

    return LinSDynamics(F, f, W)


def make_quadcost(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    np_random: Optional[Union[Generator, int]] = None,
) -> QuadCost:
    # pylint:disable=missing-function-docstring
    n_tau = state_size + ctrl_size
    np_random = np.random.default_rng(np_random)

    if stationary:
        C = np_expand_horizon(make_spd_matrix(n_dim=n_tau, rng=np_random), horizon)
        c = np_expand_horizon(np_random.normal(size=(n_tau,)), horizon)
    else:
        C = make_spd_matrix(n_dim=n_tau, sample_shape=(horizon,), rng=np_random)
        c = np_random.normal(size=(horizon, n_tau))

    C, c = map(as_float_tensor, (C, c))
    C = nt.horizon(nt.matrix(C))
    c = nt.horizon(nt.vector(c))
    return QuadCost(C, c)


def make_lqr(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    np_random: Optional[Union[Generator, int]] = None,
) -> Tuple[LinDynamics, QuadCost]:
    """Random LQR generator used in backpropagation-planning.

    Args:
        state_size: Integer size for state
        ctrl_size: Integer size for controls
        horizon: Integer number of decision steps
        stationary: Whether to create time-invariant dynamics and cost
        np_random: Numpy random number generator or integer seed

    Source::
        https://github.com/renato-scaroni/backpropagation-planning/blob/master/src/Modules/Envs/lqr.py
    """
    np_random = np.random.default_rng(np_random)
    dynamics = make_lindynamics(
        state_size, ctrl_size, horizon, stationary=stationary, np_random=np_random
    )
    cost = make_quadcost(
        state_size, ctrl_size, horizon, stationary=stationary, np_random=np_random
    )
    return dynamics, cost


def make_lqg(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool,
    np_random: Optional[Union[Generator, int]] = None,
) -> Tuple[LinSDynamics, QuadCost]:
    """Random LQG generator.

    Args:
        state_size: Integer size for state
        ctrl_size: Integer size for controls
        horizon: Integer number of decision steps
        stationary: Whether to create time-invariant dynamics and cost
        np_random: Numpy random number generator or integer seed
    """
    np_random = np.random.default_rng(np_random)
    dynamics = make_linsdynamics(
        state_size, ctrl_size, horizon, stationary=stationary, np_random=np_random
    )
    cost = make_quadcost(
        state_size, ctrl_size, horizon, stationary=stationary, np_random=np_random
    )
    return dynamics, cost


def make_lqr_linear_navigation(
    goal: Union[np.ndarray, Tuple[float, float]], beta: float, horizon: int
) -> Tuple[LinDynamics, QuadCost, Box]:
    """Goal-oriented 2D Navigation task encoded as an LQR.

    Args:
        goal: 2D coordinates of goal position
        beta: Penalty coefficient for control magnitude
        horizon: Integer number of decision steps

    Source::
        https://github.com/renato-scaroni/backpropagation-planning/blob/master/src/Modules/Envs/lqr.py
    """
    goal = np.asarray(goal)
    state_size = ctrl_size = goal.shape[0]

    F = np.concatenate([np.identity(state_size), np.identity(ctrl_size)], axis=1)
    F = np_expand_horizon(F, horizon)
    f = np.zeros((horizon, state_size))

    C = np.diag([2.0] * state_size + [2.0 * beta] * ctrl_size)
    C = np_expand_horizon(C, horizon)
    c = np.concatenate([-2.0 * goal, np.zeros((ctrl_size,))], axis=0)
    c = np_expand_horizon(c, horizon)

    bounds: Box = tuple(
        map(as_float_tensor, (s * np.ones_like(ctrl_size) for s in (-1, 1)))
    )
    F, f, C, c = map(as_float_tensor, (F, f, C, c))
    dynamics, cost = refine_lqr((F, f), (C, c))
    return dynamics, cost, bounds
