"""Random LQR problem generators."""
# pylint:disable=invalid-name,unsubscriptable-object
from __future__ import annotations

from typing import Optional
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


def stack_lqs(*systems: tuple[AnyDynamics, QuadCost]) -> tuple[AnyDynamics, QuadCost]:
    """Stack several linear quadratic problems into a batched representation.

    Returns dynamics and costs with an additional batch dimension.
    """
    dyns_costs: tuple[list[AnyDynamics], list[QuadCost]] = zip(*systems)
    dyns, costs = dyns_costs

    def stack_batch(tensors: list[Tensor]) -> Tensor:
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
) -> tuple[LinDynamics, QuadCost, Box]:
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


def expand_and_refine(
    tensor: Tensor, horizon: int, n_batch: Optional[int], base_shape: tuple[int, ...]
) -> Tensor:
    """Expand and refine tensor names with horizon and batch size information."""
    assert (
        n_batch is None or n_batch > 0
    ), f"Batch size must be null or positive, got {n_batch}"
    final_shape = (horizon,) + (() if n_batch is None else (n_batch,)) + base_shape
    names = ("H",) + (() if n_batch is None else ("B",)) + (...,)
    tensor = tensor.expand(*final_shape).refine_names(*names)
    return tensor


def make_lindynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    np_random: Optional[Union[Generator, int]] = None,
) -> LinDynamics:
    """Generate linear transition matrices."""
    # pylint:disable=too-many-arguments
    np_random = np.random.default_rng(np_random)

    n_tau = state_size + ctrl_size
    mat_shape = (state_size, n_tau)
    vec_shape = (state_size,)

    horizon_shape = () if stationary else (horizon,)
    batch_shape = () if n_batch is None else (n_batch,)
    F = np_random.normal(size=horizon_shape + batch_shape + mat_shape)
    f = np_random.normal(size=horizon_shape + batch_shape + vec_shape)

    F, f = map(as_float_tensor, (F, f))
    F = expand_and_refine(nt.matrix(F), horizon, n_batch, mat_shape)
    f = expand_and_refine(nt.vector(f), horizon, n_batch, vec_shape)
    return LinDynamics(F, f)


def make_linsdynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    np_random: Optional[Union[Generator, int]] = None,
) -> LinSDynamics:
    """Generate stochastic linear dynamics parameters."""
    # pylint:disable=too-many-arguments
    np_random = np.random.default_rng(np_random)
    F, f = make_lindynamics(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        np_random=np_random,
    )

    batch_shape = () if n_batch is None else (n_batch,)
    sample_shape = (() if stationary else (horizon,)) + batch_shape
    W = make_spd_matrix(state_size, sample_shape=sample_shape, rng=np_random)
    W = nt.matrix(as_float_tensor(W))
    W = expand_and_refine(W, horizon, n_batch, (state_size, state_size))

    return LinSDynamics(F, f, W)


def make_quadcost(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    np_random: Optional[Union[Generator, int]] = None,
) -> QuadCost:
    """Generate quadratic cost parameters."""
    # pylint:disable=too-many-arguments
    np_random = np.random.default_rng(np_random)

    n_tau = state_size + ctrl_size
    mat_shape = (n_tau, n_tau)
    vec_shape = (n_tau,)

    horizon_shape = () if stationary else (horizon,)
    batch_shape = () if n_batch is None else (n_batch,)

    C = make_spd_matrix(
        n_dim=n_tau, sample_shape=horizon_shape + batch_shape, rng=np_random
    )
    c = np_random.normal(size=horizon_shape + batch_shape + vec_shape)

    C, c = map(as_float_tensor, (C, c))
    C = expand_and_refine(nt.matrix(C), horizon, n_batch, mat_shape)
    c = expand_and_refine(nt.vector(c), horizon, n_batch, vec_shape)
    return QuadCost(C, c)


def make_lqr(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    np_random: Optional[Union[Generator, int]] = None,
) -> tuple[LinDynamics, QuadCost]:
    # pylint:disable=too-many-arguments
    """Random LQR generator used in backpropagation-planning.

    Args:
        state_size: Integer size for state
        ctrl_size: Integer size for controls
        horizon: Integer number of decision steps
        stationary: Whether to create time-invariant dynamics and cost
        n_batch: Batch size (number of LQR samples)
        np_random: Numpy random number generator or integer seed

    Source::
        https://github.com/renato-scaroni/backpropagation-planning/blob/master/src/Modules/Envs/lqr.py
    """
    np_random = np.random.default_rng(np_random)
    dynamics = make_lindynamics(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        np_random=np_random,
    )
    cost = make_quadcost(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        np_random=np_random,
    )
    return dynamics, cost


def make_lqg(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool,
    n_batch: Optional[int] = None,
    np_random: Optional[Union[Generator, int]] = None,
) -> tuple[LinSDynamics, QuadCost]:
    """Random LQG generator.

    Args:
        state_size: Integer size for state
        ctrl_size: Integer size for controls
        horizon: Integer number of decision steps
        stationary: Whether to create time-invariant dynamics and cost
        n_batch: Batch size (number of LQG samples)
        np_random: Numpy random number generator or integer seed
    """
    # pylint:disable=too-many-arguments
    np_random = np.random.default_rng(np_random)
    dynamics = make_linsdynamics(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        np_random=np_random,
    )
    cost = make_quadcost(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        np_random=np_random,
    )
    return dynamics, cost


def make_lqr_linear_navigation(
    goal: Union[np.ndarray, tuple[float, float]], beta: float, horizon: int
) -> tuple[LinDynamics, QuadCost, Box]:
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
