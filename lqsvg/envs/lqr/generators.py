"""Random LQR problem generators."""
# pylint:disable=invalid-name,unsubscriptable-object
from __future__ import annotations

from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.np_util import make_spd_matrix
from lqsvg.np_util import RNG
from lqsvg.torch.utils import as_float_tensor

from .named import refine_lqr
from .types import AnyDynamics
from .types import Box
from .types import GaussInit
from .types import LinDynamics
from .types import LinSDynamics
from .types import QuadCost
from .utils import expand_and_refine
from .utils import np_expand_horizon
from .utils import random_mat_with_eigval_range
from .utils import random_normal_matrix
from .utils import random_normal_vector
from .utils import random_spd_matrix


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


# noinspection PyPep8Naming
def make_lindynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    Fs_eigval_range: Optional[tuple[float, float]] = None,
    transition_bias: bool = True,
    rng: RNG = None,
) -> LinDynamics:
    """Generate linear transition matrices.

    Args:
        state_size: size of state vector
        ctrl_size: size of control vector
        horizon: length of the horizon
        stationary: whether dynamics vary with time
        n_batch: batch size, if any
        Fs_eigval_range: range of eigenvalues for the unnactuated system. If None,
            samples the F_s matrix entries independently from a standard normal
            distribution
        transition_bias: whether to use a non-zero bias vector for transition
            dynamics
        rng: random number generator, seed, or None
    """
    # pylint:disable=too-many-arguments
    rng = np.random.default_rng(rng)

    kwargs = dict(horizon=horizon, stationary=stationary, n_batch=n_batch, rng=rng)
    if Fs_eigval_range:
        Fs = random_mat_with_eigval_range(state_size, Fs_eigval_range, **kwargs)
    else:
        Fs = random_normal_matrix(state_size, state_size, **kwargs)
    Fa = random_normal_matrix(state_size, ctrl_size, **kwargs)
    F = torch.cat((Fs, Fa), dim="C")

    if transition_bias:
        f = random_normal_vector(state_size, **kwargs)
    else:
        f = expand_and_refine(
            torch.zeros(state_size), 1, horizon=horizon, n_batch=n_batch
        )
    return LinDynamics(F, f)


def make_linsdynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
    **linear_kwargs
) -> LinSDynamics:
    """Generate stochastic linear dynamics parameters."""
    # pylint:disable=too-many-arguments
    rng = np.random.default_rng(rng)

    F, f = make_lindynamics(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        rng=rng,
        **linear_kwargs
    )
    W = random_spd_matrix(
        state_size, horizon=horizon, stationary=stationary, n_batch=n_batch, rng=rng
    )
    return LinSDynamics(F, f, W)


def make_quadcost(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> QuadCost:
    """Generate quadratic cost parameters."""
    # pylint:disable=too-many-arguments
    rng = np.random.default_rng(rng)
    n_tau = state_size + ctrl_size

    kwargs = dict(horizon=horizon, stationary=stationary, n_batch=n_batch, rng=rng)
    C = random_spd_matrix(n_tau, **kwargs)
    c = random_normal_vector(n_tau, **kwargs)
    return QuadCost(C, c)


def make_gaussinit(
    state_size: int,
    n_batch: Optional[int] = None,
    sample_covariance: bool = False,
    rng: RNG = None,
) -> GaussInit:
    """Generate parameters for Gaussian initial state distribution."""
    # pylint:disable=invalid-name
    vec_shape = (state_size,)
    batch_shape = () if n_batch is None else (n_batch,)

    mu = torch.zeros(batch_shape + vec_shape)
    if sample_covariance:
        sig = as_float_tensor(
            make_spd_matrix(state_size, sample_shape=batch_shape, rng=rng)
        )
    else:
        sig = torch.eye(state_size)

    return GaussInit(
        mu=expand_and_refine(nt.vector(mu), 1, n_batch=n_batch),
        sig=expand_and_refine(nt.matrix(sig), 2, n_batch=n_batch),
    )


def make_lqr(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
    **linear_kwargs
) -> tuple[LinDynamics, QuadCost]:
    # pylint:disable=too-many-arguments
    """Random LQR generator used in backpropagation-planning.

    Args:
        state_size: Integer size for state
        ctrl_size: Integer size for controls
        horizon: Integer number of decision steps
        stationary: Whether to create time-invariant dynamics and cost
        n_batch: Batch size (number of LQR samples)
        rng: Numpy random number generator or integer seed

    Source::
        https://github.com/renato-scaroni/backpropagation-planning/blob/master/src/Modules/Envs/lqr.py
    """
    rng = np.random.default_rng(rng)
    dynamics = make_lindynamics(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        rng=rng,
        **linear_kwargs
    )
    cost = make_quadcost(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        rng=rng,
    )
    return dynamics, cost


def make_lqg(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool,
    n_batch: Optional[int] = None,
    rng: RNG = None,
    **linear_kwargs
) -> tuple[LinSDynamics, QuadCost]:
    """Random LQG generator.

    Args:
        state_size: Integer size for state
        ctrl_size: Integer size for controls
        horizon: Integer number of decision steps
        stationary: Whether to create time-invariant dynamics and cost
        n_batch: Batch size (number of LQG samples)
        rng: Numpy random number generator or integer seed
    """
    # pylint:disable=too-many-arguments
    rng = np.random.default_rng(rng)
    dynamics = make_linsdynamics(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        rng=rng,
        **linear_kwargs
    )
    cost = make_quadcost(
        state_size,
        ctrl_size,
        horizon,
        stationary=stationary,
        n_batch=n_batch,
        rng=rng,
    )
    return dynamics, cost


###############################################################################
# Navigation environment
###############################################################################


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

    bounds: Box = (-torch.ones(ctrl_size), torch.ones(ctrl_size))
    # Avoid tensor writing to un-writable np.array
    F, f, C, c = map(lambda x: as_float_tensor(x.copy()), (F, f, C, c))
    dynamics, cost = refine_lqr((F, f), (C, c))
    return dynamics, cost, bounds


###############################################################################
# Box-DDP environment
###############################################################################


def box_ddp_random_lqr(
    timestep: float, ctrl_coeff: float, horizon: int, rng: RNG = None
) -> tuple[LinDynamics, QuadCost, Box]:
    # pylint:disable=line-too-long
    """Generate a random, control-limited LQR as described in the Box-DDP paper.

    Taken from `Control-limited differential dynamic programming`_.

    .. _`Control-limited differential dynamic programming`: https://doi.org/10.1109/ICRA.2014.6907001
    """
    # pylint:enable=line-too-long
    assert 0 < timestep < 1

    rng = np.random.default_rng(rng)
    state_size = rng.integers(10, 100, endpoint=True)
    ctrl_size = rng.integers(1, state_size // 2, endpoint=True)

    dynamics = _box_ddp_random_dynamics(state_size, ctrl_size, timestep, horizon)
    cost = _box_ddp_random_cost(state_size, ctrl_size, timestep, ctrl_coeff, horizon)
    dynamics, cost = refine_lqr(dynamics, cost)
    bounds: Box = (-torch.ones(ctrl_size), torch.ones(ctrl_size))
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
