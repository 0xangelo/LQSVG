"""Random LQR problem generators."""
# pylint:disable=invalid-name,unsubscriptable-object
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.np_util import (
    RNG,
    make_spd_matrix,
    random_unit_col_matrix,
    random_unit_vector,
)
from lqsvg.torch.random import (
    minimal_sample_shape,
    normal_matrix,
    normal_vector,
    spd_matrix,
)
from lqsvg.torch.utils import as_float_tensor, expand_and_refine

from . import utils
from .named import refine_lqr
from .types import AnyDynamics, Box, GaussInit, LinDynamics, LinSDynamics, QuadCost


@dataclass
class LQGGenerator(DataClassJsonMixin):
    """Specifications for LQG generation.

    Args:
        n_state: dimensionality of the state vectors
        n_ctrl: dimensionality of the control (action) vectors
        horizon: task horizon
        stationary: whether dynamics and cost parameters should be
            constant over time or vary by timestep
        passive_eigval_range: range of eigenvalues for the unnactuated system
        controllable: whether to ensure the actuator dynamics (the B matrix of
            the (A,B) pair) make the system controllable
        transition_bias: whether to use a non-zero bias vector for transition
            dynamics
        rand_trans_cov: whether to sample a random SPD matrix for the
            Gaussian transition covariance or use the identity matrix.
        rand_init_cov: whether to sample a random SPD matrix for the
            Gaussian initial state covariance or use the identity matrix.
        cost_linear: whether to include a linear term in addition to the
            quadratic one in the cost
        cost_cross: whether to include state-ctrl cross terms in the quadratic
            cost matrix (C_sa and C_as)
        rng: random number generator state
    """

    # pylint:disable=too-many-instance-attributes
    n_state: int
    n_ctrl: int
    horizon: int
    stationary: bool = True
    passive_eigval_range: Optional[tuple[float, float]] = (0.0, 1.0)
    controllable: bool = False
    transition_bias: bool = False
    rand_trans_cov: bool = False
    rand_init_cov: bool = False
    cost_linear: bool = False
    cost_cross: bool = False
    rng: RNG = None

    def __call__(
        self, n_batch: Optional[int] = None
    ) -> tuple[LinSDynamics, QuadCost, GaussInit]:
        """Generates random LQG parameters.

        Generates a transition kernel, cost function and initial state
        distribution parameters.

        Args:
            n_batch: batch size, if any

        Returns:
            A tuple containing parameters for linear stochastic dynamics,
            quadratic costs, and Normal inital state distribution.
        """
        dynamics = self.make_dynamics(n_batch)
        cost = self.make_cost(n_batch)
        init = self.make_init(n_batch)
        return dynamics, cost, init

    def make_dynamics(self, n_batch: Optional[int] = None) -> LinSDynamics:
        """Generates random LQG transition dynamics.

        Args:
            n_batch: batch size, if any

        Returns:
            Parameters for linear stochastic dynamics
        """
        dynamics = make_lindynamics(
            self.n_state,
            self.n_ctrl,
            self.horizon,
            stationary=self.stationary,
            n_batch=n_batch,
            passive_eigval_range=self.passive_eigval_range,
            controllable=self.controllable,
            bias=self.transition_bias,
            rng=self.rng,
        )
        dynamics = make_linsdynamics(
            dynamics,
            stationary=self.stationary,
            n_batch=n_batch,
            sample_covariance=self.rand_trans_cov,
            rng=self.rng,
        )
        return dynamics

    def make_cost(self, n_batch: Optional[int] = None) -> QuadCost:
        """Generates random LQG cost function

        Args:
            n_batch: batch size, if any

        Returns:
            Parameters for quadratic costs
        """
        return make_quadcost(
            self.n_state,
            self.n_ctrl,
            self.horizon,
            stationary=self.stationary,
            n_batch=n_batch,
            linear=self.cost_linear,
            cross_terms=self.cost_cross,
            rng=self.rng,
        )

    def make_init(self, n_batch: Optional[int] = None) -> GaussInit:
        """Generates random LQG initial state distribution.

        Args:
            n_batch: batch size, if any

        Returns:
            Parameters for Gaussian initial state distribution
        """
        return make_gaussinit(
            state_size=self.n_state,
            n_batch=n_batch,
            sample_covariance=self.rand_init_cov,
            rng=self.rng,
        )

    @contextmanager
    def config(self, **kwargs):
        """Temporarily alter this generator's configuration and return it."""
        cache = {k: getattr(self, k) for k in kwargs}
        try:
            for k, v in kwargs.items():
                setattr(self, k, v)
            yield self
        finally:
            for k, v in cache.items():
                setattr(self, k, v)


def stack_lqs(*systems: tuple[AnyDynamics, QuadCost]) -> tuple[AnyDynamics, QuadCost]:
    """Stack several linear quadratic problems into a batched representation.

    Returns dynamics and costs with an additional batch dimension.
    """
    # noinspection PyTypeChecker
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


def make_lindynamics(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    passive_eigval_range: Optional[tuple[float, float]] = (0.0, 1.0),
    controllable: bool = False,
    bias: bool = True,
    rng: RNG = None,
) -> LinDynamics:
    """Generate linear transition matrices.

    Args:
        state_size: size of state vector
        ctrl_size: size of control vector
        horizon: length of the horizon
        stationary: whether dynamics vary with time
        n_batch: batch size, if any
        passive_eigval_range: range of eigenvalues for the unnactuated system.
            If None, samples the F_s matrix entries independently from a
            standard normal distribution
        controllable: whether to ensure the actuator dynamics (the B matrix of
            the (A,B) pair) make the system controllable
        bias: whether to use a non-zero bias vector for transition dynamics
        rng: random number generator, seed, or None

    Raises:
        ValueError: if `controllable` is True but not `stationary`
    """
    # pylint:disable=too-many-arguments
    if controllable and not stationary:
        raise ValueError("Controllable non-stationary dynamics are unsupported.")
    rng = np.random.default_rng(rng)

    Fs, _, eigvec = generate_passive(
        state_size,
        eigval_range=passive_eigval_range,
        horizon=horizon,
        stationary=stationary,
        n_batch=n_batch,
        rng=rng,
    )
    Fa = generate_active(
        Fs, ctrl_size, eigvec=eigvec, controllable=controllable, rng=rng
    )

    Fs = expand_and_refine(
        nt.matrix(as_float_tensor(Fs)), 2, horizon=horizon, n_batch=n_batch
    )
    Fa = expand_and_refine(
        nt.matrix(as_float_tensor(Fa)), 2, horizon=horizon, n_batch=n_batch
    )
    F = torch.cat((Fs, Fa), dim="C")

    if bias:
        f = random_unit_vector(
            state_size,
            sample_shape=minimal_sample_shape(horizon, stationary, n_batch),
            rng=rng,
        )
    else:
        f = np.zeros(state_size)
    f = nt.vector(as_float_tensor(f))
    f = expand_and_refine(f, 1, horizon=horizon, n_batch=n_batch)
    return LinDynamics(F, f)


def generate_passive(
    state_size: int,
    horizon: int,
    stationary: bool,
    n_batch: Optional[int] = None,
    eigval_range: Optional[tuple[float, float]] = None,
    rng: RNG = None,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Generate the unnactuated part of a linear dynamical system."""
    # pylint:disable=too-many-arguments
    sample_shape = minimal_sample_shape(horizon, stationary=stationary, n_batch=n_batch)
    if eigval_range:
        mat, eigval, eigvec = utils.random_mat_with_eigval_range(
            state_size, eigval_range=eigval_range, sample_shape=sample_shape, rng=rng
        )
    else:
        warnings.warn("Using no eigval range may lead to complex eigvals")
        mat = normal_matrix(state_size, state_size, sample_shape=sample_shape, rng=rng)
        eigval, eigvec = np.linalg.eig(mat)

    if np.iscomplex(eigval).any() or np.iscomplex(eigvec).any():
        warnings.warn("Complex value found in passive dynamics' eigvals or eigvecs")
    return mat, eigval, eigvec


def generate_active(
    passive: np.ndarray,
    ctrl_size: int,
    eigvec: Optional[np.ndarray] = None,
    controllable: bool = False,
    rng: RNG = None,
) -> np.ndarray:
    """Generate the actuated part of a linear dynamical system.

    Args:
        passive: the passive state dynamics
        ctrl_size: size of the control vector
        eigvec: optional column eigenvectors of the passive dynamics.
            Required if `controllable` is True
        controllable: whether to ensure the final linear dynamical
            system is controllable
        rng: random number generator parameter
    """
    rng = np.random.default_rng(rng)

    n_state = passive.shape[-1]
    sample_shape = passive.shape[:-2]
    # Generate initial actuator dynamics with unit norm columns
    B = random_unit_col_matrix(
        n_row=n_state, n_col=ctrl_size, sample_shape=sample_shape, rng=rng
    )

    if controllable:
        assert eigvec is not None
        # Ensure final actuator dynamics have a non-zero component in each
        # passive eigenvector direction
        while np.any(np.abs(B) < 1e-8):
            B = random_unit_col_matrix(
                n_row=n_state, n_col=ctrl_size, sample_shape=sample_shape, rng=rng
            )
        B = eigvec @ B

    return B


def make_linsdynamics(
    dynamics: LinDynamics,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    sample_covariance: bool = True,
    rng: RNG = None,
) -> LinSDynamics:
    """Generate stochastic linear dynamics from linear deterministic dynamics.

    Warning:
        This function does not check if `stationary`, and `nbatch` are
        compatible with `dynamics` (i.e., if the dynamics satisfy these
        parameters), but passing incompatible dynamics may lead to errors
        downstream.

    Args:
        dynamics: linear deterministic transition dynamics
        stationary: whether dynamics vary with time
        n_batch: batch size, if any
        sample_covariance: whether to sample a random SPD matrix for the
            Gaussian covariance or use the identity matrix.
        rng: random number generator, seed, or None
    """
    # pylint:disable=too-many-arguments
    rng = np.random.default_rng(rng)
    state_size, _, horizon = utils.dims_from_dynamics(dynamics)

    if sample_covariance:
        W = spd_matrix(
            state_size, horizon=horizon, stationary=stationary, n_batch=n_batch, rng=rng
        )
    else:
        W = expand_and_refine(
            nt.matrix(torch.eye(state_size)), 2, horizon=horizon, n_batch=n_batch
        )

    F, f = dynamics
    return LinSDynamics(F, f, W)


def make_quadcost(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = True,
    n_batch: Optional[int] = None,
    linear: bool = False,
    cross_terms: bool = False,
    rng: RNG = None,
) -> QuadCost:
    """Generate quadratic cost parameters.

    Args:
        state_size: size of state vector
        ctrl_size: size of control vector
        horizon: length of the horizon
        stationary: whether dynamics vary with time
        n_batch: batch size, if any
        linear: whether to include a linear term in addition to the quadratic
        cross_terms: whether to include state-ctrl cross terms in the quadratic
            (C_sa and C_as)
        rng: random number generator, seed, or None
    """
    # pylint:disable=too-many-arguments,too-many-locals
    rng = np.random.default_rng(rng)
    n_tau = state_size + ctrl_size

    kwargs = dict(horizon=horizon, stationary=stationary, n_batch=n_batch, rng=rng)

    C = spd_matrix(n_tau, **kwargs)
    C_s, C_a = nt.split(C, [state_size, ctrl_size], dim="C")
    C_ss, C_sa = nt.split(C_s, [state_size, ctrl_size], dim="R")
    C_as, C_aa = nt.split(C_a, [state_size, ctrl_size], dim="R")

    if not cross_terms:
        C_sa, C_as = torch.zeros_like(C_sa), torch.zeros_like(C_as)

    C_s = torch.cat((C_ss, C_sa), dim="R")
    C_a = torch.cat((C_as, C_aa), dim="R")
    C = torch.cat((C_s, C_a), dim="C")

    if linear:
        c = normal_vector(n_tau, **kwargs)
    else:
        c = expand_and_refine(
            nt.vector(torch.zeros(n_tau)), 1, horizon=horizon, n_batch=n_batch
        )
    return QuadCost(C, c)


def make_gaussinit(
    state_size: int,
    n_batch: Optional[int] = None,
    sample_covariance: bool = False,
    rng: RNG = None,
) -> GaussInit:
    """Generate parameters for Gaussian initial state distribution.

    Args:
        state_size: size of state vector
        n_batch: batch size, if any
        sample_covariance: whether to sample a random SPD matrix for the
            Gaussian covariance or use the identity matrix.
        rng: random number generator, seed, or None
    """
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
    F = utils.np_expand_horizon(F, horizon)
    f = np.zeros((horizon, state_size))

    C = np.diag([2.0] * state_size + [2.0 * beta] * ctrl_size)
    C = utils.np_expand_horizon(C, horizon)
    c = np.concatenate([-2.0 * goal, np.zeros((ctrl_size,))], axis=0)
    c = utils.np_expand_horizon(c, horizon)

    bounds = Box(-torch.ones(ctrl_size), torch.ones(ctrl_size))
    # Avoid tensor writing to un-writable np.array
    F, f, C, c = map(lambda x: as_float_tensor(x.copy()), (F, f, C, c))
    dynamics, cost = refine_lqr(LinDynamics(F, f), QuadCost(C, c))
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
    bounds = Box(-torch.ones(ctrl_size), torch.ones(ctrl_size))
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


###############################################################################
# Deprecated full LQR generation
###############################################################################


def make_lqr(
    state_size: int,
    ctrl_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    cost_linear: bool = False,
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
        cost_linear: whether to include a linear term in addition to the
            quadratic one in the cost function
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
        linear=cost_linear,
        n_batch=n_batch,
        rng=rng,
    )
    return dynamics, cost
