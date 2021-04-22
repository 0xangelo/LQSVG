"""Linear dynamics models."""
from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
from raylab.policy.modules.model import StochasticModel
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.utils import stationary_dynamics, unpack_obs
from lqsvg.torch.modules import CholeskyFactor
from lqsvg.torch.utils import assemble_cholesky, disassemble_cholesky

from .common import TVMultivariateNormal


class CovarianceCholesky(nn.Module):
    """Covariance Cholesky factor."""

    beta: float = 0.2
    size: int
    horizon: int
    stationary: bool
    ltril: nn.Parameter
    pre_diag: nn.Parameter

    def __init__(self, size: int, horizon: int, stationary: bool):
        super().__init__()
        self.horizon = horizon
        self.stationary = stationary

        h_size = 1 if stationary else horizon
        self.ltril = nn.Parameter(Tensor(h_size, size, size))
        self.pre_diag = nn.Parameter(Tensor(h_size, size))

    def factorize_(self, matrix: Tensor) -> CovarianceCholesky:
        """Set parameters to reproduce a symmetric positive definite matrix."""
        ltril, pre_diag = nt.unnamed(*disassemble_cholesky(matrix, beta=self.beta))
        self.ltril.data.copy_(ltril)
        self.pre_diag.data.copy_(pre_diag)
        return self

    def forward(self, index: Optional[IntTensor] = None) -> Tensor:
        # pylint:disable=missing-function-docstring
        ltril, pre_diag = nt.horizon(nt.matrix(self.ltril), nt.vector(self.pre_diag))
        if index is not None:
            if self.stationary:
                index = torch.zeros_like(index)
            else:
                index = torch.clamp(index, max=self.horizon - 1).int()
            ltril, pre_diag = (
                nt.index_by(x, dim="H", index=index) for x in (ltril, pre_diag)
            )
        return assemble_cholesky(ltril, pre_diag, beta=self.beta)


# noinspection PyPep8Naming
class LinearNormalParams(nn.Module):
    """Linear state-action conditional Gaussian parameters."""

    # pylint:disable=invalid-name
    n_state: int
    n_ctrl: int
    horizon: int
    stationary: bool
    F: nn.Parameter
    f: nn.Parameter
    scale_tril: CholeskyFactor

    def __init__(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.horizon = horizon
        self.stationary = stationary

        h_size = 1 if stationary else horizon
        self.F = nn.Parameter(Tensor(h_size, n_state, n_state + n_ctrl))
        self.f = nn.Parameter(Tensor(h_size, n_state))
        self.scale_tril = CholeskyFactor((h_size, n_state, n_state))
        self.reset_parameters()

    def reset_parameters(self):
        """Default parameter initialization."""
        linear = make_lindynamics(
            self.n_state, self.n_ctrl, self.horizon, stationary=self.stationary
        )
        dynamics = make_linsdynamics(linear, stationary=self.stationary)
        self.copy_(dynamics)
        self.scale_tril.reset_parameters()

    def copy_(self, dynamics: lqr.LinSDynamics) -> LinearNormalParams:
        """Set parameters to mirror a given linear stochastic dynamics."""
        if self.stationary:
            dynamics = stationary_dynamics(dynamics)
        F, f, Sigma = dynamics
        self.F.data.copy_(F)
        self.f.data.copy_(f)
        self.scale_tril.factorize_(Sigma)
        return self

    def _transition_factors(
        self, index: Optional[IntTensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        F, f, L = nt.horizon(nt.matrix(self.F), nt.vector(self.f), self.scale_tril())
        if index is not None:
            if self.stationary:
                idx = torch.zeros_like(index)
            else:
                # Timesteps after termination use last parameters
                idx = torch.clamp(index, max=self.horizon - 1).int()
            F, f, L = (nt.index_by(x, dim="H", index=idx) for x in (F, f, L))
        return F, f, L

    def forward(self, obs: Tensor, action: Tensor):
        # pylint:disable=missing-function-docstring
        obs, action = nt.vector(obs), nt.vector(action)
        state, time = unpack_obs(obs)

        # Get parameters for each timestep
        index = nt.vector_to_scalar(time)
        F, f, scale_tril = self._transition_factors(index)

        # Compute the loc for normal transitions
        tau = nt.vector_to_matrix(torch.cat([state, action], dim="R"))
        trans_loc = nt.matrix_to_vector(F @ tau + nt.vector_to_matrix(f))

        # Treat absorving states if necessary
        terminal = time.eq(self.horizon)
        loc = nt.where(terminal, state, trans_loc)
        return {"loc": loc, "scale_tril": scale_tril, "time": time}

    def as_linsdynamics(self) -> lqr.LinSDynamics:
        # pylint:disable=missing-function-docstring
        F, f, scale_tril = self._transition_factors()
        Sigma = scale_tril @ nt.transpose(scale_tril)
        return lqr.LinSDynamics(F, f, Sigma)


class LinearDynamics(StochasticModel, metaclass=abc.ABCMeta):
    """Abstraction for linear modules usable by LQG solvers."""

    n_state: int
    n_ctrl: int
    horizon: int
    F: nn.Parameter
    f: nn.Parameter
    params: LinearNormalParams

    def standard_form(self) -> lqr.LinSDynamics:
        """Returns self as parameters defining a linear stochastic system."""
        return self.params.as_linsdynamics()

    def dimensions(self) -> tuple[int, int, int]:
        """Return the state, action, and horizon size for this module."""
        return self.n_state, self.n_ctrl, self.horizon


class LinearDynamicsModule(LinearDynamics):
    """Linear stochastic model from dynamics.

    Args:
        n_state: dimensionality of the state vectors
        n_ctrl: dimensionality of the control (action) vectors
        horizon: task horizon
        stationary: whether to model stationary dynamics

    Raises:
        AssertionError: if `stationary` is True but the dynamics is not
            stationary
    """

    stationary: bool

    def __init__(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        # pylint:disable=invalid-name
        self.n_state, self.n_ctrl, self.horizon = n_state, n_ctrl, horizon
        self.stationary = stationary

        params = LinearNormalParams(n_state, n_ctrl, horizon, stationary)
        dist = TVMultivariateNormal(horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

    @classmethod
    def from_existing(
        cls, dynamics: lqr.LinSDynamics, stationary: bool
    ) -> LinearDynamicsModule:
        """Create linear dynamics module from existing linear dynamics.

        Args:
            dynamics: the linear dynamics to initialize the model with
            stationary: whether to model stationary dynamics

        Raises:
            AssertionError: if `stationary` is True but the dynamics is not
                stationary
        """
        n_state, n_ctrl, horizon = lqr.dims_from_dynamics(dynamics)
        return cls(n_state, n_ctrl, horizon, stationary).copy_(dynamics)

    def copy_(self, dynamics: lqr.LinSDynamics) -> LinearDynamicsModule:
        """Update parameters to existing linear dynamics.

        Args:
            dynamics: the linear dynamics to initialize the model with

        Raises:
            AssertionError: if `self.stationary` is True but the dynamics is
                not stationary
        """
        self.params.copy_(dynamics)
        return self

    def standard_form(self) -> lqr.LinSDynamics:
        # pylint:disable=invalid-name
        dynamics = super().standard_form()
        if self.stationary:
            F, f, W = map(self.expand_horizon, dynamics)
            dynamics = lqr.LinSDynamics(F=F, f=f, W=W)
        return dynamics

    def expand_horizon(self, tensor: Tensor) -> Tensor:
        """Expand a tensor along the horizon dim."""
        zip_names = zip(tensor.shape, tensor.names)
        new_shape = tuple(self.horizon if n == "H" else s for s, n in zip_names)
        return tensor.expand(new_shape)
