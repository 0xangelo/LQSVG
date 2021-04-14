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
from lqsvg.envs.lqr.utils import isstationary, unpack_obs

from .common import TVMultivariateNormal, assemble_scale_tril, disassemble_covariance


class CovarianceCholesky(nn.Module):
    """Covariance Cholesky factor."""

    beta: float = 0.2
    ltril: nn.Parameter
    pre_diag: nn.Parameter
    stationary: bool

    def __init__(self, sigma: Tensor, stationary: bool):
        super().__init__()
        sigma = nt.horizon(nt.matrix(sigma))
        ltril, pre_diag = nt.unnamed(*disassemble_covariance(sigma, beta=self.beta))
        self.ltril, self.pre_diag = (nn.Parameter(x) for x in (ltril, pre_diag))
        self.stationary = stationary

    def forward(self, index: Optional[IntTensor] = None) -> Tensor:
        # pylint:disable=missing-function-docstring
        ltril, pre_diag = nt.matrix(self.ltril), nt.vector(self.pre_diag)
        ltril, pre_diag = nt.horizon(ltril), nt.horizon(pre_diag)
        if index is not None:
            if self.stationary:
                index = torch.zeros_like(index)
            else:
                index = torch.clamp(index, max=self.horizon - 1)
            # noinspection PyTypeChecker
            ltril, pre_diag = (
                nt.index_by(x, dim="H", index=index) for x in (ltril, pre_diag)
            )
        return assemble_scale_tril(ltril, pre_diag, beta=self.beta)


# noinspection PyPep8Naming
class TVLinearNormalParams(nn.Module):
    """Time-varying linear state-action conditional Gaussian parameters."""

    # pylint:disable=invalid-name
    F: nn.Parameter
    f: nn.Parameter
    horizon: int
    stationary: bool

    def __init__(self, dynamics: lqr.LinSDynamics, horizon: int, stationary: bool):
        super().__init__()

        F, f, W = dynamics
        self.F = nn.Parameter(nt.unnamed(F))
        self.f = nn.Parameter(nt.unnamed(f))
        self.scale_tril = CovarianceCholesky(W, stationary=stationary)
        self.horizon = horizon
        self.stationary = stationary

    def _transition_factors(
        self, index: Optional[IntTensor] = None
    ) -> Tuple[Tensor, Tensor]:
        F, f = nt.horizon(nt.matrix(self.F)), nt.horizon(nt.vector(self.f))
        if index is not None:
            if self.stationary:
                index = torch.zeros_like(index)
            else:
                # Timesteps after termination use last parameters
                index = torch.clamp(index, max=self.horizon - 1)
            # noinspection PyTypeChecker
            F, f = (nt.index_by(x, dim="H", index=index) for x in (F, f))
        return F, f

    def forward(self, obs: Tensor, action: Tensor):
        # pylint:disable=missing-function-docstring
        obs, action = nt.vector(obs), nt.vector(action)
        state, time = unpack_obs(obs)

        # Get parameters for each timestep
        index = nt.vector_to_scalar(time)
        # noinspection PyTypeChecker
        F, f = self._transition_factors(index)
        scale_tril = self.scale_tril(index)

        # Compute the loc for normal transitions
        tau = nt.vector_to_matrix(torch.cat([state, action], dim="R"))
        trans_loc = nt.matrix_to_vector(F @ tau + nt.vector_to_matrix(f))

        # Treat absorving states if necessary
        terminal = time.eq(self.horizon)
        loc = nt.where(terminal, state, trans_loc)
        time_ = nt.where(terminal, time, time + 1)
        return {"loc": loc, "scale_tril": scale_tril, "time": time_}

    def as_linsdynamics(self) -> lqr.LinSDynamics:
        # pylint:disable=missing-function-docstring
        F, f = self._transition_factors()
        scale_tril = self.scale_tril()
        covariance_matrix = scale_tril @ nt.transpose(scale_tril)
        return lqr.LinSDynamics(F, f, covariance_matrix)


class LinearDynamics(StochasticModel, metaclass=abc.ABCMeta):
    """Abstraction for linear modules usable by LQG solvers."""

    n_state: int
    n_ctrl: int
    horizon: int
    F: nn.Parameter
    f: nn.Parameter
    params: TVLinearNormalParams

    def standard_form(self) -> lqr.LinSDynamics:
        """Returns self as parameters defining a linear stochastic system."""
        return self.params.as_linsdynamics()

    def dimensions(self) -> tuple[int, int, int]:
        """Return the state, action, and horizon size for this module."""
        return self.n_state, self.n_ctrl, self.horizon


class LinearDynamicsModule(LinearDynamics):
    """Linear stochastic model from dynamics.

    Args:
        dynamics: the linear dynamics to initialize the model with
        stationary: whether to model stationary dynamics

    Raises:
        AssertionError: if `stationary` is True but the dynamics is not
            stationary
    """

    stationary: bool

    def __init__(self, dynamics: lqr.LinSDynamics, stationary: bool):
        # pylint:disable=invalid-name
        self.n_state, self.n_ctrl, self.horizon = lqr.dims_from_dynamics(dynamics)
        self.stationary = stationary

        if stationary:
            assert isstationary(dynamics)
            F, f, W = (t.select("H", 0).align_to("H", ...) for t in dynamics)
            dynamics = lqr.LinSDynamics(F, f, W)

        params = TVLinearNormalParams(
            dynamics, horizon=self.horizon, stationary=stationary
        )
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

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


class TVLinearDynamicsModule(LinearDynamics):
    """Time-varying linear stochastic model from dynamics."""

    def __init__(self, dynamics: lqr.LinSDynamics):
        # pylint:disable=invalid-name
        self.n_state, self.n_ctrl, self.horizon = lqr.dims_from_dynamics(dynamics)
        params = TVLinearNormalParams(dynamics, horizon=self.horizon, stationary=False)
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f


class StationaryLinearDynamicsModule(LinearDynamics):
    """Linear stochastic model from dynamics."""

    def __init__(self, dynamics: lqr.LinSDynamics):
        # pylint:disable=invalid-name
        self.n_state, self.n_ctrl, self.horizon = lqr.dims_from_dynamics(dynamics)

        assert isstationary(dynamics)
        F, f, W = (t.select("H", 0).align_to("H", ...) for t in dynamics)
        stationary = lqr.LinSDynamics(F, f, W)

        params = TVLinearNormalParams(stationary, horizon=self.horizon, stationary=True)
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

    def standard_form(self) -> lqr.LinSDynamics:
        # pylint:disable=invalid-name
        dynamics = super().standard_form()
        F, f, W = map(self.expand_horizon, dynamics)
        return lqr.LinSDynamics(F=F, f=f, W=W)

    def expand_horizon(self, tensor: Tensor) -> Tensor:
        """Expand a tensor along the horizon dim."""
        zip_names = zip(tensor.shape, tensor.names)
        new_shape = tuple(self.horizon if n == "H" else s for s, n in zip_names)
        return tensor.expand(new_shape)
