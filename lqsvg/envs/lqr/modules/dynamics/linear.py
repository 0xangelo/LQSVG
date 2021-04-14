"""Linear dynamics models."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
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
    """Covariance matrix stored as Cholesky factor."""

    beta: float = 0.2

    def __init__(self, sigma: Tensor):
        super().__init__()
        assert len(sigma.shape) == 2
        ltril, pre_diag = nt.unnamed(*disassemble_covariance(sigma, beta=self.beta))
        self.ltril, self.pre_diag = (nn.Parameter(x) for x in (ltril, pre_diag))

    def _named_factors(self) -> tuple[Tensor, Tensor]:
        return nt.matrix(self.ltril), nt.vector(self.pre_diag)

    def forward(self) -> Tensor:
        # pylint:disable=missing-function-docstring
        ltril, pre_diag = self._named_factors()
        return nt.matrix(assemble_scale_tril(ltril, pre_diag, beta=self.beta))


class CovCholeskyFactor(nn.Module):
    """Covariance Cholesky factor."""

    beta: float = 0.2

    def __init__(self, sigma: Tensor, horizon: int):
        super().__init__()
        sigma = nt.horizon(nt.matrix(sigma))
        ltril, pre_diag = nt.unnamed(*disassemble_covariance(sigma, beta=self.beta))
        self.ltril, self.pre_diag = (nn.Parameter(x) for x in (ltril, pre_diag))
        self.horizon = horizon

    def forward(self, index: Optional[IntTensor] = None) -> Tensor:
        # pylint:disable=missing-function-docstring
        ltril, pre_diag = nt.matrix(self.ltril), nt.vector(self.pre_diag)
        ltril, pre_diag = nt.horizon(ltril), nt.horizon(pre_diag)
        if index is not None:
            index = torch.clamp(index, max=self.horizon - 1)
            # noinspection PyTypeChecker
            ltril = nt.index_by(ltril, dim="H", index=index)
            # noinspection PyTypeChecker
            pre_diag = nt.index_by(pre_diag, dim="H", index=index)
        return assemble_scale_tril(ltril, pre_diag, beta=self.beta)


# noinspection PyPep8Naming
class LinearNormalParams(nn.Module):
    """Linear state-action conditional Gaussian parameters."""

    # pylint:disable=invalid-name
    F: nn.Parameter
    f: nn.Parameter

    def __init__(self, dynamics: lqr.LinSDynamics, horizon: Optional[int] = None):
        super().__init__()
        assert isstationary(dynamics)
        self.F = nn.Parameter(nt.unnamed(dynamics.F.select("H", 0)))
        self.f = nn.Parameter(nt.unnamed(dynamics.f.select("H", 0)))
        self.scale_tril = CovarianceCholesky(dynamics.W.select("H", 0))
        if horizon is None:
            _, _, horizon = lqr.dims_from_dynamics(dynamics)
        self.horizon = horizon

    def _transition_factors(self) -> tuple[Tensor, Tensor]:
        return nt.matrix(self.F), nt.vector(self.f)

    def forward(self, obs: Tensor, action: Tensor):
        # pylint:disable=missing-function-docstring
        obs, action = (nt.vector(x) for x in (obs, action))
        state, time = unpack_obs(obs)
        F, f = self._transition_factors()
        scale_tril = self.scale_tril()

        tau = nt.vector_to_matrix(torch.cat([state, action], dim="R"))
        terminal = time.eq(self.horizon)
        loc = nt.where(
            terminal, state, nt.matrix_to_vector(F @ tau + nt.vector_to_matrix(f))
        )
        time_ = nt.where(terminal, time, time + 1)
        return {"loc": loc, "scale_tril": scale_tril, "time": time_}

    def as_linsdynamics(self) -> lqr.LinSDynamics:
        # pylint:disable=missing-function-docstring
        F, f = self._transition_factors()
        scale_tril = self.scale_tril()
        W = scale_tril @ nt.transpose(scale_tril)
        F, f, W = map(self.expand_and_refine_horizon, (F, f, W))
        return lqr.LinSDynamics(F=F, f=f, W=W)

    def expand_and_refine_horizon(self, tensor: Tensor) -> Tensor:
        """Expand a tensor to the horizon length and name the new dim."""
        return nt.horizon(tensor.expand((self.horizon,) + tensor.shape))


class LinearDynamics(StochasticModel, metaclass=ABCMeta):
    """Abstraction for linear modules usable by LQG solvers."""

    n_state: int
    n_ctrl: int
    horizon: int
    F: nn.Parameter
    f: nn.Parameter

    @abstractmethod
    def standard_form(self) -> lqr.LinSDynamics:
        """Returns self as parameters defining a linear stochastic system."""

    def dimensions(self) -> tuple[int, int, int]:
        """Return the state, action, and horizon size for this module."""
        return self.n_state, self.n_ctrl, self.horizon


class LinearDynamicsModule(LinearDynamics):
    """Linear stochastic model from dynamics."""

    # pylint:disable=invalid-name

    def __init__(self, dynamics: lqr.LinSDynamics):
        self.n_state, self.n_ctrl, self.horizon = lqr.dims_from_dynamics(dynamics)
        params = LinearNormalParams(dynamics, horizon=self.horizon)
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

    def standard_form(self) -> lqr.LinSDynamics:
        return self.params.as_linsdynamics()


class TVLinearNormalParams(nn.Module):
    """Time-varying linear state-action conditional Gaussian parameters."""

    # pylint:disable=invalid-name
    # noinspection PyPep8Naming
    def __init__(self, dynamics: lqr.LinSDynamics, horizon: int):
        super().__init__()

        F, f, W = dynamics
        self.F = nn.Parameter(nt.unnamed(F))
        self.f = nn.Parameter(nt.unnamed(f))
        self.scale_tril = CovCholeskyFactor(W, horizon)
        self.horizon = horizon

    def _transition_factors(
        self, index: Optional[IntTensor] = None
    ) -> Tuple[Tensor, Tensor]:
        F, f = nt.horizon(nt.matrix(self.F)), nt.horizon(nt.vector(self.f))
        if index is not None:
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


class TVLinearDynamicsModule(LinearDynamics):
    """Time-varying linear stochastic model from dynamics."""

    # pylint:disable=invalid-name

    def __init__(self, dynamics: lqr.LinSDynamics):
        self.n_state, self.n_ctrl, self.horizon = lqr.dims_from_dynamics(dynamics)
        params = TVLinearNormalParams(dynamics, self.horizon)
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

    def standard_form(self) -> lqr.LinSDynamics:
        return self.params.as_linsdynamics()
