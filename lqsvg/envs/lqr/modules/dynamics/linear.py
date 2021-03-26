"""Linear dynamics models."""
from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from raylab.policy.modules.model import StochasticModel
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import isstationary
from lqsvg.envs.lqr.utils import unpack_obs

from .common import assemble_scale_tril
from .common import disassemble_covariance
from .common import TVMultivariateNormal


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
        loc = F @ tau + nt.vector_to_matrix(f)
        return {
            "loc": nt.matrix_to_vector(loc),
            "scale_tril": scale_tril,
            "time": time + 1,
        }

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
        dist = TVMultivariateNormal()
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

    def standard_form(self) -> lqr.LinSDynamics:
        return self.params.as_linsdynamics()
