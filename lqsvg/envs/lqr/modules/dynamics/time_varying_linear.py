"""Time-varying linear dynamics models."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import unpack_obs

from .common import TVMultivariateNormal, assemble_scale_tril, disassemble_covariance
from .linear import LinearDynamics


class CovCholeskyFactor(nn.Module):
    """Covariance Cholesky factor."""

    beta: float = 0.2

    def __init__(self, sigma: Tensor):
        super().__init__()
        sigma = nt.horizon(nt.matrix(sigma))
        ltril, pre_diag = nt.unnamed(*disassemble_covariance(sigma, beta=self.beta))
        self.ltril, self.pre_diag = (nn.Parameter(x) for x in (ltril, pre_diag))

    def _named_factors(self) -> Tuple[Tensor, Tensor]:
        return nt.horizon(nt.matrix(self.ltril)), nt.horizon(nt.vector(self.pre_diag))

    def forward(self, time: Optional[IntTensor] = None) -> Tensor:
        # pylint:disable=missing-function-docstring
        ltril, pre_diag = self._named_factors()
        if time is not None:
            ltril = nt.index_by(ltril, dim="H", index=time)
            pre_diag = nt.index_by(pre_diag, dim="H", index=time)
        return nt.matrix(assemble_scale_tril(ltril, pre_diag, beta=self.beta))


class TVLinearNormalParams(nn.Module):
    """Time-varying linear state-action conditional Gaussian parameters."""

    # pylint:disable=invalid-name
    # noinspection PyPep8Naming
    def __init__(self, F: Tensor, f: Tensor, W: Tensor):
        super().__init__()

        F = nt.horizon(nt.matrix(F))
        f = nt.horizon(nt.vector(f))
        self.F = nn.Parameter(nt.unnamed(F))
        self.f = nn.Parameter(nt.unnamed(f))
        self.scale_tril = CovCholeskyFactor(W)

    def _transition_factors(self) -> Tuple[Tensor, Tensor]:
        return nt.horizon(nt.matrix(self.F)), nt.horizon(nt.vector(self.f))

    def forward(self, obs: Tensor, action: Tensor):
        # pylint:disable=missing-function-docstring
        obs, action = (nt.vector(x) for x in (obs, action))
        state, time = unpack_obs(obs)
        index = nt.vector_to_scalar(time)
        F, f = self._transition_factors()
        F, f = (nt.index_by(x, dim="H", index=index) for x in (F, f))
        scale_tril = self.scale_tril(index)

        tau = nt.vector_to_matrix(torch.cat([state, action], dim="R"))
        loc = F @ tau + nt.vector_to_matrix(f)
        return {
            "loc": nt.matrix_to_vector(loc),
            "scale_tril": nt.matrix(scale_tril),
            "time": time + 1,
        }

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
        params = TVLinearNormalParams(*dynamics)
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)
        self.F = self.params.F
        self.f = self.params.f

    def standard_form(self) -> lqr.LinSDynamics:
        return self.params.as_linsdynamics()
