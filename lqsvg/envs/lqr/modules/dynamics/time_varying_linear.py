"""Time-varying linear dynamics models."""
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from raylab.policy.modules.model import StochasticModel
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import unpack_obs

from .common import assemble_scale_tril
from .common import disassemble_covariance
from .common import TVMultivariateNormal


class CovCholeskyFactor(nn.Module):
    """Covariance Cholesky factor."""

    # pylint:disable=abstract-method
    def __init__(self, sigma: Tensor):
        super().__init__()
        sigma = sigma.refine_names("H", "R", "C")
        self.ltril, self.pre_diag = (
            nn.Parameter(x) for x in nt.unnamed(*disassemble_covariance(sigma))
        )

    def _named_factors(self) -> Tuple[Tensor, Tensor]:
        return self.ltril.refine_names("H", "R", "C"), self.pre_diag.refine_names(
            "H", "R"
        )

    def forward(self, time: Optional[Tensor] = None) -> Tensor:
        # pylint:disable=missing-function-docstring
        ltril, pre_diag = self._named_factors()
        if time is not None:
            ltril = nt.index_by(ltril, dim="H", index=time)
            pre_diag = nt.index_by(pre_diag, dim="H", index=time)
        return nt.refine_matrix_output(assemble_scale_tril(ltril, pre_diag))


class TVLinearNormalParams(nn.Module):
    """Time-varying linear state-action conditional Gaussian parameters."""

    # pylint:disable=invalid-name,abstract-method,no-self-use
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
        F, f = (
            nt.index_by(x, dim="H", index=index) for x in self._transition_factors()
        )
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


class TVLinearDynamics(StochasticModel):
    """Time-varying linear stochastic model from dynamics."""

    def __init__(self, dynamics: lqr.LinSDynamics):
        params = TVLinearNormalParams(*dynamics)
        dist = TVMultivariateNormal()
        super().__init__(params, dist)

    def standard_form(self) -> lqr.LinSDynamics:
        # pylint:disable=missing-function-docstring
        return self.params.as_linsdynamics()
