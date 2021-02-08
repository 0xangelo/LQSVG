"""Time-varying linear dynamics models."""
from typing import List
from typing import Optional
from typing import Tuple

import raylab.torch.nn.distributions as ptd
import torch
import torch.nn as nn
from raylab.policy.modules.model import StochasticModel
from raylab.torch.nn.distributions.types import DistParams
from raylab.torch.nn.distributions.types import SampleLogp
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import unpack_obs
from lqsvg.np_util import make_spd_matrix
from lqsvg.torch.utils import as_float_tensor


def softplusinv(tensor: Tensor) -> Tensor:
    """Returns the inverse softplus transformation."""
    return torch.log(tensor.exp() - 1)


def disassemble_covariance(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute cholesky factor and break it into unconstrained parameters."""
    tril = nt.cholesky(tensor, upper=False)
    ltril = nt.tril(tril, diagonal=-1)
    pre_diag = softplusinv(nt.diagonal(tril, dim1="R", dim2="C"))
    return ltril, pre_diag


def assemble_scale_tril(ltril: Tensor, pre_diag: Tensor) -> Tensor:
    """Transform uncostrained parameters into covariance cholesky factor."""
    return ltril + torch.diag_embed(nt.softplus(pre_diag))


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


def _unpack_params(
    params: DistParams, sample_shape: List[int] = ()
) -> Tuple[Tensor, Tensor, Tensor]:
    loc = params["loc"]
    scale_tril = params["scale_tril"]
    time = params["time"]
    loc, scale_tril, time = map(
        lambda x: x.expand(sample_shape + x.shape), (loc, scale_tril, time)
    )
    return loc, scale_tril, time


def _gen_sample(loc: Tensor, scale_tril: Tensor, time: Tensor) -> Tensor:
    dist = torch.distributions.MultivariateNormal(
        loc=nt.unnamed(loc), scale_tril=nt.unnamed(scale_tril)
    )
    sample = dist.rsample().refine_names(*loc.names)
    return torch.cat([sample, time.float()], dim="R")


def _logp(loc: Tensor, scale_tril: Tensor, time: Tensor, value: Tensor) -> Tensor:
    dist = torch.distributions.MultivariateNormal(
        loc=nt.unnamed(loc), scale_tril=nt.unnamed(scale_tril)
    )
    state, time_ = unpack_obs(value)
    logp_: Tensor = dist.log_prob(nt.unnamed(state)).refine_names(
        *(n for n in state.names if n != "R")
    )
    logp = nt.where(
        time_.squeeze("R") == time.squeeze("R"),
        logp_,
        torch.full_like(logp_, fill_value=float("nan")).refine_names(*logp_.names),
    )
    return logp


class TVMultivariateNormal(ptd.ConditionalDistribution):
    """Time-varying multivariate Gaussian distribution."""

    # pylint:disable=no-self-use,abstract-method
    def sample(self, params: DistParams, sample_shape: List[int] = ()) -> SampleLogp:
        loc, scale_tril, time = _unpack_params(params, sample_shape)
        sample = _gen_sample(loc, scale_tril, time).detach()
        logp = _logp(loc, scale_tril, time, sample)
        return sample, logp

    def rsample(self, params: DistParams, sample_shape: List[int] = ()) -> SampleLogp:
        loc, scale_tril, time = _unpack_params(params, sample_shape)
        sample = _gen_sample(loc, scale_tril, time)
        logp = _logp(loc, scale_tril, time, sample)
        return sample, logp

    def log_prob(self, value: Tensor, params: DistParams) -> Tensor:
        value = nt.vector(value)
        loc, scale_tril, time = _unpack_params(params)
        logp = _logp(loc, scale_tril, time, value)
        return logp


class TVLinearDynamics(StochasticModel):
    """Time-varying linear stochastic model from dynamics."""

    def __init__(self, dynamics: lqr.LinSDynamics):
        params = TVLinearNormalParams(*dynamics)
        dist = TVMultivariateNormal()
        super().__init__(params, dist)

    def standard_form(self) -> lqr.LinSDynamics:
        # pylint:disable=missing-function-docstring
        return self.params.as_linsdynamics()


class TVLinearModel(TVLinearDynamics):
    """Time-varying linear Gaussian dynamics model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        n_tau = n_state + n_ctrl
        dynamics = lqr.LinSDynamics(
            F=torch.randn(horizon, n_state, n_tau),
            f=torch.randn(horizon, n_state),
            W=torch.as_tensor(
                make_spd_matrix(n_dim=n_state, sample_shape=(horizon,)),
                dtype=torch.float32,
            ),
        )
        super().__init__(dynamics)


class InitStateDynamics(ptd.Distribution):
    """Initial state distribution as a multivariate Normal.

    All outputs are named Tensors.

    Args:
        loc: location of the distribution
        covariance_matrix: covariance matrix of the distribution. May be
            non-diagonal.
    """

    # pylint:disable=missing-class-docstring
    def __init__(self, loc: Tensor, covariance_matrix: Tensor):
        super().__init__()
        self.dist = TVMultivariateNormal()
        self.loc = nn.Parameter(nt.unnamed(loc))
        self.ltril, self.pre_diag = nt.unnamed(
            *disassemble_covariance(nt.matrix(covariance_matrix))
        )

    def scale_tril(self) -> Tensor:
        # pylint:disable=missing-function-docstring
        return nt.matrix(assemble_scale_tril(self.ltril, self.pre_diag))

    def forward(self) -> DistParams:
        # pylint:disable=missing-function-docstring
        return {
            "loc": nt.vector(self.loc),
            "scale_tril": self.scale_tril(),
            "time": nt.vector(torch.zeros_like(self.loc[..., -1:], dtype=torch.long)),
        }

    def sample(self, sample_shape: List[int] = ()) -> SampleLogp:
        params = self()
        return self.dist.sample(params, sample_shape)

    def rsample(self, sample_shape: List[int] = ()) -> SampleLogp:
        params = self()
        return self.dist.rsample(params, sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        value = nt.vector(value)
        params = self()
        return self.dist.log_prob(value, params)

    def standard_form(self) -> Tuple[Tensor, Tensor]:
        # pylint:disable=missing-function-docstring
        scale_tril = self.scale_tril()
        covariance_matrix = scale_tril @ nt.transpose(scale_tril)
        return nt.vector(self.loc), covariance_matrix


class InitStateModel(InitStateDynamics):
    """Gaussian initial state distribution model."""

    def __init__(self, n_state: int, seed: Optional[int] = None):
        loc = torch.zeros(n_state)
        covariance_matrix = as_float_tensor(make_spd_matrix(n_state, rng=seed))
        super().__init__(loc, covariance_matrix)
