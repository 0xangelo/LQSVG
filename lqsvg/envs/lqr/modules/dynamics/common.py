"""Shared implementations between init state and transition dynamics."""
from typing import List
from typing import Tuple

import raylab.torch.nn.distributions as ptd
import torch
from raylab.torch.nn.distributions.types import DistParams
from raylab.torch.nn.distributions.types import SampleLogp
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.utils import unpack_obs


def softplusinv(tensor: Tensor, *, beta: float = 1) -> Tensor:
    """Returns the inverse softplus transformation."""
    return torch.log(torch.exp(beta * tensor) - 1) / beta


def disassemble_covariance(tensor: Tensor, *, beta: float = 1) -> Tuple[Tensor, Tensor]:
    """Compute cholesky factor and break it into unconstrained parameters."""
    tril = nt.cholesky(tensor, upper=False)
    ltril = nt.tril(tril, diagonal=-1)
    pre_diag = softplusinv(nt.diagonal(tril, dim1="R", dim2="C"), beta=beta)
    return ltril, pre_diag


def assemble_scale_tril(ltril: Tensor, pre_diag: Tensor, *, beta: float = 1) -> Tensor:
    """Transform uncostrained parameters into covariance cholesky factor."""
    return ltril + torch.diag_embed(nt.softplus(pre_diag, beta=beta))


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


def _unpack_params(
    params: DistParams, sample_shape: List[int] = ()
) -> Tuple[Tensor, Tensor, Tensor]:
    loc = params["loc"]
    scale_tril = params["scale_tril"]
    time = params["time"]
    sample_names = tuple(f"B{i+1}" for i, _ in enumerate(sample_shape))
    loc, scale_tril, time = (
        x.expand(sample_shape + x.shape).refine_names(*sample_names, ...)
        for x in (loc, scale_tril, time)
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
