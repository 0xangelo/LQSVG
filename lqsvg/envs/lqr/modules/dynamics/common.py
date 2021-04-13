"""Shared implementations between init state and transition dynamics."""
from typing import List, Optional, Tuple

import raylab.torch.nn.distributions as ptd
import torch
from raylab.torch.nn.distributions.types import DistParams, SampleLogp
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs


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
    horizon: Optional[int]

    def __init__(self, horizon: Optional[int] = None):
        super().__init__()
        self.horizon = horizon

    def sample(self, params: DistParams, sample_shape: List[int] = ()) -> SampleLogp:
        loc, scale_tril, time = _unpack_params(params, sample_shape)
        sample = self._gen_sample(loc, scale_tril, time).detach()
        logp = self._logp(loc, scale_tril, time, sample)
        return sample, logp

    def rsample(self, params: DistParams, sample_shape: List[int] = ()) -> SampleLogp:
        loc, scale_tril, time = _unpack_params(params, sample_shape)
        sample = self._gen_sample(loc, scale_tril, time)
        logp = self._logp(loc, scale_tril, time, sample)
        return sample, logp

    def log_prob(self, value: Tensor, params: DistParams) -> Tensor:
        value = nt.vector(value)
        loc, scale_tril, time = _unpack_params(params)
        logp = self._logp(loc, scale_tril, time, value)
        return logp

    def _gen_sample(self, loc: Tensor, scale_tril: Tensor, time: IntTensor) -> Tensor:
        dist = torch.distributions.MultivariateNormal(
            loc=nt.unnamed(loc), scale_tril=nt.unnamed(scale_tril)
        )
        terminal = time.eq(self.horizon) if self.horizon else torch.zeros(()).bool()
        sample = nt.where(terminal, loc, dist.rsample().refine_names(*loc.names))
        return pack_obs(sample, time)

    def _logp(
        self, loc: Tensor, scale_tril: Tensor, time: Tensor, value: Tensor
    ) -> Tensor:
        # Align input tensors
        state, time_ = unpack_obs(value)
        loc = loc.align_as(state)
        time, time_ = torch.broadcast_tensors(time, time_)

        # Consider normal and absorving state transitions
        time_ = nt.vector_to_scalar(time_)
        trans_logp = self._trans_logp(loc, scale_tril, state, time_)
        absorving_logp = self._absorving_logp(loc, state, time_)

        # Filter results
        time = nt.vector_to_scalar(time)
        terminal = (
            time.eq(self.horizon) if self.horizon else torch.zeros_like(time).bool()
        )
        logp = nt.where(
            # Logp only defined at next timestep
            time.eq(time_),
            # If terminal, point mass only at the same state
            # If not terminal, density defined for states in next timestep only
            nt.where(terminal, absorving_logp, trans_logp),
            torch.full(time.shape, fill_value=float("nan")),
        )
        return logp

    @staticmethod
    def _trans_logp(
        loc: Tensor, scale_tril: Tensor, state: Tensor, time: Tensor
    ) -> Tensor:
        loc, scale_tril = nt.unnamed(loc, scale_tril)
        dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        trans_logp: Tensor = dist.log_prob(nt.unnamed(state))
        return trans_logp.refine_names(*time.names)

    @staticmethod
    def _absorving_logp(loc: Tensor, state: Tensor, time: Tensor) -> Tensor:
        return nt.where(
            nt.reduce_all(nt.isclose(loc, state), dim="R"),
            torch.zeros(time.shape),
            torch.full(time.shape, fill_value=float("nan")),
        )


def _unpack_params(
    params: DistParams, sample_shape: List[int] = ()
) -> Tuple[Tensor, Tensor, IntTensor]:
    loc = params["loc"]
    scale_tril = params["scale_tril"]
    time = params["time"]
    sample_names = tuple(f"B{i+1}" for i, _ in enumerate(sample_shape))
    loc, scale_tril, time = (
        x.expand(sample_shape + x.shape).refine_names(*sample_names, ...)
        for x in (loc, scale_tril, time)
    )
    return loc, scale_tril, time
