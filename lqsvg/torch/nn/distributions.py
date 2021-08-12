"""Shared implementations between init state and transition dynamics."""
from typing import Optional, Sequence, Tuple

import nnrl.nn.distributions as ptd
import torch
from nnrl.nn.distributions.types import DistParams, SampleLogp
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs


class TVMultivariateNormal(ptd.ConditionalDistribution):
    """Time-varying multivariate Gaussian distribution.

    The methods here expect named tensors as inputs and returns named tensors.

    The `time` tensor in the distribution parameters refers to the timestep
    preceding the next sample. Therefore, for a given `time`, the timestep
    component of the next sample will be `time + 1`.

    For this reason, `time` is allowed to take values of -1, so we can generate
    the first sample in a sample path, which starts with a timestep of 0.

    If `horizon` is set and the distribution parameters have `time == horizon`,
    then this modules simulates an absorving state distribution: a dirac delta
    at the current state.
    """

    # pylint:disable=abstract-method
    horizon: Optional[int]

    def __init__(self, horizon: Optional[int] = None):
        super().__init__()
        self.horizon = horizon

    def sample(
        self, params: DistParams, sample_shape: Sequence[int] = ()
    ) -> SampleLogp:
        loc, scale_tril, time = _unpack_params(params, sample_shape, self.horizon)
        sample = _gen_sample(loc, scale_tril, time, self.horizon).detach()
        logp = _logp(loc, scale_tril, time, sample, self.horizon)
        return sample, logp

    def rsample(
        self, params: DistParams, sample_shape: Sequence[int] = ()
    ) -> SampleLogp:
        loc, scale_tril, time = _unpack_params(params, sample_shape, self.horizon)
        sample = _gen_sample(loc, scale_tril, time, self.horizon)
        logp = _logp(loc, scale_tril, time, sample, self.horizon)
        return sample, logp

    def log_prob(self, value: Tensor, params: DistParams) -> Tensor:
        loc, scale_tril, time = _unpack_params(params, horizon=self.horizon)
        logp = _logp(loc, scale_tril, time, value, self.horizon)
        return logp


def _unpack_params(
    params: DistParams, sample_shape: Sequence[int] = (), horizon: Optional[int] = None
) -> Tuple[Tensor, Tensor, IntTensor]:
    time = params["time"]
    if horizon is None:
        loc = params["loc"]
    else:
        loc = nt.where(time.eq(horizon), params["state"], params["loc"])
    scale_tril = params["scale_tril"]

    sample_names = tuple(f"B{i+1}" for i, _ in enumerate(sample_shape))
    loc, scale_tril, time = (
        x.expand(torch.Size(sample_shape) + x.shape).refine_names(*sample_names, ...)
        for x in (loc, scale_tril, time)
    )
    return loc, scale_tril, time


def _gen_sample(
    loc: Tensor, scale_tril: Tensor, time: IntTensor, horizon: Optional[int] = None
) -> Tensor:
    next_obs = _transition(loc, scale_tril, time)
    if not horizon:
        return next_obs

    # Filter results
    # We're in an absorving state if the current timestep is the horizon
    return nt.where(time.eq(horizon), pack_obs(loc, time), next_obs)


def _transition(loc: Tensor, scale_tril: Tensor, time: IntTensor) -> Tensor:
    loc, scale_tril = nt.unnamed(loc, scale_tril)
    dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
    state = dist.rsample().refine_names(*time.names)
    return pack_obs(state, time + 1)


def _logp(
    loc: Tensor,
    scale_tril: Tensor,
    cur_time: Tensor,
    value: Tensor,
    horizon: Optional[int] = None,
) -> Tensor:
    # Align input tensors
    state, time = unpack_obs(value)
    loc, state = torch.broadcast_tensors(loc, state)
    cur_time, time = torch.broadcast_tensors(cur_time, time)

    # Consider normal state transition
    cur_time, time = nt.vector_to_scalar(cur_time, time)
    trans_logp = _trans_logp(loc, scale_tril, cur_time, state, time)
    if not horizon:
        return trans_logp

    # If horizon is set, treat absorving state transitions
    absorving_logp = _absorving_logp(loc, cur_time, state, time)

    # Filter results
    # We're in an absorving state if the current timestep is the horizon
    return nt.where(cur_time.eq(horizon), absorving_logp, trans_logp)


def _absorving_logp(
    cur_state: Tensor, cur_time: IntTensor, state: Tensor, time: IntTensor
) -> Tensor:
    # We assume time is a named scalar tensor
    cur_obs = pack_obs(cur_state, nt.scalar_to_vector(cur_time))
    obs = pack_obs(state, nt.scalar_to_vector(time))
    return nt.where(
        # Point mass only at the same state
        nt.reduce_all(nt.isclose(cur_obs, obs), dim="R"),
        torch.zeros_like(time, dtype=torch.float),
        torch.full_like(time, fill_value=float("nan"), dtype=torch.float),
    )


def _trans_logp(
    loc: Tensor,
    scale_tril: Tensor,
    cur_time: IntTensor,
    state: Tensor,
    time: IntTensor,
) -> Tensor:
    loc, scale_tril = nt.unnamed(loc, scale_tril)
    dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
    trans_logp: Tensor = dist.log_prob(nt.unnamed(state))
    trans_logp = nt.where(
        # Logp only defined at next timestep
        time.eq(cur_time + 1),
        trans_logp,
        torch.full(time.shape, fill_value=float("nan")),
    )
    # We assume time is a named scalar tensor
    return trans_logp.refine_names(*time.names)
