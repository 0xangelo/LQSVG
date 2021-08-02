"""Shared implementations between init state and transition dynamics."""
from typing import List, Optional, Tuple

import nnrl.nn.distributions as ptd
import torch
from nnrl.nn.distributions.types import DistParams, SampleLogp
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs


class TVMultivariateNormal(ptd.ConditionalDistribution):
    """Time-varying multivariate Gaussian distribution.

    This class' methods expect named tensors as inputs and returns named
    tensors.

    Note:
        The `time` tensor in the distribution parameters refers to the
        timestep preceding the next sample. Therefore, if `time = X`, the
        timestep component of the next sample will be `time + 1`.

        For this reason, `time` is allowed to take values of -1, so we
        can generate the first sample in a sample path, which starts with
        a timestep of 0.
    """

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
        loc, scale_tril, time = _unpack_params(params)
        logp = self._logp(loc, scale_tril, time, value)
        return logp

    ###########################################################################
    # Internal API
    ###########################################################################

    def _gen_sample(self, loc: Tensor, scale_tril: Tensor, time: IntTensor) -> Tensor:
        next_obs = self._transition(loc, scale_tril, time)
        if not self.horizon:
            return next_obs

        # Filter results
        # We're in an absorving state if the current timestep is the horizon
        return nt.where(time.eq(self.horizon), pack_obs(loc, time), next_obs)

    @staticmethod
    def _transition(loc: Tensor, scale_tril: Tensor, time: IntTensor) -> Tensor:
        loc, scale_tril = nt.unnamed(loc, scale_tril)
        dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        state = dist.rsample().refine_names(*time.names)
        return pack_obs(state, time + 1)

    def _logp(
        self, loc: Tensor, scale_tril: Tensor, time: Tensor, value: Tensor
    ) -> Tensor:
        # Align input tensors
        state, time_ = unpack_obs(value)
        loc, state = torch.broadcast_tensors(loc, state)
        time, time_ = torch.broadcast_tensors(time, time_)

        # Consider normal state transition
        time, time_ = nt.vector_to_scalar(time, time_)
        trans_logp = self._trans_logp(loc, scale_tril, time, state, time_)
        if not self.horizon:
            return trans_logp

        # If horizon is set, treat absorving state transitions
        absorving_logp = self._absorving_logp(loc, time, state, time_)

        # Filter results
        # We're in an absorving state if the current timestep is the horizon
        return nt.where(time.eq(self.horizon), absorving_logp, trans_logp)

    @staticmethod
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

    @staticmethod
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
