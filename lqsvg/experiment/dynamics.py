"""Functions for using dynamics models."""
from typing import Callable, Optional, Tuple

from nnrl.nn.distributions.types import SampleLogp
from nnrl.types import TensorDict
from torch import Tensor

from lqsvg.experiment.types import RecurrentDynamics, StateDynamics


def markovian_state_sampler(
    params_fn: Callable[[Tensor, Tensor], TensorDict],
    sampler_fn: Callable[[TensorDict], SampleLogp],
) -> StateDynamics:
    """Combine state-action conditional params and conditional state dist."""

    def sampler(obs: Tensor, act: Tensor) -> SampleLogp:
        params = params_fn(obs, act)
        return sampler_fn(params)

    return sampler


def recurrent_state_sampler(
    params_fn: Callable[[Tensor, Tensor, Tensor], TensorDict],
    sampler_fn: Callable[[TensorDict], SampleLogp],
) -> RecurrentDynamics:
    """Combine contextual dist params and conditional state dist."""

    def sampler(
        obs: Tensor, act: Tensor, ctx: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        params = params_fn(obs, act, ctx)
        sample, logp = sampler_fn(params)
        return sample, logp, params["context"]

    return sampler
