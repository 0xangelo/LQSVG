"""Functions for trajectories."""
import functools
from typing import Callable

import torch
from nnrl.nn import distributions as ptd
from torch import Tensor, nn

from lqsvg.torch import named as nt
from lqsvg.torch.nn import GRUGaussDynamics


@functools.singledispatch
def log_prob_fn(
    params_fn: nn.Module, dist: ptd.ConditionalDistribution
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Builds a mapping from trajectory segments to log-probabilities."""

    def func(obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        obs, act, new_obs = (x.align_to("H", ..., "R") for x in (obs, act, new_obs))

        params = params_fn(obs, act)
        return dist.log_prob(new_obs, params).sum(dim="H")

    return func


@log_prob_fn.register
def _(
    params_fn: GRUGaussDynamics, dist: ptd.ConditionalDistribution
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    """Log-probobality of a trajectory segment under a GRU dynamics model."""

    def func(obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        # noinspection PyTypeChecker
        obs_ = obs.select("H", 0)
        context = None
        logps = []
        # noinspection PyArgumentList
        for t in range(obs.size("H")):
            # noinspection PyTypeChecker
            params = params_fn(obs_, act.select("H", t), context=context)
            # noinspection PyTypeChecker
            logp_ = nt.where(
                nt.vector_to_scalar(params["time"]) == params_fn.horizon,
                torch.zeros_like(obs_.select("R", 0)),
                dist.log_prob(new_obs.select("H", t), params),
            )
            logps += [logp_]

            context = params["context"]
            obs_, _ = dist.rsample(params)

        return nt.stack_horizon(*logps).sum("H")

    return func
