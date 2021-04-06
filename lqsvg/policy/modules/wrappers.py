"""Transition model wrappers."""
from __future__ import annotations

from raylab.policy.modules.model import StochasticModel
from raylab.torch.nn.distributions.types import SampleLogp
from raylab.utils.types import TensorDict
from torch import Tensor, nn

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs

__all__ = [
    "StochasticModelWrapper",
    "LayerNormModel",
    "BatchNormModel",
    "ResidualModel",
]


class StochasticModelWrapper(StochasticModel):
    """Wraps the stochastic model to allow a modular transformation."""

    def __init__(self, model: StochasticModel):
        super().__init__(params_module=model.params, dist_module=model.dist)
        self._model = model

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        return self._model(obs, action)

    def sample(self, params: TensorDict, sample_shape: list[int] = ()) -> SampleLogp:
        return self._model.sample(params, sample_shape)

    def rsample(self, params: TensorDict, sample_shape: list[int] = ()) -> SampleLogp:
        return self._model.rsample(params, sample_shape)

    def log_prob(self, next_obs: Tensor, params: TensorDict) -> Tensor:
        return self._model.log_prob(next_obs, params)

    def cdf(self, next_obs: Tensor, params: TensorDict) -> Tensor:
        return self._model.cdf(next_obs, params)

    def icdf(self, prob, params: TensorDict) -> Tensor:
        return self._model.icdf(prob, params)

    def reproduce(self, next_obs, params: TensorDict) -> SampleLogp:
        return self._model.reproduce(next_obs, params)

    def deterministic(self, params: TensorDict) -> SampleLogp:
        return self._model.deterministic(params)


class LayerNormModel(StochasticModelWrapper):
    """Applies Layer Normalization to state inputs."""

    def __init__(self, model: StochasticModel, n_state: int):
        super().__init__(model)
        self.normalizer = nn.LayerNorm(normalized_shape=n_state)

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        state, time = unpack_obs(obs)
        state = self.normalizer(nt.unnamed(state)).refine_names(*state.names)
        obs = pack_obs(state, time)
        return self._model(obs, action)


class BatchNormModel(StochasticModelWrapper):
    """Applies Batch Normalization to state inputs."""

    def __init__(self, model: StochasticModel, n_state: int):
        super().__init__(model)
        self.normalizer = nn.BatchNorm1d(num_features=n_state)
        self.n_state = n_state

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        state, time = unpack_obs(obs)
        state = (
            self.normalizer(nt.unnamed(state).reshape(-1, self.n_state))
            .reshape_as(state)
            .refine_names(*state.names)
        )
        obs = pack_obs(state, time)
        return self._model(obs, action)


class ResidualModel(StochasticModelWrapper):
    """Predicts state transition residuals."""

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        params = self.params(obs, action)
        state, _ = unpack_obs(obs)
        params["state"] = state
        return params

    def sample(self, params: TensorDict, sample_shape: list[int] = ()) -> SampleLogp:
        residual, log_prob = self.dist.sample(params, sample_shape)
        delta, time = unpack_obs(residual)
        next_obs = pack_obs(params["state"] + delta, time)
        return next_obs, log_prob

    def rsample(self, params: TensorDict, sample_shape: list[int] = ()) -> SampleLogp:
        residual, log_prob = self.dist.rsample(params, sample_shape)
        delta, time = unpack_obs(residual)
        next_obs = pack_obs(params["state"] + delta, time)
        return next_obs, log_prob

    def log_prob(self, next_obs: Tensor, params: TensorDict) -> Tensor:
        next_state, time = unpack_obs(next_obs)
        residual = pack_obs(next_state - params["state"], time)
        return self.dist.log_prob(residual, params)

    def cdf(self, next_obs: Tensor, params: TensorDict) -> Tensor:
        next_state, time = unpack_obs(next_obs)
        residual = pack_obs(next_state - params["state"], time)
        return self.dist.cdf(residual, params)

    def icdf(self, prob, params: TensorDict) -> Tensor:
        residual = self.dist.icdf(prob, params)
        delta, time = unpack_obs(residual)
        return pack_obs(params["state"] + delta, time)

    def reproduce(self, next_obs, params: TensorDict) -> SampleLogp:
        next_state, time = unpack_obs(next_obs)
        residual = pack_obs(next_state - params["state"], time)
        residual_, log_prob_ = self.dist.reproduce(residual, params)
        delta_, time_ = unpack_obs(residual_)
        return pack_obs(params["state"] + delta_, time_), log_prob_

    def deterministic(self, params: TensorDict) -> SampleLogp:
        residual, log_prob = self.dist.deterministic(params)
        delta, time = unpack_obs(residual)
        return pack_obs(params["state"] + delta, time), log_prob
