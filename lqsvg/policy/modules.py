"""PyTorch modules for neural network agents."""
from __future__ import annotations

from gym.spaces import Box
from raylab.policy.modules.model import StochasticModel
from raylab.torch.nn.distributions.types import SampleLogp
from raylab.utils.types import TensorDict
from torch import nn
from torch import Tensor


class InputNormModel(StochasticModel):
    """StochasticModel that applies normalization layer to observation inputs."""

    def __init__(self, model: StochasticModel, obs_space: Box):
        assert isinstance(obs_space, Box)
        super().__init__(params_module=model.params, dist_module=model.dist)
        self._model = model
        self.obs_normalizer = nn.LayerNorm(normalized_shape=obs_space.shape)

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        obs = self.obs_normalizer(obs)
        params = self._model(obs, action)
        return params

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
