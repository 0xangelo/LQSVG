"""PyTorch modules for neural network agents."""
# pylint:disable=invalid-name
from __future__ import annotations

from typing import Optional
from typing import Union

import torch
from gym.spaces import Box
from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.model import StochasticModel
from raylab.torch.nn.distributions.types import SampleLogp
from raylab.utils.types import TensorDict
from torch import IntTensor
from torch import LongTensor
from torch import nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr import make_gaussinit
from lqsvg.envs.lqr.generators import make_linsdynamics
from lqsvg.envs.lqr.generators import make_quadcost
from lqsvg.envs.lqr.modules import InitStateDynamics
from lqsvg.envs.lqr.modules import LinearDynamicsModule
from lqsvg.envs.lqr.modules import QuadraticReward
from lqsvg.envs.lqr.modules import TVLinearDynamicsModule
from lqsvg.envs.lqr.utils import unpack_obs

from .utils import perturb_policy


class TVLinearFeedback(nn.Module):
    # pylint:disable=missing-docstring
    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__()
        K = torch.randn(horizon, n_ctrl, n_state)
        k = torch.randn(horizon, n_ctrl)
        self.K, self.k = (nn.Parameter(x) for x in (K, k))

    def _gains_at(self, index: Union[IntTensor, LongTensor]) -> tuple[Tensor, Tensor]:
        K = nt.horizon(nt.matrix(self.K))
        k = nt.horizon(nt.vector(self.k))
        K, k = (nt.index_by(x, dim="H", index=index) for x in (K, k))
        return K, k

    def forward(self, obs: Tensor) -> Tensor:
        obs = nt.vector(obs)
        state, time = unpack_obs(obs)

        time = nt.vector_to_scalar(time)
        K, k = self._gains_at(time)

        ctrl = K @ nt.vector_to_matrix(state) + nt.vector_to_matrix(k)
        ctrl = nt.matrix_to_vector(ctrl)
        return ctrl

    @classmethod
    def from_existing(cls, policy: lqr.Linear):
        n_state, n_ctrl, horizon = lqr.dims_from_policy(policy)
        new = cls(n_state, n_ctrl, horizon)
        new.copy(policy)
        return new

    def copy(self, policy: lqr.Linear):
        K, k = lqr.named.refine_linear_input(policy)
        self.K.data.copy_(K)
        self.k.data.copy_(nt.matrix_to_vector(k))

    def gains(self, named: bool = True) -> lqr.Linear:
        K, k = self.K, self.k
        if named:
            K = nt.horizon(nt.matrix(K))
            k = nt.horizon(nt.vector(k))
        K.grad, k.grad = self.K.grad, self.k.grad
        return K, k


class TVLinearPolicy(DeterministicPolicy):
    """Time-varying affine feedback policy as a DeterministicPolicy module."""

    K: nn.Parameter
    k: nn.Parameter

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        action_linear = TVLinearFeedback(n_state, n_ctrl, horizon)
        super().__init__(
            encoder=nn.Identity(), action_linear=action_linear, squashing=nn.Identity()
        )
        self.K = self.action_linear.K
        self.k = self.action_linear.k

    def initialize_from_optimal(self, optimal: lqr.Linear):
        # pylint:disable=missing-function-docstring
        policy = perturb_policy(optimal)
        self.action_linear.copy(policy)

    def standard_form(self) -> lqr.Linear:
        # pylint:disable=missing-function-docstring
        return self.action_linear.gains()


class TVLinearTransModel(TVLinearDynamicsModule):
    """Time-varying linear Gaussian transition model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        dynamics = make_linsdynamics(
            n_state, n_ctrl, horizon, stationary=False, sample_covariance=True
        )
        super().__init__(dynamics)


class LinearTransModel(LinearDynamicsModule):
    """Stationary linear Gaussian transition model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        dynamics = make_linsdynamics(
            n_state, n_ctrl, horizon, stationary=True, sample_covariance=True
        )
        super().__init__(dynamics)


class QuadRewardModel(QuadraticReward):
    """Time-varying quadratic reward model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        cost = make_quadcost(n_state, n_ctrl, horizon, stationary=False)
        super().__init__(cost)


class InitStateModel(InitStateDynamics):
    """Gaussian initial state distribution model."""

    def __init__(self, n_state: int, seed: Optional[int] = None):
        init = make_gaussinit(n_state, sample_covariance=True, rng=seed)
        super().__init__(init)


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
    """Applies Layer Normalization to observation inputs."""

    def __init__(self, model: StochasticModel, obs_space: Box):
        assert isinstance(obs_space, Box)
        super().__init__(model)
        self.obs_normalizer = nn.LayerNorm(normalized_shape=obs_space.shape)

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        obs = self.obs_normalizer(obs)
        params = self._model(obs, action)
        return params


class BatchNormModel(StochasticModelWrapper):
    """Applies Batch Normalization to observation inputs."""

    def __init__(self, model: StochasticModel, obs_space: Box):
        assert isinstance(obs_space, Box)
        super().__init__(model)
        self.obs_normalizer = nn.BatchNorm1d(num_features=obs_space.shape[0])

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        obs = self.obs_normalizer(obs)
        params = self._model(obs, action)
        return params
