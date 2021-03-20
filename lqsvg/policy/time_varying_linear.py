# pylint:disable=missing-docstring,invalid-name,unsubscriptable-object
from __future__ import annotations

from functools import cached_property
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from raylab.options import configure
from raylab.options import option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.modules.actor import DeterministicPolicy
from torch import IntTensor
from torch import LongTensor
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_gaussinit
from lqsvg.envs.lqr.generators import make_linsdynamics
from lqsvg.envs.lqr.generators import make_quadcost
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import InitStateDynamics
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.envs.lqr.modules import QuadraticReward
from lqsvg.envs.lqr.modules import TVLinearDynamics
from lqsvg.envs.lqr.utils import unpack_obs


def perturb_policy(policy: lqr.Linear) -> lqr.Linear:
    """Perturb policy parameters to derive sub-optimal policies.

    Adds white noise to optimal policy parameters.

    Args:
        policy: optimal policy parameters

    Returns:
        Perturbed policy parameters
    """
    # pylint:disable=invalid-name
    n_state, n_ctrl, _ = lqr.dims_from_policy(policy)
    K, k = (g + 0.5 * torch.randn_like(g) / (n_state + np.sqrt(n_ctrl)) for g in policy)
    return K, k


class TVLinearFeedback(nn.Module):
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
    def __init__(self, obs_space: Box, action_space: Box):
        n_state, n_ctrl, horizon = lqr.dims_from_spaces(obs_space, action_space)
        action_linear = TVLinearFeedback(n_state, n_ctrl, horizon)
        super().__init__(
            encoder=nn.Identity(), action_linear=action_linear, squashing=nn.Identity()
        )

    def initialize_from_optimal(self, optimal: lqr.Linear):
        policy = perturb_policy(optimal)
        self.action_linear.copy(policy)

    def standard_form(self) -> lqr.Linear:
        return self.action_linear.gains()


class TVLinearTransModel(TVLinearDynamics):
    """Time-varying linear Gaussian dynamics model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        dynamics = make_linsdynamics(n_state, n_ctrl, horizon, stationary=False)
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


class TimeVaryingLinear(nn.Module):
    # pylint:disable=abstract-method
    actor: TVLinearPolicy
    behavior: TVLinearPolicy
    model: LQGModule

    def __init__(self, obs_space: Box, action_space: Box, config: dict):
        super().__init__()
        self.actor = TVLinearPolicy(obs_space, action_space)
        self.behavior = self.actor

        n_state, n_ctrl, horizon = lqr.dims_from_spaces(obs_space, action_space)
        self.model = LQGModule(
            TVLinearTransModel(n_state, n_ctrl, horizon),
            QuadRewardModel(n_state, n_ctrl, horizon),
            InitStateModel(n_state=n_state, seed=config.get("seed", None)),
        )

    def standard_form(
        self,
    ) -> tuple[lqr.Linear, lqr.LinSDynamics, lqr.QuadCost, lqr.GaussInit]:
        actor = self.actor.standard_form()
        trans, rew, init = self.model.standard_form()
        return actor, trans, rew, init


# noinspection PyAbstractClass
@configure
@option("exploration_config/type", default="raylab.utils.exploration.GaussianNoise")
@option("exploration_config/pure_exploration_steps", default=0)
@option("explore", default=False, override=True)
class LQGPolicy(TorchPolicy):
    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    @cached_property
    def n_state(self):
        n_state, _, _ = self.space_dims()
        return n_state

    @cached_property
    def n_ctrl(self):
        _, n_ctrl, _ = self.space_dims()
        return n_ctrl

    @cached_property
    def horizon(self):
        _, _, horizon = self.space_dims()
        return horizon

    def space_dims(self):
        return lqr.dims_from_spaces(self.observation_space, self.action_space)

    def initialize_from_lqg(self, env: TorchLQGMixin):
        optimal: lqr.Linear = env.solution[0]
        self.module.actor.initialize_from_optimal(optimal)
        self.module.model.reward.copy(env.cost)

    def _make_module(
        self, obs_space: Box, action_space: Box, config: dict
    ) -> nn.Module:
        return TimeVaryingLinear(obs_space, action_space, config)

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers["actor"] = torch.optim.Adam(self.module.actor.parameters(), lr=1e-4)
