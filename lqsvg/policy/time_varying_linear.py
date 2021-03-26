# pylint:disable=missing-docstring,invalid-name,unsubscriptable-object
from __future__ import annotations

from functools import cached_property

import torch.nn as nn
from gym.spaces import Box
from raylab.options import configure
from raylab.options import option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy

from lqsvg.envs import lqr
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import LinearDynamics
from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.envs.lqr.modules.general import LQGModule

from .modules import InitStateModel
from .modules import LayerNormModel
from .modules import LinearTransModel
from .modules import QuadRewardModel
from .modules import ResidualModel
from .modules import TVLinearPolicy
from .modules import TVLinearTransModel


def glorot_init_policy(module: nn.Module):
    """Apply Glorot initialization to time-varying linear policy.

    Args:
        module: neural network PyTorch module
    """
    if isinstance(module, TVLinearPolicy):
        nn.init.xavier_uniform_(module.K)
        nn.init.constant_(module.k, 0.0)


def glorot_init_model(module: nn.Module):
    """Apply Glorot initialization to linear dynamics model.

    Args:
        module: neural network PyTorch module
    """
    if isinstance(module, LinearDynamics):
        nn.init.xavier_uniform_(module.F)
        nn.init.constant_(module.f, 0.0)


class TimeVaryingLinear(nn.Module):
    # pylint:disable=abstract-method
    actor: TVLinearPolicy
    behavior: TVLinearPolicy
    model: EnvModule

    def __init__(self, obs_space: Box, action_space: Box, config: dict):
        super().__init__()
        n_state, n_ctrl, horizon = lqr.dims_from_spaces(obs_space, action_space)

        # Policy
        self.actor = TVLinearPolicy(n_state, n_ctrl, horizon)
        self.behavior = self.actor

        # Model
        if config["stationary_model"]:
            trans = LinearTransModel(n_state, n_ctrl, horizon)
        else:
            trans = TVLinearTransModel(n_state, n_ctrl, horizon)
        if config["residual_model"]:
            trans = ResidualModel(trans)
        if config["model_input_norm"]:
            trans = LayerNormModel(trans, n_state)

        model_cls = LQGModule if isinstance(trans, LinearDynamics) else EnvModule
        self.model = model_cls(
            dims=(n_state, n_ctrl, horizon),
            trans=trans,
            reward=QuadRewardModel(n_state, n_ctrl, horizon),
            init=InitStateModel(n_state=n_state, seed=config.get("seed", None)),
        )

        # Parameter initialization
        if config["policy_initializer"] == "xavier_uniform":
            self.apply(glorot_init_policy)
        if config["model_initializer"] == "xavier_uniform":
            self.apply(glorot_init_model)

    def standard_form(
        self,
    ) -> tuple[lqr.Linear, lqr.LinSDynamics, lqr.QuadCost, lqr.GaussInit]:
        actor = self.actor.standard_form()
        trans, rew, init = self.model.standard_form()
        return actor, trans, rew, init


# noinspection PyAbstractClass
@configure
@option(
    "module/policy_initializer",
    default="xavier_uniform",
    help="""\
    How to initialize the policy's parameters. One of:
    - 'xavier_uniform'
    - 'from_optimal'
    """,
)
@option(
    "module/model_initializer",
    default="xavier_uniform",
    help="""\
    How to initialize the model's parameters. One of:
    - 'xavier_uniform'
    - 'standard_normal'
    """,
)
@option(
    "module/stationary_model",
    default=False,
    help="Whether to use a stationary linear Gaussian dynamics model.",
)
@option(
    "module/residual_model",
    default=False,
    help="Whether to predict change in state or absolute state variables.",
)
@option(
    "module/model_input_norm",
    default=False,
    help="Whether to wrap the transition model to normalize observation inputs.",
)
@option("exploration_config/type", default="raylab.utils.exploration.GaussianNoise")
@option("exploration_config/pure_exploration_steps", default=0)
@option("explore", default=False, override=True)
class LQGPolicy(TorchPolicy):
    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy
    observation_space: Box
    action_space: Box

    @property
    def n_state(self):
        n_state, _, _ = self.space_dims
        return n_state

    @property
    def n_ctrl(self):
        _, n_ctrl, _ = self.space_dims
        return n_ctrl

    @property
    def horizon(self):
        _, _, horizon = self.space_dims
        return horizon

    @cached_property
    def space_dims(self) -> tuple[int, int, int]:
        return lqr.dims_from_spaces(self.observation_space, self.action_space)

    def initialize_from_lqg(self, env: TorchLQGMixin):
        if self.config["module"]["policy_initializer"] == "from_optimal":
            optimal: lqr.Linear = env.solution[0]
            self.module.actor.initialize_from_optimal(optimal)
        self.module.model.reward.copy(env.cost)

    def _make_module(
        self, obs_space: Box, action_space: Box, config: dict
    ) -> nn.Module:
        module = TimeVaryingLinear(obs_space, action_space, config["module"])
        return module
