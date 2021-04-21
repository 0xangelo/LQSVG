# pylint:disable=missing-docstring,invalid-name,unsubscriptable-object
from __future__ import annotations

from functools import cached_property
from typing import Optional

import torch.nn as nn
from gym.spaces import Box
from raylab.options import configure, option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.modules.model import StochasticModel

from lqsvg.envs import lqr
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import LinearDynamics
from lqsvg.envs.lqr.modules.general import EnvModule, LQGModule

from .modules import (
    BatchNormModel,
    InitStateModel,
    LayerNormModel,
    LinearTransitionModel,
    QuadRewardModel,
    ResidualModel,
    TVLinearPolicy,
)


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
        trans = self._trans_model(n_state, n_ctrl, horizon, config)
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

    def _trans_model(
        self, n_state: int, n_ctrl: int, horizon: int, config: dict
    ) -> StochasticModel:
        trans = LinearTransitionModel(
            n_state, n_ctrl, horizon, stationary=config["stationary_model"]
        )

        # Wrap model if needed
        trans = self._input_processing(trans, n_state, config["model_input_norm"])
        # This should be last, otherwise we may be learning residuals between
        # normalized states and unnormalized next states
        if config["residual_model"]:
            trans = ResidualModel(trans)
        return trans

    @staticmethod
    def _input_processing(
        trans: StochasticModel, n_state: int, input_norm: Optional[str]
    ) -> StochasticModel:
        if input_norm == "LayerNorm":
            trans = LayerNormModel(trans, n_state)
        elif input_norm == "BatchNorm":
            trans = BatchNormModel(trans, n_state)
        else:
            assert (
                input_norm is None
            ), f"Invalid 'model_input_norm' option: {input_norm}"

        return trans

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
    default=dict(min_abs_eigv=0.0, max_abs_eigv=1.0),
    help="""\
    How to initialize the policy's parameters. One of:
    - 'xavier_uniform'
    - 'noisy_optimal'
    - 'stabilize_sys'
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
    default=None,
    help="""\
    Wrap the transition model to normalize observation inputs. One of:
    - LayerNorm
    - BatchNorm
    - None
    """,
)
@option("exploration_config/type", default="raylab.utils.exploration.GaussianNoise")
@option("exploration_config/pure_exploration_steps", default=0)
@option("explore", default=False, override=True)
class LQGPolicy(TorchPolicy):
    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy
    observation_space: Box
    action_space: Box
    module: TimeVaryingLinear

    def setup(self, env: TorchLQGMixin):
        policy_init = self.config["module"]["policy_initializer"]
        if policy_init == "noisy_optimal":
            optimal: lqr.Linear = env.solution[0]
            self.module.actor.noisy_(optimal)
        elif policy_init == "stabilize_sys":
            self.module.actor.stabilize_(env.dynamics)

        self.module.model.reward.copy_(env.cost)

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

    def _make_module(
        self, obs_space: Box, action_space: Box, config: dict
    ) -> nn.Module:
        module = TimeVaryingLinear(obs_space, action_space, config["module"])
        return module
