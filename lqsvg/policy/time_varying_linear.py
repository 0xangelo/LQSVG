# pylint:disable=missing-docstring,invalid-name,unsubscriptable-object
from __future__ import annotations

from typing import Optional

from gym.spaces import Box
from nnrl.nn.model import StochasticModel
from torch import nn

from lqsvg.envs import lqr
from lqsvg.torch.nn.dynamics.linear import LinearDynamics
from lqsvg.torch.nn.dynamics.segment import LinearTransitionModel
from lqsvg.torch.nn.env import EnvModule, LQGModule
from lqsvg.torch.nn.initstate import InitStateModel
from lqsvg.torch.nn.policy import TVLinearPolicy
from lqsvg.torch.nn.reward import QuadRewardModel
from lqsvg.torch.nn.wrappers import BatchNormModel, LayerNormModel, ResidualModel


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
