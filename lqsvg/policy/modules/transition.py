"""NN transition models."""
# pylint:disable=invalid-name
from __future__ import annotations

import torch
from raylab.policy.modules.model import MLPModel
from raylab.utils.types import TensorDict
from torch import Tensor

from lqsvg.envs.lqr import spaces_from_dims, unpack_obs
from lqsvg.envs.lqr.modules import LinearDynamicsModule
from .wrappers import StochasticModelWrapper

__all__ = ["LinearTransitionModel", "MLPDynamicsModel"]


class LinearTransitionModel(LinearDynamicsModule):
    """Linear Gaussian transition model."""


class MLPDynamicsModel(StochasticModelWrapper):
    """Multilayer perceptron transition model."""

    n_state: int
    n_ctrl: int
    horizon: int

    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
        hunits: tuple[int, ...],
        activation: str,
    ):
        self.n_state, self.n_ctrl, self.horizon = n_state, n_ctrl, horizon
        obs_space, act_space = spaces_from_dims(n_state, n_ctrl, horizon)
        spec = MLPModel.spec_cls(
            units=hunits, activation=activation, input_dependent_scale=False
        )
        model = MLPModel(obs_space, act_space, spec)
        super().__init__(model)

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        state, time = unpack_obs(obs)
        obs = torch.cat([state, time.float() / self.horizon], dim="R")
        return self._model(obs, action)
