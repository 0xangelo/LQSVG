"""NN transition models."""
# pylint:disable=invalid-name
from __future__ import annotations

import math

import torch
from raylab.policy.modules.model import MLPModel, StochasticModel
from raylab.utils.types import TensorDict
from torch import Tensor, nn
from torch.nn.functional import softplus

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import spaces_from_dims, unpack_obs
from lqsvg.envs.lqr.modules import LinearDynamicsModule
from lqsvg.envs.lqr.modules.dynamics.common import TVMultivariateNormal
from lqsvg.envs.lqr.modules.dynamics.linear import LinearNormalMixin

from .wrappers import StochasticModelWrapper

__all__ = [
    "LinearTransitionModel",
    "MLPDynamicsModel",
    "LinearDiagDynamicsModel",
    "SegmentStochasticModel",
]


class SegmentStochasticModel(StochasticModel):
    """Probabilistic model of trajectory segments."""

    def seg_log_prob(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Log-probability (density) of trajectory segment."""
        obs, act, new_obs = (x.align_to("H", ..., "R") for x in (obs, act, new_obs))

        return self.log_prob(new_obs, self(obs, act)).sum(dim="H")


class LinearTransitionModel(LinearDynamicsModule):
    """Linear Gaussian transition model."""


class LinearDiagNormalParams(LinearNormalMixin, nn.Module):
    """Linear state-action conditional diagonal Gaussian parameters."""

    n_state: int
    n_ctrl: int
    horizon: int
    F: nn.Parameter
    f: nn.Parameter
    pre_diag: nn.Parameter
    _softplus_beta: float = 0.2

    def __init__(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.horizon = horizon
        self.stationary = stationary

        h_size = 1 if stationary else horizon
        self.F = nn.Parameter(Tensor(h_size, n_state, n_state + n_ctrl))
        self.f = nn.Parameter(Tensor(h_size, n_state))
        self.pre_diag = nn.Parameter(Tensor(h_size, n_state))
        self.reset_parameters()

    def reset_parameters(self):
        """Default parameter initialization.

        Similar to the default initialization of `nn.Linear`.
        We initialize `pre_diag` so that the resulting covariance is the
        identity matrix.
        """
        nn.init.kaiming_uniform_(self.F, a=math.sqrt(5))
        fan_in = self.n_state + self.n_ctrl
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.f, -bound, bound)
        nn.init.constant_(self.pre_diag, 0)

    def scale_tril(self) -> Tensor:
        diag = softplus(self.pre_diag, beta=self._softplus_beta)
        return nt.matrix(torch.diag_embed(diag))


class LinearDiagDynamicsModel(SegmentStochasticModel):
    """Linear Gaussian model with diagonal covariance.

    Args:
        n_state: dimensionality of the state vectors
        n_ctrl: dimensionality of the control (action) vectors
        horizon: task horizon
        stationary: whether to model stationary dynamics
    """

    n_state: int
    n_ctrl: int
    horizon: int
    stationary: bool
    F: nn.Parameter
    f: nn.Parameter

    def __init__(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        # pylint:disable=invalid-name
        self.n_state, self.n_ctrl, self.horizon = n_state, n_ctrl, horizon
        self.stationary = stationary

        params_module = LinearDiagNormalParams(n_state, n_ctrl, horizon, stationary)
        dist_module = TVMultivariateNormal(horizon)
        super().__init__(params_module, dist_module)
        self.F = self.params.F
        self.f = self.params.f

    def dimensions(self) -> tuple[int, int, int]:
        """Return the state, action, and horizon size for this module."""
        return self.n_state, self.n_ctrl, self.horizon


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
        # pylint:disable=too-many-arguments
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
