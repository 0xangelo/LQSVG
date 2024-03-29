"""NN trajectory segment models."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import nnrl.nn as nnx
import torch
from nnrl.nn.model import StochasticModel
from nnrl.nn.model.stochastic.single import DynamicsParams, MLPModel
from nnrl.nn.networks.mlp import StateActionMLP
from nnrl.types import TensorDict
from torch import Tensor, nn
from torch.nn.functional import softplus

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.torch.nn.diagscale import DiagScale
from lqsvg.torch.nn.distributions import TVMultivariateNormal

from .linear import LinearDynamicsModule, LinearNormalParamsMixin

__all__ = [
    "LinearTransitionModel",
    "MLPDynamicsModel",
    "LinearDiagDynamicsModel",
    "GRUGaussDynamics",
]


class LinearTransitionModel(LinearDynamicsModule):
    """Linear Gaussian transition model."""


class LinearDiagNormalParams(LinearNormalParamsMixin, nn.Module):
    """Linear state-action conditional diagonal Gaussian parameters."""

    n_state: int
    n_ctrl: int
    horizon: int
    F: nn.Parameter
    f: nn.Parameter
    pre_diag: nn.Parameter
    _softplus_beta: float = 0.2

    def __init__(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        # pylint:disable=invalid-name
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


class LinearDiagDynamicsModel(StochasticModel):
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


class MLPDynamicsModel(StochasticModel):
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
        # Define input/output spaces.
        self.n_state, self.n_ctrl, self.horizon = n_state, n_ctrl, horizon
        obs_space, act_space = lqr.spaces_from_dims(n_state, n_ctrl, horizon)

        # Create encoder and Normal parameters head
        spec = MLPModel.spec_cls(
            units=hunits, activation=activation, input_dependent_scale=False
        )
        encoder = StateActionMLP(obs_space, act_space, spec)

        params = nnx.NormalParams(
            encoder.out_features,
            self.n_state,
            input_dependent_scale=spec.input_dependent_scale,
            bound_parameters=not spec.fix_logvar_bounds,
        )
        if spec.fix_logvar_bounds:
            params.max_logvar.fill_(2)
            params.min_logvar.fill_(-20)
        params = DynamicsParams(encoder, params)

        # Create conditional distribution
        dist = TVMultivariateNormal(self.horizon)
        super().__init__(params, dist)

    def forward(self, obs: Tensor, action: Tensor) -> TensorDict:
        """Map observation and action to diagonal Normal parameters."""
        state, time = lqr.unpack_obs(obs)
        obs_ = torch.cat([state, time.float() / self.horizon], dim="R")
        mlp_params = self.params(*nt.unnamed(obs_, action))

        # Convert scale vector to scale tril
        scale_tril = torch.diag_embed(mlp_params["scale"])

        return {
            "loc": mlp_params["loc"],
            "scale_tril": scale_tril,
            "time": time,
            "state": state,
        }


class GRUGaussDynamics(nn.Module):
    """Diagonal Gaussian next-state distribution using GRU cells.

    Args:
        mlp_hunits: sequence of hidden unit sizes for the encoding and decoding
            multilayer perceptrons. May be empty
        gru_hunits: sequence of hidden unit sizes for the GRU cells. This
            module requires at leat one GRU cell.
        mlp_activ: activation function for the MLPs
    """

    # pylint:disable=too-many-instance-attributes

    input_names: Sequence[str] = ("H", "B", "R")

    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
        mlp_hunits: tuple[int, ...],
        gru_hunits: tuple[int, ...],
        mlp_activ: str = "ReLU",
    ):
        # pylint:disable=too-many-arguments
        assert gru_hunits, f"Must have at least 1 GRU cell, got {gru_hunits}"
        assert all(
            h == gru_hunits[0] for h in gru_hunits
        ), "Varying sizes of GRU hidden units is unsupported"
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.horizon = horizon
        self.mlp_hunits = mlp_hunits
        self.mlp_activ = mlp_activ
        self.gru_hunits = gru_hunits

        # Slight departure from the original paper: we use the last activations
        # of the MLP (no final linear layer) as inputs to the GRU. The original
        # uses a final linear layer in the MLP projecting to the GRU hidden size
        # and sets input_size=<GRU hidden size> in the GRU.
        self.mlp_enc = nnx.FullyConnected(
            self.n_state + self.n_ctrl + 1,
            units=self.mlp_hunits,
            activation=self.mlp_activ,
        )
        self.gru = nn.GRU(
            input_size=self.mlp_enc.out_features,
            hidden_size=self.gru_hunits[0],
            num_layers=len(self.gru_hunits),
            batch_first=False,
        )
        self.mlp_dec = nnx.FullyConnected(
            self.gru_hunits[-1], units=self.mlp_hunits, activation=self.mlp_activ
        )

        self.loc_head = nn.Linear(self.mlp_dec.out_features, self.n_state)
        self.scale_tril_head = DiagScale(
            self.mlp_dec.out_features,  # Unused
            self.n_state,
            input_dependent_scale=False,
        )

        self.dist = TVMultivariateNormal(self.horizon)

    def forward(
        self, obs: Tensor, action: Tensor, context: Optional[Tensor] = None
    ) -> TensorDict:
        # pylint:disable=missing-function-docstring
        # Concatenate state and action vectors and normalize time in [0, 1]
        state, time = lqr.unpack_obs(obs)
        vec = torch.cat((state, time.float() / self.horizon, action), dim="R")
        # Ensure minimal dims for subsequent layers
        vec = nt.unnamed(vec.align_to(*self.input_names))

        # Embed, transition and decode
        z_emb = self.mlp_enc(vec)
        h_emb, new_context = self.gru(z_emb, hx=context)
        # Predict residual mean of next state instead of next state directly
        residual = self.mlp_dec(h_emb)
        # Refine and squeeze dimensions
        residual = residual.refine_names(*self.input_names).squeeze()

        # Predict Gaussian parameters
        loc = state + self.loc_head(residual)
        scale_tril = self.scale_tril_head(residual)

        return {
            "loc": loc,
            "scale_tril": scale_tril,
            "time": time,
            "state": state,
            "context": new_context,
        }
