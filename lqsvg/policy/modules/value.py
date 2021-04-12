"""Quadratic value functions as NN modules."""
from __future__ import annotations

import torch
import torch.nn as nn
from raylab.policy.modules.critic import QValue, VValue
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import Quadratic

# noinspection PyTypeChecker
from lqsvg.envs.lqr.utils import unpack_obs


def index_quadratic_parameters(
    quad: nn.Parameter, linear: nn.Parameter, const: nn.Parameter, time: IntTensor
) -> tuple[Tensor, Tensor, Tensor]:
    # pylint:disable=missing-function-docstring
    quad = nt.horizon(nt.matrix(quad))
    linear = nt.horizon(nt.vector(linear))
    const = nt.horizon(nt.scalar(const))
    index = nt.vector_to_scalar(time)
    quad, linear, const = map(
        lambda x: nt.index_by(x, dim="H", index=index), (quad, linear, const)
    )
    return quad, linear, const


class QuadVValue(VValue):
    """Quadratic time-varying state-value function."""

    n_state: int
    horizon: int
    quad: nn.Parameter
    linear: nn.Parameter
    const: nn.Parameter

    def __init__(self, quadratic: Quadratic):
        super().__init__()
        quad, _, _ = quadratic
        # noinspection PyArgumentList
        self.n_state = quad.size("C")
        # noinspection PyArgumentList
        self.horizon = quad.size("H") - 1
        self.quad, self.linear, self.const = map(
            lambda x: nn.Parameter(nt.unnamed(x)), quadratic
        )

    def forward(self, obs: Tensor) -> Tensor:
        state, time = unpack_obs(obs)
        quad, linear, const = index_quadratic_parameters(
            self.quad, self.linear, self.const, time
        )
        state = nt.vector_to_matrix(state)
        val = (
            nt.transpose(state) @ quad @ state
            + nt.transpose(nt.vector_to_matrix(linear)) @ state
            + nt.scalar_to_matrix(const)
        )
        return nt.matrix_to_scalar(val)


class QuadQValue(QValue):
    """Quadratic time-varying action-value function."""

    n_tau: int
    horizon: int
    quad: nn.Parameter
    linear: nn.Parameter
    const: nn.Parameter

    def __init__(self, quadratic: Quadratic):
        super().__init__()
        quad, _, _ = quadratic
        # noinspection PyArgumentList
        self.n_tau = quad.size("C")
        # noinspection PyArgumentList
        self.horizon = quad.size("H")
        self.quad, self.linear, self.const = map(
            lambda x: nn.Parameter(nt.unnamed(x)), quadratic
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        state, time = unpack_obs(obs)
        quad, linear, const = index_quadratic_parameters(
            self.quad, self.linear, self.const, time
        )
        vec = nt.vector_to_matrix(torch.cat([state, action], dim="R"))
        val = (
            nt.transpose(vec) @ quad @ vec
            + nt.transpose(nt.vector_to_matrix(linear)) @ vec
            + nt.scalar_to_matrix(const)
        )
        return nt.matrix_to_scalar(val)
