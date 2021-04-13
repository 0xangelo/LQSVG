"""Quadratic value functions as NN modules."""
from __future__ import annotations

import torch
import torch.nn as nn
from raylab.policy.modules.critic import QValue, VValue
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import Quadratic
from lqsvg.envs.lqr.utils import unpack_obs


def index_quadratic_parameters(
    quad: nn.Parameter, linear: nn.Parameter, const: nn.Parameter, time: IntTensor
) -> tuple[Tensor, Tensor, Tensor]:
    # pylint:disable=missing-function-docstring
    quad = nt.horizon(nt.matrix(quad))
    linear = nt.horizon(nt.vector(linear))
    const = nt.horizon(nt.scalar(const))
    index = nt.vector_to_scalar(time)
    # noinspection PyTypeChecker
    quad, linear, const = map(
        lambda x: nt.index_by(x, dim="H", index=index), (quad, linear, const)
    )
    return quad, linear, const


class QuadraticMixin:
    """Adds update and standard form to a quadratic NN module."""

    quad: nn.Parameter
    linear: nn.Parameter
    const: nn.Parameter

    def standard_form(self) -> Quadratic:
        """Return parameters in standard quadratic form.

        Returns:
            Tuple with matrix, vector, and scalar parameters, including
            their gradients (cloned)
        """
        params = (self.quad, self.linear, self.const)
        refines = (nt.matrix, nt.vector, nt.scalar)
        quadratic = tuple(nt.horizon(r(p)) for r, p in zip(refines, params))
        for tensor, param in zip(quadratic, params):
            tensor.grad = None if param.grad is None else param.grad.clone()
        # noinspection PyTypeChecker
        return quadratic

    def update(self, quadratic: Quadratic):
        """Update parameters to existing quadratic."""
        params = (self.quad, self.linear, self.const)
        for param, tensor in zip(params, quadratic):
            param.data.copy_(tensor.data)


class QuadVValue(VValue, QuadraticMixin):
    """Quadratic time-varying state-value function.

    Clones the tensors from a quadratic and sets them as parameters, avoiding
    in-place modification of the original quadratic's tensors.
    """

    n_state: int
    horizon: int

    def __init__(self, quadratic: Quadratic):
        super().__init__()
        quad, _, _ = quadratic
        # noinspection PyArgumentList
        self.n_state = quad.size("C")
        # noinspection PyArgumentList
        self.horizon = quad.size("H") - 1
        self.quad, self.linear, self.const = map(
            lambda x: nn.Parameter(nt.unnamed(x.clone())), quadratic
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


class QuadQValue(QValue, QuadraticMixin):
    """Quadratic time-varying action-value function.

    Clones the tensors from a quadratic and sets them as parameters, avoiding
    in-place modification of the original quadratic's tensors.
    """

    n_tau: int
    horizon: int

    def __init__(self, quadratic: Quadratic):
        super().__init__()
        quad, _, _ = quadratic
        # noinspection PyArgumentList
        self.n_tau = quad.size("C")
        # noinspection PyArgumentList
        self.horizon = quad.size("H")
        self.quad, self.linear, self.const = map(
            lambda x: nn.Parameter(nt.unnamed(x.clone())), quadratic
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
