"""NN module policies."""
from __future__ import annotations

from typing import Union

import torch
from raylab.policy.modules.actor import DeterministicPolicy
from torch import IntTensor
from torch import LongTensor
from torch import nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import unpack_obs

from .utils import perturb_policy

__all__ = ["TVLinearFeedback", "TVLinearPolicy"]


class TVLinearFeedback(nn.Module):
    # pylint:disable=missing-docstring,invalid-name
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

    # pylint:disable=invalid-name
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
