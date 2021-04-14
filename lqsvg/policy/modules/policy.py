"""NN module policies."""
from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import torch
from raylab.policy.modules.actor import DeterministicPolicy
from scipy.signal import place_poles
from torch import IntTensor, LongTensor, Tensor, nn

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import (
    dynamics_factors,
    isstationary,
    sample_eigvals,
    stationary_dynamics_factors,
    unpack_obs,
)

from .utils import perturb_policy

__all__ = ["TVLinearFeedback", "TVLinearPolicy"]


class TVLinearFeedback(nn.Module):
    # pylint:disable=missing-docstring,invalid-name
    horizon: int

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__()
        K = torch.randn(horizon, n_ctrl, n_state)
        k = torch.randn(horizon, n_ctrl)
        self.K, self.k = (nn.Parameter(x) for x in (K, k))
        self.horizon = horizon

    def _gains_at(self, index: Union[IntTensor, LongTensor]) -> tuple[Tensor, Tensor]:
        K = nt.horizon(nt.matrix(self.K))
        k = nt.horizon(nt.vector(self.k))
        index = torch.clamp(index, max=self.horizon - 1)
        # noinspection PyTypeChecker
        K, k = (nt.index_by(x, dim="H", index=index) for x in (K, k))
        return K, k

    def forward(self, obs: Tensor) -> Tensor:
        obs = nt.vector(obs)
        state, time = unpack_obs(obs)

        # noinspection PyTypeChecker
        K, k = self._gains_at(nt.vector_to_scalar(time))

        ctrl = K @ nt.vector_to_matrix(state) + nt.vector_to_matrix(k)
        ctrl = nt.matrix_to_vector(ctrl)
        # Return zeroed actions if in terminal state
        terminal = time.eq(self.horizon)
        return nt.where(terminal, torch.zeros_like(ctrl), ctrl)

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


def place_dynamics_poles(
    A: np.ndarray, B: np.ndarray, abs_low: float = 0.0, abs_high: float = 1.0
):
    """Compute a solution that re-scales the eigenvalues of linear dynamics."""
    # pylint:disable=invalid-name
    poles = sample_eigvals(A.shape[-1], abs_low, abs_high, size=(), rng=None)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Convergence was not reached after maxiter iterations.*",
            UserWarning,
            module="scipy.signal",
        )
        return place_poles(A, B, poles, maxiter=100)


def stabilizing_gain(
    dynamics: lqr.LinSDynamics, abs_low: float = 0.0, abs_high: float = 1.0
) -> Tensor:
    """Compute gain that stabilizes a linear dynamical system."""
    # pylint:disable=invalid-name
    F_s, F_a = stationary_dynamics_factors(dynamics)
    A, B = F_s.numpy(), F_a.numpy()
    result = place_dynamics_poles(A, B, abs_low=abs_low, abs_high=abs_high)
    minus_K = result.gain_matrix
    K = torch.as_tensor(-minus_K).to(F_a).refine_names(*F_a.names)
    return K


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

    @torch.no_grad()
    def initialize_to_stabilize(
        self, dynamics: lqr.LinSDynamics, abs_low: float, abs_high: float
    ):
        """Initilize gain matrix to make the closed-loop system stable.

        Computes a gain matrix that places the eigenvalues of the system in
        a stable range, i.e., with magnitude less that 1.

        Args:
            dynamics: the linear dynamical system
            abs_low: minimum absolute eigenvalue of the resulting system
            abs_high: maximum absolute eigenvalue of the resulting system

        Warning:
            This is only defined for stationary systems

        Raises:
            AssertionError: if the dynamics are non-stationary
        """
        # pylint:disable=invalid-name
        assert isstationary(dynamics)

        K = stabilizing_gain(dynamics, abs_low=abs_low, abs_high=abs_high)

        _, F_a = dynamics_factors(dynamics)
        K = K.expand_as(nt.transpose(F_a)).refine_names(*F_a.names)
        # k must be a column vector the size of control vectors, which are
        # multiplied by the rows of B (F_a)
        # noinspection PyTypeChecker
        k = torch.zeros_like(F_a.select("R", 0)).rename(C="R")

        self.action_linear.copy((K, k))

    def standard_form(self) -> lqr.Linear:
        # pylint:disable=missing-function-docstring
        return self.action_linear.gains()
