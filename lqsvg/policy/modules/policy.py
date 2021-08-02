"""NN module policies."""
from __future__ import annotations

import math
from typing import Union

import torch
from nnrl.nn.actor import DeterministicPolicy
from torch import IntTensor, LongTensor, Tensor, nn

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import unpack_obs

from .utils import perturb_policy, stabilizing_policy

__all__ = ["TVLinearFeedback", "TVLinearPolicy"]

from ...np_util import RNG


class TVLinearFeedback(nn.Module):
    """Non-stationary linear policy as a NN module.

    This module uses the same initialization for its parameters as `nn.Linear`.

    Args:
        n_state: size of the state vector
        n_ctrl: size of the control (action) vector
        horizon: time horizon
    """

    # pylint:disable=invalid-name
    n_state: int
    n_ctrl: int
    horizon: int
    K: nn.Parameter
    k: nn.Parameter

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.horizon = horizon
        self.K = nn.Parameter(torch.Tensor(horizon, n_ctrl, n_state))
        self.k = nn.Parameter(torch.Tensor(horizon, n_ctrl))
        self.reset_parameters()

    def reset_parameters(self):
        """Standard parameter initialization"""
        nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))
        fan_in = self.n_ctrl
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.k, -bound, bound)

    def _gains_at(
        self, index: Union[IntTensor, LongTensor, None] = None
    ) -> tuple[Tensor, Tensor]:
        K, k = nt.horizon(nt.matrix(self.K), nt.vector(self.k))
        if index is not None:
            index = torch.clamp(index, max=self.horizon - 1)
            # Assumes index is a named scalar tensor
            # noinspection PyTypeChecker
            K, k = (nt.index_by(x, dim="H", index=index) for x in (K, k))
        return K, k

    def forward(self, obs: Tensor, frozen: bool = False) -> Tensor:
        """Compute the action vector for the observed state."""
        obs = nt.vector(obs)
        state, time = unpack_obs(obs)

        # noinspection PyTypeChecker
        K, k = self._gains_at(nt.vector_to_scalar(time))
        if frozen:
            K, k = K.detach(), k.detach()

        ctrl = K @ nt.vector_to_matrix(state) + nt.vector_to_matrix(k)
        ctrl = nt.matrix_to_vector(ctrl)
        # Return zeroed actions if in terminal state
        terminal = time.eq(self.horizon)
        return nt.where(terminal, torch.zeros_like(ctrl), ctrl)

    @classmethod
    def from_existing(cls, policy: lqr.Linear):
        """Create linear feedback from linear parameters."""
        n_state, n_ctrl, horizon = lqr.dims_from_policy(policy)
        new = cls(n_state, n_ctrl, horizon)
        new.copy_(policy)
        return new

    def copy_(self, policy: lqr.Linear):
        """Set current parameters to given linear parameters."""
        K, k = policy
        self.K.data.copy_(K)
        self.k.data.copy_(k)

    def gains(self) -> lqr.Linear:
        """Return current parameters as linear parameters."""
        K, k = nt.horizon(nt.matrix(self.K), nt.vector(self.k))
        K.grad, k.grad = self.K.grad, self.k.grad
        return K, k


class TVLinearPolicy(DeterministicPolicy):
    """Time-varying affine feedback policy as a DeterministicPolicy module."""

    # pylint:disable=invalid-name
    action_linear: TVLinearFeedback
    K: nn.Parameter
    k: nn.Parameter

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__(
            encoder=nn.Identity(),
            action_linear=TVLinearFeedback(n_state, n_ctrl, horizon),
            squashing=nn.Identity(),
        )
        self.K = self.action_linear.K
        self.k = self.action_linear.k

    @classmethod
    def from_existing(cls, policy: lqr.Linear):
        """Create time-varying linear policy from linear parameters."""
        n_state, n_ctrl, horizon = lqr.dims_from_policy(policy)
        new = cls(n_state, n_ctrl, horizon)
        new.action_linear.copy_(policy)
        return new

    def noisy_(self, policy: lqr.Linear):
        """Initialize self from given linear policy plus white noise."""
        self.action_linear.copy_(perturb_policy(policy))

    @torch.no_grad()
    def stabilize_(self, dynamics: lqr.LinSDynamics, rng: RNG = None):
        """Initialize self to make the closed-loop system stable.

        Computes a dynamic gain (matrix) that places the eigenvalues of the system in
        a stable range, i.e., with magnitude less that 1.
        Initializes the static gain (vector) as zeros.

        Args:
            dynamics: the linear dynamical system
            rng: random number generator state

        Warning:
            This is only defined for stationary systems

        Raises:
            AssertionError: if the dynamics are non-stationary
        """
        self.action_linear.copy_(stabilizing_policy(dynamics, rng=rng))

    def standard_form(self) -> lqr.Linear:
        """Return self as linear function parameters."""
        return self.action_linear.gains()

    def frozen(self, obs: Tensor) -> Tensor:
        """Compute action for observation with parameters frozen.

        Useful for stochastic computation graphs where gradients should flow
        from action to previous observation but not to policy parameters.
        """
        return self.action_linear(obs, frozen=True)
