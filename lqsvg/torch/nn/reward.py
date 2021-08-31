"""NN reward models."""
from __future__ import annotations

import torch
from torch import IntTensor, Tensor, nn

from lqsvg.envs import lqr
from lqsvg.torch import named as nt
from lqsvg.torch.random import normal_vector, spd_matrix

__all__ = ["QuadraticReward", "QuadRewardModel"]


class QuadraticReward(nn.Module):
    """Quadratic reward function in NN form."""

    # pylint:disable=invalid-name,missing-docstring
    n_tau: int
    horizon: int
    C: nn.Parameter
    c: nn.Parameter

    def __init__(self, n_tau: int, horizon: int):
        super().__init__()
        self.n_tau = n_tau
        self.horizon = horizon

        self.C = nn.Parameter(Tensor(horizon, n_tau, n_tau))
        self.c = nn.Parameter(Tensor(horizon, n_tau))
        self.reset_parameters()

    def reset_parameters(self):
        """Default parameter initialization."""
        self.C.data.copy_(spd_matrix(self.n_tau, horizon=self.horizon))
        self.c.data.copy_(normal_vector(self.n_tau, horizon=self.horizon))

    @classmethod
    def from_existing(cls, cost: lqr.QuadCost) -> QuadraticReward:
        """Create reward module from existing quadratic cost parameters."""
        n_tau, horizon = lqr.dims_from_cost(cost)
        return cls(n_tau, horizon).copy_(cost)

    def copy_(self, cost: lqr.QuadCost) -> QuadraticReward:
        """Set parameters to model existing quadratic cost."""
        self.C.data.copy_(cost.C)
        self.c.data.copy_(cost.c)
        return self

    def _refined_parameters(self) -> tuple[Tensor, Tensor]:
        C, c = nt.horizon(nt.matrix(self.C), nt.vector(self.c))
        return C, c

    def _index_parameters(self, index: IntTensor) -> tuple[Tensor, Tensor]:
        refined = self._refined_parameters()
        index = torch.clamp(index, max=self.horizon - 1)
        # noinspection PyTypeChecker
        C, c = (nt.index_by(x, dim="H", index=index) for x in refined)
        return C, c

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        obs, act = (nt.vector(x) for x in (obs, act))
        state, time = lqr.unpack_obs(obs)
        tau = nt.vector_to_matrix(torch.cat([state, act], dim="R"))
        time = nt.vector_to_scalar(time)

        C, c = self._index_parameters(time)
        c = nt.vector_to_matrix(c)

        cost = nt.transpose(tau) @ C @ tau / 2 + nt.transpose(c) @ tau
        reward = nt.matrix_to_scalar(cost.neg())
        return nt.where(time.eq(self.horizon), torch.zeros_like(reward), reward)

    def standard_form(self) -> lqr.QuadCost:
        C, c = self._refined_parameters()
        return lqr.QuadCost(C=C, c=c)


class QuadRewardModel(QuadraticReward):
    """Time-varying quadratic reward model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__(n_state + n_ctrl, horizon)
