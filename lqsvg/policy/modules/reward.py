"""NN reward models."""
from __future__ import annotations

from lqsvg.envs.lqr.generators import make_quadcost
from lqsvg.envs.lqr.modules import QuadraticReward

__all__ = ["QuadRewardModel"]


class QuadRewardModel(QuadraticReward):
    """Time-varying quadratic reward model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        cost = make_quadcost(n_state, n_ctrl, horizon, stationary=False)
        super().__init__(cost)
