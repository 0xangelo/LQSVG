"""NN transition models."""
# pylint:disable=invalid-name
from __future__ import annotations

from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.modules import LinearDynamicsModule

__all__ = ["LinearTransitionModel"]


class LinearTransitionModel(LinearDynamicsModule):
    """Linear Gaussian transition model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        dynamics = make_lindynamics(n_state, n_ctrl, horizon, stationary=stationary)
        dynamics = make_linsdynamics(
            dynamics, stationary=stationary, sample_covariance=True
        )
        super().__init__(dynamics, stationary=stationary)
