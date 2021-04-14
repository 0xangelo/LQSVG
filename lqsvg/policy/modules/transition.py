"""NN transition models."""
# pylint:disable=invalid-name
from __future__ import annotations

from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.modules import LinearDynamicsModule

__all__ = ["TVLinearTransModel", "StationaryLinearTransModel"]


class TVLinearTransModel(LinearDynamicsModule):
    """Time-varying linear Gaussian transition model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        dynamics = make_lindynamics(n_state, n_ctrl, horizon, stationary=False)
        dynamics = make_linsdynamics(dynamics, stationary=False, sample_covariance=True)
        super().__init__(dynamics, stationary=False)


class StationaryLinearTransModel(LinearDynamicsModule):
    """Stationary linear Gaussian transition model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        dynamics = make_lindynamics(n_state, n_ctrl, horizon, stationary=True)
        dynamics = make_linsdynamics(dynamics, stationary=True, sample_covariance=True)
        super().__init__(dynamics, stationary=True)
