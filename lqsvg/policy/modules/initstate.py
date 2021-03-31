"""NN initial state models."""
# pylint:disable=invalid-name
from __future__ import annotations

from typing import Optional

from lqsvg.envs.lqr import make_gaussinit
from lqsvg.envs.lqr.modules import InitStateDynamics

__all__ = ["InitStateModel"]


class InitStateModel(InitStateDynamics):
    """Gaussian initial state distribution model."""

    def __init__(self, n_state: int, seed: Optional[int] = None):
        init = make_gaussinit(n_state, sample_covariance=True, rng=seed)
        super().__init__(init)
