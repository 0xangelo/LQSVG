"""NN initial state models."""
from __future__ import annotations

from typing import Optional

from lqsvg.envs.lqr import make_gaussinit

from .dynamics.gauss import InitStateModule

__all__ = ["InitStateModel"]


class InitStateModel(InitStateModule):
    """Gaussian initial state distribution model."""

    def __init__(self, n_state: int, seed: Optional[int] = None):
        super().__init__(n_state)
        init = make_gaussinit(n_state, sample_covariance=True, rng=seed)
        self.copy_(init)
