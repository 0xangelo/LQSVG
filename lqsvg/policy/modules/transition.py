"""NN transition models."""
# pylint:disable=invalid-name
from __future__ import annotations

from lqsvg.envs.lqr.modules import LinearDynamicsModule

__all__ = ["LinearTransitionModel"]


class LinearTransitionModel(LinearDynamicsModule):
    """Linear Gaussian transition model."""
