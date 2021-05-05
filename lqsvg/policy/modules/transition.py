"""NN transition models."""
# pylint:disable=invalid-name
from __future__ import annotations

from raylab.policy.modules.model import StochasticModel

from lqsvg.envs.lqr.modules import LinearDynamicsModule

__all__ = ["LinearTransitionModel", "MLPDynamicsModel"]


class LinearTransitionModel(LinearDynamicsModule):
    """Linear Gaussian transition model."""


class MLPDynamicsModel(StochasticModel):
    """Multilayer perceptron transition model."""
