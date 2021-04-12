"""Utilities for comparing complex objects."""
from lqsvg.envs.lqr import LinSDynamics, QuadCost
from lqsvg.torch import named as nt

# pylint:disable=missing-function-docstring


def allclose_dynamics(dyn1: LinSDynamics, dyn2: LinSDynamics) -> bool:
    equal = [nt.allclose(d1, d2) for d1, d2 in zip(dyn1, dyn2)]
    return all(equal)


def allclose_cost(cost1: QuadCost, cost2: QuadCost) -> bool:
    equal = [nt.allclose(c1, c2) for c1, c2 in zip(cost1, cost2)]
    return all(equal)
