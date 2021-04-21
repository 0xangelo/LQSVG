"""Common type annotations."""
from collections import namedtuple
from typing import Tuple, TypeVar

from torch import Tensor

__all__ = [
    "LQR",
    "LQG",
    "Linear",
    "Quadratic",
    "Box",
    "QuadCost",
    "LinDynamics",
    "LinSDynamics",
    "AnyDynamics",
    "GaussInit",
]

LQR = Tuple[Tensor, Tensor, Tensor, Tensor]
LQG = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
Linear = Tuple[Tensor, Tensor]
Quadratic = Tuple[Tensor, Tensor, Tensor]
Box = Tuple[Tensor, Tensor]

QuadCost = namedtuple("QuadCost", "C c")
LinDynamics = namedtuple("LinDynamics", "F f")
LinSDynamics = namedtuple("LinSDynamics", "F f W")
AnyDynamics = TypeVar("Dynamics", LinDynamics, LinSDynamics)
GaussInit = namedtuple("GaussInit", "mu sig")
