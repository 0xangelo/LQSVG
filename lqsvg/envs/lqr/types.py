"""Common type annotations."""
from typing import NamedTuple, TypeVar

from torch import Tensor

# pylint:disable=invalid-name,missing-class-docstring

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


class LQR(NamedTuple):
    F: Tensor
    f: Tensor
    C: Tensor
    c: Tensor


class LQG(NamedTuple):
    F: Tensor
    f: Tensor
    W: Tensor
    C: Tensor
    c: Tensor


class Linear(NamedTuple):
    K: Tensor
    k: Tensor


class Quadratic(NamedTuple):
    mat: Tensor
    vec: Tensor
    cst: Tensor


class QuadCost(NamedTuple):
    C: Tensor
    c: Tensor


class LinDynamics(NamedTuple):
    F: Tensor
    f: Tensor


class LinSDynamics(NamedTuple):
    F: Tensor
    f: Tensor
    W: Tensor


class GaussInit(NamedTuple):
    mu: Tensor
    sig: Tensor


class Box(NamedTuple):
    a: Tensor
    b: Tensor


AnyDynamics = TypeVar("Dynamics", LinDynamics, LinSDynamics)
