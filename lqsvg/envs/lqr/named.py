# pylint:disable=missing-docstring,invalid-name
from typing import Tuple
from typing import Union

from torch import Tensor

from .types import LinDynamics
from .types import Linear
from .types import LinSDynamics
from .types import QuadCost
from .types import Quadratic


MATRIX_NAMES = ("H", ..., "R", "C")
VECTOR_NAMES = MATRIX_NAMES[:-1]
SCALAR_NAMES = MATRIX_NAMES[:-2]


def unnamed(
    tensors: Union[Tuple[Tensor, ...], LinSDynamics, LinDynamics, QuadCost]
) -> Union[Tuple[Tensor, ...], LinSDynamics, LinDynamics, QuadCost]:
    cls = type(tensors)
    gen = (x.rename(None) for x in tensors)
    if isinstance(tensors, (LinDynamics, LinSDynamics, QuadCost)):
        return cls(*gen)
    return cls(gen)


def matrix(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*MATRIX_NAMES)


def vector(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*VECTOR_NAMES)


def scalar(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*SCALAR_NAMES)


def matrix_to_vector(tensor: Tensor) -> Tensor:
    return matrix(tensor).squeeze("C")


def matrix_to_scalar(tensor: Tensor) -> Tensor:
    return matrix(tensor).squeeze("R").squeeze("C")


def vector_to_matrix(tensor: Tensor) -> Tensor:
    return vector(tensor).align_to(..., "R", "C")


def vector_to_scalar(tensor: Tensor) -> Tensor:
    return vector(tensor).squeeze("R")


def scalar_to_matrix(tensor: Tensor) -> Tensor:
    return scalar(tensor).align_to(..., "R", "C")


def scalar_to_vector(tensor: Tensor) -> Tensor:
    return scalar(tensor).align_to(..., "R")


def refine_linear_input(linear: Linear):
    K, k = linear
    K = matrix(K)
    k = vector_to_matrix(k)
    return K, k


def refine_dynamics_input(dynamics: LinDynamics):
    F, f = dynamics
    F = matrix(F)
    f = vector_to_matrix(f)
    return LinDynamics(F, f)


def refine_sdynamics_input(dynamics: LinSDynamics):
    F, f, W = dynamics
    F, f = refine_dynamics_input((F, f))
    W = matrix(W)
    return LinSDynamics(F, f, W)


def refine_cost_input(cost: QuadCost):
    C, c = cost
    C = matrix(C)
    c = vector_to_matrix(c)
    return QuadCost(C, c)


def refine_linear_output(linear: Linear):
    K, k = linear
    K = matrix(K)
    k = matrix_to_vector(k)
    return K, k


def refine_quadratic_output(quadratic: Quadratic):
    A, b, c = quadratic
    A = matrix(A)
    b = matrix_to_vector(b)
    c = matrix_to_scalar(c)
    return A, b, c


def refine_cost_ouput(cost: QuadCost) -> QuadCost:
    C, c = cost
    C = matrix(C)
    c = matrix_to_vector(c)
    return QuadCost(C, c)


def refine_lqr(dynamics: LinDynamics, cost: QuadCost) -> Tuple[LinDynamics, QuadCost]:
    """Add dimension names to LQR parameters.

    Args:
        dynamics: transition matrix and vector
        cost: quadratic cost matrix and vector

    Returns:
        A tuple with named dynamics and cost parameters
    """
    F, f = dynamics
    C, c = cost
    F = F.refine_names("H", "R", "C")
    f = f.refine_names("H", "R")
    C = C.refine_names("H", "R", "C")
    c = c.refine_names("H", "R")
    return LinDynamics(F, f), QuadCost(C, c)
