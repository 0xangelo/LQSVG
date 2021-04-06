# pylint:disable=missing-docstring,invalid-name
from typing import Tuple, Union

from torch import Tensor

import lqsvg.torch.named as nt

from .types import LinDynamics, Linear, LinSDynamics, QuadCost, Quadratic


def unnamed(
    tensors: Union[Tuple[Tensor, ...], LinSDynamics, LinDynamics, QuadCost]
) -> Union[Tuple[Tensor, ...], LinSDynamics, LinDynamics, QuadCost]:
    cls = type(tensors)
    gen = (x.rename(None) for x in tensors)
    if isinstance(tensors, (LinDynamics, LinSDynamics, QuadCost)):
        return cls(*gen)
    return cls(gen)


def refine_linear_input(linear: Linear):
    K, k = linear
    K = nt.horizon(nt.matrix(K))
    k = nt.horizon(nt.vector_to_matrix(k))
    return K, k


def refine_dynamics_input(dynamics: LinDynamics):
    F, f = dynamics
    F = nt.horizon(nt.matrix(F))
    f = nt.horizon(nt.vector_to_matrix(f))
    return LinDynamics(F, f)


def refine_sdynamics_input(dynamics: LinSDynamics):
    F, f, W = dynamics
    F, f = refine_dynamics_input((F, f))
    W = nt.horizon(nt.matrix(W))
    return LinSDynamics(F, f, W)


def refine_cost_input(cost: QuadCost):
    C, c = cost
    C = nt.horizon(nt.matrix(C))
    c = nt.horizon(nt.vector_to_matrix(c))
    return QuadCost(C, c)


def refine_linear_output(linear: Linear):
    K, k = linear
    K = nt.horizon(nt.matrix(K))
    k = nt.horizon(nt.matrix_to_vector(k))
    return K, k


def refine_quadratic_output(quadratic: Quadratic):
    A, b, c = quadratic
    A = nt.horizon(nt.matrix(A))
    b = nt.horizon(nt.matrix_to_vector(b))
    c = nt.horizon(nt.matrix_to_scalar(c))
    return A, b, c


def refine_cost_ouput(cost: QuadCost) -> QuadCost:
    C, c = cost
    C = nt.horizon(nt.matrix(C))
    c = nt.horizon(nt.matrix_to_vector(c))
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
    F = nt.horizon(nt.matrix(F))
    f = nt.horizon(nt.vector(f))
    C = nt.horizon(nt.matrix(C))
    c = nt.horizon(nt.vector(c))
    return LinDynamics(F, f), QuadCost(C, c)
