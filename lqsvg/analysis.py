"""Utilities for measuring gradient quality and optimization landscapes."""
from __future__ import annotations

import itertools
from typing import Callable, Iterable

import numpy as np
import torch
from torch import Tensor

from lqsvg.envs import lqr
from lqsvg.estimator import PolicyLoss
from lqsvg.np_util import RNG, random_unit_vector
from lqsvg.torch import named as nt
from lqsvg.torch.utils import as_float_tensor, vector_to_tensors


def gradient_accuracy(svgs: Iterable[lqr.Linear], target: lqr.Linear) -> Tensor:
    """Compute the average cosine similarity with the target gradient."""
    cossims = [linear_feedback_cossim(g, target) for g in svgs]
    return torch.stack(cossims).mean()


def empirical_variance(svgs: Iterable[lqr.Linear]) -> Tensor:
    """Compute the average pairwise cosine similarity between gradient samples."""
    # pylint:disable=invalid-name
    cossims = [linear_feedback_cossim(a, b) for a, b in itertools.combinations(svgs, 2)]
    return torch.stack(cossims).mean()


def optimization_surface(
    f_delta: Callable[[np.ndarray], np.ndarray],
    direction: np.ndarray,
    max_scaling: float = 3.0,
    steps: int = 20,
    rng: RNG = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D approximation of a function's surface in reference to a direction.

    Evaluates a function from vectors to scalars with several linear
    combinations of a given direction with with a random one as inputs.

    Args:
        f_delta: the vector-to-scalar function. Usually an approximation of
            another function at a certain point plus the input vector. E.g.,
            this could be the first-order Taylor series expansion of another
            function around a certain point in R^d.
        direction: a vector to be linearly combined with a random one using
            varying positive weights for each. This vector is normalized to
            have a magnitude of 1, the same as the random one's.
        max_scaling: maximum weight for each vector used in the linear
            combination as input to the function
        steps: number of evenly spaced values between 0 and `max_scaling` used
            as weights for the linear combination between the random and given
            direction vectors.
        rng: random number generator parameter for repeatable results

    Returns:
        A tuple of three meshgrids (X, Y, Z) representing the weight for the
        random vector, weight for the given direction, and the function's value
        for the resulting linear combination of the two vectors.
    """
    # pylint:disable=invalid-name
    direction = direction / np.linalg.norm(direction)
    rand_dir = random_unit_vector(direction.size, rng=rng)

    scale_range = np.linspace(0, max_scaling, steps)
    X, Y = np.meshgrid(scale_range, scale_range)

    results = []
    for i, j in zip(X.reshape((-1,)), Y.reshape((-1,))):
        vector = i * rand_dir + j * direction
        results += [f_delta(vector)]
    Z = np.array(results).reshape(X.shape)

    return X, Y, Z


def delta_to_return(
    policy: lqr.Linear,
    dynamics: lqr.LinSDynamics,
    cost: lqr.QuadCost,
    init: lqr.GaussInit,
) -> Callable[[np.ndarray], np.ndarray]:
    """Creates function mapping policy parameter deltas to true LQG returns.

    Args:
        policy: the reference linear policy
        dynamics: linear stochastic dynamics
        cost: quadratic cost
        init: Gaussian initial state distribution

    Returns:
        Function mapping policy parameter deltas to the true return of the
        policy resulting from applying the deltas to the reference policy.
    """
    # pylint:disable=invalid-name
    n_state, n_ctrl, horizon = lqr.dims_from_policy(policy)
    loss = PolicyLoss(n_state, n_ctrl, horizon)
    K_0, k_0 = policy

    @torch.no_grad()
    def f_delta(delta: np.ndarray) -> np.ndarray:
        vector = nt.vector(as_float_tensor(delta))
        # pylint:disable=unbalanced-tuple-unpacking
        delta_K, delta_k = vector_to_tensors(vector, policy)
        K, k = K_0 + delta_K, k_0 + delta_k

        ret = loss((K, k), dynamics, cost, init).neg()
        return ret.numpy()

    return f_delta


def linear_feedback_cossim(linear_a: lqr.Linear, linear_b: lqr.Linear) -> Tensor:
    """Cosine similarity between the parameters of linear (affine) functions.

    Args:
        linear_a: tuple of affine function parameters
        linear_b: tuple of affine function parameters

    Returns:
        Tensor representing the (scalar) cosine similarity.
    """
    # pylint:disable=invalid-name
    Ka, ka = linear_a
    Kb, kb = linear_b
    dot_product = torch.sum(Ka * Kb) + torch.sum(ka * kb)
    norm_prod = linear_feedback_norm(linear_a) * linear_feedback_norm(linear_b)
    return dot_product / norm_prod


def linear_feedback_norm(linear: lqr.Linear) -> Tensor:
    """Norm of the parameters of a linear (affine) function.

    Uses the default norms for vectors and matrices chosen by PyTorch:
    frobenius for matrices and L2 for vectors.

    Equivalent to the norm of the flattened parameter vector.

    Args:
        linear: tuple of affine function parameters (weight matrix and bias
        column vector)

    Returns:
        Norm of the affine function's parameters
    """
    # pylint:disable=invalid-name
    K, k = linear
    K_norm = torch.linalg.norm(nt.unnamed(K), dim=(-2, -1))
    k_norm = torch.linalg.norm(nt.unnamed(k), dim=-1)
    # Following PyTorch's clip_grad_norm_ implementation
    # Reduce by horizon
    total_norm = torch.linalg.norm(torch.cat((K_norm, k_norm), dim=0), dim=0)
    return total_norm


def linear_feedback_distance(linear_a: lqr.Linear, linear_b: lqr.Linear) -> Tensor:
    """Distance between the parameters of linear (affine) functions.

    Args:
        linear_a: tuple of affine function parameters
        linear_b: tuple of affine function parameters

    Returns:
        Norm of the difference between the parameters
    """
    # pylint:disable=invalid-name
    Ka, ka = linear_a
    Kb, kb = linear_b
    return linear_feedback_norm(lqr.Linear(Ka - Kb, ka - kb))
