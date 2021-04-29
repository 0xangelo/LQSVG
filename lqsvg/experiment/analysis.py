"""Utilities for measuring gradient quality and optimization landscapes."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.np_util import RNG, random_unit_vector
from lqsvg.torch.utils import as_float_tensor, vector_to_tensors

from . import utils
from .estimators import PolicyLoss


def gradient_accuracy(svg_samples: list[lqr.Linear], target: lqr.Linear) -> float:
    """Compute the average cosine similarity with the target gradient."""
    cossims = [utils.linear_feedback_cossim(g, target) for g in svg_samples]
    return torch.stack(cossims).mean().item()


def empirical_variance(svg_samples: list[lqr.Linear]) -> float:
    """Compute the average pairwise cosine similarity between gradient samples."""
    # pylint:disable=invalid-name
    cossims = []
    for i, gi in enumerate(svg_samples):
        for gj in svg_samples[i + 1 :]:
            cossims += [utils.linear_feedback_cossim(gi, gj)]

    return torch.stack(cossims).mean().item()


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
    """Create function mapping policy parameter deltas to true LQG returns.

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
        delta_K, delta_k = vector_to_tensors(vector, policy)
        K, k = K_0 + delta_K, k_0 + delta_k

        ret = loss((K, k), dynamics, cost, init).neg()
        return ret.numpy()

    return f_delta
