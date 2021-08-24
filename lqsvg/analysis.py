"""Utilities for measuring gradient quality and optimization landscapes."""
from __future__ import annotations

import itertools
from typing import Callable, Iterable, Sequence, Tuple, Union

import numpy as np
import torch
from nnrl.nn.critic import VValue
from torch import Tensor

from lqsvg.envs import lqr
from lqsvg.estimator import analytic_value
from lqsvg.np_util import RNG, random_unit_vector
from lqsvg.torch import named as nt
from lqsvg.torch.utils import as_float_tensor, vector_to_tensors


def gradient_accuracy(svgs: Iterable[lqr.Linear], target: lqr.Linear) -> Tensor:
    """Compute the average cosine similarity with the target gradient."""
    cossims = [cosine_similarity(g, target) for g in svgs]
    return torch.stack(cossims).mean()


def gradient_precision(svgs: Iterable[lqr.Linear]) -> Tensor:
    """Compute the average pairwise cosine similarity between gradient samples."""
    # pylint:disable=invalid-name
    cossims = [cosine_similarity(a, b) for a, b in itertools.combinations(svgs, 2)]
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
    K_0, k_0 = policy

    @torch.no_grad()
    def f_delta(delta: np.ndarray) -> np.ndarray:
        vector = nt.vector(as_float_tensor(delta))
        # pylint:disable=unbalanced-tuple-unpacking
        delta_K, delta_k = vector_to_tensors(vector, policy)
        gains = lqr.Linear(K_0 + delta_K, k_0 + delta_k)

        return analytic_value(gains, init, dynamics, cost).numpy()

    return f_delta


def total_norm(tensors: Iterable[Tensor]) -> Tensor:
    """Returns the L2 norm of the flattened input tensors."""
    return torch.linalg.norm(torch.stack([torch.linalg.norm(t) for t in tensors]))


def total_distance(first: Iterable[Tensor], second: Iterable[Tensor]) -> Tensor:
    """Returns the L2 norm of the difference between input tensors."""
    return total_norm(f - s for f, s in zip(first, second))


def cosine_similarity(
    first: Union[Tensor, Sequence[Tensor]], second: Union[Tensor, Sequence[Tensor]]
) -> Tensor:
    """Returns the cosine similarity between tensors.

    Args:
        first: Tensor or sequence of tensors
        second: Tensor or sequence of tensors

    Returns:
        Scalar tensor representing the cosine similarity between the vectors of
        flattened input tensors.
    """
    assert torch.is_tensor(first) == torch.is_tensor(second)
    if torch.is_tensor(first):
        first, second = [first], [second]

    dot_product = sum(torch.sum(f * s) for f, s in zip(first, second))
    norm_prod = total_norm(first) * total_norm(second)
    return dot_product / norm_prod


def relative_error(target_val: Tensor, pred_val: Tensor) -> Tensor:
    """Returns the relative value error.

    Ref: https://en.wikipedia.org/wiki/Approximation_error
    """
    return torch.abs(1 - pred_val / target_val)


def val_err_and_grad_acc(
    val: Tensor, svg: lqr.Linear, target_val: Tensor, target_svg: lqr.Linear
) -> Tuple[Tensor, Tensor]:
    """Computes metrics for estimated gradients."""
    val_err = relative_error(target_val, val)
    grad_acc = gradient_accuracy([svg], target_svg)
    return val_err, grad_acc


@torch.no_grad()
def vvalue_err(val: Tensor, obs: Tensor, vval: VValue) -> Tensor:
    """Returns the error between the surrogate value and the state value."""
    return relative_error(vval(obs).mean(), val)
