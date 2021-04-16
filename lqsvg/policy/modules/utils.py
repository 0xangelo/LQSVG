"""Utilities for policy initialization."""
import itertools
import warnings

import numpy as np
import torch
from scipy.signal import place_poles
from torch import Tensor

from lqsvg.envs import lqr
from lqsvg.envs.lqr.utils import (
    dynamics_factors,
    isstationary,
    sample_eigvals,
    stationary_dynamics_factors,
)
from lqsvg.np_util import RNG
from lqsvg.torch import named as nt


def perturb_policy(policy: lqr.Linear) -> lqr.Linear:
    """Perturb policy parameters to derive sub-optimal policies.

    Adds white noise to optimal policy parameters.

    Args:
        policy: optimal policy parameters

    Returns:
        Perturbed policy parameters
    """
    # pylint:disable=invalid-name
    n_state, n_ctrl, _ = lqr.dims_from_policy(policy)
    K, k = (g + 0.5 * torch.randn_like(g) / (n_state + np.sqrt(n_ctrl)) for g in policy)
    return K, k


def stabilizing_policy(dynamics: lqr.LinSDynamics, rng: RNG = None) -> lqr.Linear:
    """Compute linear policy parameters that stabilize an LQG.

    Warning:
        This is only defined for stationary systems

    Raises:
        AssertionError: if the dynamics are non-stationary
    """
    # pylint:disable=invalid-name
    assert isstationary(dynamics)
    K = stabilizing_gain(dynamics, rng=rng)

    _, B = dynamics_factors(dynamics)
    K = nt.horizon(K.expand_as(nt.transpose(B)))
    # k must be a column vector the size of control vectors, equivalent to the
    # size of the columns of K
    # noinspection PyTypeChecker
    k = torch.zeros_like(K.select("C", 0))
    return K, k


def stabilizing_gain(
    dynamics: lqr.LinSDynamics,
    abs_low: float = 0.0,
    abs_high: float = 1.0,
    rng: RNG = None,
) -> Tensor:
    """Compute gain that stabilizes a linear dynamical system."""
    # pylint:disable=invalid-name
    A, B = stationary_dynamics_factors(dynamics)
    result = place_dynamics_poles(
        A.numpy(), B.numpy(), abs_low=abs_low, abs_high=abs_high, rng=rng
    )
    K = torch.empty_like(nt.transpose(B))
    K.copy_(torch.as_tensor(-result.gain_matrix))
    return K


def place_dynamics_poles(
    A: np.ndarray,
    B: np.ndarray,
    abs_low: float = 0.0,
    abs_high: float = 1.0,
    rng: RNG = None,
):
    """Compute a solution that re-places the eigenvalues of linear dynamics."""
    # pylint:disable=invalid-name
    poles = sample_eigvals(A.shape[-1], abs_low, abs_high, size=(), rng=rng)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Convergence was not reached after maxiter iterations.*",
            UserWarning,
            module="scipy.signal",
        )
        for exp in itertools.count():
            result = place_poles(A, B, poles, maxiter=2 ** exp)
            abs_poles = np.abs(result.computed_poles)
            if np.all(np.logical_and(abs_low < abs_poles, abs_poles < abs_high)):
                break
    return result
