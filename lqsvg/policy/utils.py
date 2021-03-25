"""Utilities for policy initialization."""
import numpy as np
import torch

from lqsvg.envs import lqr


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
