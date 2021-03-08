# pylint:disable=missing-module-docstring
from __future__ import annotations

import numpy as np
from gym.spaces import Box
from torch import IntTensor
from torch import Tensor

from lqsvg.np_util import np_expand

from .types import Linear
from .types import LinSDynamics


def dims_from_dynamics(dynamics: LinSDynamics) -> tuple[int, int, int]:
    """Retrieve LQG dimensions from linear Gaussian transition dynamics."""
    n_state = dynamics.F.size("R")
    n_ctrl = dynamics.F.size("C") - n_state
    horizon = dynamics.F.size("H")
    return n_state, n_ctrl, horizon


def dims_from_policy(policy: Linear) -> tuple[int, int, int]:
    """Retrieve LQG dimensions from linear feedback policy.

    Args:
        policy: linear feedback policy

    Returns:
        A tuple with state, control and horizon sizes respectively
    """
    # pylint:disable=invalid-name
    K, _ = policy
    n_state = K.size("C")
    n_ctrl = K.size("R")
    horizon = K.size("H")
    return n_state, n_ctrl, horizon


def np_expand_horizon(arr: np.ndarray, horizon: int) -> np.ndarray:
    """Expand a numpy array with a leading horizon dimension."""
    return np_expand(arr, (horizon,) + arr.shape)


def spaces_from_dims(n_state: int, n_ctrl: int, horizon: int) -> tuple[Box, Box]:
    """Constructs Gym spaces from LQR dimensions."""
    state_low = np.full(n_state, fill_value=-np.inf, dtype=np.single)
    state_high = -state_low
    observation_space = Box(
        low=np.append(state_low, np.single(0)),
        high=np.append(state_high, np.single(horizon)),
    )

    action_low = np.full(n_ctrl, fill_value=-np.inf, dtype=np.single)
    action_space = Box(low=action_low, high=-action_low)
    return observation_space, action_space


def dims_from_spaces(obs_space: Box, action_space: Box) -> tuple[int, int, int]:
    """Extracts LQR dimensions from Gym spaces."""
    n_state = obs_space.shape[0] - 1
    n_ctrl = action_space.shape[0]
    horizon = int(obs_space.high[-1])
    return n_state, n_ctrl, horizon


def unpack_obs(obs: Tensor) -> tuple[Tensor, IntTensor]:
    """Unpack observation into state variables and time.

    Expects observation as a named 'vector' tensor.
    """
    state, time = obs[..., :-1], obs[..., -1:]
    time = time.int()
    return state, time
