# pylint:disable=missing-module-docstring,unsubscriptable-object
from __future__ import annotations

from typing import Optional

import numpy as np
from gym.spaces import Box
from torch import IntTensor
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.np_util import make_spd_matrix
from lqsvg.np_util import np_expand
from lqsvg.np_util import RNG
from lqsvg.torch.utils import as_float_tensor

from .types import Linear
from .types import LinSDynamics


def expand_and_refine(
    tensor: Tensor,
    base_dim: int,
    horizon: Optional[int] = None,
    n_batch: Optional[int] = None,
) -> Tensor:
    """Expand and refine tensor names with horizon and batch size information."""
    assert (
        n_batch is None or n_batch > 0
    ), f"Batch size must be null or positive, got {n_batch}"
    assert (
        tensor.dim() >= base_dim
    ), f"Tensor must have at least {base_dim} dimensions, got {tensor.dim()}"

    shape = (
        (() if horizon is None else (horizon,))
        + (() if n_batch is None else (n_batch,))
        + tensor.shape[-base_dim:]
    )
    names = (
        (() if horizon is None else ("H",))
        + (() if n_batch is None else ("B",))
        + (...,)
    )
    tensor = tensor.expand(*shape).refine_names(*names)
    return tensor


def _sample_shape(
    horizon: int, stationary: bool = False, n_batch: Optional[int] = None
) -> tuple[int, ...]:
    horizon_shape = () if stationary else (horizon,)
    batch_shape = () if n_batch is None else (n_batch,)
    return horizon_shape + batch_shape


def random_normal_vector(
    size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    # pylint:disable=missing-function-docstring
    rng = np.random.default_rng(rng)

    vec_shape = (size,)
    shape = _sample_shape(horizon, stationary=stationary, n_batch=n_batch) + vec_shape
    vec = nt.vector(as_float_tensor(rng.normal(size=shape)))
    vec = expand_and_refine(vec, 1, horizon=horizon, n_batch=n_batch)
    return vec


def random_normal_matrix(
    row_size: int,
    col_size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    # pylint:disable=missing-function-docstring,too-many-arguments
    mat_shape = (row_size, col_size)
    shape = _sample_shape(horizon, stationary=stationary, n_batch=n_batch) + mat_shape
    mat = nt.matrix(as_float_tensor(rng.normal(size=shape)))
    mat = expand_and_refine(mat, 2, horizon=horizon, n_batch=n_batch)
    return mat


def random_spd_matrix(
    size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    # pylint:disable=missing-function-docstring
    mat = make_spd_matrix(
        size,
        sample_shape=_sample_shape(horizon, stationary=stationary, n_batch=n_batch),
        rng=rng,
    )
    mat = nt.matrix(as_float_tensor(mat))
    mat = expand_and_refine(mat, 2, horizon=horizon, n_batch=n_batch)
    return mat


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
    # state, time = nt.split(obs, [obs.size("R") - 1, 1], dim="R")
    state, time = obs[..., :-1], obs[..., -1:]
    time = time.int()
    return state, time


def wrap_sample_shape_to_size(
    sampler: callable[[int], np.ndarray], dim: int
) -> callable[[tuple[int, ...], int], np.ndarray]:
    """Converts a sampler by size to a sampler by shape.

    Computes the total size of the sample shape, calls the sampler with this
    size, and reshapes the output.

    Assumes first dim of the wrapped sampler's output corresponds to the sample
    dimension. Exception when the total sample size is 1, in which case it
    assumes the sampler's output has no sample dimension. This is usually the
    case with Numpy/Scipy.

    Args:
        sampler: function that takes an integer as argument and returns this
            many samples as numpy arrays
        dim: number of dimensions of each sample

    Returns:
        A sampler that takes a sample shape as an argument.
    """

    def wrapped(sample_shape: tuple[int, ...]) -> np.ndarray:
        sample_size = np.prod(sample_shape, dtype=int)
        arr = sampler(sample_size)
        base = arr.shape[-dim:] if dim > 0 else ()
        return np.reshape(arr, sample_shape + base)

    return wrapped
