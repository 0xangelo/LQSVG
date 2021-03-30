# pylint:disable=missing-module-docstring,unsubscriptable-object
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from gym.spaces import Box
from scipy.stats import ortho_group
from torch import IntTensor
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.np_util import make_spd_matrix
from lqsvg.np_util import np_expand
from lqsvg.np_util import RNG
from lqsvg.torch.utils import as_float_tensor

from .types import Linear
from .types import LinSDynamics


###############################################################################
# Diagnostics
###############################################################################


def isstationary(dynamics: LinSDynamics) -> bool:
    """Returns whether the dynamics are stationary (time-invariant)."""
    return (
        nt.allclose(dynamics.F, dynamics.F.select("H", 0))
        and nt.allclose(dynamics.f, dynamics.f.select("H", 0))
        and nt.allclose(dynamics.W, dynamics.W.select("H", 0))
    )


def isstable(
    dynamics: Optional[LinSDynamics] = None, eigvals: Optional[Tensor] = None
) -> np.ndarray:
    """Returns whether the unactuated dynamics are stable.

    A linear, stationary, discrete-time system is stable iff the all the
    eigenvalues of F_s live in the unit circle of the complex plane.

    Note:
        It is not obvious if this condition is generalizable to non-stationary
        systems. Thus, this function first checks if the system is stationary.

    Raises:
        AssertionError: if both `dynamics` and `eigvals` are passed/missing
        AssertionError: if the (batch of) dynamics is not stationary
    """
    assert (dynamics is not None) ^ (
        eigvals is not None
    ), "Only one of `dynamics` or `eigvals` must be passed."

    if eigvals is None:
        eigvals = stationary_eigvals(dynamics)
    # Assume the last dimension corresponds eigenvalues
    stable = np.asarray(np.all(np.absolute(eigvals) <= 1.0, axis=-1))
    return stable


def iscontrollable(dynamics: LinSDynamics) -> np.ndarray:
    """Returns whether the stationary dynamics are controllable.

    This function accepts a batch of linear dynamics.

    Note:
        It is not obvious if this condition is generalizable to non-stationary
        systems. Thus, this function first checks if the system is stationary.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    n_state, _, _ = dims_from_dynamics(dynamics)
    C = ctrb(dynamics).numpy()
    return np.linalg.matrix_rank(C) == n_state


###############################################################################
# System manipulation
###############################################################################


def dims_from_dynamics(dynamics: LinSDynamics) -> tuple[int, int, int]:
    """Retrieve LQG dimensions from linear Gaussian transition dynamics."""
    n_state = dynamics.F.size("R")
    n_ctrl = dynamics.F.size("C") - n_state
    horizon = dynamics.F.size("H")
    return n_state, n_ctrl, horizon


def dynamics_factors(dynamics: LinSDynamics) -> tuple[Tensor, Tensor]:
    """Returns the unactuated and actuaded parts of the transition matrix."""
    # pylint:disable=invalid-name
    n_state, n_ctrl, _ = dims_from_dynamics(dynamics)
    F_s, F_a = nt.split(dynamics.F, [n_state, n_ctrl], dim="C")
    return F_s, F_a


def stationary_eigvals(dynamics: LinSDynamics) -> np.ndarray:
    """Returns the eigenvalues of unactuated stationary transition dynamics.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    assert isstationary(dynamics), "Can't pass non-stationary dynamics"
    F_s, _ = dynamics_factors(dynamics)
    F_s = F_s.select("H", 0)
    # Assume last two dimensions correspond to rows and cols respectively
    eigvals, _ = np.linalg.eig(F_s.numpy())
    return eigvals


def ctrb(dynamics: LinSDynamics) -> Tensor:
    """Returns the controllability matrix for a stationary linear system.

    This function accepts batched dynamics.
    """
    # pylint:disable=invalid-name
    F_s, F_a = dynamics_factors(dynamics)
    F_s, F_a = F_s.select("H", 0), F_a.select("H", 0)
    n_state, _, _ = dims_from_dynamics(dynamics)
    # Assumes the last two dimensions correspond to rows and columns respectively
    C = torch.cat([torch.matrix_power(F_s, i) @ F_a for i in range(n_state)], dim=-1)
    return C


###############################################################################
# Random generation
###############################################################################


# noinspection PyUnresolvedReferences
def wrap_sample_shape_to_size(
    sampler: callable[[int], np.ndarray], dim: int
) -> callable[[tuple[int, ...], int], np.ndarray]:
    """Converts a sampler by size to a sampler by shape.

    Computes the total size of the sample shape, calls the sampler with this
    size, and reshapes the output.

    Args:
        sampler: function that takes an integer as argument and returns this
            many samples as numpy arrays
        dim: number of dimensions of each sample, e.g, 0 for scalars, 1 for
            vectors, 2 for matrices, and so forth

    Returns:
        A sampler that takes a sample shape as an argument.
    """

    def wrapped(sample_shape: tuple[int, ...]) -> np.ndarray:
        sample_size = np.prod(sample_shape, dtype=int)
        arr = sampler(sample_size)
        base = arr.shape[-dim:] if dim > 0 else ()
        return np.reshape(arr, sample_shape + base)

    return wrapped


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


def random_matrix_from_eigs(eigvals: np.ndarray, rng: RNG = None) -> np.ndarray:
    """Generate random matrix with specified eigenvalues.

    Supports batched inputs. Assumes `eigvals` is a named vector tensor.
    Based on::
        https://blogs.sas.com/content/iml/2012/03/30/geneate-a-random-matrix-with-specified-eigenvalues.html

    Args:
        eigvals: (batched) tensor with desired eigenvalues.
        rng: random number generator seed

    Returns:
        A random matrix with given eigenvalues.
    """
    rng = np.random.default_rng(rng)

    # Sample orthogonal matrices
    dim = eigvals.shape[-1]
    _random_orthogonal_matrix = wrap_sample_shape_to_size(
        lambda s: ortho_group.rvs(dim, size=s, random_state=rng), dim=2
    )
    # Assume last dimension is "R"
    ortho = _random_orthogonal_matrix(eigvals.shape[:-1])

    mat = ortho.transpose(*range(ortho.ndim - 2), -1, -2) @ (
        eigvals[..., np.newaxis] * ortho
    )
    return mat


def random_mat_with_eigval_range(
    size: int,
    eigval_range: tuple[float, float],
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    """Generate a random matrix with absolute eigenvalues in the desired range."""
    # pylint:disable=too-many-arguments
    assert all(
        v >= 0 for v in eigval_range
    ), f"Eigenvalue range must be positive, got {eigval_range}"
    rng = np.random.default_rng(rng)

    low, high = eigval_range
    eigvals = rng.uniform(
        low=low, high=high, size=_sample_shape(horizon, stationary, n_batch) + (size,)
    )
    # Flip sign randomly
    np.negative(eigvals, where=rng.uniform(size=eigvals.shape) < 0.5, out=eigvals)

    mat = random_matrix_from_eigs(eigvals, rng=rng)
    mat = nt.matrix(as_float_tensor(mat))
    return expand_and_refine(mat, 2, horizon=horizon, n_batch=n_batch)


###############################################################################
# Other
###############################################################################


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
    state, time = nt.split(obs, [obs.size("R") - 1, 1], dim="R")
    time = time.int()
    return state, time


def pack_obs(state: Tensor, time: IntTensor) -> Tensor:
    """Reverses the `unpack_obs` transformation."""
    return torch.cat((state, time.float()), dim="R")
