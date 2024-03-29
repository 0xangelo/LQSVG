# pylint:disable=missing-module-docstring,unsubscriptable-object
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from gym.spaces import Box
from scipy.stats import ortho_group
from torch import IntTensor, Tensor

import lqsvg.torch.named as nt
from lqsvg.np_util import RNG, np_expand

from .types import AnyDynamics, LinDynamics, Linear, LinSDynamics, QuadCost

__all__ = [
    "ctrb",
    "dims_from_cost",
    "dims_from_dynamics",
    "dims_from_spaces",
    "dims_from_policy",
    "dynamics_factors",
    "isstationary",
    "isstable",
    "iscontrollable",
    "isstabilizable",
    "is_pbh_ctrb",
    "make_controllable",
    "np_expand_horizon",
    "pack_obs",
    "wrap_sample_shape_to_size",
    "random_matrix_from_eigs",
    "random_mat_with_eigval_range",
    "sample_eigvals",
    "spaces_from_dims",
    "stationary_dynamics",
    "stationary_eigvals",
    "stationary_dynamics_factors",
    "unpack_obs",
]

###############################################################################
# Diagnostics
###############################################################################


def isstationary(dynamics: AnyDynamics) -> bool:
    """Returns whether the dynamics are stationary (time-invariant)."""
    # noinspection PyTypeChecker
    return (
        nt.allclose(dynamics.F, dynamics.F.select("H", 0))
        and nt.allclose(dynamics.f, dynamics.f.select("H", 0))
        and (
            isinstance(dynamics, LinDynamics)
            or nt.allclose(dynamics.W, dynamics.W.select("H", 0))
        )
    )


def isstable(
    dynamics: Optional[AnyDynamics] = None, eigvals: Optional[np.ndarray] = None
) -> np.ndarray:
    """Returns whether the unactuated dynamics are stable.

    A linear, stationary, discrete-time system is stable iff the all the
    eigenvalues of the passive dynamics (F_s) live in the unit circle of the
    complex plane.

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


def iscontrollable(dynamics: AnyDynamics) -> np.ndarray:
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
    C = ctrb(dynamics)
    return np.linalg.matrix_rank(C) == n_state


def is_pbh_ctrb(dynamics: AnyDynamics) -> np.ndarray:
    """Returns whether the stationary dynamics are controllable.

    Uses the Hautus Lemma for controllability::
    https://en.wikipedia.org/wiki/Hautus_lemma#Hautus_Lemma_for_controllability
    Converted to the discrete time case by checking the eigenvectors
    corresponding to the eigenvalues with magnitude greater than 1.

    This function accepts a batch of linear dynamics.

    Warning:
        This function seems to return True even for systems known to be
        uncontrollable

    Note:
        It is not obvious if this condition is generalizable to non-stationary
        systems. Thus, this function first checks if the system is stationary.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    assert isstationary(dynamics)
    A, B = map(lambda x: x.numpy(), stationary_dynamics_factors(dynamics))
    eigvals, _ = np.linalg.eig(A)

    # Align arrays with new 'test' dimension for each eigval
    A = A[..., np.newaxis, :, :]
    B = B[..., np.newaxis, :, :]
    eigvals = eigvals[..., np.newaxis, np.newaxis]

    n_state, _, _ = dims_from_dynamics(dynamics)
    lam_eye = eigvals * np.eye(n_state)
    A, lam_eye = np.broadcast_arrays(A, lam_eye)
    B = B.repeat(n_state, axis=-3)

    pbh = np.concatenate([lam_eye - A, B], axis=-1)
    return np.all(np.linalg.matrix_rank(pbh) == n_state, axis=-1)


def isstabilizable(dynamics: AnyDynamics) -> np.ndarray:
    """Returns whether the stationary dynamics are stabilizable.

    Uses the Hautus Lemma for stabilizability::
    https://en.wikipedia.org/wiki/Hautus_lemma#Hautus_Lemma_for_stabilizability
    Converted to the discrete time case by checking the eigenvectors
    corresponding to the eigenvalues with magnitude greater than 1.

    This function accepts a batch of linear dynamics.

    Warning:
        This function seems to return True even for systems known not to be
        stabilizable

    Note:
        It is not obvious if this condition is generalizable to non-stationary
        systems. Thus, this function first checks if the system is stationary.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    assert isstationary(dynamics)
    F_s, F_a = map(lambda x: x.numpy(), stationary_dynamics_factors(dynamics))
    eigvals, _ = np.linalg.eig(F_s)

    tests = []
    n_state, _, _ = dims_from_dynamics(dynamics)
    for i in range(n_state):
        pbh = np.concatenate(
            [F_s - eigvals[..., i, np.newaxis, np.newaxis] * np.eye(n_state), F_a],
            axis=-1,
        )
        unstable = np.abs(eigvals[..., i]) >= 1.0
        # If abs eigval signals instability, check PBH condition
        test = np.where(unstable, np.linalg.matrix_rank(pbh) == n_state, True)
        # assert test.shape == F_s.shape[:-2], test.shape
        tests += [test]
    return np.stack(tests, axis=-1).all(axis=-1)


###############################################################################
# System manipulation
###############################################################################


# noinspection PyTypeChecker
def stationary_dynamics(dynamics: AnyDynamics) -> AnyDynamics:
    """Retrieve stationary dynamics parameters with no horizon dimension.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    assert isstationary(
        dynamics
    ), "Can't retrieve stationary parameters from non-stationary dynamics"
    F = dynamics.F.select("H", 0)
    f = dynamics.f.select("H", 0)
    if isinstance(dynamics, LinDynamics):
        return LinDynamics(F=F, f=f)

    W = dynamics.W.select("H", 0)
    return LinSDynamics(F=F, f=f, W=W)


def dynamics_factors(dynamics: AnyDynamics) -> tuple[Tensor, Tensor]:
    """Returns the unactuated and actuaded parts of the transition matrix."""
    # pylint:disable=invalid-name
    n_state, n_ctrl, _ = dims_from_dynamics(dynamics)
    F_s, F_a = nt.split(dynamics.F, [n_state, n_ctrl], dim="C")
    return F_s, F_a


def stationary_dynamics_factors(dynamics: AnyDynamics) -> tuple[Tensor, Tensor]:
    """Returns the decomposed transition matrix of a stationary system.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    assert isstationary(dynamics)
    # noinspection PyTypeChecker
    F_s, F_a = (x.select("H", 0) for x in dynamics_factors(dynamics))
    return F_s, F_a


def stationary_eigvals(dynamics: AnyDynamics) -> np.ndarray:
    """Returns the eigenvalues of unactuated stationary transition dynamics.

    Raises:
         AssertionError: if the (batch of) dynamics is not stationary
    """
    # pylint:disable=invalid-name
    F_s, _ = stationary_dynamics_factors(dynamics)
    # Assume last two dimensions correspond to rows and cols respectively
    eigvals, _ = np.linalg.eig(F_s.numpy())
    return eigvals


def ctrb(dynamics: AnyDynamics) -> np.ndarray:
    """Returns the controllability matrix for a stationary linear system.

    This function accepts batched dynamics.
    """
    # pylint:disable=invalid-name
    A, B = (x.numpy() for x in stationary_dynamics_factors(dynamics))
    n_state, _, _ = dims_from_dynamics(dynamics)
    # Assumes the last two dimensions correspond to rows and columns respectively
    C = np.concatenate(
        [np.linalg.matrix_power(A, i) @ B for i in range(n_state)], axis=-1
    )
    return C


def make_controllable(dynamics: AnyDynamics) -> AnyDynamics:
    """Compute controllable dynamics from reference one."""
    # pylint:disable=invalid-name
    n_state, _, _ = dims_from_dynamics(dynamics)
    # noinspection PyTypeChecker
    A, B = (x.numpy() for x in stationary_dynamics_factors(dynamics))

    # Compute eigendecomp of Fs
    _, col_eigvec = np.linalg.eig(A)
    # Ensure a column of Fa has a component in each eigenvector direction
    comb = np.ones(n_state, dtype=A.dtype) / np.sqrt(n_state)
    B_col = col_eigvec @ comb[..., np.newaxis]

    F_s = torch.from_numpy(A)
    F_a = torch.from_numpy(np.concatenate([B[..., :-1], B_col], axis=-1))
    F = (
        torch.cat([F_s, F_a], dim=-1)
        .expand_as(dynamics.F)
        .refine_names(*dynamics.F.names)
    )

    if isinstance(dynamics, LinSDynamics):
        new = LinSDynamics(F=F, f=dynamics.f, W=dynamics.W)
    else:
        new = LinDynamics(F=F, f=dynamics.f)
    return new


###############################################################################
# Random generation
###############################################################################


def wrap_sample_shape_to_size(
    sampler: Callable[[int], np.ndarray], dim: int
) -> Callable[[tuple[int, ...]], np.ndarray]:
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
        arr = sampler(int(np.prod(sample_shape)))
        base = arr.shape[-dim:] if dim > 0 else ()
        return np.reshape(arr, sample_shape + base)

    return wrapped


def random_matrix_from_eigs(
    eigvals: np.ndarray, rng: RNG = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generates random matrix with specified eigenvalues.

    Supports batched inputs. Assumes `eigvals` is a vector array.
    Based on::
        https://blogs.sas.com/content/iml/2012/03/30/geneate-a-random-matrix-with-specified-eigenvalues.html

    Args:
        eigvals: (batched) array with desired eigenvalues.
        rng: random number generator seed

    Returns:
        A random matrix with given eigenvalues and its eigenvectors as a matrix
    """
    rng = np.random.default_rng(rng)

    # Sample orthogonal matrices
    dim = eigvals.shape[-1]
    _random_orthogonal_matrix = wrap_sample_shape_to_size(
        lambda s: ortho_group.rvs(dim, size=s, random_state=rng), dim=2
    )
    # Assume last dimension is "R"
    eigvecs: np.ndarray = _random_orthogonal_matrix(eigvals.shape[:-1])

    mat = eigvecs @ (eigvals[..., np.newaxis] * eigvecs.swapaxes(-2, -1))
    return mat, eigvecs


def sample_eigvals(
    num: int, abs_low: float, abs_high: float, size: tuple[int, ...], rng: RNG
) -> np.ndarray:
    """Samples values with absolutes uniformly distributed in (`low`, `high`).

    This function uses `np.linspace` and `Generator.choice` to sample
    eigenvalues with multiplicity 1.

    Flips the sign of each eigenvalue randomly.

    Warning:
        This function is very slow

    Args:
        num: number of distinct eigenvalues per sample
        abs_low: lowest absolute eigenvalue
        abs_high: highest absolute eigenvalue
        size: shape for the batch
        rng: random number generator
    """
    rng = np.random.default_rng(rng)

    space = np.linspace(start=abs_low, stop=abs_high, num=1002, endpoint=True)[1:-1]
    samples = np.stack(
        [
            rng.choice(space, size=num, replace=False)
            for _ in range(np.prod(size, dtype=int))
        ]
    )
    eigvals = samples.reshape(size + (num,))

    # Flip sign randomly
    np.negative(eigvals, where=rng.uniform(size=eigvals.shape) < 0.5, out=eigvals)

    return eigvals


def random_mat_with_eigval_range(
    size: int,
    eigval_range: tuple[float, float],
    sample_shape: tuple[int, ...] = (),
    rng: RNG = None,
    ignore_rank_defficiency: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Generate a random matrix with absolute eigenvalues in the desired range."""
    # pylint:disable=too-many-arguments
    assert all(
        v >= 0 for v in eigval_range
    ), f"Eigenvalue range must be positive, got {eigval_range}"
    rng = np.random.default_rng(rng)
    low, high = eigval_range

    eigvals = sample_eigvals(size, low, high, sample_shape, rng)
    rank_defficient = np.any(np.abs(eigvals) < 1e-8)
    # Assert state transition matrix isn't rank deficient if needed
    while rank_defficient and not ignore_rank_defficiency:
        eigvals = sample_eigvals(size, low, high, sample_shape, rng)
        rank_defficient = np.any(np.abs(eigvals) < 1e-8)

    mat, eigvecs = random_matrix_from_eigs(eigvals, rng=rng)
    return mat, eigvals, eigvecs


###############################################################################
# Other
###############################################################################


# noinspection PyArgumentList
def dims_from_dynamics(dynamics: AnyDynamics) -> tuple[int, int, int]:
    """Retrieve LQG dimensions from linear Gaussian transition dynamics."""
    n_state = dynamics.F.size("R")
    n_ctrl = dynamics.F.size("C") - n_state
    horizon = dynamics.F.size("H")
    return n_state, n_ctrl, horizon


# noinspection PyArgumentList
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


# noinspection PyArgumentList
def dims_from_cost(cost: QuadCost) -> tuple[int, int]:
    """Retrieve dimensions from quadratic cost function.

    Args:
        cost: quadratic cost function parameters as tensors

    Returns:
        A tuple with row/col size and horizon length respectively
    """
    # pylint:disable=invalid-name
    n_tau = cost.C.size("C")
    horizon = cost.C.size("H")
    return n_tau, horizon


def dims_from_spaces(obs_space: Box, action_space: Box) -> tuple[int, int, int]:
    """Extracts LQR dimensions from Gym spaces."""
    n_state = obs_space.shape[0] - 1
    n_ctrl = action_space.shape[0]
    horizon = int(obs_space.high[-1])
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


def unpack_obs(obs: Tensor) -> tuple[Tensor, IntTensor]:
    """Unpack observation into state variables and time.

    Expects observation as a named 'vector' tensor.
    """
    # noinspection PyArgumentList
    state, time = nt.split(obs, [obs.size("R") - 1, 1], dim="R")
    time = time.int()
    return state, time


def pack_obs(state: Tensor, time: IntTensor) -> Tensor:
    """Reverses the `unpack_obs` transformation."""
    return torch.cat((state, time.float()), dim="R")
