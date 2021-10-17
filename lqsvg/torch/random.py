"""Random tensor generation tooling."""
import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Generator, Tensor

from lqsvg import np_util
from lqsvg.np_util import RNG
from lqsvg.torch import named as nt
from lqsvg.torch.utils import as_float_tensor, expand_and_refine


@contextlib.contextmanager
def default_generator_seed(seed: int):
    """Temporarily set PyTorch's random seed."""
    # noinspection PyTypeChecker
    state: torch.ByteTensor = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(state)


def numpy_to_torch_generator(rng: np.random.Generator) -> Generator:
    """Returns a PyTorch random number generator from a Numpy one."""
    generator = torch.Generator()
    generator.manual_seed(rng.integers(np.iinfo(int).max).item())
    return generator


def default_rng(seed: Union[None, int, Generator] = None) -> Generator:
    """Mirrors numpy's `default_rng` to produce RNGs for Pytorch.

    Args:
        seed: a seed to initialize the generator. If passed a Generator, will
        return it unaltered. Otherwise, creates a new one. If passed an
        integer, will use it as the manual seed for the created generator.

    Returns:
        A PyTorch Generator instance
    """
    if isinstance(seed, Generator):
        return seed

    rng = Generator()
    if isinstance(seed, int):
        rng.manual_seed(seed)
    return rng


def make_spd_matrix(
    n_dim: int,
    sample_shape: Tuple[int, ...],
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = None,
    rng: np_util.RNG = None,
) -> Tensor:
    """PyTorch version of random symmetric positive-definite matrix generation."""
    return torch.as_tensor(
        np_util.make_spd_matrix(n_dim, sample_shape=sample_shape, rng=rng),
        dtype=dtype,
        device=device,
    )


def minimal_sample_shape(
    horizon: int, stationary: bool = False, n_batch: Optional[int] = None
) -> Tuple[int, ...]:
    """Minimal sample shape from horizon, stationarity, and batch size.

    This works in tandem with expand_and_refine to standardize horizon and
    batch dimensions.
    """
    horizon_shape = () if stationary else (horizon,)
    batch_shape = () if n_batch is None else (n_batch,)
    return horizon_shape + batch_shape


def normal_vector(
    size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    # pylint:disable=missing-function-docstring
    rng = np.random.default_rng(rng)

    vec_shape = (size,)
    shape = (
        minimal_sample_shape(horizon, stationary=stationary, n_batch=n_batch)
        + vec_shape
    )
    vec = nt.vector(as_float_tensor(rng.normal(size=shape)))
    vec = expand_and_refine(vec, 1, horizon=horizon, n_batch=n_batch)
    return vec


def normal_matrix(
    row_size: int, col_size: int, sample_shape: Tuple[int, ...] = (), rng: RNG = None
) -> np.ndarray:
    """Matrix with standard Normal i.i.d. entries."""
    rng = np.random.default_rng(rng)
    return rng.normal(size=sample_shape + (row_size, col_size))


def uniform_matrix(
    row_size: int,
    col_size: int,
    horizon: int,
    stationary: bool = False,
    low: float = 0.0,
    high: float = 1.0,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    """Matrix with Uniform i.i.d. entries."""
    # pylint:disable=too-many-arguments
    mat_shape = (row_size, col_size)
    shape = (
        minimal_sample_shape(horizon, stationary=stationary, n_batch=n_batch)
        + mat_shape
    )
    mat = nt.matrix(as_float_tensor(rng.uniform(low=low, high=high, size=shape)))
    mat = expand_and_refine(mat, 2, horizon=horizon, n_batch=n_batch)
    return mat


def spd_matrix(
    size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    """Generates a random, named, symmetric positive definite matrix."""
    sample_shape = minimal_sample_shape(horizon, stationary=stationary, n_batch=n_batch)
    mat = np_util.make_spd_matrix(size, sample_shape=sample_shape, rng=rng)
    mat = nt.matrix(as_float_tensor(mat))
    return expand_and_refine(mat, 2, horizon=horizon, n_batch=n_batch)


def unit_vector(
    size: int,
    horizon: int,
    stationary: bool = False,
    n_batch: Optional[int] = None,
    rng: RNG = None,
) -> Tensor:
    """Generates a random, named, unit vector."""
    sample_shape = minimal_sample_shape(horizon, stationary=stationary, n_batch=n_batch)
    vec = np_util.random_unit_vector(size, sample_shape=sample_shape, rng=rng)
    vec = nt.vector(as_float_tensor(vec))
    return expand_and_refine(vec, 1, horizon=horizon, n_batch=n_batch)


def sample_with_replacement(
    inputs: Union[Tensor, Sequence[Tensor]],
    size: int,
    dim: Union[int, str],
    rng: torch.Generator,
) -> Union[Tensor, Sequence[Tensor]]:
    """Subsamples tensor inputs along a dimension with replacement."""
    if torch.is_tensor(inputs):
        inputs = [inputs]
    # noinspection PyArgumentList
    weights = torch.ones(inputs[0].size(dim))
    idxs = torch.multinomial(weights, num_samples=size, replacement=True, generator=rng)
    outputs = tuple(nt.index_select(t, dim=dim, index=idxs.int()) for t in inputs)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
