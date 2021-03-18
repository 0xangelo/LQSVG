# pylint:disable=missing-module-docstring
from __future__ import annotations

import contextlib
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import Generator
from torch import Tensor

import lqsvg.np_util as np_util


@contextlib.contextmanager
def default_generator_seed(seed: int):
    """Temporarily set PyTorch's random seed."""
    state: torch.ByteTensor = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(state)


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


def as_float_tensor(array: np.ndarray) -> Tensor:
    """Convert numpy array to float tensor.

    Only allocates new memory if necessary, i.e., when the numpy array dtype is
    not a float32.
    """
    return torch.as_tensor(array, dtype=torch.float32)


def make_spd_matrix(
    n_dim: int,
    sample_shape: tuple[int, ...],
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """PyTorch version of random symmetric positive-definite matrix generation."""
    return torch.as_tensor(
        np_util.make_spd_matrix(n_dim, sample_shape=sample_shape),
        dtype=dtype,
        device=device,
    )
