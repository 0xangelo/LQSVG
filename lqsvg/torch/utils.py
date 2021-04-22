# pylint:disable=missing-module-docstring
from __future__ import annotations

import contextlib
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Generator, Tensor

import lqsvg.np_util as np_util
from lqsvg.torch import named as nt


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
    rng: np_util.RNG = None,
) -> Tensor:
    """PyTorch version of random symmetric positive-definite matrix generation."""
    return torch.as_tensor(
        np_util.make_spd_matrix(n_dim, sample_shape=sample_shape, rng=rng),
        dtype=dtype,
        device=device,
    )


def softplusinv(tensor: Tensor, *, beta: float = 1.0) -> Tensor:
    """Returns the inverse softplus transformation."""
    return torch.log(torch.exp(beta * tensor) - 1) / beta


def disassemble_cholesky(tensor: Tensor, *, beta: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Compute cholesky factor and break it into unconstrained parameters."""
    tril = nt.cholesky(tensor, upper=False)
    ltril = nt.tril(tril, diagonal=-1)
    pre_diag = softplusinv(nt.diagonal(tril, dim1="R", dim2="C"), beta=beta)
    return ltril, pre_diag


def assemble_cholesky(ltril: Tensor, pre_diag: Tensor, *, beta: float = 1.0) -> Tensor:
    """Transform uncostrained parameters into cholesky factor."""
    return ltril + torch.diag_embed(nt.softplus(pre_diag, beta=beta))
