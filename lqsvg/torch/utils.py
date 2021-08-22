# pylint:disable=missing-module-docstring
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from lqsvg.torch import named as nt


def as_float_tensor(array: np.ndarray) -> Tensor:
    """Convert numpy array to float tensor.

    Only allocates new memory if necessary, i.e., when the numpy array dtype is
    not a float32.
    """
    return torch.as_tensor(array, dtype=torch.float32)


def softplusinv(tensor: Tensor, *, beta: float = 1.0) -> Tensor:
    """Returns the inverse softplus transformation."""
    return torch.log(torch.exp(beta * tensor) - 1) / beta


def disassemble_cholesky(tensor: Tensor, *, beta: float = 1.0) -> Tuple[Tensor, Tensor]:
    """Compute cholesky factor and break it into unconstrained parameters."""
    tril = nt.cholesky(tensor)
    ltril = nt.tril(tril, diagonal=-1)
    pre_diag = softplusinv(nt.diagonal(tril, dim1="R", dim2="C"), beta=beta)
    return ltril, pre_diag


def assemble_cholesky(ltril: Tensor, pre_diag: Tensor, *, beta: float = 1.0) -> Tensor:
    """Transform uncostrained parameters into cholesky factor."""
    ltril, diag = nt.tril(ltril, diagonal=-1), nt.softplus(pre_diag, beta=beta)
    return ltril + torch.diag_embed(diag)


def tensors_to_vector(tensors: Iterable[Tensor]) -> Tensor:
    """Reshape and combine tensors into a vector representation."""
    vector = []
    for t in tensors:
        vector += [nt.unnamed(t).reshape(-1)]
    return nt.vector(torch.cat(vector))


def vector_to_tensors(vector: Tensor, tensors: Iterable[Tensor]) -> Iterable[Tensor]:
    """Split and reshape vector into tensors matching others' shapes."""
    split = []
    offset = 0
    vector = nt.unnamed(vector)
    for t in tensors:
        split += [vector[offset : offset + t.numel()].view_as(t).refine_names(*t.names)]
        offset += t.numel()
    return split


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
