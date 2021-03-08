"""Utilities for named tensors."""
# pylint:disable=invalid-name,missing-function-docstring
from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Union

import torch
import torch.nn.functional as F
from torch import IntTensor
from torch import LongTensor
from torch import Tensor


MATRIX_NAMES = (..., "R", "C")
VECTOR_NAMES = MATRIX_NAMES[:-1]
SCALAR_NAMES = MATRIX_NAMES[:-2]


@contextmanager
def suppress_named_tensor_warning():
    """Ignore PyTorch warning regarding experimental state of named tensors."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Named tensors .+are an experimental feature.*",
            module="torch.tensor",
        )
        yield


def unnamed(*tensors: Tensor) -> Union[Tensor, tuple[Tensor, ...]]:
    result = tuple(t.rename(None) for t in tensors)
    return result[0] if len(result) == 1 else result


def horizon(tensor: Tensor) -> Tensor:
    return tensor.refine_names("H", ...)


def matrix(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*MATRIX_NAMES)


def vector(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*VECTOR_NAMES)


def scalar(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*SCALAR_NAMES)


def matrix_to_vector(tensor: Tensor) -> Tensor:
    return matrix(tensor).squeeze("C")


def matrix_to_scalar(tensor: Tensor) -> Tensor:
    return matrix(tensor).squeeze("R").squeeze("C")


def vector_to_matrix(tensor: Tensor) -> Tensor:
    return vector(tensor).align_to(..., "R", "C")


def vector_to_scalar(tensor: Tensor) -> Tensor:
    return vector(tensor).squeeze("R")


def scalar_to_matrix(tensor: Tensor) -> Tensor:
    return scalar(tensor).align_to(..., "R", "C")


def scalar_to_vector(tensor: Tensor) -> Tensor:
    return scalar(tensor).align_to(..., "R")


def refine_matrix_input(tensor: Tensor) -> Tensor:
    return tensor.refine_names(..., "R", "C")


def refine_vector_input(tensor: Tensor) -> Tensor:
    return tensor.refine_names(..., "R").align_to(..., "R", "C")


def refine_scalar_input(tensor: Tensor) -> Tensor:
    return tensor.refine_names(...).align_to(..., "R", "C")


def refine_matrix_output(tensor: Tensor) -> Tensor:
    return tensor.refine_names(..., "R", "C")


def refine_vector_output(tensor: Tensor) -> Tensor:
    return tensor.refine_names(..., "R", "C").squeeze("C")


def refine_scalar_output(tensor: Tensor) -> Tensor:
    return tensor.refine_names(..., "R", "C").squeeze("R").squeeze("C")


def trace(tensor: Tensor) -> Tensor:
    """Returns the trace of a batched matrix.

    Assumes input is a refined matrix.
    """
    diag = torch.diagonal(tensor.rename(None), dim1=-2, dim2=-1)
    return diag.sum(-1).refine_names(*tensor.names[:-2])


def transpose(tensor: Tensor) -> Tensor:
    return tensor.transpose("R", "C").rename(R="C", C="R")


def stack_horizon(*tensors: Tensor) -> Tensor:
    return torch.cat([t.align_to("H", ...) for t in tensors], dim="H")


def index_select(
    tensor: Tensor, dim: str, index: Union[IntTensor, LongTensor]
) -> Tensor:
    aligned = tensor.align_to(dim, ...)
    selected = torch.index_select(unnamed(aligned), dim=0, index=unnamed(index))
    refined = selected.refine_names(*aligned.names)
    return refined.align_to(*tensor.names)


def index_by(tensor: Tensor, dim: str, index: Union[IntTensor, LongTensor]) -> Tensor:
    aligned = tensor.align_to(dim, ...)
    vector_index = unnamed(index).reshape(-1)
    selected = torch.index_select(unnamed(aligned), dim=0, index=vector_index)
    reshaped = selected.reshape(*(index.shape + aligned.shape[1:]))
    refined = reshaped.refine_names(*(index.names + aligned.names[1:]))
    return refined.align_to(*(n if n != dim else ... for n in tensor.names))


def diagonal(tensor: Tensor, *args, dim1: str = "R", dim2: str = "C", **kwargs):
    permuted = tensor.align_to(..., dim1, dim2)
    diag = torch.diagonal(
        unnamed(permuted), *args, dim1=-2, dim2=-1, **kwargs
    ).unsqueeze(-1)
    return diag.refine_names(*permuted.names).align_to(*tensor.names).squeeze(dim2)


def tril(tensor: Tensor, *args, **kwargs) -> Tensor:
    return torch.tril(unnamed(tensor), *args, **kwargs).refine_names(*tensor.names)


def cholesky(tensor: Tensor, *args, **kwargs) -> Tensor:
    return torch.cholesky(unnamed(tensor), *args, **kwargs).refine_names(*tensor.names)


def softplus(tensor: Tensor) -> Tensor:
    return F.softplus(unnamed(tensor)).refine_names(*tensor.names)


def allclose(inpt: Tensor, other: Tensor, *args, **kwargs) -> bool:
    return torch.allclose(unnamed(inpt), unnamed(other), *args, **kwargs)


def where(
    condition: Tensor, branch_a: Tensor, branch_b: Tensor, *args, **kwargs
) -> Tensor:
    names = branch_a.names
    condition, branch_a, branch_b = unnamed(condition, branch_a, branch_b)
    return torch.where(condition, branch_a, branch_b, *args, **kwargs).refine_names(
        *names
    )


def split(
    tensor: Tensor, split_size_or_sections: Union[int, list[int]], dim: str
) -> tuple[Tensor, ...]:
    permuted = tensor.align_to(..., dim)
    tensors = torch.split(permuted, split_size_or_sections, dim=-1)
    return tuple(
        t.refine_names(*permuted.names).align_to(*tensor.names) for t in tensors
    )
