"""Utilities for named tensors."""
# pylint:disable=invalid-name,missing-function-docstring
from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from typing import Callable, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import BoolTensor, IntTensor, LongTensor
from torch import Tensor as _Tensor

Tensor = TypeVar("Tensor", bound=_Tensor)

MATRIX_NAMES = (..., "R", "C")
VECTOR_NAMES = MATRIX_NAMES[:-1]
SCALAR_NAMES = MATRIX_NAMES[:-2]


@contextmanager
def suppress_named_tensor_warning():
    """Ignore PyTorch warning regarding experimental state of named tensors."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Named tensors .+are an experimental feature",
            category=UserWarning,
            module="torch._tensor",
        )
        yield


def variadic(func: Callable[[Tensor], Tensor]):
    @functools.wraps(func)
    def wrapped(*tensors: Tensor) -> Union[Tensor, tuple[Tensor, ...]]:
        result = tuple(func(t) for t in tensors)
        return result[0] if len(result) == 1 else result

    return wrapped


@variadic
def unnamed(tensor: Tensor) -> Tensor:
    return tensor.rename(None)


@variadic
def horizon(tensor: Tensor) -> Tensor:
    return tensor.refine_names("H", ...)


@variadic
def matrix(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*MATRIX_NAMES)


@variadic
def vector(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*VECTOR_NAMES)


@variadic
def scalar(tensor: Tensor) -> Tensor:
    return tensor.refine_names(*SCALAR_NAMES)


@variadic
def matrix_to_vector(tensor: Tensor) -> Tensor:
    return matrix(tensor).squeeze("C")


@variadic
def matrix_to_scalar(tensor: Tensor) -> Tensor:
    return matrix(tensor).squeeze("R").squeeze("C")


@variadic
def vector_to_matrix(tensor: Tensor) -> Tensor:
    return vector(tensor).align_to(..., "R", "C")


@variadic
def vector_to_scalar(tensor: Tensor) -> Tensor:
    return vector(tensor).squeeze("R")


@variadic
def scalar_to_matrix(tensor: Tensor) -> Tensor:
    return scalar(tensor).align_to(..., "R", "C")


@variadic
def scalar_to_vector(tensor: Tensor) -> Tensor:
    return scalar(tensor).align_to(..., "R")


def trace(tensor: Tensor) -> Tensor:
    """Returns the trace of a batched matrix.

    Assumes input is a refined matrix.
    """
    dim1, dim2 = tensor.names.index("R"), tensor.names.index("C")
    diag = torch.diagonal(unnamed(tensor), dim1=dim1, dim2=dim2)
    diag = vector(diag)
    # noinspection PyTypeChecker
    return diag.sum("R").refine_names(*tensor.select("R", 0).select("C", 0).names)


def transpose(tensor: Tensor) -> Tensor:
    return tensor.transpose("R", "C").rename(R="C", C="R")


def stack_horizon(*tensors: Tensor) -> Tensor:
    return torch.cat([t.align_to("H", ...) for t in tensors], dim="H")


def index_select(
    tensor: Tensor, dim: str, index: Union[IntTensor, LongTensor]
) -> Tensor:
    idim = tensor.names.index(dim)
    selected = torch.index_select(unnamed(tensor), dim=idim, index=unnamed(index))
    return selected.refine_names(*tensor.names)


def index_by(tensor: Tensor, dim: str, index: Union[IntTensor, LongTensor]) -> Tensor:
    idim = tensor.names.index(dim)
    vector_index = unnamed(index).reshape(-1)
    selected = torch.index_select(unnamed(tensor), dim=idim, index=vector_index)
    new_shape = tensor.shape[:idim] + index.shape + tensor.shape[idim + 1 :]
    reshaped = selected.reshape(new_shape)
    new_names = tensor.names[:idim] + index.names + tensor.names[idim + 1 :]
    return reshaped.refine_names(*new_names)


def diagonal(tensor: Tensor, offset: int = 0, dim1: str = "R", dim2: str = "C"):
    names = tensor.names
    idx1, idx2 = names.index(dim1), names.index(dim2)
    diag = torch.diagonal(unnamed(tensor), offset=offset, dim1=idx1, dim2=idx2)
    return diag.unsqueeze(idx2).refine_names(*names).squeeze(dim2)


def tril(tensor: Tensor, *args, **kwargs) -> Tensor:
    return torch.tril(unnamed(tensor), *args, **kwargs).refine_names(*tensor.names)


def cholesky(tensor: Tensor, *args, **kwargs) -> Tensor:
    return torch.linalg.cholesky(unnamed(tensor), *args, **kwargs).refine_names(
        *tensor.names
    )


def softplus(tensor: Tensor, *, beta: float = 1) -> Tensor:
    return F.softplus(unnamed(tensor), beta=beta).refine_names(*tensor.names)


def allclose(inpt: _Tensor, other: _Tensor, *args: object, **kwargs: object) -> bool:
    return torch.allclose(unnamed(inpt), unnamed(other), *args, **kwargs)


def isclose(inpt: Tensor, other: Tensor, *args, **kwargs) -> BoolTensor:
    names = inpt.names
    inpt, other = unnamed(inpt, other)
    return torch.isclose(inpt, other, *args, **kwargs).refine_names(*names)


def where(condition: _Tensor, branch_a: _Tensor, branch_b: _Tensor) -> _Tensor:
    names = branch_a.names
    condition, branch_a, branch_b = unnamed(condition, branch_a, branch_b)
    filtered = torch.where(condition, branch_a, branch_b)
    return filtered.refine_names(*names)


def split(
    tensor: Tensor, split_size_or_sections: Union[int, list[int]], dim: str
) -> tuple[Tensor, ...]:
    return torch.split(tensor, split_size_or_sections, dim=tensor.names.index(dim))


def reduce_all(
    tensor: BoolTensor, dim: Optional[str] = None, keepdim: bool = False, **kwargs
) -> BoolTensor:
    idx = tensor.names.index(dim) if dim else None
    names = tuple(x for x in tensor.names if not dim or keepdim or x != dim)
    result = torch.all(unnamed(tensor), dim=idx, keepdim=keepdim, **kwargs)
    return result.refine_names(*names)
