# pylint:disable=missing-module-docstring
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

import lqsvg.np_util as np_util


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
