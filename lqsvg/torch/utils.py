# pylint:disable=missing-module-docstring
import numpy as np
import torch
from torch import Tensor


def as_float_tensor(array: np.ndarray) -> Tensor:
    """Convert numpy array to float tensor.

    Only allocates new memory if necessary, i.e., when the numpy array dtype is
    not a float32.
    """
    return torch.as_tensor(array, dtype=torch.float32)
