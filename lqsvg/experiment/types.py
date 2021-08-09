"""Common type definitions."""
from typing import Callable

from torch import Tensor

QValueFn = Callable[[Tensor, Tensor], Tensor]
