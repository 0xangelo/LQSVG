"""Common type definitions."""
from typing import Callable, Sequence, Tuple

from torch import Tensor

QValueFn = Callable[[Tensor, Tensor], Tensor]
DeterministicPolicy = Callable[[Tensor], Tensor]
InitStateFn = Callable[[Sequence[int]], Tuple[Tensor, Tensor]]
StateDynamics = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
RewardFunction = Callable[[Tensor, Tensor], Tensor]
TrajectorySampler = Callable[
    [int, Sequence[int]], Tuple[Tensor, Tensor, Tensor, Tensor]
]
