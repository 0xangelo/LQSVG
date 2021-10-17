"""Common type definitions."""
from typing import Callable, Iterable, NamedTuple, Optional, Sequence, Tuple

from nnrl.nn.distributions.types import SampleLogp
from torch import Tensor

from lqsvg.torch.types import SampleShape

State = Tensor
Action = Tensor
Trajectory = Tuple[Tensor, Tensor, Tensor, Tensor]
Transition = Tuple[Tensor, Tensor, Tensor, Tensor]

InitStateFn = Callable[[SampleShape], SampleLogp]
StateDynamics = Callable[[State, Action], SampleLogp]
RecurrentDynamics = Callable[
    [State, Action, Optional[Tensor]], Tuple[State, Tensor, Tensor]
]
RewardFunction = Callable[[State, Action], Tensor]

DeterministicPolicy = Callable[[State], Action]
QValueFn = Callable[[State, Action], Tensor]

TrajectorySampler = Callable[
    [int, Sequence[int]], Tuple[Tensor, Tensor, Tensor, Tensor]
]


# Utilities
class Directory(NamedTuple):
    """Named equivalent of tuples returned by os.walk."""

    name: str
    subdirs: Iterable[str]
    files: Iterable[str]
