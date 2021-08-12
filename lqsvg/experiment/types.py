"""Common type definitions."""
from typing import Callable, Optional, Sequence, Tuple

from nnrl.nn.distributions.types import SampleLogp
from torch import Tensor

from lqsvg.torch.types import SampleShape

State = Tensor
Action = Tensor

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
