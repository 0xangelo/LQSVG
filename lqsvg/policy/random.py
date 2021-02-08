# pylint:disable=missing-module-docstring
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ray.rllib import Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import TensorType


# noinspection PyAbstractClass
class RandomPolicy(Policy):
    """Random uniform RLlib policy."""

    # pylint:disable=abstract-method,too-many-arguments
    def compute_actions(
        self,
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorType], TensorType] = None,
        prev_reward_batch: Union[List[TensorType], TensorType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[MultiAgentEpisode]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return [self.action_space.sample() for _ in obs_batch], [], {}
