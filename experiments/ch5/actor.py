# pylint:disable=missing-docstring
import torch
from torch import Tensor

from lqsvg.torch.nn import TVLinearPolicy
from lqsvg.types import DeterministicPolicy


def behavior_policy(
    policy: TVLinearPolicy, exploration: dict, rng: torch.Generator
) -> DeterministicPolicy:
    kind = exploration["type"]
    if kind is None:
        return policy
    if kind == "gaussian":
        return gaussian_behavior(policy, exploration, rng)
    raise ValueError(f"Unknown exploration type '{kind}'")


def gaussian_behavior(
    policy: TVLinearPolicy, exploration: dict, rng: torch.Generator
) -> DeterministicPolicy:
    sigma = exploration["action_noise_sigma"]

    def behavior(obs: Tensor) -> Tensor:
        act = policy(obs)
        noise = torch.randn(size=act.shape, generator=rng, device=act.device)
        return act + noise * sigma

    return behavior
