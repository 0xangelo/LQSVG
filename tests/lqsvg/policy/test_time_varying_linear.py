from __future__ import annotations

import numpy as np
import pytest
from gym.spaces import Box
from raylab.utils.exploration import GaussianNoise

from lqsvg.envs.lqr import spaces_from_dims
from lqsvg.policy.time_varying_linear import LQGPolicy


@pytest.fixture
def env_config() -> dict:
    return dict(n_state=2, n_ctrl=2, horizon=100)


@pytest.fixture
def spaces(env_config) -> tuple[Box, Box]:
    return spaces_from_dims(
        env_config["n_state"], env_config["n_ctrl"], env_config["horizon"]
    )


@pytest.fixture
def obs(spaces: tuple[Box, Box]) -> np.ndarray:
    obs_space, _ = spaces
    return obs_space.sample()


@pytest.fixture
def policy(spaces: tuple[Box, Box]) -> LQGPolicy:
    observation_space, action_space = spaces
    return LQGPolicy(observation_space, action_space, config={})


def test_does_not_explore_by_default(mocker, policy: LQGPolicy, obs: np.ndarray):
    expl = mocker.spy(GaussianNoise, "get_exploration_action")
    policy.compute_single_action(obs)

    assert "explore" in expl.call_args.kwargs
    assert expl.call_args.kwargs["explore"] is False
