from __future__ import annotations

import numpy as np
import pytest
from gym.spaces import Box
from raylab.utils.exploration import GaussianNoise

from lqsvg.envs.lqr import spaces_from_dims
from lqsvg.envs.lqr.gym import RandomLQGEnv, TorchLQGMixin
from lqsvg.policy.modules.policy import TVLinearPolicy
from lqsvg.policy.time_varying_linear import LQGPolicy


@pytest.fixture(params=(2, 3))
def n_state(request) -> int:
    return request.param


@pytest.fixture(params=(2, 3))
def n_ctrl(request) -> int:
    return request.param


@pytest.fixture(params=(1, 20))
def horizon(request) -> int:
    return request.param


@pytest.fixture
def env_config(n_state: int, n_ctrl: int, horizon: int) -> dict:
    return dict(n_state=n_state, n_ctrl=n_ctrl, horizon=horizon)


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


@pytest.fixture()
def env(env_config: dict) -> TorchLQGMixin:
    return RandomLQGEnv(env_config)


def test_inits_stabilizing_policy(mocker, env: TorchLQGMixin):
    placer = mocker.spy(TVLinearPolicy, "stabilize_")

    policy = LQGPolicy(
        env.observation_space,
        env.action_space,
        config={"policy": {"module": {"policy_initializer": "stabilize_sys"}}},
    )
    policy.setup(env)
    assert placer.called
