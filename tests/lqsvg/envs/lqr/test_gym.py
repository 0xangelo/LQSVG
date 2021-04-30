from __future__ import annotations

from typing import Callable, Union

import gym
import numpy as np
import pytest
from ray.rllib import VectorEnv

from lqsvg.envs.lqr.gym import RandomLQGEnv, RandomVectorLQG
from lqsvg.testing.compare import allclose_cost, allclose_dynamics
from lqsvg.testing.fixture import standard_fixture

EnvCreator = Callable[[dict], Union[gym.Env, VectorEnv]]


@pytest.fixture(params=(RandomLQGEnv, RandomVectorLQG), ids=lambda x: x.__name__)
def env_creator(request) -> EnvCreator:
    cls = request.param
    if issubclass(cls, RandomVectorLQG):
        return lambda config: RandomVectorLQG({"num_envs": 1, **config})

    return cls


@pytest.fixture
def config(
    n_state: int,
    n_ctrl: int,
    horizon: int,
    seed: int,
) -> dict:
    return dict(n_state=n_state, n_ctrl=n_ctrl, horizon=horizon, rng=seed)


# Test common TorchLQGMixin interface ==========================================
def test_rng(env_creator: EnvCreator, config: dict):
    config["rng"] = 42
    env1 = env_creator(config)
    env2 = env_creator(config)

    assert allclose_dynamics(env1.dynamics, env2.dynamics)
    assert allclose_cost(env1.cost, env2.cost)


def test_spaces(env_creator: EnvCreator, config: dict):
    env = env_creator(config)

    assert env.observation_space.shape[0] == config["n_state"] + 1
    assert env.observation_space.low[-1] == 0
    assert env.observation_space.high[-1] == config["horizon"]
    assert env.action_space.shape[0] == config["n_ctrl"]


def test_properties(env_creator: EnvCreator, config: dict):
    env = env_creator(config)

    assert env.horizon == config["horizon"]
    assert env.n_state == config["n_state"]
    assert env.n_ctrl == config["n_ctrl"]


def test_solution(env_creator: EnvCreator, config: dict):
    env = env_creator(config)

    pistar, qstar, vstar = env.solution
    assert pistar[0].names == tuple("H R C".split())
    assert pistar[1].names == tuple("H R".split())

    assert qstar[0].names == tuple("H R C".split())
    assert qstar[1].names == tuple("H R".split())
    assert qstar[2].names == tuple("H".split())

    assert vstar[0].names == tuple("H R C".split())
    assert vstar[1].names == tuple("H R".split())
    assert vstar[2].names == tuple("H".split())


# ==============================================================================
# Test RandomLQGEnv ============================================================
def test_reset(config: dict):
    env = RandomLQGEnv(config)

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs in env.observation_space  # pylint:disable=unsupported-membership-test


def test_step(config: dict):
    env = RandomLQGEnv(config)

    env.reset()
    act = env.action_space.sample()
    new_obs, rew, done, info = env.step(act)

    # pylint:disable=unsupported-membership-test
    assert new_obs in env.observation_space
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# ==============================================================================


num_envs = standard_fixture((1, 2, 4), "NEnvs")


@pytest.fixture
def vector_config(config: dict, num_envs: int) -> dict:
    config["num_envs"] = num_envs
    return config


# Test RandomVectorLQG =========================================================
def test_vector_init(vector_config: dict):
    env = RandomVectorLQG(vector_config)

    assert env.num_envs == vector_config["num_envs"]
    assert hasattr(env, "curr_states")
    assert env.curr_states is None


def test_vector_reset(vector_config: dict):
    env = RandomVectorLQG(vector_config)

    obs = env.vector_reset()
    assert hasattr(env, "num_envs")
    assert isinstance(obs, list)
    assert len(obs) == env.num_envs
    assert obs[0] in env.observation_space
    assert all(o in env.observation_space for o in obs)

    assert np.allclose(obs, env.curr_states)


def swap_row(arr: np.ndarray, in1: int, in2: int):
    swap = arr[in1].copy()
    arr[in1] = arr[in2]
    arr[in2] = swap


def test_reset_at(vector_config: dict):
    env = RandomVectorLQG(vector_config)
    rng = np.random.default_rng(vector_config["rng"])

    obs = np.array(env.vector_reset())
    index = rng.choice(env.num_envs)
    reset = env.reset_at(index)
    assert reset in env.observation_space

    curr_states = env.curr_states
    if env.num_envs > 1:
        swap_row(obs, 0, index)
        curr_states = env.curr_states
        swap_row(curr_states, 0, index)
        assert np.allclose(obs[1:], curr_states[1:])

    assert np.allclose(reset, curr_states[0])


def test_vector_step(vector_config: dict):
    env = RandomVectorLQG(vector_config)

    env.vector_reset()
    acts = [env.action_space.sample() for _ in range(env.num_envs)]
    new_obs, rews, dones, infos = env.vector_step(acts)
    assert all(isinstance(o, list) for o in (new_obs, rews, dones, infos))
    assert all(len(o) == env.num_envs for o in (new_obs, rews, dones, infos))

    assert all(o in env.observation_space for o in new_obs)
    assert all(isinstance(r, float) for r in rews)
    assert all(isinstance(d, bool) for d in dones)
    assert all(isinstance(i, dict) for i in infos)
