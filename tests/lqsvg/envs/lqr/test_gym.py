from typing import Any
from typing import Iterable
from typing import Type

import numpy as np
import pytest

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.gym import LQGSpec
from lqsvg.envs.lqr.gym import RandomLQGEnv
from lqsvg.envs.lqr.gym import RandomVectorLQG
from lqsvg.envs.lqr.types import LinSDynamics
from lqsvg.envs.lqr.types import QuadCost


def standard_fixture(params: Iterable[Any], name: str) -> callable:
    @pytest.fixture(params=params, ids=lambda x: f"{name}:{x}")
    def func(request):
        return request.param

    return func


n_state = standard_fixture((1, 2, 4), "NState")
n_ctrl = standard_fixture((1, 2, 3), "NCtrl")
horizon = standard_fixture((1, 4, 16), "Horizon")
gen_seed = standard_fixture((1, 2, 3), "Seed")


@pytest.fixture
def spec_cls():
    return LQGSpec


@pytest.fixture
def spec(spec_cls: Type[LQGSpec], n_state, n_ctrl, horizon, gen_seed):
    return spec_cls(
        n_state=n_state, n_ctrl=n_ctrl, horizon=horizon, gen_seed=gen_seed, num_envs=4
    )


@pytest.fixture(params=(RandomLQGEnv, RandomVectorLQG), ids=lambda x: x.__name__)
def env_creator(request):
    return request.param


def test_spec(spec: LQGSpec, n_state, n_ctrl, horizon, gen_seed):
    assert spec.n_state == n_state
    assert spec.n_ctrl == n_ctrl
    assert spec.horizon == horizon
    assert spec.gen_seed == gen_seed


def allclose_dynamics(dyn1: LinSDynamics, dyn2: LinSDynamics) -> bool:
    equal = [nt.allclose(d1, d2) for d1, d2 in zip(dyn1, dyn2)]
    return all(equal)


def allclose_cost(cost1: QuadCost, cost2: QuadCost) -> bool:
    equal = [nt.allclose(c1, c2) for c1, c2 in zip(cost1, cost2)]
    return all(equal)


def test_gen_seed(env_creator, spec):
    spec.gen_seed = 42
    env1 = env_creator(spec)
    env2 = env_creator(spec)

    assert allclose_dynamics(env1.dynamics, env2.dynamics)
    assert allclose_cost(env1.cost, env2.cost)


def test_spaces(env_creator, spec):
    env = env_creator(spec)

    assert env.observation_space.shape[0] == spec.n_state + 1
    assert env.observation_space.low[-1] == 0
    assert env.observation_space.high[-1] == spec.horizon
    assert env.action_space.shape[0] == spec.n_ctrl


def test_reset(spec):
    env = RandomLQGEnv(spec)

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs in env.observation_space  # pylint:disable=unsupported-membership-test


def test_vector_reset(spec):
    env = RandomVectorLQG(spec)

    obs = env.vector_reset()
    assert hasattr(env, "num_envs")
    assert isinstance(obs, list)
    assert len(obs) == env.num_envs
    assert obs[0] in env.observation_space
    assert all(o in env.observation_space for o in obs)


def test_vector_step(spec):
    env = RandomVectorLQG(spec)

    env.vector_reset()
    acts = [env.action_space.sample() for _ in range(env.num_envs)]
    new_obs, rews, dones, infos = env.vector_step(acts)
    assert all(isinstance(o, list) for o in (new_obs, rews, dones, infos))
    assert all(len(o) == env.num_envs for o in (new_obs, rews, dones, infos))

    assert all(o in env.observation_space for o in new_obs)
    assert all(isinstance(r, float) for r in rews)
    assert all(isinstance(d, bool) for d in dones)
    assert all(isinstance(i, dict) for i in infos)
