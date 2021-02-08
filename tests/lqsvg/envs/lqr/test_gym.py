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

from .utils import allclose_dynamics, allclose_cost


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
def spec_cls() -> Type[LQGSpec]:
    return LQGSpec


@pytest.fixture
def spec(
    spec_cls: Type[LQGSpec], n_state: int, n_ctrl: int, horizon: int, gen_seed: int
) -> LQGSpec:
    return spec_cls(
        n_state=n_state, n_ctrl=n_ctrl, horizon=horizon, gen_seed=gen_seed, num_envs=1
    )


@pytest.fixture(params=(RandomLQGEnv, RandomVectorLQG), ids=lambda x: x.__name__)
def env_creator(request):
    return request.param


# Test common TorchLQGMixin interface ==========================================
def test_spec(spec: LQGSpec, n_state, n_ctrl, horizon, gen_seed):
    assert spec.n_state == n_state
    assert spec.n_ctrl == n_ctrl
    assert spec.horizon == horizon
    assert spec.gen_seed == gen_seed


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


def test_properties(env_creator, spec):
    env = env_creator(spec)

    assert env.horizon == spec.horizon
    assert env.n_state == spec.n_state
    assert env.n_ctrl == spec.n_ctrl


def test_solution(env_creator, spec):
    env = env_creator(spec)

    pistar, qstar, vstar = env.solution()
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
def test_reset(spec: LQGSpec):
    env = RandomLQGEnv(spec)

    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs in env.observation_space  # pylint:disable=unsupported-membership-test


def test_step(spec: LQGSpec):
    env = RandomLQGEnv(spec)

    env.reset()
    act = env.action_space.sample()
    new_obs, rew, done, info = env.step(act)

    assert new_obs in env.observation_space
    assert isinstance(rew, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# ==============================================================================


num_envs = standard_fixture((1, 2, 4), "NEnvs")


@pytest.fixture
def vector_spec(spec: LQGSpec, num_envs: int) -> LQGSpec:
    spec.num_envs = num_envs
    return spec


# Test RandomVectorLQG =========================================================
def test_vector_reset(vector_spec: LQGSpec):
    env = RandomVectorLQG(vector_spec)

    obs = env.vector_reset()
    assert hasattr(env, "num_envs")
    assert isinstance(obs, list)
    assert len(obs) == env.num_envs
    assert obs[0] in env.observation_space
    assert all(o in env.observation_space for o in obs)


def test_vector_step(vector_spec: LQGSpec):
    env = RandomVectorLQG(vector_spec)

    env.vector_reset()
    acts = [env.action_space.sample() for _ in range(env.num_envs)]
    new_obs, rews, dones, infos = env.vector_step(acts)
    assert all(isinstance(o, list) for o in (new_obs, rews, dones, infos))
    assert all(len(o) == env.num_envs for o in (new_obs, rews, dones, infos))

    assert all(o in env.observation_space for o in new_obs)
    assert all(isinstance(r, float) for r in rews)
    assert all(isinstance(d, bool) for d in dones)
    assert all(isinstance(i, dict) for i in infos)
