import pytest
from gym import spaces

from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.testing.fixture import std_id
from lqsvg.torch.utils import default_generator_seed


@pytest.fixture
def seed() -> int:
    return 123


@pytest.fixture(autouse=True)
def torch_generator_state(seed: int):
    with default_generator_seed(seed):
        yield


@pytest.fixture
def n_state() -> int:
    return 2


@pytest.fixture
def n_ctrl() -> int:
    return 2


@pytest.fixture
def horizon() -> int:
    return 20


@pytest.fixture(params=(True, False), ids=std_id("Stationary"))
def stationary(request) -> bool:
    return request.param


@pytest.fixture
def lqg_generator(
    n_state: int, n_ctrl: int, horizon: int, stationary: bool, seed: int
) -> LQGGenerator:
    return LQGGenerator(n_state, n_ctrl, horizon, stationary=stationary, rng=seed)


@pytest.fixture(scope="module", params=((1,), (4,)), ids=("Obs1Dim", "Obs4Dim"))
def obs_space(request):
    return spaces.Box(-10, 10, shape=request.param)


@pytest.fixture(scope="module", params=((1,), (4,)), ids=("Act1Dim", "Act4Dim"))
def action_space(request):
    return spaces.Box(-1, 1, shape=request.param)


@pytest.fixture(scope="module")
def envs():
    from lqsvg.envs.registry import ENVS  # pylint:disable=import-outside-toplevel

    return ENVS.copy()


@pytest.fixture(
    params="""
    RandomLQG
    MockEnv
    """.split(),
    scope="module",
)
def env_name(request):
    return request.param
