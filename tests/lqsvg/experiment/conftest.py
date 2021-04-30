import pytest

from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.testing.fixture import standard_fixture
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


stationary = standard_fixture((True, False), "Stationary")


@pytest.fixture
def lqg_generator(
    n_state: int, n_ctrl: int, horizon: int, stationary: bool, seed: int
) -> LQGGenerator:
    return LQGGenerator(n_state, n_ctrl, horizon, stationary=stationary, rng=seed)
