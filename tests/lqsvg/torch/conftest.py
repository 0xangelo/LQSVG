import pytest

from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.random import default_generator_seed

seed = standard_fixture((1, 2, 3), "Seed")


@pytest.fixture(autouse=True)
def torch_generator_state(seed: int):
    with default_generator_seed(seed):
        yield
