from __future__ import annotations

import pytest

from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.random import default_generator_seed

batch_shape = standard_fixture([(), (1,), (4,), (2, 2)], "BatchShape")


@pytest.fixture
def batch_names(batch_shape: tuple[int, ...]) -> tuple[str, ...]:
    return () if not batch_shape else ("B",) if len(batch_shape) == 1 else ("H", "B")


@pytest.fixture(autouse=True)
def torch_random(seed: int):
    with default_generator_seed(seed):
        yield
