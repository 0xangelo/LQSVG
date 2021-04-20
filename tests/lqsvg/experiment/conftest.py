import pytest

from lqsvg.testing.fixture import standard_fixture


@pytest.fixture
def n_state() -> int:
    return 2


@pytest.fixture
def n_ctrl() -> int:
    return 2


@pytest.fixture
def horizon() -> int:
    return 20


@pytest.fixture
def seed() -> int:
    return 123


stationary = standard_fixture((True, False), "Stationary")
