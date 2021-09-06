import pytest

from lqsvg.testing.fixture import std_id


@pytest.fixture(params=(1, 2, 3), ids=std_id("Seed"))
def seed(request) -> int:
    return request.param
