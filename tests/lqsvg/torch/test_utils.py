import pytest
import torch

from lqsvg.torch.utils import default_generator_seed
from lqsvg.torch.utils import default_rng


@pytest.fixture(params=(None, 123, torch.Generator()))
def seed(request):
    return request.param


def test_default_rng(seed):
    rng = default_rng(seed)
    assert isinstance(rng, torch.Generator)


def test_default_generator_seed():
    random = torch.randn(10)
    with default_generator_seed(42):
        first = torch.randn(10)
    with default_generator_seed(42):
        second = torch.randn(10)

    assert not any(torch.allclose(t, random) for t in (first, second))
    assert torch.allclose(first, second)
