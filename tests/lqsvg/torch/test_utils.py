import pytest
import torch

from lqsvg.torch.utils import default_rng


@pytest.fixture(params=(None, 123, torch.Generator()))
def seed(request):
    return request.param


def test_default_rng(seed):
    rng = default_rng(seed)
    assert isinstance(rng, torch.Generator)
