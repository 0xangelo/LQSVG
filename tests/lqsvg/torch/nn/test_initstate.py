import pytest

from lqsvg.envs.lqr.modules import InitStateModule
from lqsvg.torch.nn.initstate import InitStateModel


@pytest.fixture
def module(n_state: int, seed: int) -> InitStateModel:
    return InitStateModel(n_state, seed=seed)


def test_init(module: InitStateModel, n_state: int):
    assert module.n_state == n_state
    assert isinstance(module, InitStateModule)
