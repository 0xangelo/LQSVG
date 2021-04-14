from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lqsvg.envs.lqr.utils import pack_obs
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch import named as nt
from lqsvg.torch.utils import default_generator_seed

n_state = standard_fixture((2, 3), "NState")
n_ctrl = standard_fixture((2, 3), "NCtrl")
horizon = standard_fixture((1, 3, 10), "Horizon")
seed = standard_fixture((1, 2, 3), "Seed")
batch_shape = standard_fixture([(), (1,), (4,)], "BatchShape")


@pytest.fixture(autouse=True)
def torch_generator_state(seed: int):
    with default_generator_seed(seed):
        yield


@pytest.fixture()
def obs(n_state: int, horizon: int, batch_shape: tuple[int, ...]) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,)))
    time = nt.vector(
        torch.randint_like(nt.unnamed(state[..., :1]), low=0, high=horizon)
    )
    # noinspection PyTypeChecker
    return pack_obs(state, time)


@pytest.fixture()
def last_obs(n_state: int, horizon: int, batch_shape: tuple[int, ...]) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,)))
    time = nt.vector(torch.full_like(state[..., :1], fill_value=horizon))
    # noinspection PyTypeChecker
    return pack_obs(state, time)


@pytest.fixture()
def act(n_ctrl: int, batch_shape: tuple[int, ...]) -> Tensor:
    return nt.vector(torch.randn(batch_shape + (n_ctrl,)))
