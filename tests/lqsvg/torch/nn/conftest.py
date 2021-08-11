from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch
from numpy.random import Generator
from torch import Tensor

from lqsvg.envs.lqr.utils import pack_obs, unpack_obs
from lqsvg.testing.fixture import std_id
from lqsvg.torch import named as nt
from lqsvg.torch.utils import default_generator_seed


@pytest.fixture(params=(2, 3), ids=std_id("NState"))
def n_state(request) -> int:
    return request.param


@pytest.fixture(params=(2, 3), ids=std_id("NCtrl"))
def n_ctrl(request) -> int:
    return request.param


@pytest.fixture(params=(1, 3, 10), ids=std_id("Horizon"))
def horizon(request) -> int:
    return request.param


@pytest.fixture(params=(True, False), ids=std_id("Stationary"))
def stationary(request) -> bool:
    return request.param


@pytest.fixture(params=(1, 2), ids=std_id("Seed"))
def seed(request) -> int:
    return request.param


@pytest.fixture(params=[(), (1,), (4,)], ids=std_id("BatchShape"))
def batch_shape(request) -> Tuple[int, ...]:
    return request.param


@pytest.fixture
def rng(seed: int) -> Generator:
    return np.random.default_rng(seed)


@pytest.fixture(autouse=True)
def torch_generator_state(seed: int):
    with default_generator_seed(seed):
        yield


@pytest.fixture
def batch_names(batch_shape: tuple[int, ...]) -> tuple[str, ...]:
    return () if not batch_shape else ("B",) if len(batch_shape) == 1 else ("H", "B")


@pytest.fixture()
def obs(
    n_state: int,
    horizon: int,
    batch_shape: tuple[int, ...],
    batch_names: tuple[str, ...],
) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,)))
    time = nt.vector(
        torch.randint_like(nt.unnamed(state[..., :1]), low=0, high=horizon)
    )
    # noinspection PyTypeChecker
    return pack_obs(state, time).refine_names(*batch_names, ...).requires_grad_(True)


@pytest.fixture
def new_obs(obs: Tensor) -> Tensor:
    state, time = unpack_obs(obs)
    state_ = torch.randn_like(state)
    time_ = time + 1
    return pack_obs(state_, time_).requires_grad_()


@pytest.fixture()
def last_obs(
    n_state: int,
    horizon: int,
    batch_shape: tuple[int, ...],
    batch_names: tuple[str, ...],
) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,)))
    time = nt.vector(torch.full_like(state[..., :1], fill_value=horizon))
    # noinspection PyTypeChecker
    return pack_obs(state, time).refine_names(*batch_names, ...).requires_grad_(True)


@pytest.fixture
def mix_obs(obs: Tensor, last_obs: Tensor) -> Tensor:
    _, time = unpack_obs(obs)
    mix = nt.where(torch.rand_like(time.float()) < 0.5, obs, last_obs)
    mix.retain_grad()
    return mix


@pytest.fixture()
def act(
    n_ctrl: int, batch_shape: tuple[int, ...], batch_names: tuple[str, ...]
) -> Tensor:
    return (
        nt.vector(torch.randn(batch_shape + (n_ctrl,)))
        .refine_names(*batch_names, ...)
        .requires_grad_(True)
    )
