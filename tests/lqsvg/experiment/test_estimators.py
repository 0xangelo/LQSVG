# pylint:disable=unsubscriptable-object
from __future__ import annotations

from functools import partial

import pytest
import torch
from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.model import StochasticModel

from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.experiment.estimators import MonteCarloSVG
from lqsvg.policy.modules import (
    BatchNormModel,
    InitStateModel,
    LayerNormModel,
    LinearTransitionModel,
    QuadRewardModel,
    ResidualModel,
    StochasticModelWrapper,
    TVLinearPolicy,
)


@pytest.fixture
def dims(n_state: int, n_ctrl: int, horizon: int) -> tuple[int, int, int]:
    return n_state, n_ctrl, horizon


@pytest.fixture
def policy(n_state: int, n_ctrl: int, horizon: int) -> DeterministicPolicy:
    return TVLinearPolicy(n_state, n_ctrl, horizon)


@pytest.fixture(
    params=(ResidualModel, LayerNormModel, BatchNormModel, StochasticModelWrapper)
)
def wrapper(request, n_state: int) -> callable[[StochasticModel], StochasticModel]:
    cls = request.param

    if issubclass(cls, (LayerNormModel, BatchNormModel)):
        return partial(cls, n_state=n_state)

    return cls


@pytest.fixture
def trans(
    n_state: int,
    n_ctrl: int,
    horizon: int,
    stationary: bool,
    wrapper: callable[[StochasticModel], StochasticModel],
) -> StochasticModel:
    return wrapper(
        LinearTransitionModel(n_state, n_ctrl, horizon, stationary=stationary)
    )


@pytest.fixture
def reward(n_state: int, n_ctrl: int, horizon: int) -> QuadRewardModel:
    return QuadRewardModel(n_state, n_ctrl, horizon)


@pytest.fixture
def init(n_state: int) -> InitStateModel:
    return InitStateModel(n_state)


@pytest.fixture
def model(
    dims: tuple[int, int, int],
    trans: StochasticModel,
    reward: QuadRewardModel,
    init: InitStateModel,
) -> EnvModule:
    return EnvModule(dims, trans, reward, init)


def test_monte_carlo_svg(policy: DeterministicPolicy, model: EnvModule):
    estimator = MonteCarloSVG(policy, model)
    value = estimator.value(1)
    assert torch.is_tensor(value)
