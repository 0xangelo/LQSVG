# pylint:disable=unsubscriptable-object
from __future__ import annotations

from functools import partial

import pytest
import torch
from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.model import StochasticModel

from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.experiment.models import MonteCarloSVG
from lqsvg.policy.modules import BatchNormModel
from lqsvg.policy.modules import InitStateModel
from lqsvg.policy.modules import LayerNormModel
from lqsvg.policy.modules import LinearTransModel
from lqsvg.policy.modules import QuadRewardModel
from lqsvg.policy.modules import ResidualModel
from lqsvg.policy.modules import StochasticModelWrapper
from lqsvg.policy.modules import TVLinearPolicy
from lqsvg.policy.modules import TVLinearTransModel


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


@pytest.fixture(params=(TVLinearTransModel, LinearTransModel))
def trans(
    request,
    n_state: int,
    n_ctrl: int,
    horizon: int,
    wrapper: callable[[StochasticModel], StochasticModel],
) -> StochasticModel:
    cls = request.param
    return wrapper(cls(n_state, n_ctrl, horizon))


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
