# pylint:disable=unsubscriptable-object
from __future__ import annotations

import pytest
import torch
from raylab.policy.modules.actor import DeterministicPolicy

from lqsvg.envs.lqr.modules import LinearDynamics
from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.experiment.models import MonteCarloSVG
from lqsvg.policy.modules import InitStateModel
from lqsvg.policy.modules import LinearTransModel
from lqsvg.policy.modules import QuadRewardModel
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


@pytest.fixture(params=(TVLinearTransModel, LinearTransModel))
def trans(request, n_state: int, n_ctrl: int, horizon: int) -> LinearDynamics:
    cls = request.param
    return cls(n_state, n_ctrl, horizon)


@pytest.fixture
def reward(n_state: int, n_ctrl: int, horizon: int) -> QuadRewardModel:
    return QuadRewardModel(n_state, n_ctrl, horizon)


@pytest.fixture
def init(n_state: int) -> InitStateModel:
    return InitStateModel(n_state)


@pytest.fixture
def model(
    dims: tuple[int, int, int],
    trans: LinearDynamics,
    reward: QuadRewardModel,
    init: InitStateModel,
) -> EnvModule:
    return EnvModule(dims, trans, reward, init)


def test_monte_carlo_svg(policy: DeterministicPolicy, model: EnvModule):
    estimator = MonteCarloSVG(policy, model)
    value = estimator.value(1)
    assert torch.is_tensor(value)
