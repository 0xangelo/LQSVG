# pylint:disable=unsubscriptable-object
from __future__ import annotations

from functools import partial

import pytest
import torch
from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.critic import QValue
from raylab.policy.modules.model import StochasticModel
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.envs.lqr.utils import pack_obs
from lqsvg.experiment.estimators import BootstrappedSVG, MonteCarloSVG
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
from lqsvg.policy.modules.value import QuadQValue


@pytest.fixture
def dims(n_state: int, n_ctrl: int, horizon: int) -> tuple[int, int, int]:
    return n_state, n_ctrl, horizon


@pytest.fixture
def policy(n_state: int, n_ctrl: int, horizon: int) -> TVLinearPolicy:
    return TVLinearPolicy(n_state, n_ctrl, horizon)


# noinspection PyUnresolvedReferences
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


class TestMonteCarloSVG:
    # noinspection PyUnresolvedReferences
    @pytest.fixture(
        params=(ResidualModel, LayerNormModel, BatchNormModel, StochasticModelWrapper)
    )
    def wrapper(
        self, request, n_state: int
    ) -> callable[[StochasticModel], StochasticModel]:
        cls = request.param

        if issubclass(cls, (LayerNormModel, BatchNormModel)):
            return partial(cls, n_state=n_state)

        return cls

    @pytest.fixture
    def init(self, n_state: int) -> InitStateModel:
        return InitStateModel(n_state)

    @pytest.fixture
    def model(
        self,
        dims: tuple[int, int, int],
        trans: StochasticModel,
        reward: QuadRewardModel,
        init: InitStateModel,
    ) -> EnvModule:
        return EnvModule(dims, trans, reward, init)

    @pytest.fixture
    def module(self, policy: DeterministicPolicy, model: EnvModule) -> MonteCarloSVG:
        return MonteCarloSVG(policy, model)

    @pytest.fixture(params=(1, 2, 4), ids=lambda x: f"Samples:{x}")
    def samples(self, request) -> int:
        return request.param

    def test_value(self, module: MonteCarloSVG, samples: int):
        val = module.value(samples)
        assert torch.is_tensor(val)
        assert torch.isfinite(val).all()
        assert val.shape == ()

    def test_call(self, module: MonteCarloSVG, samples: int):
        val, svg = module(samples=samples)

        assert torch.is_tensor(val)
        assert torch.isfinite(val).all()
        assert val.shape == ()

        K, k = svg
        assert torch.is_tensor(K) and torch.is_tensor(k)
        assert torch.isfinite(K).all() and torch.isfinite(k).all()


class TestBootstrappedSVG:
    # noinspection PyUnresolvedReferences
    @pytest.fixture(params=(ResidualModel, LayerNormModel, StochasticModelWrapper))
    def wrapper(
        self, request, n_state: int
    ) -> callable[[StochasticModel], StochasticModel]:
        cls = request.param

        if issubclass(cls, (LayerNormModel, BatchNormModel)):
            return partial(cls, n_state=n_state)

        return cls

    @pytest.fixture
    def qvalue(self, n_state: int, n_ctrl: int, horizon: int) -> QValue:
        return QuadQValue(n_state + n_ctrl, horizon)

    @pytest.fixture
    def module(
        self,
        policy: TVLinearPolicy,
        trans: StochasticModel,
        reward: QuadRewardModel,
        qvalue: QValue,
    ) -> BootstrappedSVG:
        return BootstrappedSVG(policy, trans, reward, qvalue)

    @pytest.fixture(params=(1, 4), ids=lambda x: f"NBatch{x}")
    def n_batch(self, request) -> int:
        return request.param

    @pytest.fixture
    def obs(self, n_state: int, horizon: int, n_batch: int) -> Tensor:
        state = torch.randn(n_batch, n_state)
        time = torch.randint(low=0, high=horizon, size=(n_batch, 1))
        # noinspection PyTypeChecker
        return pack_obs(nt.horizon(nt.vector(state)), nt.horizon(nt.vector(time)))

    @pytest.fixture(params=(0, 1, 4), ids=lambda x: f"NStep:{x}")
    def n_step(self, request) -> int:
        return request.param

    def test_call(self, module: BootstrappedSVG, obs: Tensor, n_step: int):
        val, svg = module(obs, n_step)

        assert torch.is_tensor(val)
        assert torch.isfinite(val).all()
        assert val.shape == ()

        K, k = svg
        assert torch.is_tensor(K) and torch.is_tensor(k)
        assert torch.isfinite(K).all() and torch.isfinite(k).all()
        assert not torch.allclose(K, torch.zeros([]))
        assert not torch.allclose(k, torch.zeros([]))
