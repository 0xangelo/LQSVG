from __future__ import annotations

from functools import partial
from typing import Callable

import pytest
import torch
from nnrl.nn.critic import QValue
from nnrl.nn.model import StochasticModel
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.envs.lqr.utils import pack_obs
from lqsvg.estimators import DPG, MAAC, MonteCarloSVG
from lqsvg.torch.nn.dynamics.segment import LinearTransitionModel
from lqsvg.torch.nn.initstate import InitStateModel
from lqsvg.torch.nn.policy import TVLinearPolicy
from lqsvg.torch.nn.reward import QuadRewardModel
from lqsvg.torch.nn.value import QuadQValue
from lqsvg.torch.nn.wrappers import (
    BatchNormModel,
    LayerNormModel,
    ResidualModel,
    StochasticModelWrapper,
)


@pytest.fixture
def dims(n_state: int, n_ctrl: int, horizon: int) -> tuple[int, int, int]:
    return n_state, n_ctrl, horizon


@pytest.fixture
def policy(n_state: int, n_ctrl: int, horizon: int) -> TVLinearPolicy:
    return TVLinearPolicy(n_state, n_ctrl, horizon)


@pytest.fixture(
    params=(ResidualModel, LayerNormModel, BatchNormModel, StochasticModelWrapper)
)
def wrapper(request, n_state: int) -> Callable[[StochasticModel], StochasticModel]:
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
    wrapper: Callable[[StochasticModel], StochasticModel],
) -> StochasticModel:
    return wrapper(
        LinearTransitionModel(n_state, n_ctrl, horizon, stationary=stationary)
    )


@pytest.fixture
def reward(n_state: int, n_ctrl: int, horizon: int) -> QuadRewardModel:
    return QuadRewardModel(n_state, n_ctrl, horizon)


class TestMonteCarloSVG:
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
    def module(self, policy: TVLinearPolicy, model: EnvModule) -> MonteCarloSVG:
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


class TestDPG:
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
    ) -> DPG:
        return DPG(policy, trans, reward, qvalue)

    @pytest.fixture(params=(1, 4), ids=lambda x: f"NBatch{x}")
    def n_batch(self, request) -> int:
        return request.param

    @pytest.fixture
    def state(self, n_state: int, n_batch: int) -> Tensor:
        return nt.vector(torch.randn(n_batch, n_state)).refine_names("B", ...)

    @pytest.fixture
    def obs(self, state: Tensor, horizon: int, n_batch: int) -> Tensor:
        time = torch.randint(low=0, high=horizon, size=(n_batch, 1))
        return pack_obs(state, nt.vector(time).refine_names("B", ...))

    @pytest.fixture(params=(0, 1, 4), ids=lambda x: f"NStep:{x}")
    def n_step(self, request) -> int:
        return request.param

    @staticmethod
    def check_val_svg(val: Tensor, svg: lqr.Linear):
        assert torch.is_tensor(val)
        assert torch.isfinite(val).all()
        assert val.shape == ()

        K, k = svg
        assert torch.is_tensor(K) and torch.is_tensor(k)
        assert torch.isfinite(K).all() and torch.isfinite(k).all()
        assert not torch.allclose(K, torch.zeros([]))
        assert not torch.allclose(k, torch.zeros([]))

    def test_call(self, module: DPG, obs: Tensor, n_step: int):
        val, svg = module(obs, n_step)
        self.check_val_svg(val, svg)

    @pytest.fixture
    def pre_terminal(self, state: Tensor, horizon: int, n_batch: int) -> Tensor:
        time = torch.full((n_batch, 1), fill_value=horizon - 1, dtype=torch.int)
        return pack_obs(state, nt.vector(time).refine_names("B", ...))

    def test_absorving(self, module: DPG, pre_terminal: Tensor):
        val, svg = module(pre_terminal, n_steps=4)  # Exceed the horizon
        self.check_val_svg(val, svg)


class TestMAAC(TestDPG):
    @pytest.fixture
    def module(
        self,
        policy: TVLinearPolicy,
        trans: StochasticModel,
        reward: QuadRewardModel,
        qvalue: QValue,
    ) -> MAAC:
        return MAAC(policy, trans, reward, qvalue)
