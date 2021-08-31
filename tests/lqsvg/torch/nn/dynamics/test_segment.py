from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch
from nnrl.nn.model import StochasticModel
from torch import IntTensor, Tensor

from lqsvg.envs.lqr import pack_obs, unpack_obs
from lqsvg.testing import check
from lqsvg.torch.nn.dynamics.linear import LinearDynamicsModule
from lqsvg.torch.nn.dynamics.segment import (
    GRUGaussDynamics,
    LinearDiagDynamicsModel,
    LinearTransitionModel,
    MLPDynamicsModel,
)
from lqsvg.torch.sequence import log_prob_fn

from .test_linear import DynamicsModuleTests, LinearParamsTestMixin


@pytest.fixture(params=[(), (10,)], ids=lambda x: f"HiddenUnits:{x}")
def hunits(request) -> Tuple[int, ...]:
    return request.param


@pytest.fixture(params=(None, "ReLU", "ELU", "Tanh"), ids=lambda x: f"Activation:{x}")
def activation(request) -> Optional[str]:
    return request.param


class TestLinearTransitionModel(DynamicsModuleTests, LinearParamsTestMixin):
    @pytest.fixture
    def module(
        self, n_state: int, n_ctrl: int, horizon: int, stationary: bool
    ) -> LinearTransitionModel:
        return LinearTransitionModel(n_state, n_ctrl, horizon, stationary)

    def test_init(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        module = LinearTransitionModel(n_state, n_ctrl, horizon, stationary)
        assert isinstance(module, LinearDynamicsModule)
        assert all(
            list(hasattr(module, attr) for attr in "n_state n_ctrl horizon".split())
        )


# noinspection PyMethodMayBeStatic
class SegmentModelTestMixin:
    @staticmethod
    def add_horizon(val: Tensor) -> Tensor:
        return val.expand((4,) + val.shape).refine_names("H", ...)

    @staticmethod
    def step_time(times: IntTensor, horizon: int) -> IntTensor:
        # noinspection PyArgumentList
        step = torch.arange(times.size("H")).refine_names("H")
        return (times + step.align_as(times)).clamp(max=horizon)

    @pytest.fixture
    def seg_obs(self, obs: Tensor, horizon: int) -> Tensor:
        states, times = unpack_obs(self.add_horizon(obs))
        obs = pack_obs(states, self.step_time(times, horizon))
        obs.retain_grad()
        return obs

    @pytest.fixture
    def seg_act(self, act: Tensor) -> Tensor:
        act = self.add_horizon(act)
        act.retain_grad()
        return act

    @pytest.fixture
    def seg_new_obs(self, seg_obs: Tensor, horizon: int) -> Tensor:
        states, times = unpack_obs(seg_obs)
        # Have to be careful not to exceed the horizon
        # Furthermore, in case some of the current states are already at the
        # final timestep, the new states are automatically equal, which
        # prevents problems with undefined log probs
        new_obs = pack_obs(states, (times + 1).clamp(max=horizon))
        new_obs.retain_grad()
        return new_obs

    def test_seg_log_prob(
        self,
        module: StochasticModel,
        seg_obs: Tensor,
        seg_act: Tensor,
        seg_new_obs: Tensor,
        batch_shape: tuple[int, ...],
    ):
        # pylint:disable=too-many-arguments
        seg_log_prob = log_prob_fn(module, module.dist)
        # Test shape
        log_prob = seg_log_prob(seg_obs, seg_act, seg_new_obs)
        assert torch.isfinite(log_prob).all()
        assert log_prob.shape == batch_shape

        # Test grad w.r.t. inputs
        log_prob.mean().backward()
        assert seg_obs.grad is not None
        assert seg_act.grad is not None

        # Test grad w.r.t. parameters
        check.assert_any_grads_nonzero(module)


class TestLinearDiagDynamicsModel(
    DynamicsModuleTests, SegmentModelTestMixin, LinearParamsTestMixin
):
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        return LinearDiagDynamicsModel(n_state, n_ctrl, horizon, stationary)


class TestMLPTransitionModel(SegmentModelTestMixin, DynamicsModuleTests):
    @pytest.fixture
    def module(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
        hunits: tuple[int, ...],
        activation: str,
    ) -> MLPDynamicsModel:
        # pylint:disable=too-many-arguments
        return MLPDynamicsModel(
            n_state,
            n_ctrl,
            horizon,
            hunits=hunits,
            activation=activation,
        )


class TestGRUGaussDynamics(SegmentModelTestMixin):
    @pytest.fixture(params=((1,), (2,)), ids=lambda x: f"Batch:{x}")
    def batch_shape(self, request) -> tuple[int, ...]:
        # GRU docs expect only one batch dimension
        return request.param

    @pytest.fixture
    def mlp_hunits(self, hunits: tuple[int, ...]) -> tuple[int, ...]:
        return hunits

    @pytest.fixture
    def gru_hidden(self) -> int:
        return 8

    @pytest.fixture(params=(1, 2), ids=lambda x: f"GRULayers:{x}")
    def gru_hunits(self, request, gru_hidden: int) -> tuple[int, ...]:
        return (gru_hidden,) * request.param

    @pytest.fixture
    def module(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
        mlp_hunits: tuple[int, ...],
        gru_hunits: tuple[int, ...],
    ) -> GRUGaussDynamics:
        # pylint:disable=too-many-arguments
        return GRUGaussDynamics(
            n_state, n_ctrl, horizon, mlp_hunits=mlp_hunits, gru_hunits=gru_hunits
        )

    @pytest.fixture
    def context(
        self, batch_shape: tuple[int, ...], gru_hunits: tuple[int, ...]
    ) -> Tensor:
        return torch.zeros((len(gru_hunits),) + batch_shape + (gru_hunits[0],))

    def test_forward_with_context(
        self, module: GRUGaussDynamics, obs: Tensor, act: Tensor, context: Tensor
    ):
        params = module(obs, act, context=context)
        assert "loc" in params
        assert "scale_tril" in params
        assert "time" in params
        assert "context" in params

        assert params["loc"].names == obs.names

    def test_forward_without_context(
        self, module: GRUGaussDynamics, obs: Tensor, act: Tensor
    ):
        params = module(obs, act)
        assert "loc" in params
        assert "scale_tril" in params
        assert "time" in params
        assert "context" in params

        assert params["loc"].names == obs.names
