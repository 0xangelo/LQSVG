from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lqsvg.envs.lqr.modules import LinearDynamicsModule
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.nn.transition import (
    GRUDynamicsModel,
    LinearDiagDynamicsModel,
    LinearTransitionModel,
    MLPDynamicsModel,
    SegmentStochasticModel,
)
from tests.lqsvg.envs.lqr.modules.dynamics.test_linear import (
    DynamicsModuleTests,
    LinearParamsTestMixin,
)


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
    # pylint:disable=too-few-public-methods
    def test_seg_log_prob(
        self,
        module: SegmentStochasticModel,
        obs: Tensor,
        act: Tensor,
        new_obs: Tensor,
        batch_shape: tuple[int, ...],
    ):
        # pylint:disable=too-many-arguments
        # Test shape
        log_prob = module.seg_log_prob(obs, act, new_obs)
        assert torch.isfinite(log_prob).all()
        assert log_prob.shape == batch_shape

        # Test grad w.r.t. inputs
        log_prob.mean().backward()
        assert obs.grad is not None
        assert act.grad is not None

        # Test grad w.r.t. parameters
        grads = [p.grad for p in module.parameters()]
        assert all(list(g is not None for g in grads))


class TestLinearDiagDynamicsModel(
    DynamicsModuleTests, SegmentModelTestMixin, LinearParamsTestMixin
):
    @pytest.fixture
    def module(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        return LinearDiagDynamicsModel(n_state, n_ctrl, horizon, stationary)


hunits = standard_fixture([(), (10,)], "HiddenUnits")
activation = standard_fixture((None, "ReLU", "ELU", "Tanh"), "Activation")


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


@pytest.mark.xfail(reason="Unimplemented")
class TestGRUDynamicsModel(SegmentModelTestMixin, DynamicsModuleTests):
    @pytest.fixture
    def mlp_hunits(self, hunits: int) -> int:
        return hunits

    @pytest.fixture
    def gru_hunits(self, hunits: int) -> int:
        return hunits

    @pytest.fixture
    def mlp_activ(self, activation: str) -> str:
        return activation

    @pytest.fixture
    def module(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
        mlp_hunits: tuple[int, ...],
        mlp_activ: str,
        gru_hunits: tuple[int, ...],
    ):
        # pylint:disable=too-many-arguments
        return GRUDynamicsModel(
            n_state,
            n_ctrl,
            horizon,
            mlp_hunits=mlp_hunits,
            mlp_activ=mlp_activ,
            gru_hunits=gru_hunits,
        )
