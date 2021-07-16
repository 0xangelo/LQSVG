from __future__ import annotations

import pytest

from lqsvg.envs.lqr.modules import LinearDynamicsModule
from lqsvg.policy.modules.transition import LinearTransitionModel, MLPDynamicsModel
from lqsvg.testing.fixture import standard_fixture
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


hunits = standard_fixture([(), (10,), (32, 32)], "HiddenUnits")
activation = standard_fixture((None, "ReLU", "ELU", "Tanh"), "Activation")


@pytest.mark.skip(reason="Unimplemented")
class TestMLPTransitionModel(DynamicsModuleTests):
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
