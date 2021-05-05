import pytest

from lqsvg.envs.lqr.modules import LinearDynamicsModule
from lqsvg.policy.modules.transition import LinearTransitionModel
from tests.lqsvg.envs.lqr.modules.dynamics.test_linear import DynamicsModuleTests


class TestLinearTransitionModel(DynamicsModuleTests):
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
