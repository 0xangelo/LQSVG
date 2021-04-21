from lqsvg.envs.lqr.modules import LinearDynamicsModule
from lqsvg.policy.modules.transition import LinearTransitionModel
from lqsvg.testing.fixture import standard_fixture

stationary = standard_fixture((True, False), "Stationary")


class TestLinearTransitionModel:
    # pylint:disable=too-few-public-methods
    def test_init(self, n_state: int, n_ctrl: int, horizon: int, stationary: bool):
        module = LinearTransitionModel(n_state, n_ctrl, horizon, stationary)
        assert isinstance(module, LinearDynamicsModule)
        assert all(
            list(hasattr(module, attr) for attr in "n_state n_ctrl horizon".split())
        )
