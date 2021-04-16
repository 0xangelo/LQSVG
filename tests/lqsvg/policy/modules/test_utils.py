import numpy as np
import pytest
import torch

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.utils import stationary_dynamics_factors
from lqsvg.policy.modules.utils import place_dynamics_poles, stabilizing_policy


class TestStabilizingPolicy:
    # pylint:disable=invalid-name,too-many-arguments
    @pytest.fixture(params=(2 ** i for i in range(1, 5)))
    def n_state(self, request) -> int:
        return request.param

    @pytest.fixture(params=(2 ** i for i in range(1, 5)))
    def n_ctrl(self, request) -> int:
        return request.param

    @pytest.fixture
    def dynamics(
        self, n_state: int, n_ctrl: int, horizon: int, seed: int
    ) -> lqr.LinSDynamics:
        dyn = make_lindynamics(
            n_state,
            n_ctrl,
            horizon,
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            controllable=True,
            bias=False,
            rng=seed,
        )
        return make_linsdynamics(dyn, stationary=True, rng=seed)

    def test_place_dynamics_poles(self, dynamics: lqr.LinSDynamics, seed: int):
        A, B = (x.numpy() for x in stationary_dynamics_factors(dynamics))
        result = place_dynamics_poles(A, B, rng=seed)
        K = -result.gain_matrix
        eigval, _ = np.linalg.eig(A + B @ K)
        assert np.allclose(
            np.sort(np.abs(eigval)), np.sort(np.abs(result.computed_poles))
        )

    # noinspection PyArgumentList
    def test_stabilizing_policy(
        self,
        dynamics: lqr.LinSDynamics,
        n_state: int,
        n_ctrl: int,
        horizon: int,
        seed: int,
    ):
        K, k = stabilizing_policy(dynamics, rng=seed)
        assert torch.is_tensor(K)
        assert torch.isfinite(K).all()
        assert K.size("R") == n_ctrl
        assert K.size("C") == n_state

        assert torch.is_tensor(k)
        assert nt.allclose(k, torch.zeros_like(k))
        assert k.size("R") == n_ctrl

        assert K.size("H") == k.size("H") == horizon

        A, B = (x.numpy() for x in stationary_dynamics_factors(dynamics))
        K = K.select("H", 0).numpy()
        eigval, _ = np.linalg.eig(A + B @ K)
        assert np.all(np.abs(eigval) < 1.0)
