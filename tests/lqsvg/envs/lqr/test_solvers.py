# pylint:disable=invalid-name,too-many-arguments,too-many-locals
import pytest
import torch

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import LinDynamics
from lqsvg.envs.lqr import LinSDynamics
from lqsvg.envs.lqr import NamedLQGControl
from lqsvg.envs.lqr import NamedLQGPrediction
from lqsvg.envs.lqr import NamedLQRControl
from lqsvg.envs.lqr import NamedLQRPrediction
from lqsvg.envs.lqr import QuadCost
from lqsvg.envs.lqr.generators import make_linsdynamics
from lqsvg.envs.lqr.generators import make_quadcost


@pytest.fixture
def stationary_stochastic_dynamics(n_state, n_ctrl, horizon, seed):
    dynamics = make_linsdynamics(
        state_size=n_state,
        ctrl_size=n_ctrl,
        horizon=horizon,
        stationary=True,
        np_random=seed,
    )
    return dynamics


@pytest.fixture
def stationary_deterministic_dynamics(stationary_stochastic_dynamics):
    F, f, _ = stationary_stochastic_dynamics
    return LinDynamics(F=F, f=f)


@pytest.fixture
def stationary_cost(n_state, n_ctrl, horizon, seed):
    cost = make_quadcost(
        state_size=n_state,
        ctrl_size=n_ctrl,
        horizon=horizon,
        stationary=True,
        np_random=seed,
    )
    return cost


@pytest.fixture(params=(True, False), ids=lambda x: f"TorchScript:{x}")
def torch_script(request):
    def func(mod):
        if request.param:
            mod.solver = torch.jit.script(mod.solver)
        return mod

    return func


@pytest.fixture
def lqr_control(torch_script, n_state, n_ctrl, horizon):
    controler = NamedLQRControl(n_state, n_ctrl, horizon)
    return torch_script(controler)


def is_tensor(val):
    return torch.is_tensor(val)


def check_quadratic(quadratic, horizon, size):
    A, b, c = quadratic

    assert is_tensor(A)
    assert is_tensor(b)
    assert is_tensor(c)

    assert A.shape == (horizon, size, size)
    assert b.shape == (horizon, size)
    assert c.shape == (horizon,)

    assert ~A.isnan().any()
    assert ~b.isnan().any()
    assert ~c.isnan().any()


def check_linear(linear, horizon, n1, n2):
    K, k = linear

    assert is_tensor(K)
    assert is_tensor(k)

    assert K.shape == (horizon, n1, n2)
    assert k.shape == (horizon, n1)

    assert ~K.isnan().any()
    assert ~k.isnan().any()


def test_lqr_control(
    lqr_control: NamedLQRControl,
    stationary_deterministic_dynamics: LinDynamics,
    stationary_cost: QuadCost,
    horizon: int,
    n_state: int,
    n_ctrl: int,
):
    policy, q_val, v_val = lqr_control(
        stationary_deterministic_dynamics, stationary_cost
    )

    assert isinstance(policy, tuple)
    assert len(policy) == 2
    assert isinstance(q_val, tuple)
    assert len(q_val) == 3
    assert isinstance(v_val, tuple)
    assert len(v_val) == 3

    check_linear(policy, horizon, n_ctrl, n_state)
    check_quadratic(q_val, horizon, n_state + n_ctrl)
    check_quadratic(v_val, horizon + 1, n_state)


@pytest.fixture(params=(0.0, 0.1), ids=lambda x: f"StdDev:{x}")
def stddev(request):
    return request.param


@pytest.fixture
def rand_policy(
    lqr_control: NamedLQRControl,
    stationary_deterministic_dynamics: LinDynamics,
    stationary_cost: QuadCost,
    seed: int,
):
    policy, _, _ = lqr_control(stationary_deterministic_dynamics, stationary_cost)
    K, k = policy
    torch.manual_seed(seed)
    K = K + torch.rand_like(K) * 0.2 - 0.1
    k = k + torch.rand_like(k) * 0.2 - 0.1
    return K, k


@pytest.fixture
def lqr_prediction(torch_script, n_state, n_ctrl, horizon):
    predictor = NamedLQRPrediction(n_state, n_ctrl, horizon)
    return torch_script(predictor)


def test_lqr_prediction(
    lqr_prediction: NamedLQRPrediction,
    stationary_deterministic_dynamics: LinDynamics,
    stationary_cost: QuadCost,
    rand_policy,
    horizon: int,
    n_state: int,
    n_ctrl: int,
):
    q_val, v_val = lqr_prediction(
        rand_policy, stationary_deterministic_dynamics, stationary_cost
    )
    assert isinstance(q_val, tuple)
    assert len(q_val) == 3
    assert isinstance(v_val, tuple)
    assert len(v_val) == 3

    check_quadratic(q_val, horizon, n_state + n_ctrl)
    check_quadratic(v_val, horizon + 1, n_state)


@pytest.fixture
def lqg_prediction(n_state, n_ctrl, horizon, torch_script):
    return torch_script(NamedLQGPrediction(n_state, n_ctrl, horizon))


@pytest.fixture
def lqg_control(n_state, n_ctrl, horizon, torch_script):
    return torch_script(NamedLQGControl(n_state, n_ctrl, horizon))


def test_lqg_prediction(
    lqg_prediction: NamedLQGPrediction,
    stationary_stochastic_dynamics: LinSDynamics,
    stationary_cost: QuadCost,
    rand_policy: tuple,
    n_state: int,
    n_tau: int,
    horizon: int,
):
    Qval, Vval = lqg_prediction(
        rand_policy, stationary_stochastic_dynamics, stationary_cost
    )
    check_quadratic(Qval, horizon, n_tau)
    check_quadratic(Vval, horizon + 1, n_state)


def test_lqg_control(
    lqg_control: NamedLQGControl,
    stationary_stochastic_dynamics: LinSDynamics,
    stationary_cost: QuadCost,
    n_state: int,
    n_ctrl: int,
    n_tau: int,
    horizon: int,
):
    Pi, Qval, Vval = lqg_control(stationary_stochastic_dynamics, stationary_cost)
    check_quadratic(Vval, horizon + 1, n_state)
    check_quadratic(Qval, horizon, n_tau)
    check_linear(Pi, horizon, n_ctrl, n_state)


def test_stationary_pred_equality(
    lqr_control: NamedLQRControl,
    lqg_prediction: NamedLQGPrediction,
    stationary_deterministic_dynamics: LinDynamics,
    stationary_cost: QuadCost,
    horizon: int,
    n_state: int,
):
    lqr_pi, _, lqr_val = lqr_control(stationary_deterministic_dynamics, stationary_cost)
    lqr_V, lqr_v, lqr_vc = nt.unnamed(*lqr_val)

    _, lqg_val = lqg_prediction(
        # Insert batch dimension and use column vector
        lqr_pi,
        LinSDynamics(
            F=stationary_deterministic_dynamics.F,
            f=stationary_deterministic_dynamics.f,
            W=torch.zeros(horizon, n_state, n_state),
        ),
        stationary_cost,
    )
    lqg_V, lqg_v, lqg_vc = nt.unnamed(*lqg_val)

    assert torch.allclose(lqr_V, lqg_V)
    assert torch.allclose(lqr_v, lqg_v)
    assert torch.allclose(lqr_vc, lqg_vc)


def test_stationary_ctrl_equality(
    lqr_control: NamedLQRControl,
    lqg_control: NamedLQGControl,
    stationary_deterministic_dynamics: LinSDynamics,
    stationary_cost: QuadCost,
    horizon: int,
    n_state: int,
):
    lqr_pi, _, _ = lqr_control(stationary_deterministic_dynamics, stationary_cost)
    lqg_pi, _, _ = lqg_control(
        LinSDynamics(
            F=stationary_deterministic_dynamics.F,
            f=stationary_deterministic_dynamics.f,
            W=torch.zeros(horizon, n_state, n_state),
        ),
        stationary_cost,
    )

    lqr_K, lqr_k = nt.unnamed(*lqr_pi)
    lqg_K, lqg_k = nt.unnamed(*lqg_pi)

    assert lqr_K.shape == lqg_K.shape
    assert lqr_k.shape == lqg_k.shape
    assert torch.allclose(lqr_k, lqg_k)
    assert torch.allclose(lqr_K, lqg_K)
