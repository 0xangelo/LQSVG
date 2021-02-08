import pytest


@pytest.fixture(params=(42, 69, 37), ids=lambda x: f"Seed:{x}")
def seed(request):
    return request.param


@pytest.fixture(params=(1, 3, 4), ids=lambda x: f"StateSize:{x}")
def n_state(request):
    return request.param


@pytest.fixture(params=(2, 3), ids=lambda x: f"CtrlSize:{x}")
def n_ctrl(request):
    return request.param


@pytest.fixture
def n_tau(n_state, n_ctrl):
    return n_state + n_ctrl


@pytest.fixture(params=[1] + list(range(0, 101, 10))[1:], ids=lambda x: f"Horizon:{x}")
def horizon(request):
    return request.param
