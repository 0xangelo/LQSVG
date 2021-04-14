from __future__ import annotations

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import LinSDynamics
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.modules.dynamics.linear import (
    CovarianceCholesky,
    LinearDynamics,
    LinearDynamicsModule,
)
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.utils import default_generator_seed, make_spd_matrix

batch_shape = standard_fixture([(), (1,), (4,), (2, 2)], "BatchShape")


@pytest.fixture(autouse=True)
def torch_random(seed: int):
    with default_generator_seed(seed):
        yield


@pytest.fixture
def obs(n_state: int, horizon: int, batch_shape: tuple[int, ...]) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,)))
    dummy, _ = nt.split(state, [1, n_state - 1], dim="R")
    time = torch.randint_like(nt.unnamed(dummy), low=0, high=horizon)
    time = time.refine_names(*dummy.names).int()
    return pack_obs(state, time).requires_grad_()


@pytest.fixture
def last_obs(n_state: int, horizon: int, batch_shape: tuple[int, ...]) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,)))
    dummy, _ = nt.split(state, [1, n_state - 1], dim="R")
    time = torch.full_like(dummy, fill_value=horizon).int()
    return pack_obs(state, time).requires_grad_()


@pytest.fixture
def new_obs(obs: Tensor) -> Tensor:
    state, time = unpack_obs(obs)
    state_ = torch.randn_like(state)
    time_ = time + 1
    return pack_obs(state_, time_).requires_grad_()


@pytest.fixture
def mix_obs(obs: Tensor, last_obs: Tensor) -> Tensor:
    mask = torch.bernoulli(torch.full_like(obs, fill_value=0.5)).bool()
    return torch.where(mask, obs, last_obs)


@pytest.fixture
def act(n_ctrl: int, batch_shape: tuple[int, ...]) -> Tensor:
    return nt.vector(torch.randn(batch_shape + (n_ctrl,))).requires_grad_()


# noinspection PyMethodMayBeStatic
class LinearDynamicsTests:
    def test_rsample(self, module: LinearDynamics, obs: Tensor, act: Tensor):
        params = module(obs, act)
        sample, logp = module.rsample(params)

        assert sample.shape == obs.shape
        assert sample.names == obs.names
        _, time = unpack_obs(obs)
        _, time_ = unpack_obs(sample)
        assert time.eq(time_ - 1).all()

        assert sample.grad_fn is not None
        sample.sum().backward(retain_graph=True)
        assert obs.grad is not None
        assert act.grad is not None

        assert logp.shape == tuple(s for s, n in zip(obs.shape, obs.names) if n != "R")
        assert logp.names == tuple(n for n in obs.names if n != "R")

        obs.grad, act.grad = None, None
        assert logp.grad_fn is not None
        logp.sum().backward()
        assert obs.grad is not None
        assert act.grad is not None

    def test_absorving(self, module: LinearDynamics, last_obs: Tensor, act: Tensor):
        params = module(last_obs, act)
        sample, logp = module.rsample(params)

        assert sample.shape == last_obs.shape
        assert sample.names == last_obs.names
        state, time = unpack_obs(last_obs)
        state_, time_ = unpack_obs(sample)
        assert nt.allclose(state, state_)
        assert time.eq(time_).all()

        assert sample.grad_fn is not None
        sample.sum().backward(retain_graph=True)
        assert last_obs.grad is not None
        last_state, last_time = unpack_obs(last_obs)
        expected_grad = torch.cat(
            [torch.ones_like(last_state), torch.zeros_like(last_time)], dim="R"
        )
        assert nt.allclose(last_obs.grad, expected_grad)
        assert nt.allclose(act.grad, torch.zeros_like(act))

        last_obs.grad, act.grad = None, None
        assert logp.shape == tuple(
            s for s, n in zip(last_obs.shape, last_obs.names) if n != "R"
        )
        assert logp.names == tuple(n for n in last_obs.names if n != "R")
        assert nt.allclose(logp, torch.zeros_like(logp))
        logp.sum().backward()
        assert nt.allclose(last_obs.grad, torch.zeros_like(last_obs))
        assert nt.allclose(act.grad, torch.zeros_like(act))


stationary = standard_fixture((True, False), "Stationary")


class TestLinearDynamicsModule(LinearDynamicsTests):
    @pytest.fixture
    def dynamics(
        self, n_state: int, n_ctrl: int, horizon: int, seed: int, stationary: bool
    ) -> LinSDynamics:
        # pylint:disable=too-many-arguments
        linear = make_lindynamics(
            n_state, n_ctrl, horizon, stationary=stationary, rng=seed
        )
        return make_linsdynamics(linear, stationary=stationary, rng=seed)

    @pytest.fixture
    def module(self, dynamics: LinSDynamics, stationary: bool) -> LinearDynamicsModule:
        return LinearDynamicsModule(dynamics, stationary=stationary)


@pytest.fixture
def sigma(n_tau: int, horizon: int):
    return nt.horizon(
        nt.matrix(
            make_spd_matrix(n_dim=n_tau, sample_shape=(horizon,), dtype=torch.float32)
        )
    )


def test_cov_cholesky_factor(sigma: Tensor, horizon: int):
    module = CovarianceCholesky(sigma, horizon=horizon, stationary=False)
    untimed = module()

    scale_tril = nt.cholesky(sigma)
    assert nt.allclose(scale_tril, untimed)
    assert scale_tril.names == untimed.names
