from __future__ import annotations

import pytest
import torch
from raylab.policy.modules.model import StochasticModel
from raylab.utils.types import TensorDict
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr import LinSDynamics
from lqsvg.envs.lqr.generators import make_lindynamics, make_linsdynamics
from lqsvg.envs.lqr.modules.dynamics.linear import LinearDynamics, LinearDynamicsModule
from lqsvg.envs.lqr.utils import pack_obs, unpack_obs


@pytest.fixture
def obs(
    n_state: int,
    horizon: int,
    batch_shape: tuple[int, ...],
    batch_names: tuple[str, ...],
) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,))).refine_names(
        *batch_names, ...
    )
    dummy, _ = nt.split(state, [1, n_state - 1], dim="R")
    time = torch.randint_like(nt.unnamed(dummy), low=0, high=horizon)
    time = time.refine_names(*dummy.names).int()
    return pack_obs(state, time).requires_grad_()


@pytest.fixture
def last_obs(
    n_state: int,
    horizon: int,
    batch_shape: tuple[int, ...],
    batch_names: tuple[str, ...],
) -> Tensor:
    state = nt.vector(torch.randn(batch_shape + (n_state,))).refine_names(
        *batch_names, ...
    )
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
def act(
    n_ctrl: int, batch_shape: tuple[int, ...], batch_names: tuple[str, ...]
) -> Tensor:
    return (
        nt.vector(torch.randn(batch_shape + (n_ctrl,)))
        .refine_names(*batch_names, ...)
        .requires_grad_()
    )


# noinspection PyMethodMayBeStatic
class DynamicsModuleTests:
    def test_rsample(self, module: StochasticModel, obs: Tensor, act: Tensor):
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

    def test_absorving(self, module: StochasticModel, last_obs: Tensor, act: Tensor):
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
        expected_grad = torch.cat(
            [torch.ones_like(state), torch.zeros_like(time)], dim="R"
        )
        assert nt.allclose(last_obs.grad, expected_grad)
        assert nt.allclose(act.grad, torch.zeros(()))

        last_obs.grad, act.grad = None, None
        assert logp.shape == tuple(
            s for s, n in zip(last_obs.shape, last_obs.names) if n != "R"
        )
        assert logp.names == tuple(n for n in last_obs.names if n != "R")
        assert nt.allclose(logp, torch.zeros(()))
        logp.sum().backward()
        assert nt.allclose(last_obs.grad, torch.zeros(()))
        assert nt.allclose(act.grad, torch.zeros(()))

    def test_log_prob(
        self, module: StochasticModel, obs: Tensor, act: Tensor, new_obs: Tensor
    ):
        params = module(obs, act)
        log_prob = module.log_prob(new_obs, params)
        _, time = unpack_obs(obs)
        _, time_ = unpack_obs(new_obs)
        time, time_ = nt.vector_to_scalar(time, time_)

        assert torch.is_tensor(log_prob)
        assert torch.isfinite(log_prob).all()
        assert log_prob.shape == time.shape == time_.shape
        assert log_prob.names == time.names == time_.names

        assert log_prob.grad_fn is not None
        log_prob.sum().backward()
        assert obs.grad is not None
        assert act.grad is not None
        assert not nt.allclose(obs.grad, torch.zeros(()))
        assert not nt.allclose(act.grad, torch.zeros(()))
        grads = list(p.grad for p in module.parameters())
        assert all(list(g is not None for g in grads))
        assert all(list(not torch.allclose(g, torch.zeros(())) for g in grads))


# noinspection PyMethodMayBeStatic
class LinearParamsTestMixin:
    def test_forward(self, module: StochasticModel, obs: Tensor, act: Tensor):
        params = module(obs, act)
        loc, scale_tril, time = self.check_keys(params)
        self.check_names(loc, scale_tril, time, obs)
        self.check_shapes(loc, scale_tril, time, obs)

    def check_keys(self, params: TensorDict) -> (Tensor, Tensor, Tensor):
        keys = "loc scale_tril time".split()
        assert all(list(k in params for k in keys))
        loc, scale_tril, time = (params[k] for k in keys)
        return loc, scale_tril, time

    def check_names(self, loc: Tensor, scale_tril: Tensor, time: Tensor, obs: Tensor):
        assert loc.names == obs.names
        assert scale_tril.names == obs.names + ("C",)
        assert time.names == obs.names

    def check_shapes(self, loc: Tensor, scale_tril: Tensor, time: Tensor, obs: Tensor):
        state, time_ = unpack_obs(obs)
        assert loc.shape == state.shape
        assert scale_tril.shape == state.shape + state.shape[-1:]
        assert time.shape == time_.shape


# noinspection PyMethodMayBeStatic
class TestLinearDynamicsModule(DynamicsModuleTests, LinearParamsTestMixin):
    @pytest.fixture
    def dynamics(
        self, n_state: int, n_ctrl: int, horizon: int, seed: int, stationary: bool
    ) -> LinSDynamics:
        # pylint:disable=too-many-arguments
        linear = make_lindynamics(
            n_state, n_ctrl, horizon, stationary=stationary, rng=seed
        )
        return make_linsdynamics(linear, stationary=stationary, rng=seed)

    def test_from_existing(self, dynamics: LinSDynamics, stationary: bool):
        before = tuple(x.clone() for x in dynamics)

        module = LinearDynamicsModule.from_existing(dynamics, stationary=stationary)
        for par in module.parameters():
            par.data.sub_(1.0)

        for bef, aft in zip(before, dynamics):
            assert nt.allclose(bef, aft)

    @pytest.fixture
    def module(
        self, n_state: int, n_ctrl: int, horizon: int, stationary: bool
    ) -> LinearDynamicsModule:
        return LinearDynamicsModule(n_state, n_ctrl, horizon, stationary)

    def test_standard_form(
        self, module: LinearDynamics, stationary: bool, horizon: int
    ):
        F, f, Sigma = module.standard_form()
        dummy = F.sum() + f.sum() + Sigma.sum()
        if stationary:
            # If stationary, parameters are repeated for each step within the
            # horizon and thus gradients are multiplied by horizon
            dummy = dummy / horizon
        dummy.backward()

        assert torch.allclose(module.F.grad, torch.ones([]))
        assert torch.allclose(module.f.grad, torch.ones([]))
        assert not torch.allclose(
            module.params.scale_tril.pre_diag.grad, torch.zeros([])
        )
        assert not torch.allclose(module.params.scale_tril.ltril.grad, torch.zeros([]))
