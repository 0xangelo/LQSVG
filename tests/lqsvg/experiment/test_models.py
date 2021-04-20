# pylint:disable=unsubscriptable-object
from __future__ import annotations

import functools
from typing import Optional

import pytest
import torch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.gym import RandomVectorLQG, TorchLQGMixin
from lqsvg.experiment.estimators import MonteCarloSVG
from lqsvg.experiment.models import LightningModel, RecurrentModel
from lqsvg.policy.time_varying_linear import LQGPolicy


# noinspection PyArgumentList
def assert_row_size(size: int, *tensors: Tensor):
    assert all(t.size("R") == size for t in tensors)


# noinspection PyArgumentList
def assert_horizon_len(size: int, *tensors: Tensor):
    assert all(t.size("H") == size for t in tensors)


# noinspection PyArgumentList
def assert_batch_size(size: int, *tensors: Tensor):
    assert all(t.size("B") == size for t in tensors)


def check_traj(
    traj: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    n_state: int,
    n_ctrl: int,
    horizon: int,
    n_batch: Optional[int] = None,
):
    obs, act, rew, new_obs, logp = traj

    # Include time
    assert_row_size(n_state + 1, obs, new_obs)
    assert_row_size(n_ctrl, act)
    assert_horizon_len(horizon, obs, act, rew, new_obs)
    # likelihood reduces over whole trajectory
    assert not any(n in logp.names for n in "H R C".split())
    if n_batch:
        assert_batch_size(n_batch, obs, act, rew, new_obs, logp)


@pytest.fixture
@nt.suppress_named_tensor_warning()
def env(n_state: int, n_ctrl: int, horizon: int) -> TorchLQGMixin:
    return RandomVectorLQG(
        dict(n_state=n_state, n_ctrl=n_ctrl, horizon=horizon, num_envs=1)
    )


@pytest.fixture
def rllib_policy(env: TorchLQGMixin) -> LQGPolicy:
    pol = LQGPolicy(env.observation_space, env.action_space, {})
    pol.setup(env)
    return pol


def test_lightning_model(
    rllib_policy: LQGPolicy, env: TorchLQGMixin, n_state: int, n_ctrl: int, horizon: int
):
    model = LightningModel(rllib_policy, env)

    monte_carlo = model.monte_carlo_svg
    true_mc = MonteCarloSVG(model.actor, model.mdp)

    check = functools.partial(
        check_traj, n_state=n_state, n_ctrl=n_ctrl, horizon=horizon
    )

    check(monte_carlo.rsample_trajectory([]))
    check(monte_carlo.rsample_trajectory([10]))
    # noinspection PyTypeChecker
    check(true_mc.rsample_trajectory([]))

    assert torch.isfinite(monte_carlo.value(samples=256))
    assert torch.isfinite(model.gold_standard[0])


def test_recurrent_model(rllib_policy: LQGPolicy, env: TorchLQGMixin):
    model = RecurrentModel(rllib_policy, env)

    batch_shape = (10,)
    with torch.no_grad():
        # noinspection PyTypeChecker
        trajectory = MonteCarloSVG(model.actor, model.mdp).rsample_trajectory(
            batch_shape
        )
    obs, act, _, new_obs, _ = trajectory
    log_prob = model(obs, act, new_obs)
    assert not any(n in log_prob.names for n in "H R C".split())
    assert torch.isfinite(log_prob).all()
    assert log_prob.shape == batch_shape
