# pylint:disable=missing-docstring,invalid-name
from functools import cached_property
from typing import Any, Optional
from typing import Tuple

import torch
import torch.nn as nn
from gym.spaces import Box
from ray.rllib import RolloutWorker
from raylab.options import configure
from raylab.options import option
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapDeterministicPolicy
from raylab.policy.modules.actor import DeterministicPolicy
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import make_quadcost
from lqsvg.envs.lqr.gym import RandomVectorLQG
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import TVLinearDynamics, InitStateDynamics
from lqsvg.envs.lqr.modules import QuadraticCost
from lqsvg.envs.lqr.utils import unpack_obs

# from lqsvg.policy import RandomPolicy
from lqsvg.np_util import make_spd_matrix
from lqsvg.torch.utils import as_float_tensor


class TVLinearFeedback(nn.Module):
    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__()
        K = torch.randn(horizon, n_ctrl, n_state)
        k = torch.randn(horizon, n_ctrl)
        self.K, self.k = (nn.Parameter(x) for x in (K, k))

    def _gains_at(self, index: Tensor) -> Tuple[Tensor, Tensor]:
        K = nt.horizon(nt.matrix(self.K))
        k = nt.horizon(nt.vector(self.k))
        K, k = (nt.index_by(x, dim="H", index=index) for x in (K, k))
        return K, k

    def forward(self, obs: Tensor) -> Tensor:
        obs = nt.vector(obs)
        state, time = unpack_obs(obs)

        time = nt.vector_to_scalar(time)
        K, k = self._gains_at(time)

        ctrl = K @ nt.vector_to_matrix(state) + nt.vector_to_matrix(k)
        ctrl = nt.matrix_to_vector(ctrl)
        return ctrl

    @classmethod
    def from_existing(cls, policy: lqr.Linear):
        K, _ = policy
        n_state = K.size("C")
        n_ctrl = K.size("R")
        horizon = K.size("H")
        new = cls(n_state, n_ctrl, horizon)
        new.copy(policy)
        return new

    def copy(self, policy: lqr.Linear):
        K, k = lqr.named.refine_linear_input(policy)
        self.K.data.copy_(K)
        self.k.data.copy_(nt.matrix_to_vector(k))

    def gains(self, named: bool = True) -> lqr.Linear:
        K, k = self.K.clone(), self.k.clone()
        K = nt.horizon(nt.matrix(K))
        k = nt.horizon(nt.vector(k))
        if not named:
            K, k = nt.unnamed(K, k)
        return K, k


class TVLinearPolicy(DeterministicPolicy):
    def __init__(self, obs_space: Box, action_space: Box):
        n_state, n_ctrl, horizon = lqr.dims_from_spaces(obs_space, action_space)
        action_linear = TVLinearFeedback(n_state, n_ctrl, horizon)
        super().__init__(
            encoder=nn.Identity(), action_linear=action_linear, squashing=nn.Identity()
        )

    def initialize_from_optimal(self, optimal: lqr.Linear):
        K, k = map(lambda x: x + torch.randn_like(x) * 0.5, optimal)
        self.action_linear.copy((K, k))

    def standard_form(self) -> lqr.Linear:
        return self.action_linear.gains()


class TVLinearTransModel(TVLinearDynamics):
    """Time-varying linear Gaussian dynamics model."""

    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        n_tau = n_state + n_ctrl
        dynamics = lqr.LinSDynamics(
            F=torch.randn(horizon, n_state, n_tau),
            f=torch.randn(horizon, n_state),
            W=torch.as_tensor(
                make_spd_matrix(n_dim=n_state, sample_shape=(horizon,)),
                dtype=torch.float32,
            ),
        )
        super().__init__(dynamics)


class QuadraticReward(QuadraticCost):
    # pylint:disable=abstract-method,missing-docstring
    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        cost = make_quadcost(n_state, n_ctrl, horizon, stationary=False)
        super().__init__(cost)


class InitStateModel(InitStateDynamics):
    """Gaussian initial state distribution model."""

    def __init__(self, n_state: int, seed: Optional[int] = None):
        loc = torch.zeros(n_state)
        covariance_matrix = as_float_tensor(make_spd_matrix(n_state, rng=seed))
        super().__init__(loc, covariance_matrix)


class TimeVaryingLinear(nn.Module):
    # pylint:disable=abstract-method
    def __init__(self, obs_space: Box, action_space: Box, config: dict):
        super().__init__()
        self.actor = TVLinearPolicy(obs_space, action_space)
        self.behavior = self.actor

        n_state, n_ctrl, horizon = lqr.dims_from_spaces(obs_space, action_space)
        self.init_model = InitStateModel(n_state=n_state, seed=config.get("seed", None))
        self.trans_model = TVLinearTransModel(n_state, n_ctrl, horizon)
        self.rew_model = QuadraticReward(n_state, n_ctrl, horizon)

    def standard_form(self) -> Tuple[Any, Any, Any]:
        return tuple(
            x.standard_form()
            for x in (self.actor, self.trans_model, self.rew_model, self.init_model)
        )


# noinspection PyAbstractClass
@configure
@option("exploration_config/type", "raylab.utils.exploration.GaussianNoise")
@option("exploration_config/pure_exploration_steps", 0)
class LQGPolicy(TorchPolicy):
    # pylint:disable=abstract-method
    dist_class = WrapDeterministicPolicy

    @cached_property
    def n_state(self):
        n_state, _, _ = self.space_dims()
        return n_state

    @cached_property
    def n_ctrl(self):
        _, n_ctrl, _ = self.space_dims()
        return n_ctrl

    @cached_property
    def horizon(self):
        _, _, horizon = self.space_dims()
        return horizon

    def space_dims(self):
        return lqr.dims_from_spaces(self.observation_space, self.action_space)

    def initialize_from_lqg(self, env: TorchLQGMixin):
        optimal: lqr.Linear = env.solution()[0]
        self.module.actor.initialize_from_optimal(optimal)
        self.module.rew_model.copy(env.cost)

    def _make_module(
        self, obs_space: Box, action_space: Box, config: dict
    ) -> nn.Module:
        return TimeVaryingLinear(obs_space, action_space, config)

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers["actor"] = torch.optim.Adam(self.module.actor.parameters(), lr=1e-4)


def make_worker(num_envs: int = 1) -> RolloutWorker:
    # pylint:disable=import-outside-toplevel
    import lqsvg
    from raylab.envs import get_env_creator

    lqsvg.register_all()

    # Create and initialize
    n_state, n_ctrl, horizon = 2, 2, 100
    seed = 42
    worker = RolloutWorker(
        env_creator=get_env_creator("RandomVectorLQG"),
        env_config={
            "n_state": n_state,
            "n_ctrl": n_ctrl,
            "horizon": horizon,
            "gen_seed": seed,
            "num_envs": num_envs,
        },
        num_envs=num_envs,
        policy_spec=LQGPolicy,
        # policy_spec=RandomPolicy,
        policy_config={},
        rollout_fragment_length=horizon,
        batch_mode="truncate_episodes",
        _use_trajectory_view_api=False,
    )
    assert isinstance(worker.env, RandomVectorLQG)
    assert worker.async_env.num_envs == num_envs
    worker.foreach_trainable_policy(lambda p, _: p.initialize_from_lqg(worker.env))
    return worker


def test_worker():
    # pylint:disable=import-outside-toplevel
    import numpy as np

    worker = make_worker(4)
    samples = worker.sample()
    print("Count:", samples.count)
    idxs = np.random.permutation(samples.count)[:10]
    for k in samples.keys():
        print("Key:", k)
        print(samples[k][idxs])
        print(samples[k][idxs].dtype)


if __name__ == "__main__":
    test_worker()
