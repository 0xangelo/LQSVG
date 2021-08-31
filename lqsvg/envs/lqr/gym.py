"""OpenAI Gym interface for LQG."""
from __future__ import annotations

from functools import cached_property
from typing import List, Optional, Tuple

import gym  # pylint:disable=import-self
import numpy as np
import torch
from ray.rllib.env import VectorEnv
from ray.rllib.utils.typing import EnvActionType, EnvInfoDict, EnvObsType, EnvType
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.torch.nn.dynamics.gauss import InitStateModule
from lqsvg.torch.nn.dynamics.linear import LinearDynamicsModule
from lqsvg.torch.nn.reward import QuadraticReward

from .generators import LQGGenerator
from .modules import LQGModule
from .solvers import NamedLQGControl
from .types import GaussInit, Linear, LinSDynamics, QuadCost, Quadratic
from .utils import dims_from_dynamics, spaces_from_dims

Obs = np.ndarray
Act = np.ndarray
Rew = float
Done = bool
Info = dict


# noinspection PyAttributeOutsideInit
class TorchLQGMixin:
    # pylint:disable=too-many-instance-attributes,missing-docstring
    module: LQGModule
    dynamics: LinSDynamics
    cost: QuadCost
    rho: GaussInit

    def setup(
        self,
        dynamics: LinSDynamics,
        cost: QuadCost,
        init: GaussInit,
    ):
        self.module = LQGModule(
            dims=dims_from_dynamics(dynamics),
            trans=LinearDynamicsModule.from_existing(dynamics, stationary=False),
            reward=QuadraticReward.from_existing(cost),
            init=InitStateModule.from_existing(init),
        )
        # Ensure optimizers don't update the MDP
        self.module.requires_grad_(False)
        self.dynamics, self.cost, self.rho = self.module.standard_form()
        self.observation_space, self.action_space = self._setup_spaces()

    @property
    def horizon(self):
        return self.dynamics.F.size("H")

    @property
    def n_state(self):
        return self.dynamics.F.size("R")

    @property
    def n_tau(self):
        return self.dynamics.F.size("C")

    @property
    def n_ctrl(self):
        return self.n_tau - self.n_state

    def _setup_spaces(self):
        observation_space, action_space = spaces_from_dims(
            self.n_state, self.n_ctrl, self.horizon
        )
        return observation_space, action_space

    @cached_property
    def solution(self) -> Tuple[Linear, Quadratic, Quadratic]:
        with torch.no_grad():
            solver = NamedLQGControl(self.n_state, self.n_ctrl, self.horizon)
            solution = solver(self.dynamics, self.cost)
        return solution


# noinspection PyAbstractClass
class LQGEnv(TorchLQGMixin, gym.Env):
    """Linear Quadratic Gaussian for OpenAI Gym."""

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring
    def __init__(self, dynamics: LinSDynamics, cost: QuadCost, init: GaussInit):
        self.setup(dynamics, cost, init)
        self._curr_state: Optional[Tensor] = None

    @torch.no_grad()
    def reset(self) -> Obs:
        self._curr_state, _ = self.module.init.sample()
        return self._get_obs()

    @torch.no_grad()
    def step(self, action: Act) -> Tuple[Obs, Rew, Done, Info]:
        state = self._curr_state
        action = torch.as_tensor(action, dtype=torch.float32)
        action = nt.vector(action)

        reward = self.module.reward(state, action)
        next_state, _ = self.module.trans.sample(self.module.trans(state, action))

        self._curr_state = next_state
        done = next_state[-1].long() == self.horizon
        return self._get_obs(), reward.item(), done.item(), {}

    def _get_obs(self) -> Obs:
        obs = self._curr_state
        obs = obs.detach().numpy()
        return obs.astype(self.observation_space.dtype)


# noinspection PyAbstractClass
class RandomLQGEnv(LQGEnv):
    """Random Linear Quadratic Gaussian from JSON specifications."""

    # pylint:disable=abstract-method
    def __init__(self, config: dict):
        generator = LQGGenerator(**config)
        dynamics, cost, init = generator(n_batch=None)
        super().__init__(dynamics=dynamics, cost=cost, init=init)


class RandomVectorLQG(TorchLQGMixin, VectorEnv):
    """Vectorized implementation of LQG environment.

    Attributes:
        num_envs: how many environments to simulate in parallel. Effectively
            the sample size for the initial state distribution.
    """

    num_envs: int

    def __init__(self, config: dict):
        config = config.copy()  # ensure .pop() has no side effects
        num_envs = config.pop("num_envs")

        generator = LQGGenerator(**config)
        dynamics, cost, init = generator(n_batch=None)
        self.setup(dynamics, cost, init)
        self._curr_states = None
        super().__init__(self.observation_space, self.action_space, num_envs)

    @property
    def curr_states(self) -> Optional[np.ndarray]:
        """Current vectorized state as numpy array."""
        if self._curr_states is None:
            return None
        return self._curr_states.numpy().astype(self.observation_space.dtype)

    def vector_reset(self) -> List[EnvObsType]:
        _curr_states, _ = self.module.init.sample((self.num_envs,))
        self._curr_states = _curr_states.rename(B1="B")
        return self._get_obs(self.curr_states)

    @torch.no_grad()
    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        init_state, _ = self.module.init.sample()
        self._curr_states[index] = nt.unnamed(init_state)
        return init_state.numpy().astype(self.observation_space.dtype)

    def _get_obs(self, states: np.ndarray) -> List[Obs]:
        return [o.squeeze(0) for o in np.vsplit(states, self.num_envs)]

    @torch.no_grad()
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        states = self._curr_states
        actions = np.vstack(actions).astype(self.action_space.dtype)
        actions = torch.from_numpy(actions)
        actions = nt.vector(actions)

        rewards = self.module.reward(states, actions)
        next_states, _ = self.module.trans.sample(self.module.trans(states, actions))
        dones = next_states[..., -1].long() == self.horizon
        self._curr_states = next_states

        obs = self._get_obs(self.curr_states)
        rewards = rewards.numpy().tolist()
        dones = dones.numpy().tolist()
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, infos

    def get_unwrapped(self) -> List[EnvType]:
        pass
