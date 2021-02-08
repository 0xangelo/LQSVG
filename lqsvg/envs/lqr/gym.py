"""OpenAI Gym interface for LQG."""
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import gym  # pylint:disable=import-self
import numpy as np
import torch
from dataclasses_json import DataClassJsonMixin
from ray.rllib.env import VectorEnv
from ray.rllib.utils.typing import EnvActionType
from ray.rllib.utils.typing import EnvInfoDict
from ray.rllib.utils.typing import EnvObsType
from ray.rllib.utils.typing import EnvType
from torch import Tensor

import lqsvg.torch.named as nt

from .generators import make_lqg
from .modules import InitStateDynamics
from .modules import QuadraticCost
from .modules import TVLinearDynamics
from .solvers import NamedLQGControl
from .types import Linear
from .types import LinSDynamics
from .types import QuadCost
from .types import Quadratic
from .utils import spaces_from_dims

Obs = np.ndarray
Act = np.ndarray
Rew = float
Done = bool
Info = dict


@dataclass
class LQGSpec(DataClassJsonMixin):
    """Specifications for LQG generation.

    Args:
        n_state: dimensionality of the state vectors
        n_ctrl: dimensionality of the control (action) vectors
        horizon: task horizon
        deterministic_start: whether the initial state distribution
            should be a dirac delta or multivariate diagonal Gaussian
            with mean zero and covariance as identity matrix. Has no
            effect for now
        deterministic_trans: whether the covariance for state transition
            noise should be zero or a randomly initialized positive
            definite matrix. Has no effect for now
        stationary: whether the transition kernel parameters should be
            constant over time or vary by timestep
        gen_seed: integer seed for random number generator used in
            initializing LQG parameters
        num_envs: how many environments to simulate in parallel. Effectively
            the initial state distribution batched.
    """

    # pylint:disable=too-many-instance-attributes
    n_state: int = 2
    n_ctrl: int = 2
    horizon: int = 20
    deterministic_start: bool = False
    deterministic_trans: bool = False
    stationary: bool = False
    gen_seed: Optional[int] = None
    num_envs: Optional[int] = None

    def make_lqg(self) -> Tuple[LinSDynamics, QuadCost]:
        """Returns random LQG transition kernel and cost function"""
        return make_lqg(
            state_size=self.n_state,
            ctrl_size=self.n_ctrl,
            horizon=self.horizon,
            stationary=self.stationary,
            np_random=self.gen_seed,
        )

    def init_params(self, *, vector: bool = False) -> Tuple[Tensor, Tensor]:
        """Compute parameters of the multivariate Normal for initial states.

        Args:
            vector: whether to return batched parameters for use in a
                vectorized environment

        Returns:
            A tuple containing mean and covariance tensors
        """
        mean = torch.zeros(self.n_state, names=("R",))
        cov = torch.eye(self.n_state).refine_names("R", "C")
        if vector and self.num_envs:
            mean, cov = (
                x.expand((self.num_envs,) + x.shape).refine_names("B", ...)
                for x in (mean, cov)
            )
        return mean, cov


# noinspection PyAttributeOutsideInit
class TorchLQGMixin:
    # pylint:disable=too-many-instance-attributes,missing-docstring
    def setup(
        self,
        dynamics: LinSDynamics,
        cost: QuadCost,
        init_mean: Tensor,
        init_cov: Tensor,
    ):
        self.dynamics = dynamics
        self.cost = cost

        self.observation_space, self.action_space = self._setup_spaces()

        self._trans = TVLinearDynamics(dynamics)
        self._cost = QuadraticCost(cost)
        self._rho = InitStateDynamics(init_mean, init_cov)

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

    @torch.no_grad()
    def solution(self) -> Tuple[Linear, Quadratic, Quadratic]:
        solver = NamedLQGControl(self.n_state, self.n_ctrl, self.horizon)
        solution = solver(self.dynamics, self.cost)
        return solution


# noinspection PyAbstractClass
class LQGEnv(TorchLQGMixin, gym.Env):
    """Linear Quadratic Gaussian for OpenAI Gym."""

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring
    def __init__(
        self,
        dynamics: LinSDynamics,
        cost: QuadCost,
        init_mean: Tensor,
        init_cov: Tensor,
    ):
        self.setup(dynamics, cost, init_mean, init_cov)
        self._curr_state: Optional[Tensor] = None

    @torch.no_grad()
    def reset(self) -> Obs:
        self._curr_state, _ = self._rho.sample()
        return self._get_obs()

    @torch.no_grad()
    def step(self, action: Act) -> Tuple[Obs, Rew, Done, Info]:
        state = self._curr_state
        action = torch.as_tensor(action, dtype=torch.float32)
        action = nt.vector(action)

        reward = self._cost(state, action)
        next_state, _ = self._trans.sample(self._trans(state, action))

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
    def __init__(self, spec: LQGSpec):
        dynamics, cost = spec.make_lqg()
        init_mean, init_cov = spec.init_params()
        super().__init__(
            dynamics=dynamics, cost=cost, init_mean=init_mean, init_cov=init_cov
        )


class RandomVectorLQG(TorchLQGMixin, VectorEnv):
    """Vectorized implementation of LQG environment."""

    def __init__(self, spec: LQGSpec):
        assert spec.num_envs is not None
        dynamics, cost = spec.make_lqg()
        init_mean, init_cov = spec.init_params(vector=True)
        self.setup(dynamics, cost, init_mean=init_mean, init_cov=init_cov)
        self.spec = spec
        self._curr_states: Optional[Tensor] = None
        super().__init__(self.observation_space, self.action_space, spec.num_envs)

    def vector_reset(self) -> List[EnvObsType]:
        self._curr_states, _ = self._rho.sample()
        return self._get_obs(self._curr_states)

    @torch.no_grad()
    def reset_at(self, index: int) -> EnvObsType:
        init_states, _ = self._rho.sample()
        self._curr_states[index] = nt.unnamed(init_states)[index]
        return init_states[index].numpy().astype(self.observation_space.dtype)

    def _get_obs(self, states: Tensor) -> List[Obs]:
        obs = states
        obs = obs.detach().numpy()
        obs = obs.astype(self.observation_space.dtype)
        return [o.squeeze(0) for o in np.vsplit(obs, self.num_envs)]

    @torch.no_grad()
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        states = self._curr_states
        actions = np.vstack(actions).astype(self.action_space.dtype)
        actions = torch.from_numpy(actions)
        actions = nt.vector(actions)

        rewards = self._cost(states, actions)
        next_states, _ = self._trans.sample(self._trans(states, actions))
        dones = next_states[..., -1].long() == self.horizon

        self._curr_states = next_states
        rewards = rewards.numpy().tolist()
        dones = dones.numpy().tolist()
        return (
            self._get_obs(next_states),
            rewards,
            dones,
            [{} for _ in range(self.num_envs)],
        )

    def get_unwrapped(self) -> List[EnvType]:
        pass
