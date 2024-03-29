"""Dummy gym.Env subclasses."""
import gym
import numpy as np
import torch
from gym.spaces import Box
from ray.rllib.utils import override


class MockEnv(gym.Env):
    """Dummy environment with continuous action space."""

    def __init__(self, _):
        self.horizon = 200
        self.time = 0

        low = np.array([-1] * 3 + [0], dtype=np.float32)
        high = np.array([1] * 4, dtype=np.float32)
        self.observation_space = Box(low=low, high=high)

        action_dim = 3
        self.action_space = Box(high=1, low=-1, shape=(action_dim,), dtype=np.float32)

        self.goal = torch.zeros(*self.observation_space.shape)[..., :-1]
        self.state = None

    @override(gym.Env)
    def reset(self):
        self.time = 0
        self.state = self.observation_space.sample()
        self.state[-1] = 0
        return self.state

    @override(gym.Env)
    def step(self, action):
        self.time += 1
        self.state[:3] = np.clip(
            self.state[:3] + action,
            self.observation_space.low[:3],
            self.observation_space.high[:3],
        )
        self.state[-1] = self.time / self.horizon
        reward = np.linalg.norm((self.state[:3] - self.goal.numpy()), axis=-1)
        return self.state, reward, self.time >= self.horizon, {}

    # noinspection PyUnusedLocal
    @staticmethod
    def reward_fn(state, action, next_state):
        # pylint:disable=missing-docstring,unused-argument
        return torch.norm(next_state[..., :3], dim=-1)

    def dynamics_fn(self, state, action):
        state, time = state[..., :3], state[..., 3:]
        new_state = state + action
        new_state = torch.max(
            torch.min(new_state, torch.from_numpy(self.action_space.high)),
            torch.from_numpy(self.action_space.low),
        )

        time = time * self.horizon
        new_time = torch.clamp((time + 1) / self.horizon, min=0, max=1)
        return torch.cat([new_state, new_time], dim=-1), None

    def render(self, mode="human"):
        pass
