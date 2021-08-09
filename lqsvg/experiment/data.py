"""Utilities for data collection in LQG envs."""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import torch
from nnrl.types import TensorDict
from ray.rllib import RolloutWorker, SampleBatch
from torch import Tensor
from torch.utils.data import Dataset, random_split

from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.torch import named as nt

from .tqdm_util import collect_with_progress
from .utils import group_batch_episodes

DeterministicPolicy = Callable[[Tensor], Tensor]
InitStateFn = Callable[[Sequence[int]], Tuple[Tensor, Tensor]]
StateDynamics = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
RewardFunction = Callable[[Tensor, Tensor], Tensor]
TrajectorySampler = Callable[
    [int, Sequence[int]], Tuple[Tensor, Tensor, Tensor, Tensor]
]


def split_dataset(
    dataset: Dataset, train_val_test_split: Tuple[float, float, float]
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split generic dataset into train, validation, and test datasets."""
    train, val, test = train_val_test_split
    # noinspection PyTypeChecker
    total_samples = len(dataset)
    train_samples = int(train * total_samples)
    holdout_samples = total_samples - train_samples
    val_samples = int((val / (val + test)) * holdout_samples)
    test_samples = holdout_samples - val_samples

    train_data, val_data, test_data = random_split(
        dataset, (train_samples, val_samples, test_samples)
    )
    return train_data, val_data, test_data


def markovian_state_sampler(
    params_fn: Callable[[Tensor, Tensor], TensorDict],
    sampler_fn: Callable[[TensorDict], Tuple[Tensor, Tensor]],
) -> StateDynamics:
    """Combine state-action conditional params and conditional state dist."""

    def sampler(obs: Tensor, act: Tensor) -> Tuple[Tensor, Tensor]:
        params = params_fn(obs, act)
        return sampler_fn(params)

    return sampler


def trajectory_sampler(
    policy: DeterministicPolicy,
    init_fn: InitStateFn,
    dynamics: StateDynamics,
    reward_fn: RewardFunction,
) -> TrajectorySampler:
    """Reparameterized trajectory sampler."""

    def sample_trajectory(
        horizon: int,
        sample_shape: Sequence[int] = torch.Size(),
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample full trajectory.

        Sample trajectory using initial state, dynamics and reward models with
        a deterministic actor.

        Args:
            horizon: length of the trajectory
            sample_shape: shape for batched trajectory samples

        Returns:
            Trajectory sample following the policy and its corresponding
            log-likelihood under the model
        """
        obs, logp = init_fn(sample_shape)
        sample_names = (f"B{i + 1}" for i, _ in enumerate(sample_shape))
        obs = obs.refine_names(*sample_names, ...)
        obss, acts, rews = [obs], [], []
        for _ in range(horizon):
            act = policy(obs)
            rew = reward_fn(obs, act)
            # No sample_shape needed since initial states are batched
            new_obs, logp_t = dynamics(obs, act)

            obss += [new_obs]
            acts += [act]
            rews += [rew]
            logp = logp + logp_t
            obs = new_obs

        obs = nt.stack_horizon(*obss)
        act = nt.stack_horizon(*acts)
        rew = nt.stack_horizon(*rews)
        return obs, act, rew, logp

    return sample_trajectory


###############################################################################
# RLlib
###############################################################################


def batched(trajs: list[SampleBatch], key: str) -> Tensor:
    """Automatically stack sample rows and convert to tensor."""
    return torch.stack([torch.from_numpy(traj[key]) for traj in trajs], dim=0)


def check_worker_config(worker: RolloutWorker):
    """Verify that worker collects full trajectories on an LQG."""
    assert worker.rollout_fragment_length == worker.env.horizon * worker.num_envs
    assert worker.batch_mode == "truncate_episodes"
    assert isinstance(worker.env, TorchLQGMixin)


def collect_trajs_from_worker(
    worker: RolloutWorker, total_trajs: int, progress: bool = False
) -> list[SampleBatch]:
    """Use rollout worker to collect trajectories in the environment."""
    sample_batch = collect_with_progress(worker, total_trajs, prog=progress)
    sample_batch = group_batch_episodes(sample_batch)

    trajs = sample_batch.split_by_episode()
    traj_counts = [t.count for t in trajs]
    assert all(c == worker.env.horizon for c in traj_counts), traj_counts
    total_ts = sum(t.count for t in trajs)
    assert total_ts == total_trajs * worker.env.horizon, total_ts

    return trajs
