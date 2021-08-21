"""Utilities for data collection in LQG envs."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from nnrl.nn.distributions.types import SampleLogp
from nnrl.types import TensorDict
from ray.rllib import RolloutWorker, SampleBatch
from torch import Tensor
from torch.utils.data import Dataset, random_split
from tqdm.auto import tqdm

from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.torch import named as nt
from lqsvg.types import (
    DeterministicPolicy,
    InitStateFn,
    RecurrentDynamics,
    RewardFunction,
    StateDynamics,
    TrajectorySampler,
)


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
    sampler_fn: Callable[[TensorDict], SampleLogp],
) -> StateDynamics:
    """Combine state-action conditional params and conditional state dist."""

    def sampler(obs: Tensor, act: Tensor) -> SampleLogp:
        params = params_fn(obs, act)
        return sampler_fn(params)

    return sampler


def recurrent_state_sampler(
    params_fn: Callable[[Tensor, Tensor, Tensor], TensorDict],
    sampler_fn: Callable[[TensorDict], SampleLogp],
) -> RecurrentDynamics:
    """Combine contextual dist params and conditional state dist."""

    def sampler(
        obs: Tensor, act: Tensor, ctx: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        params = params_fn(obs, act, ctx)
        sample, logp = sampler_fn(params)
        return sample, logp, params["context"]

    return sampler


def trajectory_sampler(
    policy: DeterministicPolicy,
    init_fn: InitStateFn,
    dynamics: StateDynamics,
    reward_fn: RewardFunction,
) -> TrajectorySampler:
    """Full trajectory sampler."""

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


def train_val_sizes(total: int, train_frac: float) -> Tuple[int, int]:
    """Compute train and validation dataset sizes from total size."""
    train_samples = int(total * train_frac)
    val_samples = total - train_samples
    return train_samples, val_samples


class TensorSeqDataset(Dataset[Tuple[Tensor, ...]]):
    """Dataset wrapping sequence tensors.

    Args:
        *tensors: tensors that have the same size of the first and second
            dimensions.
        seq_len: length of sub-sequences to sample from tensors
    """

    # noinspection PyArgumentList
    def __init__(self, *tensors: Tensor, seq_len: int):
        tensors = tuple(t.align_to("B", "H", ...) for t in tensors)
        trajs: int = tensors[0].size("B")
        horizon: int = tensors[0].size("H")
        assert all(
            t.size("B") == trajs and t.size("H") == horizon for t in tensors
        ), "Size mismatch between tensors"

        # Pytorch Lightning deepcopies the dataloader when using overfit_batches=True
        # Deepcopying is incompatible with named tensors for some reason
        self.tensors = nt.unnamed(*tensors)
        self.seq_len = seq_len
        self.seqs_per_traj = horizon - self.seq_len - 1
        self._len = trajs * self.seqs_per_traj

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        traj_idx = index // self.seqs_per_traj
        timestep_start = index % self.seqs_per_traj
        # noinspection PyTypeChecker
        return tuple(
            t[traj_idx, timestep_start : timestep_start + self.seq_len]
            for t in self.tensors
        )

    def __len__(self) -> int:
        return self._len


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


def group_batch_episodes(samples: SampleBatch) -> SampleBatch:
    """Return the sample batch with rows grouped by episode id.

    Moreover, rows are sorted by timestep.

    Warning:
        Modifies the sample batch in-place
    """
    # Assume "t" is the timestep key in the sample batch
    sorted_timestep_idxs = np.argsort(samples["t"])
    for key, val in samples.items():
        samples[key] = val[sorted_timestep_idxs]

    # Stable sort is important so that we don't alter the order
    # of timesteps
    sorted_episode_idxs = np.argsort(samples[SampleBatch.EPS_ID], kind="stable")
    for key, val in samples.items():
        samples[key] = val[sorted_episode_idxs]

    return samples


def num_complete_episodes(samples: SampleBatch) -> int:
    """Return the number of complete episodes in a SampleBatch."""
    num_eps = len(np.unique(samples[SampleBatch.EPS_ID]))
    num_dones = np.sum(samples[SampleBatch.DONES]).item()
    assert (
        num_dones <= num_eps
    ), f"More done flags than episodes: dones={num_dones}, episodes={num_eps}"
    return num_dones


def collect_with_progress(worker, total_trajs, prog: bool = True) -> SampleBatch:
    """Collect sample batches with progress monitoring."""
    with tqdm(
        total=total_trajs, desc="Collecting", unit="traj", disable=not prog
    ) as pbar:
        sample_batch: SampleBatch = worker.sample()
        eps = num_complete_episodes(sample_batch)
        while eps < total_trajs:
            old_eps = eps
            sample_batch = sample_batch.concat(worker.sample())
            eps = num_complete_episodes(sample_batch)
            pbar.update(eps - old_eps)

    return sample_batch
