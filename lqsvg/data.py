"""Utilities for data collection in LQG envs."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from more_itertools import all_equal, first
from nnrl.nn.distributions.types import SampleLogp
from nnrl.types import TensorDict
from ray.rllib import RolloutWorker, SampleBatch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm.auto import tqdm

from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.torch import named as nt
from lqsvg.torch.nn.env import EnvModule
from lqsvg.types import (
    DeterministicPolicy,
    InitStateFn,
    RecurrentDynamics,
    RewardFunction,
    StateDynamics,
    Trajectory,
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


def environment_sampler(
    env: EnvModule,
) -> Callable[[DeterministicPolicy, int], Trajectory]:
    """Creates a trajectory sampler from an environment module."""
    dynamics = markovian_state_sampler(env.trans, env.trans.sample)

    def sampler(policy: DeterministicPolicy, trajectories: int) -> Trajectory:
        obs, logp = env.init.sample([trajectories])
        obs = obs.refine_names("B", ...)
        obss, acts, rews, logps = [obs], [], [], [logp]
        for _ in range(env.horizon):
            act = policy(obs)
            rew = env.reward(obs, act)
            new_obs, logp = dynamics(obs, act)

            obss += [new_obs]
            acts += [act]
            rews += [rew]
            logps += [logp]
            obs = new_obs

        obs, act, rew, logp = itertools.starmap(
            nt.stack_horizon, (obss, acts, rews, logps)
        )
        return obs, act, rew, logp

    return sampler


def trajectory_sampler(
    policy: DeterministicPolicy,
    init_fn: InitStateFn,
    dynamics: StateDynamics,
    reward_fn: RewardFunction,
) -> TrajectorySampler:
    """Creates a full trajectory sampler."""

    def sample_trajectory(horizon: int, sample_shape: Sequence[int] = ()) -> Trajectory:
        """Samples full trajectory.

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


@nt.variadic
def merge_horizon_and_batch_dims(tensor: Tensor) -> Tensor:
    """Flattens the 'H' and 'B' dimensions into a single 'B' dimension."""
    return tensor.flatten("H B".split(), "B")


def obs_trajectory_to_transitions(obs: Tensor) -> Tuple[Tensor, Tensor]:
    """Splits an observation sequence into current and next observation pairs."""
    aligned = obs.align_to("H", ...)
    curr, succ = aligned[:-1], aligned[1:]
    return curr.align_as(obs), succ.align_as(obs)


def split_along_batch_dim(
    tensors: Sequence[Tensor], split_sizes: Sequence[int], rng: torch.Generator
) -> Sequence[Sequence[Tensor]]:
    """Randomly splits tensors along the 'B' dimension in given split sizes."""
    if not tensors:
        return ((),) * len(split_sizes)

    # noinspection PyArgumentList
    bsize = tensors[0].size("B")
    indices = torch.split(
        torch.randperm(bsize, generator=rng), split_size_or_sections=split_sizes
    )
    # noinspection PyTypeChecker
    return tuple(
        tuple(nt.index_select(t, "B", idxs) for t in tensors) for idxs in indices
    )


@dataclass
class TensorDataSpec:
    """Specifications for TensorDataModule."""

    train_batch_size: int
    val_batch_size: int
    train_fraction: float = 0.9


class TensorDataModule(pl.LightningDataModule):
    """Data module holding tensors."""

    train_dataset: TensorDataset
    val_dataset: TensorDataset
    spec_cls = TensorDataSpec

    def __init__(
        self, *tensors: Tensor, spec: Union[TensorDataSpec, dict], rng: torch.Generator
    ):
        super().__init__()
        assert tensors, "Empty tensor list"
        self.tensors = tuple(t.align_to("B", ...) for t in tensors)
        self.size = self.tensors[0].size("B")
        self.names = tuple(t.names for t in self.tensors)
        self.spec = self.spec_cls(**spec) if isinstance(spec, dict) else spec
        self.rng = rng

    def setup(self, stage: Optional[str] = None) -> None:
        spec = self.spec
        train_val_tensors = split_along_batch_dim(
            self.tensors, train_val_sizes(self.size, spec.train_fraction), self.rng
        )
        train_tensors, val_tensors = itertools.starmap(nt.unnamed, train_val_tensors)
        self.train_dataset = TensorDataset(*train_tensors)
        self.val_dataset = TensorDataset(*val_tensors)

    def train_dataloader(self) -> DataLoader:
        batch_size = self.spec.train_batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        batch_size = self.spec.val_batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    def on_after_batch_transfer(
        self, batch: Tuple[Tensor, ...], dataloader_idx: int
    ) -> Tuple[Tensor, ...]:
        del dataloader_idx
        return tuple(t.refine_names(*names) for t, names in zip(batch, self.names))


@dataclass
class SequenceDataSpec:
    """Specifications for SequenceDataModule."""

    train_batch_size: int
    val_batch_size: int
    seq_len: int
    train_fraction: float = 0.9


class SequenceDataModule(pl.LightningDataModule):
    """Data module holding sequence tensors."""

    train_dataset: Dataset
    val_dataset: Dataset
    spec_cls = SequenceDataSpec

    def __init__(
        self,
        *tensors: Tensor,
        spec: Union[SequenceDataSpec, dict],
        rng: torch.Generator,
    ):
        super().__init__()
        assert tensors, "Empty tensor list"
        self.tensors = tuple(t.align_to("B", "H", ...) for t in tensors)
        self.sequences = self.tensors[0].size("B")
        self.names = tuple(t.names for t in self.tensors)
        self.spec = self.spec_cls(**spec) if isinstance(spec, dict) else spec
        self.rng = rng

    def setup(self, stage: Optional[str] = None) -> None:
        spec = self.spec
        train_tensors, val_tensors = split_along_batch_dim(
            self.tensors, train_val_sizes(self.sequences, spec.train_fraction), self.rng
        )
        self.train_dataset = TensorSeqDataset(*train_tensors, seq_len=spec.seq_len)
        self.val_dataset = TensorSeqDataset(*val_tensors, seq_len=spec.seq_len)

    def train_dataloader(self) -> DataLoader:
        batch_size = self.spec.train_batch_size
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        batch_size = self.spec.val_batch_size
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    def on_after_batch_transfer(
        self, batch: Tuple[Tensor, ...], dataloader_idx: int
    ) -> Tuple[Tensor, ...]:
        del dataloader_idx
        return tuple(t.refine_names(*names) for t, names in zip(batch, self.names))


class TensorSeqDataset(Dataset[Tuple[Tensor, ...]]):
    """Dataset wrapping sequence tensors.

    Note:
        May refactor this to use `torch.Tensor.unfold`.

    Args:
        *tensors: tensors that have the same size of the first and second
            dimensions.
        seq_len: length of sub-sequences to sample from tensors
    """

    # noinspection PyArgumentList
    def __init__(self, *tensors: Tensor, seq_len: int):
        tensors = tuple(t.align_to("B", "H", ...) for t in tensors)
        msg = "Size mismatch between tensors in dim {}"
        assert all_equal(t.size("B") for t in tensors), msg.format("B")
        assert all_equal(t.size("H") for t in tensors), msg.format("H")

        # Pytorch Lightning deepcopies the dataloader when using overfit_batches=True
        # Deepcopying is incompatible with named tensors for some reason
        self.tensors = nt.unnamed(*tensors)
        self.seq_len = seq_len
        self.seqs_per_traj = first(tensors).size("H") - self.seq_len - 1
        self._len = first(tensors).size("B") * self.seqs_per_traj

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
