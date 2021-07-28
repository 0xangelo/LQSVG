"""Utilities for data collection in LQG envs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from dataclasses_json import DataClassJsonMixin
from ray.rllib import RolloutWorker, SampleBatch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from lqsvg.envs.lqr.gym import TorchLQGMixin

from .tqdm_util import collect_with_progress
from .utils import group_batch_episodes


@dataclass
class DataModuleSpec(DataClassJsonMixin):
    """Specifications for creating the data module.

    Attributes:
        total_trajs: Number of trajectories for the full dataset
        train_val_test_split: Fractions of the dataset for training, validation
            and testing respectively
        batch_size: Size of minibatch for dynamics model training
        shuffle: set to ``True`` to have the data reshuffled
            at every epoch (default: ``True``).
    """

    total_trajs: int = 100
    train_val_test_split: tuple[float, float, float] = (0.7, 0.2, 0.1)
    batch_size: int = 64
    shuffle: bool = True

    def __post_init__(self):
        assert np.allclose(sum(self.train_val_test_split), 1.0), "Invalid dataset split"


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


def split_dataset(
    dataset: Dataset, train_val_test_split: tuple[float, float, float]
) -> tuple[Dataset, Dataset, Dataset]:
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


class DataModule(pl.LightningDataModule):
    """Generic datamodule for dynamics model training."""

    spec_cls = DataModuleSpec

    def __init__(self, rollout_worker: RolloutWorker, spec: DataModuleSpec):
        super().__init__()
        self.worker = rollout_worker
        check_worker_config(self.worker)

        self.spec = spec
        self.full_dataset = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def collect_trajectories(self, prog: bool = True) -> list[SampleBatch]:
        """Sample trajectories with rollout worker and build dataset."""
        return collect_trajs_from_worker(
            self.worker, self.spec.total_trajs, progress=prog
        )

    def build_dataset(self, prog: bool = False):
        """Create the full dataset of experiences."""

    def setup(self, stage: Optional[str] = None):
        del stage
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(
            self.full_dataset, self.spec.train_val_test_split
        )

    def train_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        spec = self.spec
        dataloader = DataLoader(
            self.train_dataset, shuffle=spec.shuffle, batch_size=spec.batch_size
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        dataloader = DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.spec.batch_size
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        dataloader = DataLoader(
            self.test_dataset, shuffle=False, batch_size=self.spec.batch_size
        )
        return dataloader


class TrajectoryData(DataModule):
    """Data module for on-policy trajectory data in LQG envs."""

    def build_dataset(self, prog: bool = False):
        trajs = self.collect_trajectories(prog=prog)
        self.full_dataset = self.trajectory_dataset(trajs)

    @staticmethod
    def trajectory_dataset(trajs: list[SampleBatch]) -> TensorDataset:
        """Concat and convert a list of trajectories into a tensor dataset."""
        dataset = TensorDataset(
            batched(trajs, SampleBatch.CUR_OBS),
            batched(trajs, SampleBatch.ACTIONS),
            batched(trajs, SampleBatch.NEXT_OBS),
        )
        assert len(dataset) == len(trajs)
        return dataset


class TransitionData(DataModule):
    """Data module for on-policy transition data in LQG envs."""

    def build_dataset(self, prog: bool = True):
        """Sample trajectories with rollout worker and build dataset."""
        trajs = collect_trajs_from_worker(
            self.worker, self.spec.total_trajs, progress=prog
        )
        self.full_dataset = self.transition_dataset(trajs)

    @staticmethod
    def transition_dataset(trajs: list[SampleBatch]) -> TensorDataset:
        """Convert a list of trajectories into a transition tensor dataset."""
        transitions = SampleBatch.concat_samples(trajs)

        dataset = TensorDataset(
            torch.from_numpy(transitions[SampleBatch.CUR_OBS]),
            torch.from_numpy(transitions[SampleBatch.ACTIONS]),
            torch.from_numpy(transitions[SampleBatch.NEXT_OBS]),
        )
        assert len(dataset) == transitions.count
        return dataset


def build_trajectory_datamodule(worker: RolloutWorker, **kwargs):
    # pylint:disable=missing-function-docstring
    data_spec = DataModuleSpec(**kwargs)
    datamodule = TrajectoryData(worker, data_spec)
    return datamodule
