"""Utilities for data collection in LQG envs."""
from dataclasses import dataclass
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
from dataclasses_json import DataClassJsonMixin
from ray.rllib import RolloutWorker
from ray.rllib import SampleBatch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset

from lqsvg.envs.lqr.gym import TorchLQGMixin

from tqdm_util import collect_with_progress  # pylint:disable=wrong-import-order
from utils import group_batch_episodes  # pylint:disable=wrong-import-order


@dataclass
class DataModuleSpec(DataClassJsonMixin):
    """Specifications for creating the data module.

    Attributes:
        total_trajs: Number of trajectories for the full dataset
        holdout_ratio: Fraction of trajectories to use as validation dataset
        max_holdout: Maximum number of trajectories to use as validation dataset
        batch_size: Size of minibatch for dynamics model training
        shuffle: set to ``True`` to have the data reshuffled
            at every epoch (default: ``True``).
    """

    total_trajs: int = 100
    holdout_ratio: float = 0.2
    max_holdout: Optional[int] = None
    batch_size: int = 64
    shuffle: bool = True


def batched(trajs: List[SampleBatch], key: str) -> Tensor:
    """Automatically stack sample rows and convert to tensor."""
    return torch.stack([torch.from_numpy(traj[key]) for traj in trajs], dim=0)


class TrajectoryData(pl.LightningDataModule):
    """Data module for on-policy trajectory data in LQG envs."""

    # pylint:disable=abstract-method
    spec_cls = DataModuleSpec

    def __init__(self, rollout_worker: RolloutWorker, spec: DataModuleSpec):
        super().__init__()
        self.worker = rollout_worker
        assert (
            self.worker.rollout_fragment_length
            == self.worker.env.horizon * self.worker.num_envs
        )
        assert self.worker.batch_mode == "truncate_episodes"
        assert isinstance(self.worker.env, TorchLQGMixin)

        self.spec = spec
        self.full_dataset = None
        self.train_dataset, self.val_dataset = None, None

    def collect_trajectories(self):
        """Sample trajectories with rollout worker and build dataset."""
        sample_batch = collect_with_progress(self.worker, self.spec.total_trajs)

        sample_batch = group_batch_episodes(sample_batch)

        trajs = sample_batch.split_by_episode()
        traj_counts = [t.count for t in trajs]
        assert all(c == self.worker.env.horizon for c in traj_counts), traj_counts
        total_ts = sum(t.count for t in trajs)
        assert total_ts == self.spec.total_trajs * self.worker.env.horizon, total_ts

        self.full_dataset = self.trajectory_dataset(trajs)

    def setup(self, stage: Optional[str] = None):
        del stage
        spec = self.spec
        val_trajs = min(
            int(spec.holdout_ratio * spec.total_trajs),
            spec.max_holdout or spec.total_trajs,
        )
        train_trajs = spec.total_trajs - val_trajs
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, (train_trajs, val_trajs)
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

    def check_dataloaders(self):
        """For testing only."""
        env = self.worker.env
        horizon = env.horizon
        n_state = env.n_state
        n_ctrl = env.n_ctrl

        for dataloader in self.train_dataloader(), self.val_dataloader():
            batch_size = min(self.spec.batch_size, len(dataloader.dataset))
            batch = next(iter(dataloader))
            assert len(batch) == 3
            obs, act, new_obs = batch
            assert obs.shape == (
                batch_size,
                horizon,
                n_state + 1,
            ), f"{obs.shape}, B: {batch_size}, H: {horizon}, dim(S): {n_state}"
            assert act.shape == (
                batch_size,
                horizon,
                n_ctrl,
            ), f"{act.shape}, B: {batch_size}, H: {horizon}, dim(A): {n_ctrl}"
            assert new_obs.shape == (
                batch_size,
                horizon,
                n_state + 1,
            ), f"{new_obs.shape}, B: {batch_size}, H: {horizon}, dim(S): {n_state}"

    @staticmethod
    def trajectory_dataset(trajs: List[SampleBatch]) -> TensorDataset:
        """Concat and convert a list of trajectories into a tensor dataset."""
        dataset = TensorDataset(
            batched(trajs, SampleBatch.CUR_OBS),
            batched(trajs, SampleBatch.ACTIONS),
            batched(trajs, SampleBatch.NEXT_OBS),
        )
        assert len(dataset) == len(trajs)
        return dataset


def build_datamodule(worker: RolloutWorker, **kwargs):
    # pylint:disable=missing-function-docstring
    data_spec = DataModuleSpec(**kwargs)
    datamodule = TrajectoryData(worker, data_spec)
    datamodule.collect_trajectories()
    datamodule.setup()
    return datamodule


def test_datamodule():
    # pylint:disable=missing-function-docstring,import-outside-toplevel
    from policy import make_worker

    # Create and initialize
    worker = make_worker()
    datamodule = build_datamodule(worker)
    datamodule.check_dataloaders()


if __name__ == "__main__":
    test_datamodule()
