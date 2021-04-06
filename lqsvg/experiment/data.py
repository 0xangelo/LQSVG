"""Utilities for data collection in LQG envs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from dataclasses_json import DataClassJsonMixin
from ray.rllib import RolloutWorker, SampleBatch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

import lqsvg
import lqsvg.torch.named as nt
from lqsvg.envs.lqr.gym import TorchLQGMixin

from .tqdm_util import collect_with_progress
from .utils import group_batch_episodes
from .worker import make_worker


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


class TrajectoryData(pl.LightningDataModule):
    """Data module for on-policy trajectory data in LQG envs."""

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
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def collect_trajectories(self, prog: bool = True):
        """Sample trajectories with rollout worker and build dataset."""
        sample_batch = collect_with_progress(
            self.worker, self.spec.total_trajs, prog=prog
        )
        sample_batch = group_batch_episodes(sample_batch)

        trajs = sample_batch.split_by_episode()
        traj_counts = [t.count for t in trajs]
        assert all(c == self.worker.env.horizon for c in traj_counts), traj_counts
        total_ts = sum(t.count for t in trajs)
        assert total_ts == self.spec.total_trajs * self.worker.env.horizon, total_ts

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

    def setup(self, stage: Optional[str] = None):
        del stage
        spec = self.spec
        train_trajs = int(spec.train_val_test_split[0] * spec.total_trajs)
        holdout_trajs = spec.total_trajs - train_trajs
        val_trajs = int(
            (spec.train_val_test_split[1] / sum(spec.train_val_test_split[1:]))
            * holdout_trajs
        )
        test_trajs = holdout_trajs - val_trajs

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, (train_trajs, val_trajs, test_trajs)
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

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def prepare_data(self, *args, **kwargs):
        pass


def build_datamodule(worker: RolloutWorker, **kwargs):
    # pylint:disable=missing-function-docstring
    data_spec = DataModuleSpec(**kwargs)
    datamodule = TrajectoryData(worker, data_spec)
    return datamodule


def check_dataloaders(datamodule):
    """For testing only."""
    env = datamodule.worker.env
    horizon = env.horizon
    n_state = env.n_state
    n_ctrl = env.n_ctrl

    for loader in (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ):
        batch_size = min(datamodule.spec.batch_size, len(loader.dataset))
        batch = next(iter(loader))
        assert len(batch) == 3
        obs, act, new_obs = batch

        def tensor_info(tensor: Tensor, dim: int, batch_size: int = batch_size) -> str:
            return f"{tensor.shape}, B: {batch_size}, H: {horizon}, dim: {dim}"

        assert obs.shape == (batch_size, horizon, n_state + 1), tensor_info(
            obs, n_state
        )
        assert act.shape == (batch_size, horizon, n_ctrl), tensor_info(act, n_ctrl)
        assert new_obs.shape == (batch_size, horizon, n_state + 1), tensor_info(
            new_obs, n_state
        )


def test_datamodule():
    # pylint:disable=missing-function-docstring
    lqsvg.register_all()
    # Create and initialize
    with nt.suppress_named_tensor_warning():
        worker = make_worker(
            env_config=dict(n_state=2, n_ctrl=2, horizon=100, num_envs=10)
        )
    datamodule = build_datamodule(worker, total_trajs=100)
    datamodule.collect_trajectories()
    datamodule.setup()
    check_dataloaders(datamodule)


if __name__ == "__main__":
    test_datamodule()
