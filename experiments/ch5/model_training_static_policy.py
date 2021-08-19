# pylint:disable=missing-docstring
import functools
import logging
import os.path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import click
import numpy as np
import pytorch_lightning as pl
import ray
import torch
import wandb
from data import TrajectorySegmentDataset
from model import LightningModel
from pytorch_lightning.utilities.seed import seed_everything
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from wandb.sdk import wandb_config

import lqsvg.envs.lqr.utils as lqg_util
import lqsvg.torch.named as nt
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment import utils
from lqsvg.experiment.data import trajectory_sampler
from lqsvg.experiment.dynamics import markovian_state_sampler
from lqsvg.np_util import RNG
from lqsvg.torch.nn.policy import TVLinearPolicy

RESULTS_DIR = os.path.abspath("./results")
WANDB_DIR = RESULTS_DIR


@dataclass
class DataSpec:
    trajectories: int
    train_batch_size: int
    val_loss_batch_size: int
    val_grad_batch_size: int
    segment_len: int
    train_frac: float = 0.9

    def __post_init__(self):
        assert self.train_frac < 1.0

    def train_val_sizes(self, total: int) -> Tuple[int, int]:
        """Compute train and validation dataset sizes from total size."""
        train_samples = int(total * self.train_frac)
        val_samples = total - train_samples
        return train_samples, val_samples


class DataModule(pl.LightningDataModule):
    tensors: Tuple[Tensor, Tensor, Tensor]
    train_dataset: Dataset
    val_seg_dataset: Dataset
    val_state_dataset: Dataset

    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, spec: DataSpec):
        super().__init__()
        self.spec = spec
        assert self.spec.segment_len <= lqg.horizon, "Invalid trajectory segment length"

        sample_fn = trajectory_sampler(
            policy,
            lqg.init.sample,
            markovian_state_sampler(lqg.trans, lqg.trans.sample),
            lqg.reward,
        )
        self.sample_fn = functools.partial(sample_fn, horizon=lqg.horizon)

    def prepare_data(self) -> None:
        with torch.no_grad():
            obs, act, _, _ = self.sample_fn(sample_shape=[self.spec.trajectories])
        obs, act = (x.rename(B1="B") for x in (obs, act))
        decision_steps = torch.arange(obs.size("H") - 1).int()
        self.tensors = (
            nt.index_select(obs, "H", decision_steps),
            act,
            nt.index_select(obs, "H", decision_steps + 1),
        )
        # noinspection PyArgumentList
        assert all(t.size("B") == obs.size("B") for t in self.tensors)

    def setup(self, stage: Optional[str] = None):
        n_trajs = self.spec.trajectories
        train_traj_idxs, val_traj_idxs = torch.split(
            torch.randperm(n_trajs),
            split_size_or_sections=self.spec.train_val_sizes(n_trajs),
        )
        # noinspection PyTypeChecker
        train_trajs, val_trajs = (
            tuple(nt.index_select(t, "B", idxs) for t in self.tensors)
            for idxs in (train_traj_idxs, val_traj_idxs)
        )
        self.train_dataset = TrajectorySegmentDataset(
            *train_trajs, segment_len=self.spec.segment_len
        )
        self.val_seg_dataset = TrajectorySegmentDataset(
            *val_trajs, segment_len=self.spec.segment_len
        )
        val_obs = val_trajs[0]
        self.val_state_dataset = TensorDataset(
            nt.unnamed(val_obs.flatten(["H", "B"], "B"))
        )

    def train_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.spec.train_batch_size
        )
        return dataloader

    def val_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        # pylint:disable=arguments-differ
        # For loss evaluation
        seg_loader = DataLoader(
            self.val_seg_dataset,
            shuffle=False,
            batch_size=self.spec.val_loss_batch_size,
        )
        # For gradient estimation
        state_loader = DataLoader(
            self.val_state_dataset,
            shuffle=True,
            batch_size=self.spec.val_grad_batch_size,
        )
        return seg_loader, state_loader

    def test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        # pylint:disable=arguments-differ
        return self.val_dataloader()


def extra_tags(config: dict) -> Tuple[str, ...]:
    extra = ()
    if config["n_state"] > config["n_ctrl"]:
        extra += ("underactuated",)
    return extra


def wandb_init(
    name: str,
    config: dict,
    project="LQG-SVG",
    entity="angelovtt",
    tags: Sequence[str] = (),
    reinit=True,
    **kwargs,
) -> wandb.sdk.wandb_run.Run:
    # pylint:disable=too-many-arguments
    return wandb.init(
        name=name,
        config=config,
        project=project,
        entity=entity,
        dir=WANDB_DIR,
        tags=("ch5", utils.calver()) + extra_tags(config) + tuple(tags),
        reinit=reinit,
        **kwargs,
    )


class Experiment(tune.Trainable):
    # pylint:disable=abstract-method,too-many-instance-attributes
    run: wandb.sdk.wandb_run.Run
    rng: RNG
    generator: LQGGenerator
    lqg: LQGModule
    policy: TVLinearPolicy
    model: LightningModel
    datamodule: DataModule
    trainer: pl.Trainer

    def setup(self, config: dict):
        wandb_kwargs = config.pop("wandb")
        self.run = wandb_init(config=config, **wandb_kwargs)
        seed_everything(self.hparams.seed)
        self.rng = np.random.default_rng(self.hparams.seed)
        self.generator = LQGGenerator(
            n_state=self.hparams.n_state,
            n_ctrl=self.hparams.n_ctrl,
            horizon=self.hparams.horizon,
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            controllable=True,
            rng=self.rng,
        )
        self._make_modules()
        self._make_trainer()

    @property
    def hparams(self) -> wandb_config.Config:
        return self.run.config

    def _make_modules(self):
        with nt.suppress_named_tensor_warning():
            dynamics, cost, init = self.generator()
        lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
        policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon)
        policy.stabilize_(dynamics, rng=self.rng)
        self.lqg, self.policy = lqg, policy
        self.model = LightningModel(lqg, policy, self.hparams)
        self.datamodule = DataModule(lqg, policy, DataSpec(**self.hparams.datamodule))

    def _make_trainer(self):
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        self.trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            callbacks=[pl.callbacks.EarlyStopping("val/loss/dataloader_idx_0")],
            num_sanity_val_steps=0,  # avoid evaluating gradients in the beginning?
            checkpoint_callback=False,  # don't save last model checkpoint
            **self.hparams.trainer,
        )

    def step(self) -> dict:
        with self.run:
            self._log_env_info()
            with utils.suppress_dataloader_warnings(num_workers=True, shuffle=True):
                self.trainer.validate(self.model, datamodule=self.datamodule)
                self.trainer.fit(self.model, datamodule=self.datamodule)
                final_eval = self.trainer.test(self.model, datamodule=self.datamodule)

        return {tune.result.DONE: True, **final_eval[0]}

    def _log_env_info(self):
        dynamics = self.lqg.trans.standard_form()
        eigvals = lqg_util.stationary_eigvals(dynamics)
        tests = {
            "stability": lqg_util.isstable(eigvals=eigvals),
            "controllability": lqg_util.iscontrollable(dynamics),
        }
        self.run.summary.update(tests)
        self.run.summary.update({"passive_eigvals": wandb.Histogram(eigvals)})


def run_with_tune(name: str = "ModelSearch"):
    ray.init(logging_level=logging.WARNING)

    models = [
        # {"type": "linear"},
        # {"type": "mlp", "kwargs": {"hunits": (10, 10), "activation": "ReLU"}},
        {"type": "gru", "kwargs": {"mlp_hunits": (), "gru_hunits": (10, 10)}},
        {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
    ]
    config = {
        "wandb": {"name": name},
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "seed": None,
        "n_state": 8,
        "n_ctrl": 8,
        "horizon": 50,
        "pred_horizon": [8],
        "model": tune.grid_search(models),
        "zero_q": False,
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 128,
            "segment_len": 4,
        },
        "trainer": {
            "max_epochs": 20,
            # don't show progress bar for model training
            "progress_bar_refresh_rate": 0,
            # don't print summary before training
            "weights_summary": None,
            "track_grad_norm": 2,
            # "val_check_interval": 0.5,
            # "gpus": 1,
        },
    }
    tune.run(Experiment, config=config, num_samples=200, local_dir=RESULTS_DIR)
    ray.shutdown()


def run_simple():
    config = {
        "wandb": {"name": "Debug", "mode": "offline"},
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "seed": 42,
        "n_state": 8,
        "n_ctrl": 8,
        "horizon": 50,
        "pred_horizon": 8,
        "model": {
            "type": "gru",
            "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10, 10)},
        },
        "zero_q": False,
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 200,
            "segment_len": 8,
        },
        "trainer": dict(
            max_epochs=5,
            fast_dev_run=True,
            track_grad_norm=2,
            # overfit_batches=10,
            weights_summary="full",
            # limit_train_batches=10,
            # limit_val_batches=10,
            # profiler="simple",
            val_check_interval=0.5,
            gpus=1,
        ),
    }
    experiment = Experiment(config)
    experiment.train()


@click.command()
@click.option("--name", type=str)
@click.option("--debug/--no-debug", default=False)
def main(name: str, debug: bool):
    if debug:
        run_simple()
    else:
        run_with_tune(name)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
