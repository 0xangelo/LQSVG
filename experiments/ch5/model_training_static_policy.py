# pylint:disable=missing-docstring
import functools
import logging
import os.path
from dataclasses import dataclass
from typing import Optional, Tuple

import click
import numpy as np
import pytorch_lightning as pl
import ray
import torch
import wandb.sdk
from model import LightningModel
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from wandb_util import env_info, wandb_init

import lqsvg.torch.named as nt
from lqsvg.data import (
    TensorSeqDataset,
    markovian_state_sampler,
    train_val_sizes,
    trajectory_sampler,
)
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment import utils as exp_utils
from lqsvg.torch.nn.policy import TVLinearPolicy

RESULTS_DIR = os.path.abspath("./results")


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
        obs, act = (x.rename(B1="B").align_to("H", ...) for x in (obs, act))
        self.tensors = (obs[:-1], act, obs[1:])
        # noinspection PyArgumentList
        assert all(t.size("B") == obs.size("B") for t in self.tensors)

    def setup(self, stage: Optional[str] = None):
        n_trajs = self.spec.trajectories
        train_traj_idxs, val_traj_idxs = torch.split(
            torch.randperm(n_trajs),
            split_size_or_sections=train_val_sizes(n_trajs, self.spec.train_frac),
        )
        # noinspection PyTypeChecker
        train_trajs, val_trajs = (
            tuple(nt.index_select(t, "B", idxs) for t in self.tensors)
            for idxs in (train_traj_idxs, val_traj_idxs)
        )
        self.train_dataset = TensorSeqDataset(
            *train_trajs, seq_len=self.spec.segment_len
        )
        self.val_seg_dataset = TensorSeqDataset(
            *val_trajs, seq_len=self.spec.segment_len
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


def make_modules(
    generator: LQGGenerator, hparams: wandb.sdk.wandb_config.Config
) -> Tuple[LQGModule, TVLinearPolicy, LightningModel]:
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
    policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
        dynamics, rng=generator.rng
    )
    model = LightningModel(lqg, policy, hparams)
    return lqg, policy, model


class Experiment(tune.Trainable):
    # pylint:disable=abstract-method,too-many-instance-attributes
    run: wandb.sdk.wandb_run.Run

    def setup(self, config: dict):
        wandb_kwargs = config.pop("wandb")
        self.run = wandb_init(config=config, **wandb_kwargs)
        pl.seed_everything(self.hparams.seed)

    @property
    def hparams(self) -> wandb.sdk.wandb_config.Config:
        return self.run.config

    def step(self) -> dict:
        generator = LQGGenerator(
            n_state=self.hparams.n_state,
            n_ctrl=self.hparams.n_ctrl,
            horizon=self.hparams.horizon,
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            controllable=True,
            rng=np.random.default_rng(self.hparams.seed),
        )
        lqg, policy, model = make_modules(generator, self.hparams)
        datamodule = DataModule(lqg, policy, DataSpec(**self.hparams.datamodule))
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            callbacks=[pl.callbacks.EarlyStopping("val/loss/dataloader_idx_0")],
            num_sanity_val_steps=0,  # avoid evaluating gradients in the beginning?
            checkpoint_callback=False,  # don't save last model checkpoint
            **self.hparams.trainer,
        )

        with self.run as run:
            run.summary.update(env_info(lqg))
            run.summary.update({"trainable_parameters": model.num_parameters()})
            with exp_utils.suppress_dataloader_warnings(num_workers=True, shuffle=True):
                trainer.validate(model, datamodule=datamodule)
                trainer.fit(model, datamodule=datamodule)
                final_eval = trainer.test(model, datamodule=datamodule)

        return {tune.result.DONE: True, **final_eval[0]}


def run_with_tune(name: str = "ModelSearch"):
    ray.init(logging_level=logging.WARNING)

    # models = [
    #     {"type": "linear"},
    #     {"type": "mlp", "kwargs": {"hunits": (10, 10), "activation": "ReLU"}},
    #     {"type": "gru", "kwargs": {"mlp_hunits": (), "gru_hunits": (10, 10)}},
    #     {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
    # ]
    config = {
        "wandb": {"name": name},
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        # "seed": tune.grid_search(list(range(123, 133))),
        "seed": 123,
        "n_state": 2,
        "n_ctrl": 2,
        "horizon": 50,
        "pred_horizon": [0, 2, 4, 8],
        "model": {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
        "zero_q": False,
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 256,
            "segment_len": 4,
        },
        "trainer": {
            "max_epochs": 1000,
            # don't show progress bar for model training
            # "progress_bar_refresh_rate": 0,
            # don't print summary before training
            # "weights_summary": None,
            "weights_summary": "full",
            "track_grad_norm": 2,
            # "val_check_interval": 0.5,
            # "gpus": 1,
        },
    }
    Experiment(config).train()
    tune.run(Experiment, config=config, num_samples=1, local_dir=RESULTS_DIR)
    ray.shutdown()


def run_simple():
    config = {
        "wandb": {"name": "Debug", "mode": "offline"},
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 123,
        "n_state": 2,
        "n_ctrl": 2,
        "horizon": 50,
        "pred_horizon": [0, 2, 4, 8],
        "model": {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
        "zero_q": False,
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 256,
            "segment_len": 4,
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
