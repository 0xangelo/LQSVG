# pylint:disable=missing-docstring
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import ray
import torch
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from wandb.sdk import wandb_config

import lqsvg.envs.lqr.utils as lqg_util
import lqsvg.experiment.utils as utils
import lqsvg.torch.named as nt
import wandb
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment.data import split_dataset
from lqsvg.experiment.estimators import MAAC, AnalyticSVG
from lqsvg.np_util import RNG
from lqsvg.policy.modules import QuadQValue, TVLinearPolicy
from lqsvg.policy.modules.transition import (
    LinearDiagDynamicsModel,
    SegmentStochasticModel,
)

BATCH = Tuple[Tensor, Tensor, Tensor]


class LightningModel(pl.LightningModule):
    model: SegmentStochasticModel
    estimator: MAAC
    lqg: LQGModule
    policy: TVLinearPolicy
    qval: QuadQValue
    gold_standard: Tuple[Tensor, lqr.Linear]

    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy):
        super().__init__()
        self.lqg = lqg
        self.policy = policy
        self.qval = QuadQValue(lqg.n_state + lqg.n_ctrl, lqg.horizon)
        self.qval.match_policy_(
            policy.standard_form(),
            lqg.trans.standard_form(),
            lqg.reward.standard_form(),
        )
        self.model = LinearDiagDynamicsModel(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, stationary=True
        )
        self.estimator = MAAC(policy, self.model, lqg.reward, self.qval)
        self.compute_gold_standards()

    def compute_gold_standards(self):
        self.gold_standard = AnalyticSVG(self.policy, self.lqg)()

    # noinspection PyArgumentList
    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Log-likelihood of (batched) trajectory segment."""
        return self.model.seg_log_prob(obs, act, new_obs) / obs.size("H")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def _compute_loss_on_batch(self, batch: BATCH, _: int) -> Tensor:
        obs, act, new_obs = (x.refine_names("B", "H", "R") for x in batch)
        return -self(obs, act, new_obs).mean()

    def training_step(self, batch: BATCH, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: BATCH, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        grad_acc = self._compute_grad_acc_on_batch(batch)
        self.log("val/loss", loss)
        self.log("val/grad_acc", grad_acc)
        return loss

    def test_step(self, batch: BATCH, batch_idx: int):
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        grad_acc = self._compute_grad_acc_on_batch(batch)
        self.log("test/loss", loss)
        self.log("test/grad_acc", grad_acc)

    def _compute_grad_acc_on_batch(self, batch: BATCH) -> Tensor:
        obs, _, _ = batch
        obs = obs.flatten(["B", "H"], "B")
        estim = self.estimator(
            obs,
        )


@dataclass
class DataSpec:
    train_val_split: (float, float)
    train_batch_size: int
    val_batch_size: int


class DataModule(pl.LightningDataModule):
    full_dataset: Dataset
    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, spec: DataSpec):
        super().__init__()
        self.spec = spec

    def setup(self, stage: Optional[str] = None):
        del stage
        self.train_dataset, self.val_dataset, _ = split_dataset(
            self.full_dataset, self.spec.train_val_split + (0,)
        )

    def train_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        spec = self.spec
        dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=spec.train_batch_size
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        dataloader = DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.spec.val_batch_size
        )
        return dataloader


# noinspection PyAbstractClass
class Experiment(tune.Trainable):
    # pylint:disable=abstract-method,too-many-instance-attributes
    run: wandb.sdk.wandb_run.Run
    rng: RNG
    generator: LQGGenerator
    lqg: LQGModule
    policy: TVLinearPolicy
    model: LightningModel
    datamodule: pl.LightningDataModule
    trainer: pl.Trainer

    def setup(self, config: dict):
        self._init_wandb(config)
        self.rng = np.random.default_rng(self.hparams.seed)
        self.make_generator()
        self.make_modules()
        self.make_trainer()

    def _init_wandb(self, config: dict):
        self.run = wandb.init(
            name="LinearML",
            config=config,
            project="LQG-SVG",
            entity="angelovtt",
            tags=["ch5"],
            reinit=True,
            mode="online",
        )

    @property
    def hparams(self) -> wandb_config.Config:
        return self.run.config

    def make_generator(self):
        self.generator = LQGGenerator(
            n_state=self.hparams.n_state,
            n_ctrl=self.hparams.n_ctrl,
            horizon=self.hparams.horizon,
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            controllable=True,
            rng=self.rng,
        )

    def make_modules(self):
        with nt.suppress_named_tensor_warning():
            dynamics, cost, init = self.generator()
        lqg = LQGModule.from_existing(dynamics, cost, init)
        policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon)
        policy.stabilize_(dynamics, rng=self.rng)
        self.lqg, self.policy = lqg, policy
        self.model = LightningModel(lqg, policy)

    def make_trainer(self):
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        self.trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            num_sanity_val_steps=2,
            max_epochs=self.hparams.max_epochs,
            progress_bar_refresh_rate=0,  # don't show progress bar for model training
            weights_summary=None,  # don't print summary before training
            checkpoint_callback=False,  # don't save last model checkpoint
        )

    def step(self) -> dict:
        with self.run:
            self.log_env_info()
            self.build_dataset()
            with utils.suppress_dataloader_warning():
                self.trainer.fit(self.model, datamodule=self.datamodule)
                final_eval = self.trainer.test(self.model, datamodule=self.datamodule)

        return {tune.result.DONE: True, **final_eval}

    def log_env_info(self):
        dynamics = self.lqg.trans.standard_form()
        eigvals = lqg_util.stationary_eigvals(dynamics)
        tests = {
            "stability": lqg_util.isstable(eigvals=eigvals),
            "controllability": lqg_util.iscontrollable(dynamics),
        }
        self.run.summary.update(tests)
        self.run.summary.update({"passive_eigvals": wandb.Histogram(eigvals)})

    def build_dataset(self):
        pass

    def cleanup(self):
        self.run.finish()


def main():
    ray.init()


if __name__ == "__main__":
    main()
