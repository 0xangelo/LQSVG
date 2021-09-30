# pylint:disable=missing-docstring
import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import click
import pytorch_lightning as pl
import ray
import torch
from actor import behavior_policy
from critic import LightningQValue, LightningReward, TDBatch
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from wandb.sdk import wandb_config, wandb_run
from wandb_util import WANDB_DIR, env_info, wandb_init

from lqsvg import data, lightning
from lqsvg.envs import lqr
from lqsvg.random import RNG, make_rng
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, TVLinearPolicy
from lqsvg.types import DeterministicPolicy


def make_modules(
    rng: RNG, hparams: dict
) -> Tuple[LQGModule, TVLinearPolicy, DeterministicPolicy, LightningQValue]:
    generator = lqr.LQGGenerator(
        stationary=True, controllable=True, rng=rng.numpy, **hparams["env_config"]
    )
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
    policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
        dynamics, rng.numpy
    )
    behavior = behavior_policy(policy, hparams["exploration"], rng.torch)
    qvalue = LightningQValue(lqg, policy, hparams)
    return lqg, policy, behavior, qvalue


@dataclass
class DataSpec:
    trajectories: int
    train_batch_size: int
    val_batch_size: int
    train_frac: float = 0.9


class DataModule(pl.LightningDataModule):
    tensors: Tuple[Tensor, Tensor, Tensor, Tensor]
    train_dataset: TensorDataset
    val_dataset: TensorDataset

    def __init__(
        self,
        lqg: LQGModule,
        policy: DeterministicPolicy,
        spec: DataSpec,
        rng: torch.Generator,
    ):
        super().__init__()
        self.spec = spec
        self.rng = rng

        sampler = data.environment_sampler(lqg)
        self.sample_fn = functools.partial(sampler, policy)

    def prepare_data(self) -> None:
        with torch.no_grad():
            obs, act, rew, _ = self.sample_fn(self.spec.trajectories)
        obs, act, rew = (t.align_to("H", "B", ...) for t in (obs, act, rew))
        self.tensors = (obs[:-1], act, rew, obs[1:])

    def setup(self, stage: Optional[str] = None) -> None:
        train_trajs, val_trajs = data.split_along_batch_dim(
            self.tensors,
            data.train_val_sizes(self.spec.trajectories, self.spec.train_frac),
            self.rng,
        )
        train_trans, val_trans = (
            tuple(t.flatten(["H", "B"], "B") for t in tensors)
            for tensors in (train_trajs, val_trajs)
        )
        self.train_dataset = TensorDataset(*nt.unnamed(*train_trans))
        self.val_dataset = TensorDataset(*nt.unnamed(*val_trans))

    def train_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        return DataLoader(
            self.train_dataset, batch_size=self.spec.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        # pylint:disable=arguments-differ
        # Shuffle for SVG estimator
        return DataLoader(
            self.val_dataset, batch_size=self.spec.val_batch_size, shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def on_after_batch_transfer(self, batch: TDBatch, _: int) -> TDBatch:
        obs, act, rew, new_obs = (t.refine_names("B", ...) for t in batch)
        obs, act, new_obs = nt.vector(obs, act, new_obs)
        return obs, act, rew, new_obs


class Experiment(tune.Trainable):
    _run: wandb_run.Run = None

    @property
    def run(self) -> wandb_run.Run:
        if self._run is None:
            config = self.config.copy()
            wandb_kwargs = config.pop("wandb")
            self._run = wandb_init(config=config, **wandb_kwargs)
        return self._run

    @property
    def hparams(self) -> wandb_config.Config:
        return self.run.config

    def step(self) -> dict:
        rng = make_rng(self.hparams.seed)
        lqg, _, behavior, qvalue = make_modules(rng, self.hparams.as_dict())
        datamodule = DataModule(
            lqg, behavior, DataSpec(**self.hparams.datamodule), rng.torch
        )

        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            num_sanity_val_steps=0,  # avoid evaluating gradients in the beginning?
            checkpoint_callback=False,  # don't save last model checkpoint
            **self.hparams.trainer,
        )

        with self.run as run:
            run.summary.update(env_info(lqg))
            run.summary.update({"trainable_parameters": qvalue.num_parameters()})
            with lightning.suppress_dataloader_warnings(num_workers=True, shuffle=True):
                trainer.validate(qvalue, datamodule=datamodule)
                trainer.fit(qvalue, datamodule=datamodule)
                final_eval = trainer.test(qvalue, datamodule=datamodule)

        return {tune.result.DONE: True, **final_eval[0]}


class RewardExperiment(tune.Trainable):
    _run: wandb_run.Run = None

    @property
    def run(self) -> wandb_run.Run:
        if self._run is None:
            config = self.config.copy()
            wandb_kwargs = config.pop("wandb")
            self._run = wandb_init(config=config, **wandb_kwargs)
        return self._run

    @property
    def hparams(self) -> wandb_config.Config:
        return self.run.config

    def step(self) -> dict:
        # ===== MAKE MODELS =====
        rng = make_rng(self.hparams.seed)
        generator = lqr.LQGGenerator(
            stationary=True,
            controllable=True,
            rng=rng.numpy,
            **self.hparams["env_config"],
        )
        with nt.suppress_named_tensor_warning():
            dynamics, cost, init = generator()

        lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
        policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
            dynamics, rng.numpy
        )
        behavior = behavior_policy(policy, self.hparams["exploration"], rng.torch)
        model = LightningReward(lqg, policy, self.hparams.as_dict(), rng)
        # ===== =====

        datamodule = DataModule(
            lqg, behavior, DataSpec(**self.hparams.datamodule), rng.torch
        )

        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            num_sanity_val_steps=0,  # avoid evaluating gradients in the beginning?
            checkpoint_callback=False,  # don't save last model checkpoint
            **self.hparams.trainer,
        )

        with self.run as run:
            run.summary.update(env_info(lqg))
            run.summary.update({"trainable_parameters": model.num_parameters()})
            with lightning.suppress_dataloader_warnings(num_workers=True, shuffle=True):
                trainer.validate(model, datamodule=datamodule)
                trainer.fit(model, datamodule=datamodule)
                final_eval = trainer.test(model, datamodule=datamodule)

        return {tune.result.DONE: True, **final_eval[0]}


@click.group()
def main():
    pass


def base_config() -> dict:
    return {
        "loss": "VGL(1)",
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "polyak": 0.995,
        "seed": 123,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 50,
            "passive_eigval_range": (0.9, 1.1),
        },
        "exploration": {"type": "gaussian", "action_noise_sigma": 0.3},
        "model": {"type": "quad"},
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_batch_size": 256,
        },
        "trainer": dict(
            max_epochs=40,
            progress_bar_refresh_rate=0,  # don't show model training progress bar
            weights_summary=None,  # don't print summary before training
            # track_grad_norm=2,
            val_check_interval=0.5,
        ),
    }


@main.command()
def reward():
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    ray.init(logging_level=logging.WARNING)

    config = {
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 50,
            "passive_eigval_range": (0.9, 1.1),
        },
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_batch_size": 256,
        },
        "trainer": dict(
            max_epochs=40,
            progress_bar_refresh_rate=0,  # don't show model training progress bar
            weights_summary=None,  # don't print summary before training
            # track_grad_norm=2,
            val_check_interval=0.5,
        ),
        "wandb": {"name": "RewardLearning", "mode": "online"},
        "exploration": {
            "type": tune.grid_search(["gaussian", None]),
            "action_noise_sigma": 0.3,
        },
        "seed": tune.grid_search(list(range(123, 133))),
    }
    tune.run(
        RewardExperiment,
        config=config,
        num_samples=1,
        local_dir=WANDB_DIR,
        callbacks=[],
    )
    ray.shutdown()


@main.command()
def sweep():
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    ray.init(logging_level=logging.WARNING)

    config = {
        **base_config(),
        "wandb": {"name": "ValueLearning", "mode": "online"},
        "loss": "TD(1)",
        "polyak": tune.grid_search([0, 0.995]),
        "exploration": {
            "type": tune.grid_search(["gaussian", None]),
            "action_noise_sigma": 0.3,
        },
        "seed": tune.grid_search(list(range(123, 133))),
        "model": {"type": tune.grid_search(["quad"]), "hunits": (10, 10)},
    }
    tune.run(
        Experiment, config=config, num_samples=1, local_dir=WANDB_DIR, callbacks=[]
    )
    ray.shutdown()


@main.command()
def debug():
    config = {
        **base_config(),
        "wandb": {"name": "DEBUG", "mode": "disabled"},
        "loss": "TD(1)",
        "trainer": dict(
            track_grad_norm=2,
            fast_dev_run=True,
            weights_summary="full",
        ),
    }
    Experiment(config).train()


if __name__ == "__main__":
    main()
