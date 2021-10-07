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
import wandb.sdk
from ray import tune
from ray.tune.logger import NoopLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lqsvg import data, lightning
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.random import RNG, make_rng
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, TVLinearPolicy
from lqsvg.types import DeterministicPolicy

# isort: off
# pylint:disable=wrong-import-order
from actor import behavior_policy
from model import LightningModel, ValBatch
from wandb_util import WANDB_DIR, env_info, wandb_init


@dataclass
class DataSpec:
    trajectories: int
    train_batch_size: int
    val_loss_batch_size: int
    val_grad_batch_size: int
    seq_len: int
    train_frac: float = 0.9

    def __post_init__(self):
        assert self.train_frac < 1.0


class DataModule(pl.LightningDataModule):
    tensors: Tuple[Tensor, Tensor, Tensor]
    train_dataset: Dataset
    val_seq_dataset: Dataset
    val_state_dataset: Dataset

    def __init__(
        self,
        lqg: LQGModule,
        behavior: DeterministicPolicy,
        spec: DataSpec,
        rng: torch.Generator,
    ):
        super().__init__()
        self.spec = spec
        assert self.spec.seq_len <= lqg.horizon, "Invalid trajectory segment length"

        sampler = data.environment_sampler(lqg)
        self.sample_fn = functools.partial(sampler, behavior)
        self.rng = rng

    def prepare_data(self) -> None:
        with torch.no_grad():
            obs, act, _, _ = self.sample_fn(self.spec.trajectories)
        obs = obs.align_to("H", "B", ...)
        act = act.align_to("H", "B", ...)
        self.tensors = (obs[:-1], act, obs[1:])

    def setup(self, stage: Optional[str] = None):
        spec = self.spec
        train_trajs, val_trajs = data.split_along_batch_dim(
            self.tensors,
            data.train_val_sizes(spec.trajectories, spec.train_frac),
            self.rng,
        )
        self.train_dataset = data.TensorSeqDataset(*train_trajs, seq_len=spec.seq_len)
        self.val_seq_dataset = data.TensorSeqDataset(*val_trajs, seq_len=spec.seq_len)
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
        seq_loader = DataLoader(
            self.val_seq_dataset,
            shuffle=False,
            batch_size=self.spec.val_loss_batch_size,
        )
        # For gradient estimation
        state_loader = DataLoader(
            self.val_state_dataset,
            shuffle=True,
            batch_size=self.spec.val_grad_batch_size,
        )
        return seq_loader, state_loader

    def test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        # pylint:disable=arguments-differ
        return self.val_dataloader()

    def on_after_batch_transfer(self, batch: ValBatch, dataloader_idx: int) -> ValBatch:
        if dataloader_idx == 0:
            return tuple(t.refine_names("B", "H", "R") for t in batch)

        return tuple(t.refine_names("B", "R") for t in batch)


def make_modules(
    rng: RNG, hparams: dict
) -> Tuple[LQGModule, TVLinearPolicy, DeterministicPolicy, LightningModel]:
    generator = LQGGenerator(
        stationary=True, controllable=True, rng=rng.numpy, **hparams["env_config"]
    )
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
    policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
        dynamics, rng=rng.numpy
    )
    behavior = behavior_policy(policy, hparams["exploration"], rng.torch)
    model = LightningModel(lqg, policy, hparams)
    return lqg, policy, behavior, model


class Experiment(tune.Trainable):
    _run: wandb.sdk.wandb_run.Run = None

    def setup(self, config: dict):
        pl.seed_everything(config["seed"])

    @property
    def run(self) -> wandb.sdk.wandb_run.Run:
        if self._run is None:
            config = self.config.copy()
            wandb_kwargs = config.pop("wandb")
            self._run = wandb_init(config=config, **wandb_kwargs)
        return self._run

    @property
    def hparams(self) -> wandb.sdk.wandb_config.Config:
        return self.run.config

    def step(self) -> dict:
        rng = make_rng(self.hparams.seed)
        lqg, _, behavior, model = make_modules(rng, self.hparams.as_dict())
        datamodule = DataModule(
            lqg, behavior, DataSpec(**self.hparams.datamodule), rng.torch
        )
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            # callbacks=[pl.callbacks.EarlyStopping("val/loss")],
            num_sanity_val_steps=0,  # avoid evaluating gradients in the beginning?
            checkpoint_callback=False,  # don't save last model checkpoint
            **self.hparams.trainer,
        )

        with self.run as run:
            run.summary.update(env_info(lqg))
            run.summary.update({"trainable_parameters": model.num_parameters()})
            with lightning.suppress_dataloader_warnings(num_workers=True, shuffle=True):
                trainer.validate(model, datamodule=datamodule, verbose=False)
                trainer.fit(model, datamodule=datamodule)
                final_eval = trainer.test(model, datamodule=datamodule, verbose=False)

        return {tune.result.DONE: True, **final_eval[0]}


@click.group()
def main():
    pass


def base_config() -> dict:
    return {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 124,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 50,
            "passive_eigval_range": (0.9, 1.1),
        },
        "exploration": {"type": "gaussian", "action_noise_sigma": 0.3},
        "model": {"type": "linear"},
        "pred_horizon": [0, 2, 4, 8],
        "zero_q": False,
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 256,
            "seq_len": 4,
        },
        "trainer": dict(
            max_epochs=40,
            progress_bar_refresh_rate=0,  # don't show model training progress bar
            weights_summary=None,  # don't print summary before training
            track_grad_norm=2,
        ),
    }


@main.command()
@click.option("--name", type=str)
def sweep(name: str = "ModelSearch"):
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["WANDB_SILENT"] = "true"
    ray.init(logging_level=logging.WARNING)

    models = [
        {"type": "linear"},
        {"type": "mlp", "kwargs": {"hunits": (10, 10), "activation": "ReLU"}},
        {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
    ]
    config = {
        **base_config(),
        "wandb": {"name": name, "tags": ["ModelBasedPrediction"]},
        "seed": tune.grid_search(list(range(138, 143))),
        "exploration": {
            "type": tune.grid_search([None, "gaussian"]),
            "action_noise_sigma": 0.3,
        },
        "model": tune.grid_search(models),
    }
    tune.run(
        Experiment, config=config, num_samples=1, local_dir=WANDB_DIR, callbacks=[]
    )
    ray.shutdown()


@main.command()
def debug():
    config = {
        **base_config(),
        "wandb": {"name": "Debug", "mode": "disabled"},
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
            # gpus=1,
        ),
    }
    Experiment(
        config, logger_creator=functools.partial(NoopLogger, logdir=os.devnull)
    ).train()


if __name__ == "__main__":
    main()
