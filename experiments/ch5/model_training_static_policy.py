# pylint:disable=missing-docstring
import functools
import logging
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
from wandb_util import WANDB_DIR, env_info, wandb_init

from lqsvg import data
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.experiment import utils as exp_utils
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, TVLinearPolicy
from lqsvg.types import DeterministicPolicy


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

    def __init__(self, lqg: LQGModule, behavior: DeterministicPolicy, spec: DataSpec):
        super().__init__()
        self.spec = spec
        assert self.spec.seq_len <= lqg.horizon, "Invalid trajectory segment length"

        sampler = data.environment_sampler(lqg)
        self.sample_fn = functools.partial(sampler, behavior)

    def prepare_data(self) -> None:
        with torch.no_grad():
            obs, act, _, _ = self.sample_fn(self.spec.trajectories)
        obs = obs.align_to("H", "B", ...)
        act = act.align_to("H", "B", ...)
        self.tensors = (obs[:-1], act, obs[1:])

    def setup(self, stage: Optional[str] = None):
        spec = self.spec
        train_traj_idxs, val_traj_idxs = torch.split(
            torch.randperm(spec.trajectories),
            split_size_or_sections=data.train_val_sizes(
                spec.trajectories, spec.train_frac
            ),
        )
        # noinspection PyTypeChecker
        train_trajs, val_trajs = (
            tuple(nt.index_select(t, "B", idxs) for t in self.tensors)
            for idxs in (train_traj_idxs, val_traj_idxs)
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


def gaussian_behavior(
    policy: TVLinearPolicy, exploration: dict, seed: int
) -> DeterministicPolicy:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sigma = exploration["action_noise_sigma"]

    def behavior(obs: Tensor) -> Tensor:
        act = policy(obs)
        noise = torch.randn(size=act.shape, generator=generator, device=act.device)
        return act + noise * sigma

    return behavior


def behavior_policy(
    policy: TVLinearPolicy, exploration: dict, seed: int
) -> DeterministicPolicy:
    kind = exploration["type"]
    if kind is None:
        return policy
    if kind == "gaussian":
        return gaussian_behavior(policy, exploration, seed)
    raise ValueError(f"Unknown exploration type '{kind}'")


def make_modules(
    generator: LQGGenerator, hparams: dict
) -> Tuple[LQGModule, TVLinearPolicy, DeterministicPolicy, LightningModel]:
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
    policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
        dynamics, rng=generator.rng
    )
    behavior = behavior_policy(policy, hparams["exploration"], hparams["seed"])
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
        generator = LQGGenerator(
            stationary=True,
            controllable=True,
            rng=np.random.default_rng(self.hparams.seed),
            **self.hparams.env_config,
        )
        lqg, _, behavior, model = make_modules(generator, self.hparams.as_dict())
        datamodule = DataModule(lqg, behavior, DataSpec(**self.hparams.datamodule))
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            callbacks=[pl.callbacks.EarlyStopping("val/loss")],
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

    models = [
        {"type": "linear"},
        {"type": "mlp", "kwargs": {"hunits": (10, 10), "activation": "ReLU"}},
        {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
    ]
    config = {
        "wandb": {"name": name},
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": tune.grid_search(list(range(128, 143))),
        # "seed": 124,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 50,
            "passive_eigval_range": (0.9, 1.1),
        },
        "exploration": {
            "type": tune.grid_search([None, "gaussian"]),
            "action_noise_sigma": 0.3,
        },
        "model": tune.grid_search(models),
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
    tune.run(Experiment, config=config, num_samples=1, local_dir=WANDB_DIR)
    ray.shutdown()


def run_simple():
    config = {
        "wandb": {"name": "Debug", "mode": "offline"},
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 123,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 50,
            "passive_eigval_range": (0.9, 1.1),
        },
        "pred_horizon": [0, 2, 4, 8],
        "zero_q": False,
        "model": {"type": "gru", "kwargs": {"mlp_hunits": (10,), "gru_hunits": (10,)}},
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 256,
            "seq_len": 4,
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
