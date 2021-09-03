# pylint:disable=missing-docstring
import functools
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import ray
import torch
from critic import LightningQValue
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from wandb.sdk import wandb_config, wandb_run
from wandb_util import WANDB_DIR, env_info, wandb_init

from lqsvg import data
from lqsvg.envs import lqr
from lqsvg.experiment import utils as exp_utils
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, TVLinearPolicy


def make_modules(
    generator: lqr.LQGGenerator, hparams: dict
) -> Tuple[LQGModule, TVLinearPolicy, LightningQValue]:
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
    policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
        dynamics, rng=generator.rng
    )
    qvalue = LightningQValue(lqg, policy, hparams)
    return lqg, policy, qvalue


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

    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, spec: DataSpec):
        super().__init__()
        self.spec = spec

        sampler = data.environment_sampler(lqg)
        self.sample_fn = functools.partial(sampler, policy)

    def prepare_data(self) -> None:
        with torch.no_grad():
            obs, act, rew, _ = self.sample_fn(self.spec.trajectories)
        obs, act, rew = (t.align_to("H", "B", ...) for t in (obs, act, rew))
        self.tensors = (obs[:-1], act, rew, obs[1:])

    def setup(self, stage: Optional[str] = None) -> None:
        idxs = torch.randperm(self.spec.trajectories)
        split_sizes = data.train_val_sizes(self.spec.trajectories, self.spec.train_frac)
        train_traj_idxs, val_traj_idxs = torch.split(idxs, split_sizes)
        # noinspection PyTypeChecker
        train_trajs, val_trajs = (
            tuple(nt.index_select(t, "B", i) for t in self.tensors)
            for i in (train_traj_idxs, val_traj_idxs)
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
        generator = lqr.LQGGenerator(
            stationary=True,
            controllable=True,
            rng=np.random.default_rng(self.hparams.seed),
            **self.hparams.env_config,
        )
        lqg, policy, qvalue = make_modules(generator, self.hparams.as_dict())
        datamodule = DataModule(lqg, policy, DataSpec(**self.hparams.datamodule))

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
            with exp_utils.suppress_dataloader_warnings(num_workers=True, shuffle=True):
                trainer.validate(qvalue, datamodule=datamodule)
                trainer.fit(qvalue, datamodule=datamodule)
                final_eval = trainer.test(qvalue, datamodule=datamodule)

        return {tune.result.DONE: True, **final_eval[0]}


def main():
    ray.init(logging_level=logging.WARNING)

    config = {
        "wandb": {"name": "ValueGradientLearning", "mode": "online"},
        "loss": "VGL(1)",
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "polyak": tune.grid_search([0, 0.995]),
        "seed": tune.grid_search(list(range(123, 128))),
        # "seed": 123,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 50,
            "passive_eigval_range": (0.9, 1.1),
        },
        "model": {"type": tune.grid_search(["mlp", "quad"]), "hunits": (10, 10)},
        "datamodule": {
            "trajectories": 2000,
            "train_batch_size": 128,
            "val_batch_size": 128,
        },
        "trainer": dict(
            max_epochs=40,
            progress_bar_refresh_rate=0,  # don't show model training progress bar
            weights_summary=None,  # don't print summary before training
            track_grad_norm=2,
            val_check_interval=0.5,
        ),
    }
    tune.run(Experiment, config=config, num_samples=1, local_dir=WANDB_DIR)
    ray.shutdown()


if __name__ == "__main__":
    main()
