# pylint:disable=missing-docstring
import functools
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from data import train_val_sizes
from model import LightningQValue
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from wandb.sdk import wandb_config, wandb_run
from wandb_util import env_info, wandb_init

from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment import utils as exp_utils
from lqsvg.experiment.data import trajectory_sampler
from lqsvg.experiment.dynamics import markovian_state_sampler
from lqsvg.torch import named as nt
from lqsvg.torch.nn.policy import TVLinearPolicy


def make_modules(
    generator: LQGGenerator, hparams: wandb_config.Config
) -> Tuple[LQGModule, TVLinearPolicy, LightningQValue]:
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init)
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

        sample_fn = trajectory_sampler(
            policy,
            lqg.init.sample,
            markovian_state_sampler(lqg.trans, lqg.trans.sample),
            lqg.reward,
        )
        self.sample_fn = functools.partial(sample_fn, horizon=lqg.horizon)

    def prepare_data(self) -> None:
        with torch.no_grad():
            obs, act, rew, _ = self.sample_fn(sample_shape=[self.spec.trajectories])
        obs, act, rew = (t.rename(B1="B").align_to("H", ...) for t in (obs, act, rew))
        self.tensors = (obs[:-1], act, rew, obs[1:])

    def setup(self, stage: Optional[str] = None) -> None:
        idxs = torch.randperm(self.spec.trajectories)
        split_sizes = train_val_sizes(self.spec.trajectories, self.spec.train_frac)
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
    run: wandb_run.Run

    def setup(self, config: dict):
        wandb_kwargs = config.pop("wandb")
        self.run = wandb_init(config=config, **wandb_kwargs)
        pl.seed_everything(self.hparams.seed)

    @property
    def hparams(self) -> wandb_config.Config:
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
        lqg, policy, qvalue = make_modules(generator, self.hparams)
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
    pass


if __name__ == "__main__":
    main()
