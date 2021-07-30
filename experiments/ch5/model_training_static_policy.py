# pylint:disable=missing-docstring
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import click
import numpy as np
import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.utilities.seed import seed_everything
from ray import tune
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from wandb.sdk import wandb_config

import lqsvg.envs.lqr.utils as lqg_util
import lqsvg.torch.named as nt
import wandb
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment import utils
from lqsvg.experiment.analysis import gradient_accuracy
from lqsvg.experiment.estimators import MAAC, AnalyticSVG, MonteCarloSVG
from lqsvg.np_util import RNG
from lqsvg.policy.modules import QuadQValue, TVLinearPolicy
from lqsvg.policy.modules.transition import (
    LinearDiagDynamicsModel,
    MLPDynamicsModel,
    SegmentStochasticModel,
)

Batch = Tuple[Tensor, Tensor, Tensor]
ValBatch = Union[Batch, Sequence[Tensor]]


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    model: SegmentStochasticModel
    estimator: MAAC
    lqg: LQGModule
    policy: TVLinearPolicy
    qval: QuadQValue
    gold_standard: Tuple[Tensor, lqr.Linear]

    def __init__(
        self, lqg: LQGModule, policy: TVLinearPolicy, hparams: wandb_config.Config
    ):
        super().__init__()
        self.lqg = lqg
        self.policy = policy
        self.hparams.update(hparams)

        # NN modules
        self.qval = QuadQValue(lqg.n_state + lqg.n_ctrl, lqg.horizon)
        self.qval.match_policy_(
            policy.standard_form(),
            lqg.trans.standard_form(),
            lqg.reward.standard_form(),
        )
        self.qval.requires_grad_(False)
        self.model = self.make_model()

        # Gradients
        self.estimator = MAAC(policy, self.model, lqg.reward, self.qval)
        self.compute_gold_standards()

    def make_model(self) -> SegmentStochasticModel:
        if self.hparams.model["type"] == "linear":
            return LinearDiagDynamicsModel(
                self.lqg.n_state, self.lqg.n_ctrl, self.lqg.horizon, stationary=True
            )
        return MLPDynamicsModel(
            self.lqg.n_state,
            self.lqg.n_ctrl,
            self.lqg.horizon,
            **self.hparams.model["kwargs"]
        )

    def compute_gold_standards(self):
        self.gold_standard = AnalyticSVG(self.policy, self.lqg)()

    # noinspection PyArgumentList
    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Log-likelihood of (batched) trajectory segment."""
        # pylint:disable=arguments-differ
        return self.model.seg_log_prob(obs, act, new_obs) / obs.size("H")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    # noinspection PyTypeChecker
    @staticmethod
    def _refine_batch(batch: Batch) -> Batch:
        return tuple(x.refine_names("B", "H", "R") for x in batch)

    def _compute_loss_on_batch(self, batch: Batch) -> Tensor:
        obs, act, new_obs = batch
        return -self(obs, act, new_obs).mean()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        batch = self._refine_batch(batch)
        loss = self._compute_loss_on_batch(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self._compute_and_log_loss_or_grad_acc(batch, dataloader_idx, stage="val")

    def test_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self._compute_and_log_loss_or_grad_acc(batch, dataloader_idx, stage="test")

    def _compute_and_log_loss_or_grad_acc(
        self, batch: ValBatch, dataloader_idx: int, *, stage: str
    ):
        if dataloader_idx == 0:
            # Compute log-prob of traj segments
            loss = self._compute_loss_on_batch(self._refine_batch(batch))
            self.log(stage + "/loss", loss)
        else:
            # Compute gradient acc using uniformly sampled states
            grad_acc = self._compute_grad_acc_on_batch(batch[0].refine_names("B", "R"))
            self.log(stage + "/grad_acc", grad_acc)

    @torch.enable_grad()
    def _compute_grad_acc_on_batch(self, batch: Tensor) -> Tensor:
        obs = batch
        _, svg = self.estimator(obs, n_steps=self.hparams.pred_horizon)
        return torch.as_tensor(gradient_accuracy([svg], self.gold_standard[1]))


class TrajectorySegmentDataset(Dataset):
    # noinspection PyArgumentList
    def __init__(self, obs: Tensor, act: Tensor, new_obs: Tensor, segment_len: int):
        # Pytorch Lightning deepcopies the dataloader when using overfit_batches=True
        # Deepcopying is incompatible with named tensors for some reason
        self.tensors = nt.unnamed(
            *(x.align_to("B", "H", ...) for x in (obs, act, new_obs))
        )
        self.segment_len = segment_len
        horizon: int = obs.size("H")
        trajs: int = obs.size("B")
        self.segs_per_traj = horizon - segment_len + 1
        self._len = trajs * self.segs_per_traj

    def __getitem__(self, index) -> Batch:
        traj_idx = index // self.segs_per_traj
        timestep_start = index % self.segs_per_traj
        # noinspection PyTypeChecker
        return tuple(
            t[traj_idx, timestep_start : timestep_start + self.segment_len]
            for t in self.tensors
        )

    def __len__(self) -> int:
        return self._len


@dataclass
class DataSpec:
    train_frac: float
    train_batch_size: int
    val_loss_batch_size: int
    val_grad_batch_size: int
    segment_len: int

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

    def __init__(self, spec: DataSpec):
        super().__init__()
        self.spec = spec

    def build(self, lqg: LQGModule, policy: TVLinearPolicy, trajectories: int):
        assert self.spec.segment_len <= lqg.horizon, "Invalid trajectory segment length"
        rollout = MonteCarloSVG(policy, lqg)
        with torch.no_grad():
            obs, act, _, new_obs, _ = rollout.rsample_trajectory(
                torch.Size([trajectories])
            )
        obs, act, new_obs = (x.rename(B1="B") for x in (obs, act, new_obs))
        self.tensors = (obs, act, new_obs)
        # noinspection PyArgumentList
        assert all(t.size("B") == obs.size("B") for t in self.tensors)

    def setup(self, stage: Optional[str] = None):
        # noinspection PyArgumentList
        n_trajs = self.tensors[0].size("B")
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


# noinspection PyAbstractClass
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
        self._init_wandb(config)
        seed_everything(self.hparams.seed)
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
            tags=["ch5", utils.calver()],
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
        lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)
        policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon)
        policy.stabilize_(dynamics, rng=self.rng)
        self.lqg, self.policy = lqg, policy
        self.model = LightningModel(lqg, policy, self.hparams)
        self.datamodule = DataModule(
            DataSpec(train_frac=0.9, **self.hparams.datamodule)
        )

    def make_trainer(self):
        logger = pl.loggers.WandbLogger(
            save_dir=self.run.dir, log_model=False, experiment=self.run
        )
        self.trainer = pl.Trainer(
            default_root_dir=self.run.dir,
            logger=logger,
            callbacks=[pl.callbacks.EarlyStopping("val/loss/dataloader_idx_0")],
            num_sanity_val_steps=0,  # avoid evaluating gradients in the beginning?
            checkpoint_callback=False,  # don't save last model checkpoint
            **self.hparams.trainer
        )

    def step(self) -> dict:
        with self.run:
            self.log_env_info()
            self.build_dataset()
            with utils.suppress_dataloader_warning():
                self.trainer.fit(self.model, datamodule=self.datamodule)
                final_eval = self.trainer.test(self.model, datamodule=self.datamodule)

        return {tune.result.DONE: True, **final_eval[0]}

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
        self.datamodule.build(
            self.lqg, self.policy, trajectories=self.hparams.trajectories
        )

    def cleanup(self):
        self.run.finish()


def run_with_tune():
    ray.init(logging_level=logging.WARNING)

    config = {
        "learning_rate": 1e-3,
        "seed": tune.grid_search(list(range(42, 52))),
        "n_state": 2,
        "n_ctrl": 2,
        "horizon": 20,
        "pred_horizon": 4,
        "trajectories": 10000,
        "model": {"type": "linear"},
        "datamodule": {
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 200,
            "segment_len": 4,
        },
        "trainer": {
            "max_epochs": 1000,
            # don't show progress bar for model training
            "progress_bar_refresh_rate": 0,
            # don't print summary before training
            "weights_summary": None,
        },
    }
    tune.run(Experiment, config=config, num_samples=1, local_dir="./results")
    ray.shutdown()


def run_simple():
    config = {
        "learning_rate": 5e-4,
        "seed": 42,
        "n_state": 2,
        "n_ctrl": 2,
        "horizon": 100,
        "pred_horizon": 8,
        "trajectories": 5000,
        "model": {
            "type": "mlp",
            "kwargs": {
                "hunits": (10,),
                "activation": "ReLU",
            },
        },
        "datamodule": {
            "train_batch_size": 128,
            "val_loss_batch_size": 128,
            "val_grad_batch_size": 200,
            "segment_len": 1,
        },
        "trainer": dict(
            max_epochs=100,
            # fast_dev_run=True,
            track_grad_norm=2,
            # overfit_batches=10,
            weights_summary="full",
            # limit_train_batches=10,
            # limit_val_batches=10,
            # profiler="simple",
            val_check_interval=0.5,
        ),
    }
    experiment = Experiment(config)
    experiment.train()


@click.command()
@click.option("--debug/--no-debug", default=False)
def main(debug: bool):
    if debug:
        run_simple()
    else:
        run_with_tune()


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
