# pylint:disable=missing-docstring
import contextlib
import functools
import logging
import os
from typing import Callable, Tuple

import click
import pytorch_lightning as pl
import ray
import torch
import yaml
from nnrl.nn.utils import update_polyak
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from lqsvg import analysis, data, estimator, lightning
from lqsvg.envs import lqr
from lqsvg.experiment.utils import calver
from lqsvg.random import RNG, make_rng
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, QuadRewardModel, TVLinearPolicy
from lqsvg.torch.sequence import log_prob_fn
from lqsvg.types import DeterministicPolicy

# isort: off
# pylint:disable=wrong-import-order
from actor import behavior_policy
from critic import td_modules as critic_modules, qval_constructor, TDBatch
from model import make_model as dynamics_model
from wandb_util import WANDB_DIR

DynamicsBatch = Tuple[Tensor, Tensor, Tensor]
RewardBatch = Tuple[Tensor, Tensor, Tensor]


def prediction_problem(
    rng: RNG, config: dict
) -> (LQGModule, TVLinearPolicy, DeterministicPolicy):
    generator = lqr.LQGGenerator(
        stationary=True, controllable=True, rng=rng.numpy, **config["env_config"]
    )
    with nt.suppress_named_tensor_warning():
        dynamics, cost, init = generator()

    lqg = LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)

    policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon)
    policy.stabilize_(dynamics, rng.numpy)

    behavior = behavior_policy(policy, config["exploration"], rng.torch)
    return lqg, policy, behavior


def models(
    rng: RNG, lqg: LQGModule, policy: TVLinearPolicy, config: dict
) -> (nn.Module, nn.Module, nn.Module):
    dmodel = dynamics_model(lqg.n_state, lqg.n_ctrl, lqg.horizon, config["dynamics"])
    rmodel = QuadRewardModel(
        lqg.n_state, lqg.n_ctrl, lqg.horizon, stationary=True, rng=rng.numpy
    )
    qmodel = nn.Module()
    qval_fn = qval_constructor(policy, config["qvalue"])
    qmodel.qval, qmodel.target_qval, qmodel.target_vval = critic_modules(
        policy, qval_fn, config["qvalue"], rng.numpy
    )
    return dmodel, rmodel, qmodel


def dynamics_loss(dynamics: nn.Module) -> Callable[[DynamicsBatch], Tensor]:
    log_prob = log_prob_fn(dynamics, dynamics.dist)

    def loss(batch: DynamicsBatch) -> Tensor:
        obs, act, new_obs = batch
        return -log_prob(obs, act, new_obs).mean()

    return loss


def reward_loss(reward: nn.Module) -> Callable[[RewardBatch], Tensor]:
    def loss(batch: RewardBatch) -> Tensor:
        obs, act, rew = batch
        return torch.mean(0.5 * torch.square(reward(obs, act) - rew))

    return loss


def td_mse(qmodel: nn.Module) -> Callable[[TDBatch], Tensor]:
    def loss(batch: TDBatch) -> Tensor:
        obs, act, rew, new_obs = batch
        with torch.no_grad():
            target = nt.unnamed(rew + qmodel.target_vval(new_obs))
        loss_fn = nn.MSELoss()
        values = nt.unnamed(*qmodel.qval.q_values(obs, act))
        return torch.stack([loss_fn(v, target) for v in values]).sum()

    return loss


def mage_loss(
    qmodel: nn.Module, policy: DeterministicPolicy, dmodel: nn.Module, rmodel: nn.Module
) -> Callable[[TDBatch], Tensor]:
    def loss(batch: TDBatch) -> Tensor:
        obs, _, _, _ = batch
        act = policy(obs)
        rew = rmodel(obs, act)
        new_obs, _ = dmodel.rsample(dmodel(obs, act))

        value = qmodel.qval(obs, act)
        target = rew + qmodel.target_vval(new_obs)

        delta = value - target
        (act_grad,) = torch.autograd.grad(
            delta, act, grad_outputs=torch.ones_like(delta), create_graph=True
        )
        loss = torch.norm(act_grad, dim=-1).mean() + 0.05 * torch.square(delta).mean()
        return loss

    return loss


def td_rerror(qmodel: nn.Module) -> Callable[[TDBatch], Tensor]:
    def loss(batch: TDBatch) -> Tensor:
        obs, act, rew, new_obs = batch

        pred = qmodel.qval(obs, act)  # (B,)
        target = rew + qmodel.target_vval(new_obs)  # (B,)

        relative_error = analysis.relative_error(pred, target).mean()
        return relative_error

    return loss


class LightningModelPlus(lightning.Lightning):
    # pylint:disable=too-many-ancestors
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[lightning.BatchType], Tensor],
        config: dict,
        val_loss: Callable[[lightning.BatchType], Tensor],
    ):
        super().__init__(model, loss, config)
        self.val_loss = val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.qval.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def validation_step(self, batch: lightning.BatchType, _: int) -> Tensor:
        loss = self.val_loss(batch)
        self.log("val/loss", loss)
        return loss

    def on_train_batch_end(self, *args, **kwargs):  # pylint:disable=unused-argument
        update_polyak(self.model.qval, self.model.target_qval, self.hparams.polyak)


def datamodules(
    rng: RNG, lqg: LQGModule, behavior: DeterministicPolicy, config: dict
) -> (pl.LightningDataModule, pl.LightningDataModule, pl.LightningDataModule):
    sampler = data.environment_sampler(lqg)
    sample_fn = functools.partial(sampler, behavior)

    with torch.no_grad():
        obs, act, rew, _ = sample_fn(config["trajectories"])
    obs, act, rew = (t.align_to("H", "B", ...) for t in (obs, act, rew))
    obs, new_obs = obs[:-1], obs[1:]

    dynamics_dm = data.SequenceDataModule(
        obs, act, new_obs, spec=config["dynamics_dm"], rng=rng.torch
    )

    obs, act, rew, new_obs = data.merge_horizon_and_batch_dims(obs, act, rew, new_obs)
    reward_dm = data.TensorDataModule(
        obs, act, rew, spec=config["reward_dm"], rng=rng.torch
    )
    qvalue_dm = data.TensorDataModule(
        obs, act, rew, new_obs, spec=config["qvalue_dm"], rng=rng.torch
    )
    return dynamics_dm, reward_dm, qvalue_dm


def maac_grad_stats(
    lqg: LQGModule,
    policy: TVLinearPolicy,
    dmodel: nn.Module,
    rmodel: nn.Module,
    qmodel: nn.Module,
) -> Callable[[pl.LightningDataModule], dict]:
    def stats(datamodule: pl.LightningDataModule) -> dict:
        maac = estimator.maac_estimator(
            policy,
            data.markovian_state_sampler(dmodel, dmodel.rsample),
            rmodel,
            qmodel.qval,
        )
        dataset = TensorDataset(nt.unnamed(datamodule.tensors[0]))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        svgs = []
        for batch in dataloader:
            (obs,) = batch
            obs = obs.refine_names(*datamodule.tensors[0].names)
            svgs += [maac(obs, 4)[1]]

        dynamics, cost, init = lqg.standard_form()
        _, target = estimator.analytic_svg(policy, init, dynamics, cost)
        return {
            "grad_accuracy": analysis.gradient_accuracy(svgs, target).item(),
            "grad_precision": analysis.gradient_precision(svgs).item(),
        }

    return stats


def dpg_grad_stats(
    lqg: LQGModule, policy: TVLinearPolicy, qmodel: nn.Module
) -> Callable[[pl.LightningDataModule], dict]:
    def stats(datamodule: pl.LightningDataModule) -> dict:
        dpg = estimator.mfdpg_estimator(policy, qmodel.qval)
        dataset = TensorDataset(nt.unnamed(datamodule.tensors[0]))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        svgs = []
        for batch in dataloader:
            (obs,) = batch
            obs = obs.refine_names(*datamodule.tensors[0].names)
            svgs += [dpg(obs)[1]]

        dynamics, cost, init = lqg.standard_form()
        _, target = estimator.analytic_svg(policy, init, dynamics, cost)
        return {
            "grad_accuracy": analysis.gradient_accuracy(svgs, target).item(),
            "grad_precision": analysis.gradient_precision(svgs).item(),
        }

    return stats


class Experiment(tune.Trainable):
    def setup(self, config: dict):
        pl.seed_everything(config["seed"])

    def step(self) -> dict:
        # pylint:disable=too-many-locals
        config = self.config
        rng = make_rng(config["seed"])
        lqg, policy, behavior = prediction_problem(rng, config)
        dmodel, rmodel, qmodel = models(rng, lqg, policy, config["model"])

        pl_dynamics = lightning.Lightning(
            dmodel, dynamics_loss(dmodel), config["model"]["dynamics"]
        )
        pl_reward = lightning.Lightning(
            rmodel, reward_loss(rmodel), config["model"]["reward"]
        )
        if "mage" in config["strategy"]:
            qloss = mage_loss(qmodel, policy, dmodel, rmodel)
        else:
            qloss = td_mse(qmodel)
        pl_qvalue = LightningModelPlus(
            qmodel, qloss, config["model"]["qvalue"], td_rerror(qmodel)
        )

        dynamics_dm, reward_dm, qvalue_dm = datamodules(rng, lqg, behavior, config)

        dl_ctx = lightning.suppress_dataloader_warnings(num_workers=True)
        dm_ctx = lightning.suppress_datamodule_warnings()
        with dl_ctx, dm_ctx:
            dynamics_info = lightning.train_lite(
                pl_dynamics, dynamics_dm, config["model"]["dynamics"]
            )
            reward_info = lightning.train_lite(
                pl_reward, reward_dm, config["model"]["reward"]
            )
            qvalue_info = lightning.train_lite(
                pl_qvalue, qvalue_dm, config["model"]["qvalue"]
            )

        if "maac" in config["strategy"]:
            grad_stats = maac_grad_stats(lqg, policy, dmodel, rmodel, qmodel)
        else:
            grad_stats = dpg_grad_stats(lqg, policy, qmodel)
        return {
            "dynamics": dynamics_info,
            "reward": reward_info,
            "qvalue": qvalue_info,
            tune.result.DONE: True,
            **grad_stats(qvalue_dm),
        }


@contextlib.contextmanager
def logging_setup():
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["WANDB_SILENT"] = "true"
    with open(".local_ip", "rt") as file:  # pylint:disable=unspecified-encoding
        ray.init(logging_level=logging.WARNING, _node_ip_address=file.read().rstrip())
    yield
    ray.shutdown()


@click.group()
def main():
    pass


@main.command()
@logging_setup()
def sweep():
    config = {
        "strategy": tune.grid_search(["maac", "mage", "maac+mage"]),
        "seed": tune.grid_search(list(range(120, 140))),
        "env_config": {
            "n_state": 4,
            "n_ctrl": 4,
            "horizon": 100,
            "passive_eigval_range": (0.9, 1.1),
        },
        "exploration": {"type": "gaussian", "action_noise_sigma": 0.3},
        "trajectories": 2_000,
        "dynamics_dm": {
            "train_batch_size": 128,
            "val_batch_size": 128,
            "seq_len": 8,
        },
        "reward_dm": {
            "train_batch_size": 64,
            "val_batch_size": 64,
        },
        "qvalue_dm": {
            "train_batch_size": 128,
            "val_batch_size": 128,
        },
        "model": {
            "dynamics": {
                "type": "linear",
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "max_epochs": 20,
            },
            "reward": {
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "max_epochs": 20,
            },
            "qvalue": {
                "type": "quad",
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "max_epochs": 20,
                "polyak": 0.995,
            },
        },
    }

    logger = WandbLoggerCallback(
        name="PredictionFull",
        project="ch5",
        entity="angelovtt",
        tags=[calver(), "PredictionFull"],
        dir=WANDB_DIR,
    )
    tune.run(Experiment, config=config, local_dir=WANDB_DIR, callbacks=[logger])


@main.command()
@click.argument("strategy", type=str)
@logging_setup()
def debug(strategy: str):
    config = {
        "strategy": strategy,
        "seed": 120,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 100,
            "passive_eigval_range": (0.9, 1.1),
        },
        "exploration": {"type": "gaussian", "action_noise_sigma": 0.3},
        "trajectories": 2_000,
        "dynamics_dm": {
            "train_batch_size": 128,
            "val_batch_size": 128,
            "seq_len": 8,
        },
        "reward_dm": {
            "train_batch_size": 64,
            "val_batch_size": 64,
        },
        "qvalue_dm": {
            "train_batch_size": 128,
            "val_batch_size": 128,
        },
        "model": {
            "dynamics": {
                "type": "linear",
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "max_epochs": 1,
            },
            "reward": {
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "max_epochs": 1,
            },
            "qvalue": {
                "type": "quad",
                "learning_rate": 1e-3,
                "weight_decay": 0,
                "max_epochs": 1,
                "polyak": 0.995,
            },
        },
    }
    out = Experiment(config).train()
    del out["config"]
    print(yaml.dump(out))


if __name__ == "__main__":
    main()
