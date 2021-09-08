"""Full policy optimization via Stochastic Value Gradients."""
# pylint:disable=missing-docstring
import itertools
import logging
import os
from collections import deque
from functools import partial
from typing import Callable, Generator, Sequence, Tuple

import click
import numpy as np
import pytorch_lightning as pl
import ray
import torch
import wandb.sdk
import yaml
from critic import modules as critic_modules
from model import make_model as dynamics_model
from ray import tune
from ray.tune.logger import NoopLogger
from torch import Tensor, nn
from wandb_util import WANDB_DIR, wandb_init, with_prefix

from lqsvg import data, lightning
from lqsvg.envs import lqr
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, QuadQValue, QuadRewardModel, TVLinearPolicy
from lqsvg.torch.sequence import log_prob_fn

DynamicsBatch = Tuple[Tensor, Tensor, Tensor]
RewardBatch = Tuple[Tensor, Tensor, Tensor]
Replay = Sequence[data.Trajectory]


def make_model(
    lqg: LQGModule, policy: TVLinearPolicy, rng: Generator, config: dict
) -> nn.Module:
    # pylint:disable=unused-argument
    model = nn.Module()
    if config["perfect_model"]:
        model.dynamics = lqg.trans
        model.reward = lqg.reward
        model.qval = QuadQValue(
            n_tau=lqg.n_state + lqg.n_ctrl, horizon=lqg.horizon, rng=rng
        )
    else:
        model.dynamics = dynamics_model(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, config["dynamics"]
        )
        model.reward = QuadRewardModel(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, stationary=True, rng=rng
        )
        model.qval, model.target_qval, model.target_vval = critic_modules(
            policy, config["qvalue"], rng
        )
    return model


def dynamics_loss(dynamics: nn.Module) -> Callable[[DynamicsBatch], Tensor]:
    log_prob = log_prob_fn(dynamics, dynamics.dist)

    def loss(batch: DynamicsBatch) -> Tensor:
        obs, act, new_obs = (t.refine_names("B", "H", "R") for t in batch)
        return -log_prob(obs, act, new_obs).mean()

    return loss


def reward_loss(reward: nn.Module) -> Callable[[RewardBatch], Tensor]:
    def loss(batch: RewardBatch) -> Tensor:
        obs, act, rew = batch
        obs = obs.refine_names("B", "R")
        act = act.refine_names("B", "R")
        rew = rew.refine_names("B")
        return torch.mean(0.5 * torch.square(reward(obs, act) - rew))

    return loss


def model_trainer(
    model: nn.Module, config: dict
) -> Callable[[Replay, Generator], dict]:
    if config["perfect_model"]:
        return lambda *_: {}

    pl_dynamics = lightning.Lightning(
        model.dynamics, dynamics_loss(model.dynamics), config
    )
    pl_reward = lightning.Lightning(model.reward, reward_loss(model.reward), config)

    def train(dataset: Replay, rng: Generator) -> dict:
        obs, act, rew, _ = map(partial(torch.cat, dim="B"), zip(*dataset))
        obs, next_obs = data.obs_trajectory_to_transitions(obs)
        dynamics_dm = data.SequenceDataModule(
            obs, act, next_obs, spec=config["dynamics_dm"], rng=rng
        )
        reward_dm = data.TensorDataModule(
            *data.merge_horizon_and_batch_dims(obs, act, rew),
            spec=config["reward_dm"],
            rng=rng
        )
        dynamics_info = lightning.train_lite(pl_dynamics, dynamics_dm, config)
        reward_info = lightning.train_lite(pl_reward, reward_dm, config)
        return {
            **with_prefix("dynamics/", dynamics_info),
            **with_prefix("reward/", reward_info),
        }

    return train


def policy_trainer(
    policy: TVLinearPolicy, model: nn.Module, config: dict
) -> Callable[[Replay, Generator], dict]:
    # pylint:disable=unused-argument
    def optimize(dataset: Replay, rng: Generator) -> dict:
        return {}

    return optimize


def episode_stats(dataset: Replay) -> dict:
    batched = map(partial(torch.cat, dim="B"), zip(*dataset))
    obs, act, rew, logp = (t.align_to("B", "H", ...)[-100:] for t in batched)

    rets = rew.sum("H")
    log_likelihoods = logp.sum("H")
    return {
        "obs_mean": obs.mean().item(),
        "obs_std": obs.std().item(),
        "act_mean": act.mean().item(),
        "act_std": act.std().item(),
        "episode_return_max": rets.max().item(),
        "episode_return_mean": rets.mean().item(),
        "episode_return_min": rets.min().item(),
        "episode_logp_mean": log_likelihoods.mean().item(),
    }


def timesteps(trajectories: data.Trajectory):
    """Returns the number of timesteps in a trajectory batch."""
    actions = trajectories[1]
    # noinspection PyArgumentList
    return actions.size("H") * actions.size("B")


def policy_optimization_(
    lqg: LQGModule, policy: TVLinearPolicy, config: dict
) -> Generator[dict, None, None]:
    """Trains a policy via Stochastic Value Gradients."""
    rng = np.random.default_rng(config["seed"])
    model = make_model(lqg, policy, rng, config["model"])

    collect = data.environment_sampler(lqg)
    optimize_model_ = model_trainer(model, config["model"])
    optimize_policy_ = policy_trainer(policy, model, config)

    dataset = deque(maxlen=config["replay_size"] // lqg.horizon)
    for _ in itertools.count():
        metrics = {}
        trajectories = collect(policy, config["trajs_per_iter"])
        dataset.append(trajectories)
        metrics.update(optimize_model_(dataset, rng))
        metrics.update(optimize_policy_(dataset, rng))

        metrics[tune.result.TIMESTEPS_THIS_ITER] = timesteps(trajectories)
        metrics.update(with_prefix("collect/", episode_stats(dataset)))
        yield metrics


def create_env(config: dict) -> LQGModule:
    generator = lqr.LQGGenerator(
        stationary=True, controllable=True, rng=config["seed"], **config["env_config"]
    )
    dynamics, cost, init = generator()
    return LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)


def init_policy(lqg: LQGModule, config: dict) -> TVLinearPolicy:
    return TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon).stabilize_(
        lqg.trans.standard_form(), rng=config["seed"]
    )


class Experiment(tune.Trainable):
    coroutine: Generator[dict, None, None]
    _run: wandb.sdk.wandb_run.Run = None

    def setup(self, config: dict):
        pl.seed_everything(config["seed"])
        with nt.suppress_named_tensor_warning():
            lqg = create_env(config)
        policy = init_policy(lqg, config)
        self.coroutine = policy_optimization_(lqg, policy, config)

    @property
    def run(self) -> wandb.sdk.wandb_run.Run:
        if self._run is None:
            config = self.config.copy()
            wandb_kwargs = config.pop("wandb")
            self._run = wandb_init(config=config, **wandb_kwargs)
        return self._run

    def step(self) -> dict:
        metrics = next(self.coroutine)
        self.run.log(metrics)
        return metrics

    def cleanup(self):
        self.run.finish()
        self.coroutine.close()


@click.group()
def main():
    pass


def base_config() -> dict:
    return {
        "seed": 124,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 100,
            "passive_eigval_range": (0.9, 1.1),
        },
        "replay_size": int(1e5),
        "trajs_per_iter": 1,
        "model": {
            "perfect_model": True,
            "dynamics": {},
            "qvalue": {},
        },
    }


@main.command()
def sweep():
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    ray.init(logging_level=logging.WARNING)

    config = {
        **base_config(),
        "wandb": {"name": "SVG", "mode": "disabled"},
    }

    tune.run(
        Experiment,
        config=config,
        num_samples=1,
        stop={tune.result.TIMESTEPS_TOTAL: int(1e3)},
        local_dir=WANDB_DIR,
    )
    ray.shutdown()


@main.command()
def debug():
    config = {
        **base_config(),
        "wandb": {"name": "DEBUG", "mode": "disabled"},
    }
    exp = Experiment(config, logger_creator=partial(NoopLogger, logdir=os.devnull))
    print(yaml.dump({k: v for k, v in exp.train().items() if k != "config"}))


if __name__ == "__main__":
    main()
