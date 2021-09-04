"""Full policy optimization via Stochastic Value Gradients."""
# pylint:disable=missing-docstring
import logging
from collections import deque
from functools import partial
from typing import Callable, Generator, Sequence

import pytorch_lightning as pl
import ray
import torch
import wandb.sdk
from ray import tune
from wandb_util import WANDB_DIR, wandb_init, with_prefix

from lqsvg import data
from lqsvg.envs import lqr
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, TVLinearPolicy


def make_model(
    lqg: LQGModule, policy: TVLinearPolicy, config: dict
) -> pl.LightningModule:
    # pylint:disable=unused-argument
    return pl.LightningModule()


def model_trainer(
    config: dict,
) -> Callable[[pl.LightningModule, Sequence[data.Trajectory]], dict]:
    # pylint:disable=unused-argument
    def train(model: pl.LightningModule, dataset: Sequence[data.Trajectory]) -> dict:
        pass

    return train


def policy_trainer(
    config: dict,
) -> Callable[[TVLinearPolicy, pl.LightningModule, Sequence[data.Trajectory]], dict]:
    # pylint:disable=unused-argument
    def optimize(
        policy: TVLinearPolicy,
        model: pl.LightningModule,
        dataset: Sequence[data.Trajectory],
    ) -> dict:
        pass

    return optimize


def episode_stats(dataset: Sequence[data.Trajectory]) -> dict:
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


def policy_optimization_(
    lqg: LQGModule, policy: TVLinearPolicy, config: dict
) -> Generator[dict, None, None]:
    """Trains a policy via Stochastic Value Gradients."""
    collect = data.environment_sampler(lqg)
    optimize_model_ = model_trainer(config)
    optimize_policy_ = policy_trainer(config)

    model = make_model(lqg, policy, config)
    dataset = deque(maxlen=config["replay_size"] // lqg.horizon)
    for i in range(config["iterations"]):
        metrics = {}
        trajectories = collect(policy, config["trajs_per_iter"])
        dataset.append(trajectories)
        metrics.update(optimize_model_(model, dataset))
        metrics.update(optimize_policy_(policy, model, dataset))

        metrics.update(with_prefix("collect/", episode_stats(dataset)))
        metrics.update(done=(i + 1 == config["iterations"]))
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

        if metrics["done"]:
            metrics[tune.result.DONE] = True
            self.coroutine.close()
        return metrics

    def cleanup(self):
        self.run.finish()


def main():
    ray.init(logging_level=logging.WARNING)

    config = {"wandb": {"mode": "offline"}}

    tune.run(Experiment, config=config, num_samples=1, local_dir=WANDB_DIR)
    ray.shutdown()


if __name__ == "__main__":
    main()
