"""Full policy optimization via Stochastic Value Gradients."""
# pylint:disable=missing-docstring
import logging
from collections import deque
from typing import Generator, Sequence

import pytorch_lightning as pl
import ray
import wandb.sdk
from ray import tune
from wandb_util import WANDB_DIR, wandb_init

from lqsvg import data
from lqsvg.torch.nn import LQGModule, TVLinearPolicy


def make_model(
    lqg: LQGModule, policy: TVLinearPolicy, config: dict
) -> pl.LightningModule:
    pass


def optimize_model_(
    model: pl.LightningModule, dataset: Sequence[data.Trajectory], config: dict
) -> dict:
    pass


def optimize_policy_(
    policy: TVLinearPolicy,
    model: pl.LightningModule,
    dataset: Sequence[data.Trajectory],
    config: dict,
) -> dict:
    pass


def episode_stats(dataset: Sequence[data.Trajectory]) -> dict:
    pass


def policy_optimization_(lqg: LQGModule, policy: TVLinearPolicy, config: dict):
    """Trains a policy via Stochastic Value Gradients."""
    collect = data.environment_sampler(lqg)
    model = make_model(lqg, policy, config)

    dataset = deque(maxlen=config["replay_size"] // lqg.horizon)
    for i in range(config["iterations"]):
        metrics = {}
        trajectories = collect(policy, config["trajs_per_iter"])
        dataset.append(trajectories)
        metrics.update(optimize_model_(model, dataset, config))
        metrics.update(optimize_policy_(policy, model, dataset, config))

        metrics.update(episode_stats(dataset))
        metrics.update(done=(i + 1 == config["iterations"]))
        yield metrics


def create_env(config: dict) -> LQGModule:
    pass


def init_policy(lqg: LQGModule, config: dict) -> TVLinearPolicy:
    pass


class Experiment(tune.Trainable):
    coroutine: Generator[dict, None, None]
    _run: wandb.sdk.wandb_run.Run = None

    def setup(self, config: dict):
        pl.seed_everything(config["seed"])
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
