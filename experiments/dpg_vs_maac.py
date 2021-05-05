# pylint:disable=missing-docstring
import logging
import os
from collections import deque
from typing import Union

import numpy as np
import ray
import torch
import wandb
from ray import tune
from torch import Tensor
from torch.optim import Optimizer

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment.estimators import DPG, MAAC, MonteCarloSVG
from lqsvg.experiment.utils import calver
from lqsvg.np_util import RNG
from lqsvg.policy.modules import QuadQValue, TVLinearPolicy


# noinspection PyAbstractClass
class Experiment(tune.Trainable):
    # pylint:disable=abstract-method,too-many-instance-attributes
    rng: RNG
    generator: LQGGenerator
    lqg: LQGModule
    policy: TVLinearPolicy
    rollout: MonteCarloSVG
    qvalue: QuadQValue
    optimizer: Optimizer
    estimator: Union[DPG, MAAC]
    n_step: int

    def setup(self, config: dict):
        self._init_wandb(config)
        self.rng = np.random.default_rng(config["seed"])
        self.make_generator()
        self.make_modules()
        self.make_optimizer()
        self.make_estimator(config)
        self._init_stats()

    def _init_wandb(self, config: dict):
        # pylint:disable=attribute-defined-outside-init
        os.environ["WANDB_SILENT"] = "true"
        cwd = config.pop("wandb_dir")
        tags = config.pop("wandb_tags", [])
        self.run = wandb.init(
            dir=cwd,
            name="DPGvsMAAC (on-policy)",
            config=config,
            project="LQG-SVG",
            entity="angelovtt",
            tags=[calver()] + tags,
            reinit=True,
            mode="online",
        )

    def make_generator(self):
        self.generator = LQGGenerator(
            n_state=2,
            n_ctrl=2,
            horizon=20,
            stationary=True,
            passive_eigval_range=(0.5, 1.5),
            controllable=True,
            rng=self.rng,
        )

    def make_modules(self):
        with nt.suppress_named_tensor_warning():
            dynamics, cost, init = self.generator()
        lqg = LQGModule.from_existing(dynamics, cost, init)
        policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon)
        policy.stabilize_(dynamics, rng=self.rng)
        qvalue = QuadQValue(lqg.n_state + lqg.n_ctrl, lqg.horizon)
        self.lqg, self.policy, self.qvalue = lqg, policy, qvalue
        self.rollout = MonteCarloSVG(policy, lqg)

    def make_optimizer(self):
        self.optimizer = torch.optim.Adam(self.policy.parameters())

    def make_estimator(self, config: dict):
        ecls = DPG if config["estimator"] == "dpg" else MAAC
        self.estimator = ecls(self.policy, self.lqg.trans, self.lqg.reward, self.qvalue)
        self.n_step = config["K"]

    def _init_stats(self):
        # pylint:disable=attribute-defined-outside-init
        self._return_history = deque([0], maxlen=100)

    def step(self):
        obs = self.lqg_rollout()

        self.update_qvalue()
        self.optimizer.zero_grad()
        loss = self.estimator.surrogate(obs, n_steps=self.n_step).neg()
        loss.backward()
        self.optimizer.step()

        return self.get_stats()

    def lqg_rollout(self) -> Tensor:
        n_trajs = int(np.ceil(self.config["B"] / self.lqg.horizon))
        with torch.no_grad():
            obs, _, rew, _, _ = self.rollout.rsample_trajectory(torch.Size([n_trajs]))

        obs = obs.flatten(["H", "B1"], "B")
        rets = rew.sum("H").tolist()
        self._return_history.extend(rets)
        return obs

    def update_qvalue(self):
        self.qvalue.match_policy_(
            self.policy.standard_form(),
            self.lqg.trans.standard_form(),
            self.lqg.reward.standard_form(),
        )

    def get_stats(self) -> dict:
        info = {
            "episode_reward_mean": np.mean(self._return_history),
            "episode_reward_max": np.max(self._return_history),
            "episode_reward_min": np.min(self._return_history),
        }
        self.run.log(info)
        return info

    def cleanup(self):
        self.run.finish()


def main():
    ray.init(logging_level=logging.WARNING)
    config = {
        "wandb_dir": os.getcwd(),
        "wandb_tags": "unstable controllable".split(),
        "seed": tune.grid_search(list(range(10))),
        "estimator": tune.grid_search("dpg maac".split()),
        "K": 6,
        "B": 200,
    }
    tune.run(
        Experiment,
        config=config,
        num_samples=10,
        stop=dict(training_iteration=1000),
        local_dir="./results",
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
