# pylint:disable=missing-docstring
import json
import logging
import os
from collections import deque
from typing import Union

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch import Tensor, nn
from torch.optim import Optimizer

import lqsvg.torch.named as nt
import wandb
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.envs.lqr.solvers import NamedLQGControl
from lqsvg.experiment.estimators import (
    DPG,
    MAAC,
    AnalyticSVG,
    ExpectedValue,
    MonteCarloSVG,
)
from lqsvg.experiment.utils import calver
from lqsvg.np_util import RNG
from lqsvg.torch.nn.policy import TVLinearPolicy
from lqsvg.torch.nn.value import QuadQValue


# noinspection PyAbstractClass
class HparamSearch(tune.Trainable):
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
        self.rng = np.random.default_rng(self.run.config.seed)
        self.make_generator()
        self.make_modules()
        self.make_optimizer()
        self.make_estimator()
        self._init_stats()

    def _init_wandb(self, config: dict):
        # pylint:disable=attribute-defined-outside-init
        os.environ["WANDB_SILENT"] = "true"
        cwd = config.pop("wandb_dir")
        tags = config.pop("wandb_tags", [])
        self.run = wandb.init(
            dir=cwd,
            name="SuboptimalityGapSearch",
            config=config,
            project="LQG-SVG",
            entity="angelovtt",
            tags=[calver()] + list(tags),
            reinit=True,
            mode="online",
        )

    def make_generator(self):
        self.generator = LQGGenerator(
            n_state=self.run.config.env_dim,
            n_ctrl=self.run.config.env_dim,
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
        policy.stabilize_(dynamics, rng=self.generator.rng)
        qvalue = QuadQValue(lqg.n_state + lqg.n_ctrl, lqg.horizon)
        self.lqg, self.policy, self.qvalue = lqg, policy, qvalue
        self.rollout = MonteCarloSVG(policy, lqg)

    def make_optimizer(self):
        name = self.run.config.optimizer
        if name == "Adam":
            cls = torch.optim.Adam
        elif name == "SGD":
            cls = torch.optim.SGD
        else:
            raise ValueError(f"Invalid optimizer type {name}")
        self.optimizer = cls(self.policy.parameters(), lr=self.run.config.learning_rate)

    def make_estimator(self):
        # pylint:disable=attribute-defined-outside-init
        ecls = DPG if self.run.config.estimator == "dpg" else MAAC
        self.estimator = ecls(self.policy, self.lqg.trans, self.lqg.reward, self.qvalue)
        self.n_step = self.run.config.K
        # noinspection PyAttributeOutsideInit
        self._golden_standard = AnalyticSVG(policy=self.policy, model=self.lqg)

    def _init_stats(self):
        # pylint:disable=attribute-defined-outside-init
        self._return_history = deque([0], maxlen=100)
        dynamics, cost, init = self.lqg.standard_form()
        solver = NamedLQGControl(self.lqg.n_state, self.lqg.n_ctrl, self.lqg.horizon)
        _, _, vstar = solver(dynamics, cost)
        vstar = tuple(v.select("H", 0) for v in vstar)
        self._optimal_value = -ExpectedValue()(init, vstar).item()

    def step(self):
        obs = self.lqg_rollout()

        self.update_qvalue()
        self.optimizer.zero_grad()
        loss = self.estimator.surrogate(obs, n_steps=self.n_step).neg()
        loss.backward()
        info = self.postprocess_svg()
        self.optimizer.step()

        return self.get_stats(info)

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

    def postprocess_svg(self):
        max_norm = self.run.config.clip_grad_norm
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=max_norm
        )
        return dict(grad_norm=grad_norm.item())

    def get_stats(self, info) -> dict:
        stats = {
            "true_value": self._golden_standard.value().item(),
            "optimal_value": self._optimal_value,
            "episode_reward_mean": np.mean(self._return_history),
            "episode_reward_max": np.max(self._return_history),
            "episode_reward_min": np.min(self._return_history),
            **info,
        }
        self.run.log(stats)
        return stats

    def cleanup(self):
        self.run.finish()


def main():
    ray.init(logging_level=logging.WARNING)
    space = {
        "wandb_dir": os.getcwd(),
        "wandb_tags": "unstable controllable".split(),
        "seed": 42,
        "estimator": "maac",
        "K": 8,
        "B": tune.qrandint(20, 400, 20),
        "optimizer": "SGD",
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "clip_grad_norm": 100,
    }

    for env_dim in range(10, 11):
        hyperopt_search = HyperOptSearch(
            {"env_dim": env_dim, **space},
            metric="true_value",
            mode="max",
            random_state_seed=42,
        )
        name = f"HparamSearch-Dim{env_dim}"
        analysis = tune.run(
            HparamSearch,
            name=name,
            search_alg=hyperopt_search,
            local_dir="./results",
            stop={"time_total_s": 300},
            num_samples=1000,
        )
        best = analysis.get_best_config("true_value", mode="max")
        with open(os.path.join("./results", name, "search_best.json"), "w") as file:
            json.dump(best, file)

    ray.shutdown()


if __name__ == "__main__":
    main()
