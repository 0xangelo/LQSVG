# pylint:disable=missing-docstring
import numpy as np
import ray
import torch
from ray import tune
from raylab.policy.modules.model import StochasticModel
from torch.optim import Optimizer
from wandb.sdk import wandb_config

import lqsvg.torch.named as nt
import wandb
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.experiment.estimators import MAAC
from lqsvg.np_util import RNG
from lqsvg.policy.modules import QuadQValue, TVLinearPolicy


# noinspection PyAbstractClass
class Experiment(tune.Trainable):
    # pylint:disable=abstract-method,too-many-instance-attributes
    run: wandb.sdk.wandb_run.Run
    rng: RNG
    generator: LQGGenerator
    lqg: LQGModule
    policy: TVLinearPolicy
    qvalue: QuadQValue
    model: StochasticModel
    optimizer: Optimizer
    estimator: MAAC

    def setup(self, config: dict):
        self._init_wandb(config)
        self.rng = np.random.default_rng(self.hparams.seed)
        self.make_generator()
        self.make_modules()
        self.make_optimizer()

    def _init_wandb(self, config: dict):
        self.run = wandb.init(
            name="LinearML",
            config=config,
            project="LQG-SVG",
            entity="angelovtt",
            tags=["ch5"],
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
        lqg = LQGModule.from_existing(dynamics, cost, init)
        policy = TVLinearPolicy(lqg.n_state, lqg.n_ctrl, lqg.horizon)
        policy.stabilize_(dynamics, rng=self.rng)
        qvalue = QuadQValue(lqg.n_state + lqg.n_ctrl, lqg.horizon)
        self.lqg, self.policy, self.qvalue = lqg, policy, qvalue
        self.model = None

    def make_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate
        )


def main():
    ray.init()


if __name__ == "__main__":
    main()
