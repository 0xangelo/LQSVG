# pylint:disable=unsubscriptable-object
"""Compilation of LQG modules."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from lqsvg.envs import lqr

from .dynamics import InitStateDynamics
from .dynamics import TVLinearDynamics
from .reward import QuadraticReward


class LQGModule(nn.Module):
    """Linear Quadratic Gaussian as neural network module."""

    def __init__(
        self, trans: TVLinearDynamics, reward: QuadraticReward, init: InitStateDynamics
    ):
        super().__init__()
        self.trans = trans
        self.reward = reward
        self.init = init

    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Computes the trajectory log-likelihood."""
        return self.log_prob(obs, act, new_obs)

    def log_prob(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Log-likelihood of trajectory.

        Treats actions as constants.
        """
        obs, act, new_obs = (x.align_to("H", ..., "R") for x in (obs, act, new_obs))

        init_logp = self.init.log_prob(obs.select(dim="H", index=0))
        trans_logp = self.trans.log_prob(new_obs, self.trans(obs, act)).sum(dim="H")

        return init_logp + trans_logp

    def standard_form(self) -> tuple[lqr.LinSDynamics, lqr.QuadCost, lqr.GaussInit]:
        """Submodules as a collection of matrices."""
        trans = self.trans.standard_form()
        cost = self.reward.standard_form()
        init = self.init.standard_form()
        return trans, cost, init
