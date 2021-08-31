"""Compilation of LQG modules."""
from typing import Tuple

from nnrl.nn.model import StochasticModel
from torch import Tensor, nn

from lqsvg.envs import lqr

from .dynamics.linear import LinearDynamics, LinearDynamicsModule
from .initstate import InitStateModule
from .reward import QuadraticReward


class EnvModule(nn.Module):
    """Environment dynamics as a neural network module."""

    n_state: int
    n_ctrl: int
    horizon: int

    def __init__(
        self,
        dims: Tuple[int, int, int],
        trans: StochasticModel,
        reward: QuadraticReward,
        init: InitStateModule,
    ):
        super().__init__()
        self.n_state, self.n_ctrl, self.horizon = dims
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


class LQGModule(EnvModule):
    """Linear Quadratic Gaussian as neural network module."""

    trans: LinearDynamics

    def __init__(
        self,
        dims: Tuple[int, int, int],
        trans: LinearDynamics,
        reward: QuadraticReward,
        init: InitStateModule,
    ):
        super().__init__(dims, trans, reward, init)

    @classmethod
    def from_existing(
        cls, dynamics: lqr.LinSDynamics, cost: lqr.QuadCost, init: lqr.GaussInit
    ) -> "LQGModule":
        """Create LQG from existing components' parameters."""
        dims = lqr.dims_from_dynamics(dynamics)
        trans = LinearDynamicsModule.from_existing(dynamics, stationary=False)
        reward = QuadraticReward.from_existing(cost)
        init = InitStateModule.from_existing(init)
        return cls(dims, trans, reward, init)

    def standard_form(self) -> Tuple[lqr.LinSDynamics, lqr.QuadCost, lqr.GaussInit]:
        """Submodules as a collection of matrices."""
        trans = self.trans.standard_form()
        cost = self.reward.standard_form()
        init = self.init.standard_form()
        return trans, cost, init
