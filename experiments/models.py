# pylint:disable=missing-docstring
from typing import List
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import InitStateDynamics

from policy import LQGPolicy  # pylint:disable=wrong-import-order


# noinspection PyMethodMayBeStatic
class ExpectedValue(nn.Module):
    # pylint:disable=invalid-name,no-self-use
    def forward(self, rho: Tuple[Tensor, Tensor], vval: lqr.Quadratic):
        """Expected cost given mean and covariance matrix of the initial state.

        https://en.wikipedia.org/wiki/Quadratic_form_(statistics)#Expectation.
        """
        V, v, c = vval
        V = nt.refine_matrix_input(V)
        v = nt.refine_vector_input(v)
        c = nt.refine_scalar_input(c)
        mean, cov = rho
        mean = nt.refine_vector_input(mean)
        cov = nt.refine_matrix_input(cov)

        value = (
            nt.trace(cov @ V).align_to(..., "R", "C") / 2
            + nt.transpose(mean) @ V @ mean
            + nt.transpose(v) @ mean
            + c
        )
        return nt.refine_scalar_output(value)


class PolicyLoss(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
    ):
        super().__init__()
        self.predict = lqr.NamedLQGPrediction(n_state, n_ctrl, horizon)
        self.expected = ExpectedValue()

    def forward(
        self,
        policy: lqr.Linear,
        dynamics: lqr.LinDynamics,
        cost: lqr.QuadCost,
        rho: Tuple[Tensor, Tensor],
    ):
        _, vval = self.predict(policy, dynamics, cost)
        vval = tuple(x.select("H", 0) for x in vval)
        cost = self.expected(rho, vval)
        return cost


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    def __init__(self, policy: LQGPolicy, env: TorchLQGMixin):
        super().__init__()
        self.module = policy.module

        self.dynamics = env.dynamics
        self.cost = env.cost
        self.rho = env.rho

        self.policy_loss = PolicyLoss(env.n_state, env.n_ctrl, env.horizon)

    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Batched trajectory log prob."""
        # pylint:disable=arguments-differ
        obs, act, new_obs = (x.align_to("H", ..., "R") for x in (obs, act, new_obs))

        init_logp = self.module.init_model.log_prob(obs.select(dim="H", index=0))
        trans_params = self.module.trans_model(obs, act)
        trans_logp = self.module.trans_model.log_prob(new_obs, trans_params).sum(
            dim="H"
        )

        return init_logp + trans_logp

    def rsample(
        self, sample_shape: List[int] = ()
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample trajectory for Stochastic Value Gradients.

        Sample trajectory using model with reparameterization trick and
        deterministic actor.
        """
        horizon = self.module.trans_model.params.F.size(0)
        batch = []
        obs, _ = self.module.init_model.rsample(sample_shape)
        for _ in range(horizon):
            act = self.module.actor(obs)
            rew = self.module.rew_model(obs, act)
            # No sample_shape needed, already handled via batch of initial states
            new_obs, _ = self.module.trans_model.rsample(
                self.module.trans_model(obs, act)
            )

            batch += [(obs, act, rew, new_obs)]
            obs = new_obs

        obs, act, rew, new_obs = (nt.stack_horizon(*x) for x in zip(*batch))
        return obs, act, rew, new_obs

    def configure_optimizers(self):
        params = nn.ParameterList(
            [
                p
                for m in (self.module.init_model, self.module.trans_model)
                for p in m.parameters()
            ]
        )
        optim = torch.optim.Adam(params, lr=1e-3)
        return optim

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        obs, act, new_obs = (x.refine_names("B", "H", "R") for x in batch)
        loss = -self(obs, act, new_obs).mean()
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        return self.training_step(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        # pylint:disable=arguments-differ
        del validation_step_outputs
        self.value_gradient_info()

    def value_gradient_info(self):
        policy, dynamics, cost, rho = self.module.standard_form()

        model = dynamics, cost, rho
        ground_truth = self.dynamics, self.cost, self.rho
        self.log("learned_loss", self.policy_loss(policy, *model))
        self.log("true_loss", self.policy_loss(policy, *ground_truth))


def test_lightning_model():
    from policy import make_worker

    worker = make_worker()
    model = LightningModel(worker.get_policy(), worker.env)
    assert isinstance(model.rho, InitStateDynamics)

    def print_traj(traj):
        obs, act, rew, new_obs = traj
        print(
            f"""\
        Obs: {obs.shape}, {obs.names}
        Act: {act.shape}, {act.names}
        Rew: {rew.shape}, {rew.names}
        New Obs: {new_obs.shape}, {new_obs.names}
        """
        )

    print_traj(model.rsample(()))
    print_traj(model.rsample((10,)))

    obs, act, _, new_obs = model.rsample((100,))
    traj_logp = model(obs, act, new_obs)
    print(f"Traj logp: {traj_logp}, {traj_logp.shape}, {traj_logp.names}")


if __name__ == "__main__":
    test_lightning_model()
