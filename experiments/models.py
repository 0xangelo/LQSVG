# pylint:disable=missing-docstring,unsubscriptable-object
from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.policy.time_varying_linear import LQGPolicy

from utils import linear_feedback_norm  # pylint:disable=wrong-import-order


class ExpectedValue(nn.Module):
    # pylint:disable=invalid-name,no-self-use
    # noinspection PyMethodMayBeStatic
    def forward(self, rho: tuple[Tensor, Tensor], vval: lqr.Quadratic):
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
        rho: tuple[Tensor, Tensor],
    ):
        _, vval = self.predict(policy, dynamics, cost)
        vval = tuple(x.select("H", 0) for x in vval)
        cost = self.expected(rho, vval)
        return cost


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    actor: nn.Module
    model: nn.Module
    mdp: nn.Module
    policy_loss: nn.Module
    early_stop_on: str = "early_stop_on"

    def __init__(self, policy: LQGPolicy, env: TorchLQGMixin):
        super().__init__()
        self.actor = policy.module.actor
        self.model = policy.module.model
        self.mdp = env.module
        self.policy_loss = PolicyLoss(env.n_state, env.n_ctrl, env.horizon)

    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Batched trajectory log prob."""
        # pylint:disable=arguments-differ
        return self.model.log_prob(obs, act, new_obs)

    def rsample_trajectory(
        self, sample_shape: torch.Size = torch.Size()
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample trajectory for Stochastic Value Gradients.

        Sample trajectory using model with reparameterization trick and
        deterministic actor.

        Args:
            sample_shape: shape for batched trajectory samples

        Returns:
            Trajectory sample following the policy
        """
        sample_shape = torch.Size(sample_shape)
        horizon = self.model.horizon
        batch = []
        obs, _ = self.model.init.rsample(sample_shape)
        obs = obs.refine_names(*(f"B{i+1}" for i in range(len(sample_shape))), ...)
        for _ in range(horizon):
            act = self.actor(obs)
            rew = self.model.reward(obs, act)
            # No sample_shape needed since initial states are batched
            new_obs, _ = self.model.trans.rsample(self.model.trans(obs, act))

            batch += [(obs, act, rew, new_obs)]
            obs = new_obs

        obs, act, rew, new_obs = (nt.stack_horizon(*x) for x in zip(*batch))
        return obs, act, rew, new_obs

    def rsample_return(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample return for Stochastic Value Gradients.

        Sample reparameterized trajectory and compute empirical return.

        Args:
            sample_shape: shape for batched return samples

        Returns:
            Sample of the policy return
        """
        _, _, rew, _ = self.rsample_trajectory(sample_shape)
        # Reduce over horizon
        ret = rew.sum("H")
        return ret

    def monte_carlo_value(self, samples: int = 1) -> Tensor:
        """Monte Carlo estimate of the Stochastic Value.

        Args:
            samples: number of samples

        Returns:
            A tensor representing the Monte Carlo estimate of the policy's
            expected return
        """
        rets = self.rsample_return((samples,))
        # Reduce over the first (and only) batch dimension
        mc_performance = rets.mean("B1")
        return mc_performance

    def analytic_value(self, ground_truth: bool = False) -> Tensor:
        """Compute the analytic policy performance using the value function.

        Solves the prediction problem for the current policy and uses the
        resulting state-value function to compute the expected return.

        Args:
            ground_truth: whether to use the dynamics of the MDP to derive
               the true value (gold standard)

        Returns:
            A tensor representing the policy's expected return
        """
        policy = self.actor.standard_form()
        model = self.mdp.standard_form() if ground_truth else self.model.standard_form()
        value = -self.policy_loss(policy, *model)
        return value

    def monte_carlo_svg(self, samples: int = 1) -> tuple[Tensor, lqr.Linear]:
        """Monte Carlo estimate of the Stochastic Value Gradient.

        Args:
            samples: number of samples

        Returns:
            A tuple with the expected policy return (as a tensor) and an
            iterable of gradients of the expected return with respect to
            each policy parameter
        """
        mc_performance = self.monte_carlo_value(samples)
        svg = self._current_policy_svg(mc_performance)
        return mc_performance, svg

    def analytic_svg(self, ground_truth: bool = False) -> tuple[Tensor, lqr.Linear]:
        """Compute the analytic SVG using the value function.

        Solves the prediction problem for the current policy and uses the
        resulting state-value function to compute the expected return and its
        gradient w.r.t. policy parameters.

        Args:
            ground_truth: whether to use the dynamics of the MDP to derive
               the true SVG (gold standard)

        Returns:
            A tuple with the expected policy return (as a tensor) and an
            iterable of gradients of the expected return w.r.t. each policy
            parameter
        """
        value = self.analytic_value(ground_truth=ground_truth)
        svg = self._current_policy_svg(value)
        return value, svg

    def _current_policy_svg(self, value: Tensor) -> lqr.Linear:
        # pylint:disable=invalid-name
        self.actor.zero_grad(set_to_none=True)
        value.backward()
        K, k = self.actor.standard_form()
        return K.grad.clone(), k.grad.clone()

    def configure_optimizers(self):
        params = nn.ParameterList(
            list(self.model.trans.parameters()) + list(self.model.init.parameters())
        )
        optim = torch.optim.Adam(params, lr=1e-3)
        return optim

    def _compute_loss_on_batch(self, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        obs, act, new_obs = (x.refine_names("B", "H", "R") for x in batch)
        return -self(obs, act, new_obs).mean()

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        loss = self._compute_loss_on_batch(batch)
        self.log("train/loss", loss)
        self.log(self.early_stop_on, loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        loss = self._compute_loss_on_batch(batch)
        self.log("val/loss", loss)
        self.log(self.early_stop_on, loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        # pylint:disable=arguments-differ
        del validation_step_outputs
        self.value_gradient_info("val")

    def test_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int, dataloader_idx: int
    ):
        # pylint:disable=arguments-differ
        del batch_idx, dataloader_idx
        loss = self._compute_loss_on_batch(batch)
        self.log("test/loss", loss)

    def test_epoch_end(self, test_step_outputs):
        # pylint:disable=arguments-differ
        del test_step_outputs
        self.value_gradient_info("test")

    def value_gradient_info(self, prefix: Optional[str] = None):
        with torch.set_grad_enabled(True):
            mc_val, mc_svg = self.monte_carlo_svg(samples=1000)
            analytic_val, analytic_svg = self.analytic_svg(ground_truth=False)
            true_val, true_svg = self.analytic_svg(ground_truth=False)

        prfx = "" if prefix is None else prefix + "/"
        self.log(prfx + "monte_carlo_value", mc_val)
        self.log(prfx + "monte_carlo_svg_norm", linear_feedback_norm(mc_svg))
        self.log(prfx + "analytic_value", analytic_val)
        self.log(prfx + "analytic_svg_norm", linear_feedback_norm(analytic_svg))
        self.log(prfx + "true_value", true_val)
        self.log(prfx + "true_svg_norm", linear_feedback_norm(true_svg))


def test_lightning_model():
    from policy import make_worker

    worker = make_worker(env_config=dict(n_state=2, n_ctrl=2, horizon=100, num_envs=1))
    model = LightningModel(worker.get_policy(), worker.env)

    def print_traj(traj):
        obs, act, rew, new_obs = traj
        print(
            f"""\
        Obs: {obs.shape}, {obs.names}
        Act: {act.shape}, {act.names}
        Rew: {rew.shape}, {rew.names}
        New Obs: {new_obs.shape}, {new_obs.names}\
        """
        )

    print_traj(model.rsample_trajectory([]))
    print_traj(model.rsample_trajectory([10]))

    obs, act, _, new_obs = model.rsample_trajectory([100])
    traj_logp = model(obs, act, new_obs)
    print(f"Traj logp: {traj_logp}, {traj_logp.shape}")

    print("Monte Carlo value:", model.monte_carlo_value(samples=1000))
    print("Analytic value:", model.analytic_value(ground_truth=False))
    print("True value:", model.analytic_value(ground_truth=True))


if __name__ == "__main__":
    test_lightning_model()
