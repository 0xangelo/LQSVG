# pylint:disable=missing-docstring,unsubscriptable-object
from __future__ import annotations

import itertools
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from raylab.policy.modules.actor import DeterministicPolicy
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.envs.lqr.modules import TVLinearNormalParams
from lqsvg.policy.time_varying_linear import LQGPolicy

from .utils import linear_feedback_cossim
from .utils import linear_feedback_norm


class ExpectedValue(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, rho: lqr.GaussInit, vval: lqr.Quadratic):
        """Expected cost given mean and covariance matrix of the initial state.

        https://en.wikipedia.org/wiki/Quadratic_form_(statistics)#Expectation.
        """
        # pylint:disable=invalid-name,no-self-use
        V, v, c = vval
        v = nt.vector_to_matrix(v)
        c = nt.scalar_to_matrix(c)
        mean, cov = rho
        mean = nt.vector_to_matrix(mean)

        value = (
            nt.trace(cov @ V).align_to(..., "R", "C") / 2
            + nt.transpose(mean) @ V @ mean
            + nt.transpose(v) @ mean
            + c
        )
        return nt.matrix_to_scalar(value)


class PolicyLoss(nn.Module):
    def __init__(self, n_state: int, n_ctrl: int, horizon: int):
        super().__init__()
        self.predict = lqr.NamedLQGPrediction(n_state, n_ctrl, horizon)
        self.expected = ExpectedValue()

    def forward(
        self,
        policy: lqr.Linear,
        dynamics: lqr.LinSDynamics,
        cost: lqr.QuadCost,
        rho: lqr.GaussInit,
    ):
        _, vval = self.predict(policy, dynamics, cost)
        vval = tuple(x.select("H", 0) for x in vval)
        cost = self.expected(rho, vval)
        return cost


def policy_svg(policy: DeterministicPolicy, value: Tensor) -> lqr.Linear:
    """Computes the policy SVG from the estimated return."""
    # pylint:disable=invalid-name
    policy.zero_grad(set_to_none=True)
    value.backward()
    K, k = policy.standard_form()
    return K.grad.clone(), k.grad.clone()


class MonteCarloSVG(nn.Module):
    """Computes the Monte Carlo aproximation for SVGs."""

    def __init__(self, policy: DeterministicPolicy, model: LQGModule):
        super().__init__()
        self.policy = policy
        self.model = model

    def rsample_trajectory(
        self, sample_shape: torch.Size = torch.Size()
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample trajectory for Stochastic Value Gradients.

        Sample trajectory using model with reparameterization trick and
        deterministic actor.

        Args:
            sample_shape: shape for batched trajectory samples

        Returns:
            Trajectory sample following the policy and its corresponding
            log-likelihood under the model
        """
        sample_shape = torch.Size(sample_shape)

        batch = []
        obs, logp = self.model.init.rsample(sample_shape)
        obs = obs.refine_names(*(f"B{i+1}" for i, _ in enumerate(sample_shape)), ...)
        for _ in range(self.model.horizon):
            act = self.policy(obs)
            rew = self.model.reward(obs, act)
            # No sample_shape needed since initial states are batched
            new_obs, logp_t = self.model.trans.rsample(self.model.trans(obs, act))

            batch += [(obs, act, rew, new_obs)]
            logp = logp + logp_t
            obs = new_obs

        obs, act, rew, new_obs = (nt.stack_horizon(*x) for x in zip(*batch))
        return obs, act, rew, new_obs, logp

    def rsample_return(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Sample return for Stochastic Value Gradients.

        Sample reparameterized trajectory and compute empirical return.

        Args:
            sample_shape: shape for batched return samples

        Returns:
            Sample of the policy return
        """
        _, _, rew, _, _ = self.rsample_trajectory(sample_shape)
        # Reduce over horizon
        ret = rew.sum("H")
        return ret

    def value(self, samples: int = 1) -> Tensor:
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

    def forward(self, samples: int = 1) -> tuple[Tensor, lqr.Linear]:
        """Monte Carlo estimate of the Stochastic Value Gradient.

        Args:
            samples: number of samples

        Returns:
            A tuple with the expected policy return (as a tensor) and an
            iterable of gradients of the expected return with respect to
            each policy parameter
        """
        mc_performance = self.value(samples)
        svg = policy_svg(self.policy, mc_performance)
        return mc_performance, svg


class AnalyticSVG(nn.Module):
    """Computes the SVG analytic via LQG prediction."""

    def __init__(self, policy: DeterministicPolicy, model: LQGModule):
        super().__init__()
        self.policy = policy
        self.model = model
        self.policy_loss = PolicyLoss(model.n_state, model.n_ctrl, model.horizon)

    def value(self) -> Tensor:
        """Compute the analytic policy performance using the value function.

        Solves the prediction problem for the current policy and uses the
        resulting state-value function to compute the expected return.

        Returns:
            A tensor representing the policy's expected return
        """
        policy = self.policy.standard_form()
        dynamics, cost, init = self.model.standard_form()
        value = -self.policy_loss(policy, dynamics, cost, init)
        return value

    def forward(self) -> tuple[Tensor, lqr.Linear]:
        """Compute the analytic SVG using the value function.

        Solves the prediction problem for the current policy and uses the
        resulting state-value function to compute the expected return and its
        gradient w.r.t. policy parameters.

        Returns:
            A tuple with the expected policy return (as a tensor) and an
            iterable of gradients of the expected return w.r.t. each policy
            parameter
        """
        value = self.value()
        svg = policy_svg(self.policy, value)
        return value, svg


def glorot_init_model(model: nn.Module):
    """Apply Glorot initialization to every time-varying linear submodule.

    Args:
        model: neural network PyTorch module
    """

    def initialize(module: nn.Module):
        if isinstance(module, TVLinearNormalParams):
            nn.init.xavier_uniform_(module.F)
            nn.init.constant_(module.f, 0.0)

    model.apply(initialize)


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    actor: DeterministicPolicy
    model: LQGModule
    mdp: LQGModule
    gold_standard: tuple[Tensor, lqr.Linear]
    early_stop_on: str = "val/loss"

    def __init__(self, policy: LQGPolicy, env: TorchLQGMixin):
        super().__init__()
        self.actor = policy.module.actor
        self.model = policy.module.model
        self.mdp = env.module
        self.monte_carlo_svg = MonteCarloSVG(self.actor, self.model)
        self.analytic_svg = AnalyticSVG(self.actor, self.model)

        self.hparams.learning_rate = 1e-3
        self.hparams.mc_samples = 256

        self.gold_standard = AnalyticSVG(self.actor, self.mdp)()
        glorot_init_model(self.model)

    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Batched trajectory log prob."""
        # pylint:disable=arguments-differ
        return self.model.log_prob(obs, act, new_obs)

    def configure_optimizers(self):
        params = nn.ParameterList(
            itertools.chain(self.model.trans.parameters(), self.model.init.parameters())
        )
        optim = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optim

    def _compute_loss_on_batch(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        del batch_idx
        obs, act, new_obs = (x.refine_names("B", "H", "R") for x in batch)
        return -self(obs, act, new_obs).mean()

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("val/loss", loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        # pylint:disable=arguments-differ
        del validation_step_outputs
        self.value_gradient_info("val")

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("test/loss", loss)

    def test_epoch_end(self, test_step_outputs):
        # pylint:disable=arguments-differ
        del test_step_outputs
        self.log_gold_standard()
        self.value_gradient_info("test")

    def value_gradient_info(self, prefix: Optional[str] = None):
        true_val, true_svg = self.gold_standard
        with torch.enable_grad():
            mc_val, mc_svg = self.monte_carlo_svg(samples=self.hparams.mc_samples)
            analytic_val, analytic_svg = self.analytic_svg()
        self.zero_grad(set_to_none=True)

        prfx = "" if prefix is None else prefix + "/"
        self.log(prfx + "monte_carlo_value", mc_val)
        self.log(prfx + "monte_carlo_svg_norm", linear_feedback_norm(mc_svg))
        self.log(prfx + "analytic_value", analytic_val)
        self.log(prfx + "analytic_svg_norm", linear_feedback_norm(analytic_svg))

        self.log(prfx + "monte_carlo_diff", mc_val - true_val)
        self.log(prfx + "analytic_diff", analytic_val - true_val)
        self.log(prfx + "monte_carlo_cossim", linear_feedback_cossim(mc_svg, true_svg))
        self.log(
            prfx + "analytic_cossim", linear_feedback_cossim(analytic_svg, true_svg)
        )

    def log_gold_standard(self):
        """Logs gold standard value and gradient."""
        true_val, true_svg = self.gold_standard
        self.log("true_value", true_val)
        self.log("true_svg_norm", linear_feedback_norm(true_svg))


@nt.suppress_named_tensor_warning()
def test_lightning_model():
    from .worker import make_worker

    worker = make_worker(env_config=dict(n_state=2, n_ctrl=2, horizon=100, num_envs=1))
    model = LightningModel(worker.get_policy(), worker.env)

    def print_traj(traj):
        obs, act, rew, new_obs, logp = traj
        print(
            f"""\
        Obs: {obs.shape}, {obs.names}
        Act: {act.shape}, {act.names}
        Rew: {rew.shape}, {rew.names}
        New Obs: {new_obs.shape}, {new_obs.names}
        Logp: {logp.shape}, {logp.names}\
        """
        )

    monte_carlo = model.monte_carlo_svg
    analytic = model.analytic_svg
    true_mc = MonteCarloSVG(model.actor, model.mdp)

    print("Model sample:")
    print_traj(monte_carlo.rsample_trajectory([]))
    print("Batched model sample:")
    print_traj(monte_carlo.rsample_trajectory([10]))
    print("MDP sample:")
    print_traj(true_mc.rsample_trajectory([]))

    obs, act, _, new_obs, sample_logp = monte_carlo.rsample_trajectory([10])
    print(f"RSample logp: {sample_logp}, {sample_logp.shape}")
    traj_logp = model(obs, act, new_obs)
    print(f"Traj logp: {traj_logp}, {traj_logp.shape}")
    print("Model logp of MDP sample:")
    obs, act, _, new_obs, _ = true_mc.rsample_trajectory([10])
    traj_logp = model(obs, act, new_obs)
    print(traj_logp, traj_logp.shape)

    print("Monte Carlo value:", monte_carlo.value(samples=256))
    print("Analytic value:", analytic.value())
    print("True value:", model.gold_standard[0])


if __name__ == "__main__":
    test_lightning_model()
