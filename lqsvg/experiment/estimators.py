"""Stochastic Value Gradient estimation utilities."""
from __future__ import annotations

import torch
import torch.nn as nn
from raylab.policy.modules.critic import QValue
from raylab.policy.modules.model import StochasticModel
from torch import Tensor

from lqsvg.envs import lqr
from lqsvg.envs.lqr.modules import LQGModule, QuadraticReward
from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.policy.modules import TVLinearPolicy
from lqsvg.torch import named as nt


class ExpectedValue(nn.Module):
    # pylint:disable=missing-docstring
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
            + nt.transpose(mean) @ V @ mean / 2
            + nt.transpose(v) @ mean
            + c
        )
        return nt.matrix_to_scalar(value)


class PolicyLoss(nn.Module):
    # pylint:disable=missing-docstring
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


def policy_svg(policy: TVLinearPolicy, value: Tensor) -> lqr.Linear:
    """Computes the policy SVG from the estimated return."""
    # pylint:disable=invalid-name
    policy.zero_grad(set_to_none=True)
    value.backward()
    K, k = policy.standard_form()
    return K.grad.clone(), k.grad.clone()


class AnalyticSVG(nn.Module):
    """Computes the SVG analytically via LQG prediction."""

    def __init__(self, policy: TVLinearPolicy, model: LQGModule):
        super().__init__()
        self.policy = policy
        self.model = model
        self.policy_loss = PolicyLoss(model.n_state, model.n_ctrl, model.horizon)

    def value(self) -> Tensor:
        """Compute the analytic policy performance using the value function.

        Solves the prediction problem for the current policy and uses the
        resulting state-value function to compute the expected return.

        Note:
            Ensures the LQG model is put in evaluation mode and the policy in
            training mode.

        Returns:
            A tensor representing the policy's expected return
        """
        self.policy.train()
        self.model.eval()
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
            A tuple with the expected policy return (as a tensor) and a tuple
            of gradients of the expected return w.r.t. each policy parameter
        """
        value = self.value()
        svg = policy_svg(self.policy, value)
        return value, svg


class MonteCarloSVG(nn.Module):
    """Computes the Monte Carlo aproximation for SVGs."""

    def __init__(self, policy: TVLinearPolicy, model: EnvModule):
        super().__init__()
        self.policy = policy
        self.model = model

    def rsample_trajectory(
        self, sample_shape: torch.Size = torch.Size()
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample trajectory for Stochastic Value Gradients.

        Sample trajectory using model with reparameterization trick and
        deterministic actor.

        Note:
            Ensures the environment model is put in evaluation mode and the
            policy in training mode.

        Args:
            sample_shape: shape for batched trajectory samples

        Returns:
            Trajectory sample following the policy and its corresponding
            log-likelihood under the model
        """
        sample_shape = torch.Size(sample_shape)
        self.policy.train()
        self.model.eval()

        batch = []
        # noinspection PyTypeChecker
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
        # noinspection PyTypeChecker
        rets = self.rsample_return((samples,))
        # Reduce over the first (and only) batch dimension
        mc_performance = rets.mean("B1")
        return mc_performance

    def forward(self, samples: int = 1) -> tuple[Tensor, lqr.Linear]:
        """Monte Carlo estimate of the Stochastic Value Gradient.

        Args:
            samples: number of samples

        Returns:
            A tuple with the expected policy return (as a tensor) and a tuple
            of gradients of the expected return w.r.t. each policy parameter
        """
        mc_performance = self.value(samples)
        svg = policy_svg(self.policy, mc_performance)
        return mc_performance, svg


class BootstrappedSVG(nn.Module):
    """Computes a bootstrapped SVG estimate via the DPG theorem."""

    def __init__(
        self,
        policy: TVLinearPolicy,
        trans_model: StochasticModel,
        rew_model: QuadraticReward,
        qvalue_model: QValue,
    ):
        super().__init__()
        self.policy = policy
        self.transition = trans_model
        self.reward = rew_model
        self.qvalue = qvalue_model

    def surrogate(self, obs: Tensor, n_steps: int = 0) -> Tensor:
        """Monte Carlo estimate of the surrogate objective function.

        Note:
            Ensures the transition, reward, and value function models are put
            in evaluation mode and the policy in training mode.

        Args:
            obs: starting state samples
            n_steps: number of steps to rollout the model before bootstrapping

        Returns:
            The average surrogate objective as a tensor.
        """
        self.policy.train()
        self.transition.eval()
        self.reward.eval()
        self.qvalue.eval()
        # This is the only action through which gradients can propagate
        act = self.policy(obs)
        partial_return = torch.zeros(())
        for _ in range(n_steps):
            partial_return = partial_return + self.reward(obs, act)
            obs, _ = self.transition.rsample(self.transition(obs, act))
            act = self.policy(obs).detach()

        nstep_return = partial_return + self.qvalue(obs, act)
        return nstep_return.mean()

    def forward(self, obs: Tensor, n_steps: int = 0) -> tuple[Tensor, lqr.Linear]:
        """Monte Carlo estimate of the bootstrapped Stochastic Value Gradient.

        Args:
            obs: starting state samples
            n_steps: number of steps to rollout the model before bootstrapping

        Returns:
            A tuple with the average surrogate objective (as a tensor) and a
            tuple of gradients of the surrogate w.r.t. each policy parameter
        """
        surrogate = self.surrogate(obs, n_steps)
        svg = policy_svg(self.policy, surrogate)
        return surrogate, svg
