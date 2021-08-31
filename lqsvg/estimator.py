"""Stochastic Value Gradient estimation utilities."""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
from nnrl.nn.critic import QValue
from nnrl.nn.model import StochasticModel
from torch import Tensor, autograd, nn

from lqsvg.data import markovian_state_sampler, trajectory_sampler
from lqsvg.envs import lqr
from lqsvg.torch import named as nt
from lqsvg.torch.nn.env import EnvModule, LQGModule
from lqsvg.torch.nn.policy import TVLinearPolicy
from lqsvg.torch.nn.reward import QuadraticReward
from lqsvg.types import (
    DeterministicPolicy,
    QValueFn,
    RecurrentDynamics,
    RewardFunction,
    StateDynamics,
)

# Model-based estimator
MBEstimator = Callable[[Tensor, int], Tuple[Tensor, lqr.Linear]]

# Model-free estimator
MFEstimator = Callable[[Tensor], Tuple[Tensor, lqr.Linear]]

###############################################################################
# Functional
###############################################################################


def expected_value(rho: lqr.GaussInit, vval: lqr.Quadratic) -> Tensor:
    """Expected state value under a Gaussian initial state distribution.

    https://en.wikipedia.org/wiki/Quadratic_form_(statistics)#Expectation.
    """
    # pylint:disable=invalid-name
    V, v, c = vval
    v = nt.vector_to_matrix(v)
    c = nt.scalar_to_matrix(c)
    mean, cov = rho
    mean = nt.vector_to_matrix(mean)

    value = (
        nt.scalar_to_matrix(nt.trace(cov @ V)) / 2
        + nt.transpose(mean) @ V @ mean / 2
        + nt.transpose(v) @ mean
        + c
    )
    return nt.matrix_to_scalar(value)


def on_policy_value_functions(
    policy: lqr.Linear, dynamics: lqr.LinSDynamics, cost: lqr.QuadCost
) -> Tuple[lqr.Quadratic, lqr.Quadratic]:
    """Computes the action- and state-value functions for a policy.

    Infers LQG dimensions (number of states and actions and horizon length)
    from the dynamics.

    Args:
        policy: parameters of a linear feedback policy in canonical form
        dynamics: linear Gaussian parameters in canonical form
        cost: quadratic cost coefficients in canonical form

    Returs:
        The action- and state-value functions, respectively, as quadratic
        function coefficients in canonical form.
    """
    n_state, n_ctrl, horizon = lqr.dims_from_dynamics(dynamics)
    predictor = lqr.NamedLQGPrediction(n_state, n_ctrl, horizon)
    return predictor(policy, dynamics, cost)


def analytic_value(
    policy: lqr.Linear,
    init: lqr.GaussInit,
    dynamics: lqr.LinSDynamics,
    cost: lqr.QuadCost,
) -> Tensor:
    """Compute the analytic policy performance using the value function.

    Solves the prediction problem for the current policy and uses the
    resulting state-value function to compute the expected return.

    Args:
        policy: parameters of a linear feedback policy in canonical form
        init: Gaussian initial state mean and covariance
        dynamics: linear Gaussian parameters in canonical form
        cost: quadratic cost coefficients in canonical form

    Returns:
        A tensor representing the policy's expected return
    """
    _, vval = on_policy_value_functions(policy, dynamics, cost)
    # noinspection PyTypeChecker
    vval = lqr.Quadratic(*(x.select("H", 0) for x in vval))
    # Have to negate here since vval predicts costs
    value = -expected_value(init, vval)
    return value


def policy_svg(
    policy: TVLinearPolicy,
    value: Tensor,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: bool = False,
) -> lqr.Linear:
    """Computes the policy SVG from the estimated return.

    Args:
        policy: The time-varying linear policy module
        value: Scalar tensor representing the policy's value
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (bool, optional): If ``False``, specifying inputs that were not
            used when computing outputs (and therefore their grad is always zero)
            is an error. Defaults to ``False``.

    Returns:
        A named tuple with attributes
        - K: The value gradient w.r.t. the dynamic gain
        - k: The value gradient w.r.t. the static gain
    """
    # pylint:disable=invalid-name
    grad_K, grad_k = autograd.grad(
        value,
        (policy.K, policy.k),
        retain_graph=retain_graph,
        create_graph=create_graph,
        only_inputs=True,  # Avoid side-effects by computing policy gradients only
        allow_unused=allow_unused,
    )
    return lqr.Linear(grad_K, grad_k)


def analytic_svg(
    policy: TVLinearPolicy,
    init: lqr.GaussInit,
    dynamics: lqr.LinSDynamics,
    cost: lqr.QuadCost,
) -> Tuple[Tensor, lqr.Linear]:
    """Ground-truth policy value and SVG.

    Solves the prediction problem for the current policy and uses the
    resulting state-value function to compute the expected return and its
    gradient w.r.t. policy parameters.

    Returns:
        A tuple with the expected policy return (as a tensor) and a tuple
        of gradients of the expected return w.r.t. each policy parameter
    """
    value = analytic_value(policy.standard_form(), init, dynamics, cost)
    svg = policy_svg(policy, value)
    return value, svg


def mfdpg_surrogate(
    policy: DeterministicPolicy, qvalue: QValueFn
) -> Callable[[Tensor], Tensor]:
    """Returns the surrogate objective function for model-free DPG.

    Args:
        policy: differentiable function of policy parameters and states to
            actions
        qvalue: differentiable function of states and actions to expected
            returns

    Returns:
        A callable mapping a batch of states to the estimated surrogate
        objective
    """

    def surrogate(obs: Tensor) -> Tensor:
        return qvalue(obs, policy(obs)).mean()

    return surrogate


def model_free_estimator(
    policy: TVLinearPolicy, surrogate: Callable[[Tensor], Tensor]
) -> MFEstimator:
    """Model-free value gradient estimator.

    Args:
        policy: linear feedback policy module
        surrogate: function mapping states to surrogate objective for
            differentiation

    Returns:
        Callable from starting state samples to average surrogate objective
        (as a tensor) and its gradients w.r.t. each policy parameter
    """

    def estimator(obs: Tensor) -> Tuple[Tensor, lqr.Linear]:
        surr = surrogate(obs)
        svg = policy_svg(policy, surr)
        return surr, svg

    return estimator


def mfdpg_estimator(policy: TVLinearPolicy, qvalue: QValueFn) -> MFEstimator:
    """Returns the model-free DPG estimator for a given policy and Q-value."""
    surrogate = mfdpg_surrogate(policy, qvalue)
    return model_free_estimator(policy, surrogate)


def dpg_surrogate(
    policy: DeterministicPolicy,
    frozen_policy: DeterministicPolicy,
    dynamics: StateDynamics,
    reward_fn: RewardFunction,
    qvalue: QValueFn,
) -> Callable[[Tensor, int], Tensor]:
    """Returns the surrogate value function for DPG(K).

    Args:
        policy: differentiable function of policy parameters and states to
            actions
        frozen_policy: differentiable function of states to actions but
            blocking action gradients from flowing to parameters
        dynamics: reparameterized state transition sampler
        reward_fn: differentiable function of states and actions to rewards
        qvalue: differentiable function of states and actions to expected
            returns

    Returns:
        A callable mapping a state batch and prediction horizon to the
        estimated surrogate value
    """

    def surrogate(obs: Tensor, n_steps: int = 0) -> Tensor:
        # This is the only action whose gradients w.r.t. policy parameters will
        # be computed
        act = policy(obs)
        partial_return = torch.zeros(())
        for _ in range(n_steps):
            partial_return = partial_return + reward_fn(obs, act)
            obs, _ = dynamics(obs, act)
            act = frozen_policy(obs)

        nstep_return = partial_return + qvalue(obs, act)
        return nstep_return.mean()

    return surrogate


def nstep_estimator(
    policy: TVLinearPolicy, surrogate: Callable[[Tensor, int], Tensor]
) -> MBEstimator:
    """Returns a model-based lookahead value gradient estimator.

    Args:
        policy: linear feedback policy module
        surrogate: function mapping states and number of steps to surrogate
            objective for differentiation

    Returns:
        Callable from starting state samples and prediction horizon to
        surrogate value and its gradients w.r.t. each policy parameter
    """

    def estimator(obs: Tensor, steps: int) -> Tuple[Tensor, lqr.Linear]:
        surr = surrogate(obs, steps)
        svg = policy_svg(policy, surr)
        return surr, svg

    return estimator


def dpg_estimator(
    policy: TVLinearPolicy,
    dynamics: StateDynamics,
    reward_fn: RewardFunction,
    qvalue: QValueFn,
) -> MBEstimator:
    """Returns the model-based DPG estimator.

    Args:
        policy: linear feedback policy module
        dynamics: reparameterized state transition sampler
        reward_fn: differentiable function of states and actions to rewards
        qvalue: differentiable function of states and actions to expected
            returns

    Returns:
        The model-based lookahed version of the DPG estimator
    """
    surrogate = dpg_surrogate(policy, policy.frozen, dynamics, reward_fn, qvalue)
    return nstep_estimator(policy, surrogate)


def maac_markovian(
    dynamics: StateDynamics,
    policy: DeterministicPolicy,
    reward_fn: RewardFunction,
    qvalue: QValueFn,
) -> Callable[[Tensor, int], Tensor]:
    """Returns the surrogate value function for MAAC with markovian model.

    Args:
        dynamics: reparameterized markovian state transition sampler
        policy: differentiable function of policy parameters and states to
            actions
        reward_fn: differentiable function of states and actions to rewards
        qvalue: differentiable function of states and actions to expected
            returns

    Returns:
        A callable mapping a state batch and prediction horizon to the
        estimated surrogate value
    """

    def surrogate(obs: Tensor, n_steps: int) -> Tensor:
        act = policy(obs)
        partial_return = torch.zeros(())
        for _ in range(n_steps):
            partial_return = partial_return + reward_fn(obs, act)
            obs, _ = dynamics(obs, act)
            act = policy(obs)

        nstep_return = partial_return + qvalue(obs, act)
        return nstep_return.mean()

    return surrogate


def maac_recurrent(
    dynamics: RecurrentDynamics,
    policy: DeterministicPolicy,
    reward_fn: RewardFunction,
    qvalue: QValueFn,
) -> Callable[[Tensor, int], Tensor]:
    """Returns the surrogate value function for MAAC with recurrent model.

    Args:
        dynamics: reparameterized recurrent state transition sampler
        policy: differentiable function of policy parameters and states to
            actions
        reward_fn: differentiable function of states and actions to rewards
        qvalue: differentiable function of states and actions to expected
            returns

    Returns:
        A callable mapping a state batch and prediction horizon to the
        estimated surrogate value
    """

    def surrogate(obs: Tensor, n_steps: int) -> Tensor:
        ctx = None
        act = policy(obs)
        partial_return = torch.zeros(())
        for _ in range(n_steps):
            partial_return = partial_return + reward_fn(obs, act)
            obs, _, ctx = dynamics(obs, act, ctx)
            act = policy(obs)

        nstep_return = partial_return + qvalue(obs, act)
        return nstep_return.mean()

    return surrogate


def maac_estimator(
    policy: TVLinearPolicy,
    dynamics: Union[StateDynamics, RecurrentDynamics],
    reward_fn: RewardFunction,
    qvalue: QValueFn,
    recurrent: bool = False,
) -> Callable[[Tensor, int], Tuple[Tensor, lqr.Linear]]:
    """Returns the value gradient estimator from MAAC.

    Args:
        policy: linear feedback policy module
        dynamics: reparameterized state transition sampler
        reward_fn: differentiable function of states and actions to rewards
        qvalue: differentiable function of states and actions to expected
            returns
        recurrent: whether the dynamics model is recurrent

    Returns:
        A model-based n-step SVG estimator
    """
    surrogate_fn = maac_recurrent if recurrent else maac_markovian
    surrogate = surrogate_fn(dynamics, policy, reward_fn, qvalue)
    return nstep_estimator(policy, surrogate)


###############################################################################
# Deprecated
###############################################################################


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
            nt.scalar_to_matrix(nt.trace(cov @ V)) / 2
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
        state_transitioner = markovian_state_sampler(model.trans, model.trans.rsample)
        # noinspection PyTypeChecker
        self._rsampler = trajectory_sampler(
            policy, model.init.sample, state_transitioner, model.reward
        )

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
        self.policy.train()
        self.model.eval()
        all_obs, act, rew, logp = self._rsampler(self.model.horizon, sample_shape)
        decision_steps = torch.arange(self.model.horizon).int()
        obs = nt.index_select(all_obs, "H", decision_steps)
        new_obs = nt.index_select(all_obs, "H", decision_steps + 1)
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


class DPG(nn.Module):
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
        # This is the only action whose gradients w.r.t. policy parameters will
        # be computed
        act = self.policy(obs)
        partial_return = torch.zeros(())
        for _ in range(n_steps):
            partial_return = partial_return + self.reward(obs, act)
            obs, _ = self.transition.rsample(self.transition(obs, act))
            act = self.policy.frozen(obs)

        nstep_return = partial_return + self.qvalue(obs, act)
        return nstep_return.mean()

    def forward(self, obs: Tensor, n_steps: int = 0) -> tuple[Tensor, lqr.Linear]:
        """Monte Carlo estimate of the Stochastic Value Gradient.

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


class MAAC(DPG):
    """Computes a bootstrapped SVG estimate as in MAAC."""

    def surrogate(self, obs: Tensor, n_steps: int = 0) -> Tensor:
        self.policy.train()
        self.transition.eval()
        self.reward.eval()
        self.qvalue.eval()
        act = self.policy(obs)
        partial_return = torch.zeros(())
        for _ in range(n_steps):
            partial_return = partial_return + self.reward(obs, act)
            obs, _ = self.transition.rsample(self.transition(obs, act))
            act = self.policy(obs)

        nstep_return = partial_return + self.qvalue(obs, act)
        return nstep_return.mean()
