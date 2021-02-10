"""
Linear Quadratic Regulator (LQR):
Please see http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
for notation and more details on LQR.
"""
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .named import refine_cost_input
from .named import refine_dynamics_input
from .named import refine_linear_input
from .named import refine_linear_output
from .named import refine_quadratic_output
from .named import refine_sdynamics_input
from .named import unnamed
from .types import LinDynamics
from .types import Linear
from .types import LinSDynamics
from .types import LQG
from .types import LQR
from .types import QuadCost
from .types import Quadratic


def transpose(tensor: Tensor):
    """Transposes a batched matrix by the last two dimensions."""
    return tensor.transpose(-2, -1)


# noinspection PyMethodMayBeStatic
class BuildQuadraticMixin:
    """Adds methods to progressively build batched quadratic parameters."""

    # pylint:disable=missing-function-docstring,no-self-use,invalid-name

    def init_quadratic(self):
        A: List[Tensor] = []
        b: List[Tensor] = []
        c: List[Tensor] = []
        return A, b, c

    def assign_quadratic(
        self, quads: Tuple[List[Tensor], List[Tensor], List[Tensor]], quad: Quadratic
    ):
        As, bs, cs = quads
        A, b, c = quad
        As.insert(0, A)
        bs.insert(0, b)
        cs.insert(0, c)

    def stack_quadratic(
        self,
        quad: Tuple[List[Tensor], List[Tensor], List[Tensor]],
    ):
        A, b, c = quad
        return torch.stack(A), torch.stack(b), torch.stack(c)


# noinspection PyMethodMayBeStatic
class BuildLinearMixin:
    """Adds methods to progressively build batched linear parameters."""

    # pylint:disable=missing-function-docstring,no-self-use,invalid-name
    def init_linear(self):
        K: List[Tensor] = []
        k: List[Tensor] = []
        return K, k

    def assign_linear(self, linears: Tuple[List[Tensor], List[Tensor]], linear: Linear):
        Ks, ks = linears
        K, k = linear
        Ks.insert(0, K)
        ks.insert(0, k)

    def stack_linear(self, linears: Tuple[List[Tensor], List[Tensor]]):
        Ks, ks = linears
        return torch.stack(Ks), torch.stack(ks)


# noinspection PyPep8Naming,PyMethodMayBeStatic
class LQRPrediction(BuildQuadraticMixin, nn.Module):
    """Linear Quadratic Regulator prediction.

    Computes the cost-to-go functions for a given time-varying linear policy in
    a linear quadratic problem.

    Expects and returns all tensors with 4 dimensions: horizon, batch, row,
    and column.

    Args:
        n_state: size of state vector
        n_ctrl: size of action vector
        horizon: number of decision steps
    """

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring,no-self-use
    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        horizon: int,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.horizon = horizon

        self.n_tau = self.n_state + self.n_ctrl

    def forward(self, policy: Linear, dynamics: LinDynamics, cost: QuadCost):
        # pylint:disable=arguments-differ
        system = self.standardize_system(dynamics, cost)

        v_val = self.init_quadratic()
        q_val = self.init_quadratic()

        Vs = self.final_V(system)
        self.assign_quadratic(v_val, Vs)

        for i in range(self.horizon):  # Effectively solving backwards through time
            Qs, Vs = self.single_step(
                self.system_at(system, self.horizon - i - 1),
                self.policy_at(policy, self.horizon - i - 1),
                Vs,
            )
            self.assign_quadratic(q_val, Qs)
            self.assign_quadratic(v_val, Vs)

        return self.stack_quadratic(q_val), self.stack_quadratic(v_val)

    def standardize_system(self, dynamics: LinDynamics, cost: QuadCost):
        F, f = dynamics
        C, c = cost
        return F, f, C, c

    def final_V(self, system: LQR) -> Quadratic:
        F, _, _, _ = system
        batch_shape = F.shape[1:-2]
        n_state = self.n_state
        mat_shape = batch_shape + (n_state, n_state)
        vec_shape = mat_shape[:-1] + (1,)
        scl_shape = mat_shape[:-2] + (1, 1)

        Vs = (torch.zeros(mat_shape), torch.zeros(vec_shape), torch.zeros(scl_shape))
        return Vs

    def system_at(self, system: LQR, time: int):
        F, f, C, c = system
        return F[time], f[time], C[time], c[time]

    def policy_at(self, policy: Linear, time: int):
        K, k = policy
        return K[time], k[time]

    def single_step(self, system: LQR, Ks: Linear, Vs: Quadratic):
        Qs = self.compute_Q(system, Vs)
        Vs = self.compute_V(Qs, Ks)
        return Qs, Vs

    def compute_Q(self, system: LQR, Vs: Quadratic):
        F, f, C, c = system
        V, v, vc = Vs

        FV = transpose(F) @ V
        Q = C + FV @ F
        q = c + FV @ f + transpose(F) @ v
        qc = transpose(f) @ V @ f / 2 + transpose(f) @ v + vc
        return Q, q, qc

    def compute_V(self, Qs: Quadratic, Ks: Linear):
        # pylint:disable=too-many-locals
        n_state = self.n_state
        Q, q, qc = Qs
        Q_uu = Q[..., n_state:, n_state:]
        Q_ux = Q[..., n_state:, :n_state]
        q_u = q[..., n_state:, :]
        Q_xx = Q[..., :n_state, :n_state]
        Q_xu = Q[..., :n_state, n_state:]
        q_x = q[..., :n_state, :]

        K, k = Ks
        KTQ_uu = transpose(K) @ Q_uu

        V = Q_xx + Q_xu @ K + transpose(K) @ Q_ux + KTQ_uu @ K
        v = Q_xu @ k + KTQ_uu @ k + q_x + transpose(K) @ q_u
        vc = transpose(k) @ Q_uu @ k / 2 + transpose(k) @ q_u + qc
        return V, v, vc


# noinspection PyPep8Naming,PyMethodOverriding
class LQRControl(BuildLinearMixin, LQRPrediction):
    """Linear Quadratic Regulator control.

    Computes the optimal time-varying linear policy and cost-to-go functions for
    a given linear quadratic problem.

    Args:
        n_state: size of state vector
        n_ctrl: size of action vector
        horizon: number of decision steps
    """

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring,no-self-use
    def forward(self, dynamics: LinDynamics, cost: QuadCost):
        # pylint:disable=arguments-differ
        system = self.standardize_system(dynamics, cost)

        policy = self.init_linear()
        v_val = self.init_quadratic()
        q_val = self.init_quadratic()

        Vs = self.final_V(system)
        self.assign_quadratic(v_val, Vs)

        for i in range(self.horizon):  # Effectively solving backwards through time
            Qs, Ks, Vs = self.single_step(
                self.system_at(system, self.horizon - i - 1), Vs
            )
            self.assign_quadratic(q_val, Qs)
            self.assign_linear(policy, Ks)
            self.assign_quadratic(v_val, Vs)

        return (
            self.stack_linear(policy),
            self.stack_quadratic(q_val),
            self.stack_quadratic(v_val),
        )

    def single_step(
        self, system: LQR, Vs: Quadratic
    ):  # pylint:disable=arguments-differ
        Qs = self.compute_Q(system, Vs)
        Ks = self.compute_K(Qs)
        Vs = self.compute_V(Qs, Ks)
        return Qs, Ks, Vs

    def compute_K(self, Qs: Quadratic):
        n_state = self.n_state
        Q, q, _ = Qs
        Q_uu = Q[..., n_state:, n_state:]
        Q_ux = Q[..., n_state:, :n_state]
        q_u = q[..., n_state:, :]

        inv_Q_uu = Q_uu.inverse()

        K = -inv_Q_uu @ Q_ux
        k = -inv_Q_uu @ q_u
        return K, k


# noinspection PyPep8Naming,PyMethodMayBeStatic
class LQGMixin:
    """Modifies LQR to handle time-varying stochastic linear quadratic systems."""

    # pylint:disable=invalid-name,missing-function-docstring,no-self-use
    def standardize_system(self, dynamics: LinSDynamics, cost: QuadCost):
        F, f, W = dynamics
        C, c = cost
        return F, f, W, C, c

    def final_V(self, system: LQG) -> Quadratic:
        F, _, _, _, _ = system
        batch_shape = F.shape[1:-2]
        n_state = self.n_state
        mat_shape = batch_shape + (n_state, n_state)
        vec_shape = mat_shape[:-1] + (1,)
        scl_shape = mat_shape[:-2] + (1, 1)

        Vs = (torch.zeros(mat_shape), torch.zeros(vec_shape), torch.zeros(scl_shape))
        return Vs

    def system_at(self, system: LQG, time: int):
        F, f, W, C, c = system
        return F[time], f[time], W[time], C[time], c[time]

    def compute_Q(self, system: LQG, Vs: Quadratic):
        F, f, W, C, c = system
        V, v, vc = Vs

        FV = transpose(F) @ V
        Q = C + FV @ F
        q = c + FV @ f + transpose(F) @ v
        qc = (
            transpose(f) @ V @ f / 2
            + transpose(f) @ v
            + vc
            + self.matrix_trace(W @ V) / 2
        )
        return Q, q, qc

    def matrix_trace(self, mat: Tensor) -> Tensor:
        """Returns the trace of a matrix as a unitary matrix (keeps dims)."""
        return torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)


# noinspection PyPep8Naming
class LQGPrediction(LQGMixin, LQRPrediction):
    """Linear Quadratic Gaussian prediction

    Computes the cost-to-go functions for a given time-varying linear policy
    and time-varying stochastic linear quadratic system.

    Args:
        n_state: size of state vector
        n_ctrl: size of action vector
        horizon: number of decision steps
    """

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring,no-self-use
    def forward(self, policy: Linear, dynamics: LinSDynamics, cost: QuadCost):
        system = self.standardize_system(dynamics, cost)

        v_val = self.init_quadratic()
        q_val = self.init_quadratic()

        Vs = self.final_V(system)
        self.assign_quadratic(v_val, Vs)

        for i in range(self.horizon):  # Effectively solving backwards through time
            Qs, Vs = self.single_step(
                self.system_at(system, self.horizon - i - 1),
                self.policy_at(policy, self.horizon - i - 1),
                Vs,
            )
            self.assign_quadratic(q_val, Qs)
            self.assign_quadratic(v_val, Vs)

        return self.stack_quadratic(q_val), self.stack_quadratic(v_val)

    def single_step(self, system: LQG, Ks: Linear, Vs: Quadratic):
        Qs = self.compute_Q(system, Vs)
        Vs = self.compute_V(Qs, Ks)
        return Qs, Vs


# noinspection PyPep8Naming
class LQGControl(LQGMixin, LQRControl):
    """Linear Quadratic Gaussian control.

    Computes the optimal policy and cost-to-go functions for a time-varying
    stochastic linear quadratic system.

    Args:
        n_state: size of state vector
        n_ctrl: size of action vector
        horizon: number of decision steps
    """

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring,no-self-use
    def forward(self, dynamics: LinSDynamics, cost: QuadCost):
        system = self.standardize_system(dynamics, cost)

        policy = self.init_linear()
        v_val = self.init_quadratic()
        q_val = self.init_quadratic()

        Vs = self.final_V(system)
        self.assign_quadratic(v_val, Vs)

        for i in range(self.horizon):  # Effectively solving backwards through time
            Qs, Ks, Vs = self.single_step(
                self.system_at(system, self.horizon - i - 1), Vs
            )
            self.assign_quadratic(q_val, Qs)
            self.assign_linear(policy, Ks)
            self.assign_quadratic(v_val, Vs)

        return (
            self.stack_linear(policy),
            self.stack_quadratic(q_val),
            self.stack_quadratic(v_val),
        )

    def single_step(self, system: LQG, Vs: Quadratic):
        Qs = self.compute_Q(system, Vs)
        Ks = self.compute_K(Qs)
        Vs = self.compute_V(Qs, Ks)
        return Qs, Ks, Vs


# ==============================================================================
# Named Tensors
# ==============================================================================


class NamedLQRPrediction(nn.Module):
    """Adapter between core LQRPrediction and named tensors.

    Allows using named tensors with a TorchScript-compatible solver.
    """

    # pylint:disable=abstract-method,invalid-name,missing-function-docstring,no-self-use
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.solver = LQRPrediction(*args, **kwargs)

    def forward(self, policy: Linear, dynamics: LinDynamics, cost: QuadCost):
        policy = refine_linear_input(policy)
        dynamics = refine_dynamics_input(dynamics)
        cost = refine_cost_input(cost)

        policy, dynamics, cost = map(unnamed, (policy, dynamics, cost))
        qval, vval = self.solver(policy, dynamics, cost)

        qval = refine_quadratic_output(qval)
        vval = refine_quadratic_output(vval)
        return qval, vval


class NamedLQRControl(nn.Module):
    """Adapter between core LQRControl and named tensors.

    Allows using named tensors with a TorchScript-compatible solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.solver = LQRControl(*args, **kwargs)

    def forward(self, dynamics: LinDynamics, cost: QuadCost):
        # pylint:disable=missing-function-docstring
        dynamics = refine_dynamics_input(dynamics)
        cost = refine_cost_input(cost)

        dynamics, cost = map(unnamed, (dynamics, cost))
        pistar, qstar, vstar = self.solver(dynamics, cost)

        pistar = refine_linear_output(pistar)
        qstar = refine_quadratic_output(qstar)
        vstar = refine_quadratic_output(vstar)
        return pistar, qstar, vstar


class NamedLQGPrediction(nn.Module):
    """Adapter between core LQGPrediction and named tensors.

    Allows using named tensors with a TorchScript-compatible solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.solver = LQGPrediction(*args, **kwargs)

    def forward(self, policy: Linear, dynamics: LinSDynamics, cost: QuadCost):
        # pylint:disable=missing-function-docstring
        policy = refine_linear_input(policy)
        dynamics = refine_sdynamics_input(dynamics)
        cost = refine_cost_input(cost)

        policy, dynamics, cost = map(unnamed, (policy, dynamics, cost))
        qval, vval = self.solver(policy, dynamics, cost)

        qval = refine_quadratic_output(qval)
        vval = refine_quadratic_output(vval)
        return qval, vval


class NamedLQGControl(nn.Module):
    """Adapter between core LQGControl and named tensors.

    Allows using named tensors with a TorchScript-compatible solver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.solver = LQGControl(*args, **kwargs)

    def forward(self, dynamics: LinSDynamics, cost: QuadCost):
        # pylint:disable=missing-function-docstring
        dynamics = refine_sdynamics_input(dynamics)
        cost = refine_cost_input(cost)

        dynamics, cost = map(unnamed, (dynamics, cost))
        pistar, qstar, vstar = self.solver(dynamics, cost)

        pistar = refine_linear_output(pistar)
        qstar = refine_quadratic_output(qstar)
        vstar = refine_quadratic_output(vstar)
        return pistar, qstar, vstar
