# pylint:disable=missing-docstring
from typing import Callable, Tuple

import pytorch_lightning as pl
import torch
from nnrl.nn.actor import DeterministicPolicy
from nnrl.nn.critic import HardValue, MLPQValue, QValue, QValueEnsemble, VValue
from nnrl.nn.model import StochasticModel
from nnrl.nn.utils import update_polyak
from nnrl.types import TensorDict
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor, autograd, nn
from wandb_util import with_prefix

from lqsvg import analysis, estimator
from lqsvg.envs import lqr
from lqsvg.torch import named as nt
from lqsvg.torch.nn import (
    LQGModule,
    QuadQValue,
    QuadraticReward,
    QuadVValue,
    TVLinearPolicy,
)

TDBatch = Tuple[Tensor, Tensor, Tensor, Tensor]


def refine_tdbatch(batch: TDBatch) -> TDBatch:
    obs, act, rew, new_obs = (t.refine_names("B", ...) for t in batch)
    obs, act, new_obs = nt.vector(obs, act, new_obs)
    return obs, act, rew, new_obs


def value_learning(module: "LightningQValue", batch: TDBatch) -> Tensor:
    """Returns the temporal difference error induced by the value function."""
    module.qval: ClippedQValue

    obs, act, rew, new_obs = batch
    with torch.no_grad():
        target = nt.unnamed(rew + module.target_vval(new_obs))
    loss_fn = nn.MSELoss()
    values = nt.unnamed(*module.qval.q_values(obs, act))
    return torch.stack([loss_fn(v, target) for v in values]).sum()


def value_gradient_learning(module: "LightningQValue", batch: TDBatch) -> Tensor:
    """Returns the value gradient error induced by the value function."""
    policy: DeterministicPolicy = module.policy
    dynamics: StochasticModel = module.lqg.trans
    reward: QuadraticReward = module.lqg.reward

    obs, _, _, _ = batch
    act = policy(obs)
    rew = reward(obs, act)
    new_obs, _ = dynamics.rsample(dynamics(obs, act))

    value = module.qval(obs, act)
    target = rew + module.target_vval(new_obs)

    (act_grad,) = autograd.grad(torch.sum(value - target), act, create_graph=True)
    loss = torch.norm(act_grad)
    return loss


LOSS = {
    "TD(1)": value_learning,
    "VGL(1)": value_gradient_learning,
}


class ClippedQValue(QValue):
    """Q-value computed as the minimum among Q-values in an ensemble."""

    def __init__(self, q_values: QValueEnsemble):
        super().__init__()
        self.q_values = q_values

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        values = self.q_values(obs, action)
        mininum, _ = torch.stack(nt.unnamed(*values), dim=0).min(dim=0)
        return mininum.refine_names(*values[0].names)


def td_modules(
    policy: DeterministicPolicy, qval_fn: Callable[..., QValue], hparams: AttributeDict
) -> Tuple[ClippedQValue, ClippedQValue, VValue]:
    """Creates modules for Streamlined Off-Policy TD learning."""
    qval = ClippedQValue(QValueEnsemble([qval_fn() for _ in range(2)]))
    target_qval = ClippedQValue(QValueEnsemble([qval_fn() for _ in range(2)]))
    target_qval.requires_grad_(False)
    # Syncronize initial Q values
    target_qval.load_state_dict(qval.state_dict())

    # Target state-values
    target_noise = hparams.model.get("target_policy_sigma", 0.3)
    if target_noise:
        target_policy = DeterministicPolicy.add_gaussian_noise(
            policy, noise_stddev=target_noise
        )
    else:
        target_policy = policy
    target_vval = HardValue(target_policy, target_qval)
    return qval, target_qval, target_vval


def vgl_modules(
    policy: DeterministicPolicy, qval_fn: Callable[..., QValue], _: AttributeDict
) -> Tuple[QValue, QValue, VValue]:
    """Creates modules for Value Gradient Learning (VGL)."""
    qval = qval_fn()

    target_qval = qval_fn().requires_grad_(False)
    target_qval.load_state_dict(qval.state_dict())

    target_vval = HardValue(policy, qval)
    return qval, target_qval, target_vval


def qval_constructor(
    policy: TVLinearPolicy, hparams: AttributeDict
) -> Callable[..., QValue]:
    n_state, n_ctrl, horizon = lqr.dims_from_policy(policy.standard_form())
    kind = hparams.model["type"]
    if kind == "mlp":
        obs_space, act_space = lqr.spaces_from_dims(n_state, n_ctrl, horizon)
        spec = MLPQValue.spec_cls(units=hparams.model["hunits"], activation="ReLU")
        return lambda: MLPQValue(obs_space, act_space, spec)
    if kind == "quad":
        rng = hparams.seed
        return lambda: QuadQValue(n_state + n_ctrl, horizon, rng)

    raise ValueError(f"Unknown Qvalue type {kind}")


def modules(
    policy: TVLinearPolicy, hparams: AttributeDict
) -> Tuple[QValue, QValue, VValue]:
    qval_fn = qval_constructor(policy, hparams)
    if hparams.loss == "TD(1)":
        return td_modules(policy, qval_fn, hparams)
    if hparams.loss == "VGL(1)":
        return vgl_modules(policy, qval_fn, hparams)

    raise ValueError(f"No modules defined for loss '{hparams.loss}'")


class LightningQValue(pl.LightningModule):
    # pylint:disable=too-many-ancestors,arguments-differ
    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, hparams: dict):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.lqg = lqg
        self.policy = policy
        self.hparams.update(hparams)

        # NN modules
        self.qval, self.target_qval, self.target_vval = modules(policy, self.hparams)
        # Estimator
        self.estimator = estimator.mfdpg_estimator(self.policy, self.qval)

        # Groud-truth
        dynamics, cost, init = lqg.standard_form()
        true_val, true_svg = estimator.analytic_svg(policy, init, dynamics, cost)
        # Register targets as buffers so they are moved to device
        self.register_buffer("true_val", true_val)
        self.register_buffer("true_svg_K", true_svg.K)
        self.register_buffer("true_svg_k", true_svg.k)

        # For VValue error
        self._vval = QuadVValue.from_policy(
            policy.standard_form(),
            lqg.trans.standard_form(),
            lqg.reward.standard_form(),
        ).requires_grad_(False)

    def num_parameters(self) -> int:
        """Returns the number of trainable parameters"""
        return sum(p.numel() for p in self.qval.parameters() if p.requires_grad)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.qval.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def forward(self, batch: TDBatch) -> Tensor:
        return LOSS[self.hparams.loss](self, batch)

    def training_step(self, batch: TDBatch, batch_idx: int) -> Tensor:
        del batch_idx
        batch = refine_tdbatch(batch)
        loss = self(batch)
        self.log("train/loss", loss)
        with torch.no_grad():
            self.log_dict(with_prefix("train/", self._evaluate_on(batch)))
        return loss

    def on_train_batch_end(self, *args, **kwargs):  # pylint:disable=unused-argument
        update_polyak(self.qval, self.target_qval, self.hparams.polyak)

    def validation_step(self, batch: TDBatch, batch_idx: int) -> None:
        del batch_idx
        batch = refine_tdbatch(batch)
        self.log_dict(with_prefix("val/", self._evaluate_on(batch)))

    def test_step(self, batch: TDBatch, batch_idx: int) -> None:
        del batch_idx
        batch = refine_tdbatch(batch)
        self.log_dict(with_prefix("test/", self._evaluate_on(batch)))

    def _evaluate_on(self, batch: TDBatch) -> TensorDict:
        obs, act, rew, new_obs = batch
        td_error = analysis.relative_error(
            rew + self.target_vval(new_obs), self.qval(obs, act)
        ).mean()
        with torch.enable_grad():
            val, svg = self.estimator(obs)
        return {
            "relative_td_error": td_error,
            "relative_vval_err": analysis.relative_error(val, self._vval(obs).mean()),
            "grad_acc": analysis.gradient_accuracy(
                [svg], lqr.Linear(self.true_svg_K, self.true_svg_k)
            ),
        }

    def log_grad_norm(self, grad_norm_dict: TensorDict) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
