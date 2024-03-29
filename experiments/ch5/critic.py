# pylint:disable=missing-docstring
from typing import Callable, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from nnrl.nn.actor import DeterministicPolicy
from nnrl.nn.critic import HardValue, MLPQValue, QValue, QValueEnsemble, VValue
from nnrl.nn.model import StochasticModel
from nnrl.nn.utils import update_polyak
from nnrl.types import TensorDict
from numpy.random import Generator
from torch import Tensor, autograd, nn
from wandb_util import with_prefix

from lqsvg import analysis, estimator
from lqsvg.envs import lqr
from lqsvg.random import RNG
from lqsvg.torch import named as nt
from lqsvg.torch.nn import (
    LQGModule,
    QuadQValue,
    QuadraticReward,
    QuadRewardModel,
    QuadVValue,
    TVLinearPolicy,
)
from lqsvg.torch.random import default_generator_seed

TDBatch = Tuple[Tensor, Tensor, Tensor, Tensor]
RewardBatch = Tuple[Tensor, Tensor, Tensor]


def value_learning(module: nn.Module, batch: TDBatch) -> Tensor:
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

    delta = value - target
    (act_grad,) = autograd.grad(
        delta, act, grad_outputs=torch.ones_like(delta), create_graph=True
    )
    loss = torch.norm(act_grad, dim=-1).mean()
    return loss


def mage_loss(module: "LightningQValue", batch: TDBatch) -> Tensor:
    """Returns the regularized bootstrapped action-gradient error."""
    # pylint:disable=too-many-locals
    module.qval: ClippedQValue
    policy: DeterministicPolicy = module.policy
    dynamics: StochasticModel = module.lqg.trans
    reward: QuadraticReward = module.lqg.reward
    module.target_vval: HardValue

    obs, _, _, _ = batch
    act = policy(obs)
    rew = reward(obs, act)
    new_obs, _ = dynamics.rsample(dynamics(obs, act))

    value = module.qval(obs, act)
    target = rew + module.target_vval(new_obs)

    delta = value - target
    grad_out = nt.unnamed(torch.ones_like(delta))
    (act_grad,) = autograd.grad(delta, act, grad_outputs=grad_out, create_graph=True)
    loss = torch.norm(act_grad, dim=-1).mean() + 0.05 * torch.square(delta).mean()
    return loss


LOSS = {"TD(1)": value_learning, "VGL(1)": value_gradient_learning, "MAGE": mage_loss}


class ClippedQValue(QValue):
    """Q-value computed as the minimum among Q-values in an ensemble."""

    def __init__(self, q_values: QValueEnsemble):
        super().__init__()
        self.q_values = q_values

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Computes the clipped Q-value.

        Args:
            obs: The observation (*, O)
            action: The action (*, A)

        Returns:
            The clipped Q-value (*,)
        """
        values = self.q_values(obs, action)
        mininum, _ = torch.stack(nt.unnamed(*values), dim=0).min(dim=0)
        return mininum.refine_names(*values[0].names)


def td_modules(
    policy: DeterministicPolicy,
    qval_fn: Callable[[Generator], QValue],
    hparams: dict,
    rng: Generator,
) -> Tuple[ClippedQValue, ClippedQValue, VValue]:
    """Creates modules for Streamlined Off-Policy TD learning."""
    qval = ClippedQValue(QValueEnsemble([qval_fn(rng) for _ in range(2)]))
    target_qval = ClippedQValue(QValueEnsemble([qval_fn(rng) for _ in range(2)]))
    target_qval.requires_grad_(False)
    # Syncronize initial Q values
    target_qval.load_state_dict(qval.state_dict())

    # Target state-values
    target_noise = hparams.get("target_policy_sigma", 0.3)
    if target_noise:
        target_policy = DeterministicPolicy.add_gaussian_noise(
            policy, noise_stddev=target_noise
        )
    else:
        target_policy = policy
    target_vval = HardValue(target_policy, target_qval)
    return qval, target_qval, target_vval


def vgl_modules(
    policy: DeterministicPolicy,
    qval_fn: Callable[[Generator], QValue],
    _: dict,
    rng: Generator,
) -> Tuple[QValue, QValue, VValue]:
    """Creates modules for Value Gradient Learning (VGL)."""
    qval = qval_fn(rng)

    target_qval = qval_fn(rng).requires_grad_(False)
    target_qval.load_state_dict(qval.state_dict())

    target_vval = HardValue(policy, qval)
    return qval, target_qval, target_vval


def qval_constructor(
    policy: TVLinearPolicy, hparams: dict
) -> Callable[[Generator], QValue]:
    n_state, n_ctrl, horizon = lqr.dims_from_policy(policy.standard_form())
    kind = hparams["type"]
    if kind == "mlp":

        def constructor(rng: Generator) -> MLPQValue:
            obs_space, act_space = lqr.spaces_from_dims(n_state, n_ctrl, horizon)
            spec = MLPQValue.spec_cls(units=hparams["hunits"], activation="ReLU")
            with default_generator_seed(rng.integers(np.iinfo(int).max)):
                return MLPQValue(obs_space, act_space, spec)

    elif kind == "quad":

        def constructor(rng: Generator) -> QuadQValue:
            return QuadQValue(n_state + n_ctrl, horizon, rng)

    else:
        raise ValueError(f"Unknown Qvalue type {kind}")

    return constructor


def modules(
    policy: TVLinearPolicy, hparams: dict, rng: Generator
) -> Tuple[QValue, QValue, VValue]:
    qval_fn = qval_constructor(policy, hparams["model"])
    loss = hparams["loss"]
    if loss in {"TD(1)", "MAGE"}:
        return td_modules(policy, qval_fn, hparams["model"], rng)
    if loss == "VGL(1)":
        return vgl_modules(policy, qval_fn, {}, rng)

    raise ValueError(f"No modules defined for loss '{loss}'")


class LightningQValue(pl.LightningModule):
    # pylint:disable=too-many-ancestors,arguments-differ,too-many-instance-attributes
    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, hparams: dict):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.lqg = lqg
        self.policy = policy
        self.hparams.update(hparams)

        # NN modules
        self.qval, self.target_qval, self.target_vval = modules(
            policy, self.hparams, np.random.default_rng(self.hparams.seed)
        )
        # Estimator
        self.estimator = estimator.mfdpg_estimator(self.policy, self.qval)

        # Groud-truth
        dynamics, cost, init = lqg.standard_form()
        true_val, true_svg = estimator.analytic_svg(policy, init, dynamics, cost)
        # Register targets as buffers so they are moved to device
        self.register_buffer("true_val", true_val)
        self.register_buffer("true_svg_K", true_svg.K)
        self.register_buffer("true_svg_k", true_svg.k)
        quad_q, quad_v = estimator.on_policy_value_functions(
            policy.standard_form(), dynamics, cost
        )
        self.true_qval = QuadQValue.from_existing(quad_q).requires_grad_(False)
        self.true_vval = QuadVValue.from_existing(quad_v).requires_grad_(False)

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
        loss = self(batch)
        self.log("train/loss", loss)
        with torch.no_grad():
            self.log_dict(with_prefix("train/", self._evaluate_on(batch)))
        return loss

    def on_train_batch_end(self, *args, **kwargs):  # pylint:disable=unused-argument
        update_polyak(self.qval, self.target_qval, self.hparams.polyak)

    def validation_step(self, batch: TDBatch, batch_idx: int) -> None:
        del batch_idx
        self.log_dict(with_prefix("val/", self._evaluate_on(batch)))

    def test_step(self, batch: TDBatch, batch_idx: int) -> None:
        del batch_idx
        self.log_dict(with_prefix("test/", self._evaluate_on(batch)))

    def _evaluate_on(self, batch: TDBatch) -> TensorDict:
        obs = batch[0]
        with torch.enable_grad():
            val, svg = self.estimator(obs)
        true_vval = self.true_vval(obs).mean()
        return {
            "relative_vval_err": analysis.relative_error(val, true_vval),
            "grad_acc": analysis.cosine_similarity(
                svg, lqr.Linear(self.true_svg_K, self.true_svg_k)
            ),
            **qval_metrics(self, batch),
        }

    def log_grad_norm(self, grad_norm_dict: TensorDict) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


@torch.enable_grad()
def qval_metrics(module: "LightningQValue", batch: TDBatch) -> TensorDict:
    # pylint:disable=too-many-locals
    obs = batch[0].requires_grad_()  # (B, O)
    act = module.policy(obs)  # (B, A)
    rew = module.lqg.reward(obs, act)  # (B,)
    new_obs, _ = module.lqg.trans.rsample(module.lqg.trans(obs, act))  # (B, O)

    pred_qval = module.qval(obs, act)  # (B,)
    boot_qval = rew + module.target_vval(new_obs)  # (B,)
    true_qval = module.true_qval(obs, act)  # (B,)

    td_rel_error = analysis.relative_error(boot_qval, pred_qval).mean()  # ()
    rel_error = analysis.relative_error(true_qval, pred_qval).mean()

    grad_out = nt.unnamed(torch.ones_like(pred_qval))  # (B,)
    # (B, O), (B, A)
    (true_ograd,) = torch.autograd.grad(
        true_qval, obs, grad_outputs=grad_out, retain_graph=True
    )
    (true_agrad,) = torch.autograd.grad(
        true_qval, act, grad_outputs=grad_out, retain_graph=True
    )
    (pred_ograd,) = torch.autograd.grad(
        pred_qval, obs, grad_outputs=grad_out, retain_graph=True
    )
    (pred_agrad,) = torch.autograd.grad(
        pred_qval, act, grad_outputs=grad_out, retain_graph=True
    )
    (boot_ograd,) = torch.autograd.grad(
        boot_qval, obs, grad_outputs=grad_out, retain_graph=True
    )
    (boot_agrad,) = torch.autograd.grad(boot_qval, act, grad_outputs=grad_out)

    td_ograd_acc = analysis.cosine_similarity(pred_ograd, boot_ograd, dim=-1).mean()
    td_agrad_acc = analysis.cosine_similarity(pred_agrad, boot_agrad, dim=-1).mean()
    ograd_acc = analysis.cosine_similarity(pred_ograd, true_ograd, dim=-1).mean()
    agrad_acc = analysis.cosine_similarity(pred_agrad, true_agrad, dim=-1).mean()

    return {
        "bootstrap/relative_qval_err": td_rel_error,
        "bootstrap/obs_grad_acc": td_ograd_acc,
        "bootstrap/action_grad_acc": td_agrad_acc,
        "relative_qval_err": rel_error,
        "obs_grad_acc": ograd_acc,
        "action_grad_acc": agrad_acc,
    }


class LightningReward(pl.LightningModule):
    # pylint:disable=too-many-ancestors,too-many-instance-attributes,arguments-differ
    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, hparams: dict, rng: RNG):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.lqg = lqg
        self.policy = policy
        self.hparams.update(hparams)

        # NN modules
        self.rmodel = QuadRewardModel(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, stationary=True, rng=rng.numpy
        )

        def reward_loss(reward: nn.Module) -> Callable[[RewardBatch], Tensor]:
            def loss(batch: RewardBatch) -> Tensor:
                obs, act, rew = batch
                return torch.mean(0.5 * torch.square(reward(obs, act) - rew))

            return loss

        self.loss = reward_loss(self.rmodel)

    def num_parameters(self) -> int:
        """Returns the number of trainable parameters"""
        return sum(p.numel() for p in self.rmodel.parameters() if p.requires_grad)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.rmodel.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def forward(self, batch: TDBatch) -> Tensor:
        obs, act, rew, _ = batch
        return self.loss((obs, act, rew))

    def training_step(self, batch: TDBatch, batch_idx: int) -> Tensor:
        del batch_idx
        loss = self(batch)
        self.log("train/loss", loss)
        with torch.no_grad():
            self.log_dict(with_prefix("train/", self._evaluate_on(batch)))
        return loss

    def validation_step(self, batch: TDBatch, batch_idx: int) -> None:
        del batch_idx
        self.log_dict(with_prefix("val/", self._evaluate_on(batch)))

    def test_step(self, batch: TDBatch, batch_idx: int) -> None:
        del batch_idx
        self.log_dict(with_prefix("test/", self._evaluate_on(batch)))

    def _evaluate_on(self, batch: TDBatch) -> TensorDict:
        obs, act, rew, _ = batch
        loss = self.loss((obs, act, rew))
        with torch.enable_grad():
            act.requires_grad_().retain_grad()
            obs.requires_grad_().retain_grad()
            targ = self.lqg.reward(obs, act).mean()
            ograd = torch.autograd.grad(targ, obs, retain_graph=True)
            agrad = torch.autograd.grad(targ, act)

            pred = self.rmodel(obs, act).mean()
            ograd_ = torch.autograd.grad(pred, obs, retain_graph=True)
            agrad_ = torch.autograd.grad(pred, act)
            obs_grad_acc = analysis.cosine_similarity(ograd, ograd_)
            act_grad_acc = analysis.cosine_similarity(agrad, agrad_)

        return {
            "loss": loss,
            "obs_grad_acc": obs_grad_acc,
            "act_grad_acc": act_grad_acc,
        }

    def log_grad_norm(self, grad_norm_dict: TensorDict) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
