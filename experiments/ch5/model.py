# pylint:disable=missing-docstring
import functools
from typing import Callable, Dict, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from nnrl.nn.actor import DeterministicPolicy
from nnrl.nn.critic import ClippedQValue, HardValue, MLPQValue, QValueEnsemble, VValue
from nnrl.nn.utils import update_polyak
from nnrl.types import TensorDict
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor, nn
from wandb.sdk import wandb_config

from lqsvg.analysis import gradient_accuracy
from lqsvg.data import markovian_state_sampler, recurrent_state_sampler
from lqsvg.envs import lqr
from lqsvg.envs.lqr import dims_from_policy, spaces_from_dims
from lqsvg.envs.lqr.modules import LQGModule, QuadraticReward
from lqsvg.estimators import MBEstimator, analytic_svg, maac_estimator, mfdpg_estimator
from lqsvg.torch.nn.dynamics.segment import (
    GRUGaussDynamics,
    LinearDiagDynamicsModel,
    MLPDynamicsModel,
    log_prob_fn,
)
from lqsvg.torch.nn.policy import TVLinearPolicy
from lqsvg.torch.nn.value import QuadQValue, QuadVValue, ZeroQValue

SegBatch = Tuple[Tensor, Tensor, Tensor]
ValBatch = Union[SegBatch, Sequence[Tensor]]
TDBatch = Tuple[Tensor, Tensor, Tensor, Tensor]


@functools.singledispatch
def make_estimator(
    model: nn.Module, policy: TVLinearPolicy, reward: QuadraticReward, qval: QuadQValue
) -> MBEstimator:
    return maac_estimator(
        policy, markovian_state_sampler(model, model.rsample), reward, qval
    )


@make_estimator.register
def _(
    model: GRUGaussDynamics,
    policy: TVLinearPolicy,
    reward: QuadraticReward,
    qval: QuadQValue,
) -> MBEstimator:
    return maac_estimator(
        policy,
        recurrent_state_sampler(model, model.dist.rsample),
        reward,
        qval,
        recurrent=True,
    )


def make_model(
    n_state: int, n_ctrl: int, horizon: int, hparams: AttributeDict
) -> nn.Module:
    if hparams.model["type"] == "linear":
        return LinearDiagDynamicsModel(n_state, n_ctrl, horizon, stationary=True)
    if hparams.model["type"] == "gru":
        return GRUGaussDynamics(n_state, n_ctrl, horizon, **hparams.model["kwargs"])
    return MLPDynamicsModel(n_state, n_ctrl, horizon, **hparams.model["kwargs"])


def relative_error(target_val: Tensor, pred_val: Tensor) -> Tensor:
    """Returns the relative value error.

    Ref: https://en.wikipedia.org/wiki/Approximation_error
    """
    return torch.abs(1 - pred_val / target_val)


def val_err_and_grad_acc(
    val: Tensor, svg: lqr.Linear, target_val: Tensor, target_svg: lqr.Linear
) -> Tuple[Tensor, Tensor]:
    """Computes metrics for estimated gradients."""
    val_err = relative_error(target_val, val)
    grad_acc = gradient_accuracy([svg], target_svg)
    return val_err, grad_acc


@torch.no_grad()
def vvalue_err(val: Tensor, obs: Tensor, vval: VValue) -> Tensor:
    """Returns the error between the surrogate value and the state value."""
    return relative_error(vval(obs).mean(), val)


def refine_segbatch(batch: SegBatch) -> SegBatch:
    # noinspection PyTypeChecker
    return tuple(x.refine_names("B", "H", "R") for x in batch)


def wrap_log_prob(
    log_prob: Callable[[Tensor, Tensor, Tensor], Tensor]
) -> Callable[[SegBatch], Tensor]:
    def wrapped(batch: SegBatch) -> Tensor:
        return log_prob(*batch)

    return wrapped


def empirical_kl(
    source_logp: Callable[[SegBatch], Tensor], other_logp: Callable[[SegBatch], Tensor]
) -> Callable[[SegBatch], Tensor]:
    """Returns a function of samples to empirical KL divergence.

    Args:
        source_logp: likelihood function of distribution from which samples are
            collected
        other_logp: likelihood function of another distribution
    """

    def kld(batch: SegBatch) -> Tensor:
        return torch.mean(source_logp(batch) - other_logp(batch))

    return kld


def with_prefix(prefix: str, dictionary: dict) -> dict:
    """Returns a dictionary copy with prefixed keys."""
    return {prefix + k: v for k, v in dictionary.items()}


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    def __init__(
        self, lqg: LQGModule, policy: TVLinearPolicy, hparams: wandb_config.Config
    ):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.add_module("_lqg", lqg)
        self.add_module("_policy", policy)
        self.hparams.update(hparams)

        # NN modules
        self.model = make_model(lqg.n_state, lqg.n_ctrl, lqg.horizon, self.hparams)
        self.seg_log_prob = log_prob_fn(self.model, self.model.dist)
        self.kl_with_dynamics = empirical_kl(
            wrap_log_prob(log_prob_fn(lqg.trans, lqg.trans.dist)),
            wrap_log_prob(self.seg_log_prob),
        )

        # Estimators
        if self.hparams.zero_q:
            qval = ZeroQValue()
        else:
            qval = QuadQValue.from_policy(
                policy.standard_form(),
                lqg.trans.standard_form(),
                lqg.reward.standard_form(),
            ).requires_grad_(False)
        self.estimator = make_estimator(self.model, policy, lqg.reward, qval)
        self.add_module("_qval", qval)

        # Ground-truth
        dynamics, cost, init = lqg.standard_form()
        true_val, true_svg = analytic_svg(policy, init, dynamics, cost)
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

    def forward(self, batch: SegBatch) -> Tensor:
        """Negative log-likelihood of (batched) trajectory segment."""
        # pylint:disable=arguments-differ
        obs, act, new_obs = batch
        # noinspection PyArgumentList
        return -self.seg_log_prob(obs, act, new_obs) / obs.size("H")

    def num_parameters(self) -> int:
        """Returns the number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    # noinspection PyArgumentList
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch: SegBatch, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        loss = self(refine_segbatch(batch)).mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self.log_dict(with_prefix("val/", self._evaluate_on(batch, dataloader_idx)))

    def test_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self.log_dict(with_prefix("test/", self._evaluate_on(batch, dataloader_idx)))

    def _evaluate_on(self, batch: ValBatch, dataloader_idx: int) -> TensorDict:
        if dataloader_idx == 0:
            return self._eval_on_traj_seg(batch)

        return self._eval_on_uniform_states(batch[0])

    def _eval_on_traj_seg(self, batch: SegBatch) -> TensorDict:
        metrics = {}
        self.model.eval()
        batch = refine_segbatch(batch)
        # Compute log-prob of traj segments
        metrics["loss"] = self(batch).mean()
        # Compute KL with true dynamics
        metrics["empirical_kl"] = self.kl_with_dynamics(batch)
        return metrics

    def _eval_on_uniform_states(self, obs: Tensor) -> TensorDict:
        metrics = {}
        # Avoid:
        # RuntimeError: cudnn RNN backward can only be called in training mode
        self.model.train()
        obs = obs.refine_names("B", "R")
        horizons = self.hparams.pred_horizon
        horizons = [horizons] if isinstance(horizons, int) else horizons
        for horizon in horizons:
            with torch.enable_grad():
                val, svg = self.estimator(obs, horizon)
            true_svg = lqr.Linear(self.true_svg_K, self.true_svg_k)
            value_err, grad_acc = val_err_and_grad_acc(
                val, svg, self.true_val, true_svg
            )
            metrics[f"{horizon}/relative_value_err"] = value_err
            metrics[f"{horizon}/grad_acc"] = grad_acc
            metrics[f"{horizon}/relative_vval_err"] = vvalue_err(val, obs, self._vval)
        return metrics

    def log_grad_norm(self, grad_norm_dict: Dict[str, Tensor]) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


def make_value(
    policy: TVLinearPolicy, hparams: AttributeDict
) -> Tuple[QValueEnsemble, QValueEnsemble, VValue]:
    """Creates modules for Streamlined Off-Policy TD learning."""
    n_state, n_ctrl, horizon = dims_from_policy(policy.standard_form())
    obs_space, act_space = spaces_from_dims(n_state, n_ctrl, horizon)
    spec = MLPQValue.spec_cls(units=hparams.hunits, activation="ReLU")
    qvals = [MLPQValue(obs_space, act_space, spec) for _ in range(2)]
    qval = QValueEnsemble(qvals)

    # Loss
    target_policy = DeterministicPolicy.add_gaussian_noise(policy, noise_stddev=0.3)
    qvals_ = [MLPQValue(obs_space, act_space, spec) for _ in range(2)]
    target_qval = QValueEnsemble(qvals_).requires_grad_(False)
    target_vval = HardValue(target_policy, target_qval)
    return qval, target_qval, target_vval


class LightningQValue(pl.LightningModule):  # pylint:disable=too-many-ancestors
    def __init__(
        self, lqg: LQGModule, policy: TVLinearPolicy, hparams: wandb_config.Config
    ):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.add_module("_lqg", lqg)
        self.policy = policy
        self.hparams.update(hparams)

        # NN modules
        self.qvals, self.targ_qvals, self.target_vval = make_value(policy, self.hparams)
        # Estimator
        self.estimator = mfdpg_estimator(self.policy, ClippedQValue(self.qvals))

        # Groud-truth
        dynamics, cost, init = lqg.standard_form()
        true_val, true_svg = analytic_svg(policy, init, dynamics, cost)
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

    def forward(self, batch: TDBatch) -> Tensor:
        """Returns the temporal difference error induced by the value function."""
        # pylint:disable=arguments-differ
        obs, act, rew, new_obs = batch
        with torch.no_grad():
            target = rew + self.target_vval(new_obs)
        loss_fn = nn.MSELoss()
        values = self.qvals(obs, act)
        return torch.stack([loss_fn(v, target) for v in values]).sum()

    def num_parameters(self) -> int:
        """Returns the number of trainable parameters"""
        return sum(p.numel() for p in self.qvals.parameters() if p.requires_grad)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.qvals.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch: TDBatch, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        loss = self(batch)
        self.log("train/loss", loss)
        with torch.no_grad():
            self.log_dict(with_prefix("train/", self._evaluate_on(batch)))
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        # pylint:disable=arguments-differ,unused-argument
        update_polyak(self.qvals, self.targ_qvals, self.hparams.polyak)

    def validation_step(self, batch: TDBatch, batch_idx: int) -> None:
        # pylint:disable=arguments-differ
        del batch_idx
        self.log_dict(with_prefix("val/", self._evaluate_on(batch)))

    def test_step(self, batch: TDBatch, batch_idx: int) -> None:
        # pylint:disable=arguments-differ
        del batch_idx
        self.log_dict(with_prefix("test/", self._evaluate_on(batch)))

    def _evaluate_on(self, batch: TDBatch) -> TensorDict:
        obs, act, rew, new_obs = batch
        td_error = relative_error(rew + self.target_vval(new_obs), self.qvals(obs, act))
        with torch.enable_grad():
            val, svg = self.estimator(obs)
        return {
            "relative_td_error": td_error,
            "relative_vval_err": vvalue_err(val, obs, self._vval),
            "grad_acc": gradient_accuracy(
                [svg], lqr.Linear(self.true_svg_K, self.true_svg_k)
            ),
        }

    def log_grad_norm(self, grad_norm_dict: Dict[str, Tensor]) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
