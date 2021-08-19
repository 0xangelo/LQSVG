# pylint:disable=missing-docstring
import functools
from typing import Callable, Dict, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from nnrl.nn.critic import VValue
from nnrl.types import TensorDict
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor, nn
from wandb.sdk import wandb_config

from lqsvg.envs import lqr
from lqsvg.envs.lqr.modules import LQGModule, QuadraticReward
from lqsvg.experiment.analysis import gradient_accuracy
from lqsvg.experiment.dynamics import markovian_state_sampler, recurrent_state_sampler
from lqsvg.experiment.estimators import analytic_svg, maac_estimator
from lqsvg.torch.nn.dynamics.segment import (
    GRUGaussDynamics,
    LinearDiagDynamicsModel,
    MLPDynamicsModel,
    log_prob_fn,
)
from lqsvg.torch.nn.policy import TVLinearPolicy
from lqsvg.torch.nn.value import QuadQValue, QuadVValue, ZeroQValue

Batch = Tuple[Tensor, Tensor, Tensor]
ValBatch = Union[Batch, Sequence[Tensor]]
Estimator = Callable[[Tensor, int], Tuple[Tensor, lqr.Linear]]


@functools.singledispatch
def make_estimator(
    model: nn.Module, policy: TVLinearPolicy, reward: QuadraticReward, qval: QuadQValue
) -> Estimator:
    return maac_estimator(
        policy, markovian_state_sampler(model, model.rsample), reward, qval
    )


@make_estimator.register
def _(
    model: GRUGaussDynamics,
    policy: TVLinearPolicy,
    reward: QuadraticReward,
    qval: QuadQValue,
) -> Estimator:
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


def val_mse_and_grad_acc(
    val: Tensor, svg: lqr.Linear, target_val: Tensor, target_svg: lqr.Linear
) -> Tuple[Tensor, Tensor]:
    """Computes metrics for estimated gradients."""
    mse = torch.square(target_val - val)
    grad_acc = torch.as_tensor(gradient_accuracy([svg], target_svg)).to(val)
    return mse, grad_acc


@torch.no_grad()
def batch_val_mse(val: Tensor, obs: Tensor, vval: VValue) -> Tensor:
    """MSE between the surrogate value and the average V-value function."""
    batch_val = vval(obs)
    return torch.square(batch_val - val)


def _refine_batch(batch: Batch) -> Batch:
    # noinspection PyTypeChecker
    return tuple(x.refine_names("B", "H", "R") for x in batch)


def wrap_log_prob(
    log_prob: Callable[[Tensor, Tensor, Tensor], Tensor]
) -> Callable[[Batch], Tensor]:
    def wrapped(batch: Batch) -> Tensor:
        return log_prob(*batch)

    return wrapped


def empirical_kl(
    source_logp: Callable[[Batch], Tensor], other_logp: Callable[[Batch], Tensor]
) -> Callable[[Batch], Tensor]:
    """Returns a function of samples to empirical KL divergence.

    Args:
        source_logp: likelihood function of distribution from which samples are
            collected
        other_logp: likelihood function of another distribution
    """

    def kld(batch: Batch) -> Tensor:
        return torch.mean(source_logp(batch) - other_logp(batch))

    return kld


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    model: nn.Module
    seg_log_prob: Callable[[Tensor, Tensor, Tensor], Tensor]
    kl_with_dynamics: Callable[[Batch], Tensor]
    estimator: Estimator

    def __init__(
        self, lqg: LQGModule, policy: TVLinearPolicy, hparams: wandb_config.Config
    ):
        super().__init__()
        self.hparams.update(hparams)

        # NN modules
        self.model = make_model(lqg.n_state, lqg.n_ctrl, lqg.horizon, self.hparams)
        self.seg_log_prob = log_prob_fn(self.model, self.model.dist)
        self.kl_with_dynamics = empirical_kl(
            wrap_log_prob(log_prob_fn(lqg.trans, lqg.trans.dist)),
            wrap_log_prob(self.seg_log_prob),
        )

        # Gradients
        if self.hparams.zero_q:
            qval = ZeroQValue()
        else:
            qval = QuadQValue.from_policy(
                policy.standard_form(),
                lqg.trans.standard_form(),
                lqg.reward.standard_form(),
            ).requires_grad_(False)
        self.estimator = make_estimator(self.model, policy, lqg.reward, qval)
        dynamics, cost, init = lqg.standard_form()
        true_val, true_svg = analytic_svg(policy, init, dynamics, cost)
        # Register targets as buffers so they are moved to device
        self.register_buffer("true_val", true_val)
        self.register_buffer("true_svg_K", true_svg.K)
        self.register_buffer("true_svg_k", true_svg.k)

        # For batch MSE
        self._vval = QuadVValue.from_policy(
            policy.standard_form(),
            lqg.trans.standard_form(),
            lqg.reward.standard_form(),
        ).requires_grad_(False)
        # Register modules to cast them to appropriate device
        self._extra_modules = nn.ModuleList([lqg, policy, qval])

    def log_grad_norm(self, grad_norm_dict: Dict[str, Tensor]) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )

    # noinspection PyArgumentList
    def forward(self, batch: Batch) -> Tensor:
        """Negative log-likelihood of (batched) trajectory segment."""
        # pylint:disable=arguments-differ
        obs, act, new_obs = batch
        return -self.seg_log_prob(obs, act, new_obs) / obs.size("H")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def on_train_start(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log("trainable_parameters", n_params)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        loss = self(_refine_batch(batch)).mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        metrics = self._compute_eval_metrics(batch, dataloader_idx)
        self.log_dict({"val/" + k: v for k, v in metrics.items()})

    def test_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        metrics = self._compute_eval_metrics(batch, dataloader_idx)
        self.log_dict({"test/" + k: v for k, v in metrics.items()})

    def _compute_eval_metrics(self, batch: ValBatch, dataloader_idx: int) -> TensorDict:
        metrics = {}
        if dataloader_idx == 0:
            self.model.eval()
            batch = _refine_batch(batch)
            # Compute log-prob of traj segments
            metrics["loss"] = self(batch).mean()
            # Compute KL with true dynamics
            metrics["empirical_kl"] = self.kl_with_dynamics(batch)
        else:
            # Avoid:
            # RuntimeError: cudnn RNN backward can only be called in training mode
            self.model.train()
            # Compute gradient acc using uniformly sampled states
            obs = batch[0].refine_names("B", "R")
            horizons = self.hparams.pred_horizon
            horizons = [horizons] if isinstance(horizons, int) else horizons
            for horizon in horizons:
                with torch.enable_grad():
                    val, svg = self.estimator(obs, horizon)
                true_svg = lqr.Linear(self.true_svg_K, self.true_svg_k)
                val_mse, grad_acc = val_mse_and_grad_acc(
                    val, svg, self.true_val, true_svg
                )
                metrics[f"{horizon}/val_mse"] = val_mse
                metrics[f"{horizon}/grad_acc"] = grad_acc
                metrics[f"{horizon}/batch_val_mse"] = batch_val_mse(
                    val, obs, self._vval
                )
        return metrics
