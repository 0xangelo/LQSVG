# pylint:disable=missing-docstring
import functools
from typing import Callable, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
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
from lqsvg.torch.nn.value import QuadQValue, ZeroQValue

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


@torch.enable_grad()
def val_mse_and_grad_acc(
    estimator: Estimator, obs: Tensor, pred_horizon: int, targets: (Tensor, lqr.Linear)
) -> Tuple[Tensor, Tensor]:
    """Computes metrics for evaluating gradient estimators."""
    target_val, target_grad = targets
    val, svg = estimator(obs, pred_horizon)
    mse = torch.square(target_val - val)
    grad_acc = torch.as_tensor(gradient_accuracy([svg], target_grad))
    return mse, grad_acc


def _refine_batch(batch: Batch) -> Batch:
    # noinspection PyTypeChecker
    return tuple(x.refine_names("B", "H", "R") for x in batch)


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    model: nn.Module
    seg_log_prob: Callable[[Tensor, Tensor, Tensor], Tensor]
    estimator: Estimator
    true_val: Tensor
    true_svg: lqr.Linear

    def __init__(
        self, lqg: LQGModule, policy: TVLinearPolicy, hparams: wandb_config.Config
    ):
        super().__init__()
        self.hparams.update(hparams)

        # NN modules
        self.model = make_model(lqg.n_state, lqg.n_ctrl, lqg.horizon, self.hparams)
        self.seg_log_prob = log_prob_fn(self.model, self.model.dist)

        # Gradients
        if self.hparams.zero_q:
            qval = ZeroQValue()
        else:
            qval = QuadQValue(lqg.n_state + lqg.n_ctrl, lqg.horizon)
            qval.match_policy_(
                policy.standard_form(),
                lqg.trans.standard_form(),
                lqg.reward.standard_form(),
            )
            qval.requires_grad_(False)
        self.estimator = make_estimator(self.model, policy, lqg.reward, qval)
        dynamics, cost, init = lqg.standard_form()
        self.true_val, self.true_svg = analytic_svg(policy, init, dynamics, cost)

    # noinspection PyArgumentList
    def forward(self, batch: Batch) -> Tensor:
        """Negative log-likelihood of (batched) trajectory segment."""
        # pylint:disable=arguments-differ
        obs, act, new_obs = batch
        return -self.seg_log_prob(obs, act, new_obs) / obs.size("H")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def on_train_start(self) -> None:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
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
            # Compute log-prob of traj segments
            loss = self(_refine_batch(batch)).mean()
            metrics["loss"] = loss
        else:
            # Compute gradient acc using uniformly sampled states
            obs = batch[0].refine_names("B", "R")
            horizons = self.hparams.pred_horizon
            horizons = [horizons] if isinstance(horizons, int) else horizons
            for horizon in horizons:
                val_mse, grad_acc = val_mse_and_grad_acc(
                    self.estimator, obs, horizon, (self.true_val, self.true_svg)
                )
                metrics[f"val_mse_{horizon}"] = val_mse
                metrics[f"grad_acc_{horizon}"] = grad_acc
        return metrics
