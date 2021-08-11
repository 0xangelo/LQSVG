# pylint:disable=missing-docstring
import functools
from typing import Callable, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
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
def _compute_grad_acc_on_batch(
    estimator: Estimator, obs: Tensor, pred_horizon: int, target: lqr.Linear
) -> Tensor:
    _, svg = estimator(obs, pred_horizon)
    return torch.as_tensor(gradient_accuracy([svg], target))


def _refine_batch(batch: Batch) -> Batch:
    # noinspection PyTypeChecker
    return tuple(x.refine_names("B", "H", "R") for x in batch)


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    model: nn.Module
    seg_log_prob: Callable[[Tensor, Tensor, Tensor], Tensor]
    estimator: Estimator
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
        _, self.true_svg = analytic_svg(policy, init, dynamics, cost)

    # noinspection PyArgumentList
    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Log-likelihood of (batched) trajectory segment."""
        # pylint:disable=arguments-differ
        return self.seg_log_prob(obs, act, new_obs) / obs.size("H")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

    def _compute_loss_on_batch(self, batch: Batch) -> Tensor:
        obs, act, new_obs = batch
        return -self(obs, act, new_obs).mean()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        batch = _refine_batch(batch)
        loss = self._compute_loss_on_batch(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self._compute_and_log_loss_or_grad_acc(batch, dataloader_idx, stage="val")

    def test_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self._compute_and_log_loss_or_grad_acc(batch, dataloader_idx, stage="test")

    def _compute_and_log_loss_or_grad_acc(
        self, batch: ValBatch, dataloader_idx: int, *, stage: str
    ):
        if dataloader_idx == 0:
            # Compute log-prob of traj segments
            loss = self._compute_loss_on_batch(_refine_batch(batch))
            self.log(stage + "/loss", loss)
        else:
            # Compute gradient acc using uniformly sampled states
            obs = batch[0].refine_names("B", "R")
            if isinstance(self.hparams.pred_horizon, list):
                for steps in self.hparams.pred_horizon:
                    grad_acc = _compute_grad_acc_on_batch(
                        self.estimator, obs, steps, self.true_svg
                    )
                    self.log(stage + f"/grad_acc_{steps}", grad_acc)
            else:
                grad_acc = _compute_grad_acc_on_batch(
                    self.estimator, obs, self.hparams.pred_horizon, self.true_svg
                )
                self.log(stage + "/grad_acc", grad_acc)
