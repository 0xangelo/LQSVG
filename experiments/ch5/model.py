# pylint:disable=missing-docstring
import functools
from typing import Callable, Dict, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from nnrl.types import TensorDict
from torch import Tensor, nn
from wandb_util import with_prefix

from lqsvg import analysis, data, estimator
from lqsvg.envs import lqr
from lqsvg.torch.nn import (
    GRUGaussDynamics,
    LinearDiagDynamicsModel,
    LQGModule,
    MLPDynamicsModel,
    QuadQValue,
    QuadVValue,
    TVLinearPolicy,
    ZeroQValue,
)
from lqsvg.torch.sequence import log_prob_fn

SeqBatch = Tuple[Tensor, Tensor, Tensor]
ValBatch = Union[SeqBatch, Sequence[Tensor]]


@functools.singledispatch
def make_estimator(
    model: nn.Module, policy: TVLinearPolicy, reward: nn.Module, qval: nn.Module
) -> estimator.MBEstimator:
    dynamics = data.markovian_state_sampler(model, model.rsample)
    return estimator.maac_estimator(policy, dynamics, reward, qval)


@make_estimator.register
def _(
    model: GRUGaussDynamics, policy: TVLinearPolicy, reward: nn.Module, qval: nn.Module
) -> estimator.MBEstimator:
    dynamics = data.recurrent_state_sampler(model, model.dist.rsample)
    return estimator.maac_estimator(policy, dynamics, reward, qval, recurrent=True)


def make_model(n_state: int, n_ctrl: int, horizon: int, hparams: dict) -> nn.Module:
    kind = hparams["type"]
    if kind == "linear":
        return LinearDiagDynamicsModel(n_state, n_ctrl, horizon, stationary=True)
    if kind == "gru":
        return GRUGaussDynamics(n_state, n_ctrl, horizon, **hparams["kwargs"])
    if kind == "mlp":
        return MLPDynamicsModel(n_state, n_ctrl, horizon, **hparams["kwargs"])
    raise ValueError(f"Unknown model type '{kind}'")


def refine_segbatch(batch: SeqBatch) -> SeqBatch:
    # noinspection PyTypeChecker
    return tuple(x.refine_names("B", "H", "R") for x in batch)


def wrap_log_prob(
    log_prob: Callable[[Tensor, Tensor, Tensor], Tensor]
) -> Callable[[SeqBatch], Tensor]:
    def wrapped(batch: SeqBatch) -> Tensor:
        return log_prob(*batch)

    return wrapped


def empirical_kl(
    source_logp: Callable[[SeqBatch], Tensor], other_logp: Callable[[SeqBatch], Tensor]
) -> Callable[[SeqBatch], Tensor]:
    """Returns a function of samples to empirical KL divergence.

    Args:
        source_logp: likelihood function of distribution from which samples are
            collected
        other_logp: likelihood function of another distribution
    """

    def kld(batch: SeqBatch) -> Tensor:
        return torch.mean(source_logp(batch) - other_logp(batch))

    return kld


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors,too-many-instance-attributes
    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, hparams: dict):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.lqg = lqg
        self.policy = policy
        self.hparams.update(hparams)

        # Ground-truth
        dynamics, cost, init = lqg.standard_form()
        true_val, true_svg = estimator.analytic_svg(policy, init, dynamics, cost)
        # Register targets as buffers so they are moved to device
        self.register_buffer("true_val", true_val)
        self.register_buffer("true_svg_K", true_svg.K)
        self.register_buffer("true_svg_k", true_svg.k)
        quad_qval, quad_vval = estimator.on_policy_value_functions(
            policy.standard_form(), dynamics, cost
        )
        self.true_qval = QuadQValue.from_existing(quad_qval).requires_grad_(False)
        self.true_vval = QuadVValue.from_existing(quad_vval).requires_grad_(False)

        # NN modules
        self.model = make_model(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, self.hparams.model
        )
        self.seg_log_prob = log_prob_fn(self.model, self.model.dist)
        self.kl_with_dynamics = empirical_kl(
            wrap_log_prob(log_prob_fn(lqg.trans, lqg.trans.dist)),
            wrap_log_prob(self.seg_log_prob),
        )

        # Estimators
        qval = ZeroQValue() if self.hparams.zero_q else self.true_qval
        self.estimator = make_estimator(self.model, policy, lqg.reward, qval)

    def forward(self, batch: SeqBatch) -> Tensor:
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

    def training_step(self, batch: SeqBatch, batch_idx: int) -> Tensor:
        # pylint:disable=arguments-differ
        del batch_idx
        loss = self(refine_segbatch(batch)).mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self.log_dict(
            with_prefix("val/", self._evaluate_on(batch, dataloader_idx)),
            add_dataloader_idx=False,
        )

    def test_step(self, batch: ValBatch, batch_idx: int, dataloader_idx: int):
        # pylint:disable=arguments-differ
        del batch_idx
        self.log_dict(
            with_prefix("test/", self._evaluate_on(batch, dataloader_idx)),
            add_dataloader_idx=False,
        )

    def _evaluate_on(self, batch: ValBatch, dataloader_idx: int) -> TensorDict:
        if dataloader_idx == 0:
            return self._eval_on_traj_segment(batch)

        return self._eval_on_uniform_states(batch[0])

    def _eval_on_traj_segment(self, batch: SeqBatch) -> TensorDict:
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
            value_err, grad_acc = analysis.val_err_and_grad_acc(
                val, svg, self.true_val, lqr.Linear(self.true_svg_K, self.true_svg_k)
            )
            metrics[f"{horizon}/relative_value_err"] = value_err
            metrics[f"{horizon}/grad_acc"] = grad_acc
            metrics[f"{horizon}/relative_vval_err"] = analysis.relative_error(
                self.true_vval(obs).mean(), val
            )
        return metrics

    def log_grad_norm(self, grad_norm_dict: Dict[str, Tensor]) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
