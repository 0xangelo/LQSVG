# pylint:disable=missing-docstring
from typing import Tuple

import pytorch_lightning as pl
import torch
from nnrl.nn.actor import DeterministicPolicy
from nnrl.nn.critic import HardValue, MLPQValue, QValue, QValueEnsemble, VValue
from nnrl.nn.utils import update_polyak
from nnrl.types import TensorDict
from pytorch_lightning.utilities import AttributeDict
from torch import Tensor, nn
from wandb_util import with_prefix

from lqsvg import analysis, estimator
from lqsvg.envs import lqr
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, QuadQValue, QuadVValue, TVLinearPolicy

TDBatch = Tuple[Tensor, Tensor, Tensor, Tensor]


class ClippedQValue(QValue):
    """Q-value computed as the minimum among Q-values in an ensemble."""

    def __init__(self, q_values: QValueEnsemble):
        super().__init__()
        self.q_values = q_values

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        values = self.q_values(obs, action)
        mininum, _ = torch.stack(nt.unnamed(*values), dim=0).min(dim=0)
        return mininum.refine_names(*values[0].names)


def make_value(
    policy: TVLinearPolicy, hparams: AttributeDict
) -> Tuple[QValueEnsemble, QValueEnsemble, VValue]:
    """Creates modules for Streamlined Off-Policy TD learning."""
    n_state, n_ctrl, horizon = lqr.dims_from_policy(policy.standard_form())

    kind = hparams.model["type"]
    if kind == "mlp":
        obs_space, act_space = lqr.spaces_from_dims(n_state, n_ctrl, horizon)
        spec = MLPQValue.spec_cls(units=hparams.model["hunits"], activation="ReLU")
        qvals = [MLPQValue(obs_space, act_space, spec) for _ in range(2)]
        qvals_ = [MLPQValue(obs_space, act_space, spec) for _ in range(2)]
    elif kind == "quad":
        rng = hparams.seed
        qvals = [QuadQValue(n_state + n_ctrl, horizon, rng) for _ in range(2)]
        qvals_ = [QuadQValue(n_state + n_ctrl, horizon, rng) for _ in range(2)]
    else:
        raise ValueError(f"Unknown type {kind}")
    qvals = QValueEnsemble(qvals)
    target_qvals = QValueEnsemble(qvals_).requires_grad_(False)
    # Syncronize initial Q values
    target_qvals.load_state_dict(qvals.state_dict())

    # Loss
    target_policy = DeterministicPolicy.add_gaussian_noise(
        policy, noise_stddev=hparams.model.get("target_policy_sigma", 0.3)
    )
    target_vval = HardValue(target_policy, ClippedQValue(target_qvals))
    return qvals, target_qvals, target_vval


def refine_tdbatch(batch: TDBatch) -> TDBatch:
    obs, act, rew, new_obs = (t.refine_names("B", ...) for t in batch)
    obs, act, new_obs = nt.vector(obs, act, new_obs)
    return obs, act, rew, new_obs


class LightningQValue(pl.LightningModule):  # pylint:disable=too-many-ancestors
    def __init__(self, lqg: LQGModule, policy: TVLinearPolicy, hparams: dict):
        super().__init__()
        # Register modules to cast them to appropriate device
        self.add_module("_lqg", lqg)
        self.policy = policy
        self.hparams.update(hparams)

        # NN modules
        self.qvals, self.targ_qvals, self.target_vval = make_value(policy, self.hparams)
        # Estimator
        self.estimator = estimator.mfdpg_estimator(
            self.policy, ClippedQValue(self.qvals)
        )

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

    def forward(self, batch: TDBatch) -> Tensor:
        """Returns the temporal difference error induced by the value function."""
        # pylint:disable=arguments-differ
        obs, act, rew, new_obs = batch
        with torch.no_grad():
            target = nt.unnamed(rew + self.target_vval(new_obs))
        loss_fn = nn.MSELoss()
        values = nt.unnamed(*self.qvals(obs, act))
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
        batch = refine_tdbatch(batch)
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
        batch = refine_tdbatch(batch)
        self.log_dict(with_prefix("val/", self._evaluate_on(batch)))

    def test_step(self, batch: TDBatch, batch_idx: int) -> None:
        # pylint:disable=arguments-differ
        del batch_idx
        batch = refine_tdbatch(batch)
        self.log_dict(with_prefix("test/", self._evaluate_on(batch)))

    def _evaluate_on(self, batch: TDBatch) -> TensorDict:
        obs, act, rew, new_obs = batch
        prediction = self.qvals.clipped(nt.unnamed(*self.qvals(obs, act)))
        td_error = analysis.relative_error(
            rew + self.target_vval(new_obs), prediction
        ).mean()
        with torch.enable_grad():
            val, svg = self.estimator(obs)
        return {
            "relative_td_error": td_error,
            "relative_vval_err": analysis.vvalue_err(val, obs, self._vval),
            "grad_acc": analysis.gradient_accuracy(
                [svg], lqr.Linear(self.true_svg_K, self.true_svg_k)
            ),
        }

    def log_grad_norm(self, grad_norm_dict: TensorDict) -> None:
        # Override original: set prog_bar=False to reduce clutter
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
