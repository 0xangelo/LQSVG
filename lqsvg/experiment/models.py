# pylint:disable=missing-docstring,unsubscriptable-object
from __future__ import annotations

import itertools
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from raylab.policy.modules.actor import DeterministicPolicy
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr
from lqsvg.envs.lqr.gym import TorchLQGMixin
from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.envs.lqr.modules.general import EnvModule
from lqsvg.experiment.estimators import AnalyticSVG, MonteCarloSVG
from lqsvg.policy.time_varying_linear import LQGPolicy

from .utils import linear_feedback_cossim, linear_feedback_norm


class LightningModel(pl.LightningModule):
    # pylint:disable=too-many-ancestors
    actor: DeterministicPolicy
    model: EnvModule
    mdp: LQGModule
    gold_standard: tuple[Tensor, lqr.Linear]
    early_stop_on: str = "val/loss"

    def __init__(self, policy: LQGPolicy, env: TorchLQGMixin):
        super().__init__()
        self.actor = policy.module.actor
        self.model = policy.module.model
        self.mdp = env.module
        self.monte_carlo_svg = MonteCarloSVG(self.actor, self.model)

        self.analytic_svg = None
        if isinstance(self.model, LQGModule):
            self.analytic_svg = AnalyticSVG(self.actor, self.model)

        self.gold_standard = AnalyticSVG(self.actor, self.mdp)()

    # noinspection PyArgumentList
    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        """Batched trajectory log prob."""
        # pylint:disable=arguments-differ
        return self.model.log_prob(obs, act, new_obs) / obs.size("H")

    def configure_optimizers(self):
        params = nn.ParameterList(
            itertools.chain(self.model.trans.parameters(), self.model.init.parameters())
        )
        optim = torch.optim.Adam(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optim

    def _compute_loss_on_batch(
        self, batch: tuple[Tensor, Tensor, Tensor], _: int
    ) -> Tensor:
        obs, act, new_obs = (x.refine_names("B", "H", "R") for x in batch)
        return -self(obs, act, new_obs).mean()

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        self.model.train()
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int):
        # pylint:disable=arguments-differ
        loss = self._compute_loss_on_batch(batch, batch_idx)
        self.log("test/loss", loss)

    def test_epoch_end(self, test_step_outputs):
        # pylint:disable=arguments-differ
        del test_step_outputs
        self.log_gold_standard()
        self.value_gradient_info("test")

    def value_gradient_info(self, prefix: str = ""):
        self.log_monte_carlo(prefix + "/")
        if self.analytic_svg is not None:
            self.log_analytic(prefix + "/")

    def log_monte_carlo(self, prefix: str = ""):
        with torch.enable_grad():
            mc_val, mc_svg = self.monte_carlo_svg(samples=self.hparams.mc_samples)
        self.zero_grad(set_to_none=True)

        true_val, true_svg = self.gold_standard
        self.log(prefix + "monte_carlo_value", mc_val)
        self.log(prefix + "monte_carlo_svg_norm", linear_feedback_norm(mc_svg))
        self.log(prefix + "monte_carlo_abs_diff", torch.abs(mc_val - true_val))
        self.log(
            prefix + "monte_carlo_cossim", linear_feedback_cossim(mc_svg, true_svg)
        )
        self.log(prefix + "empirical_grad_var", self.empirical_variance([mc_svg]))

    def empirical_variance(self, existing: Optional[list[lqr.Linear]] = None) -> float:
        samples = existing or []

        n_grads = self.hparams.empvar_samples - len(samples)
        with torch.enable_grad():
            samples += [
                self.monte_carlo_svg(samples=self.hparams.mc_samples)[1]
                for _ in range(n_grads)
            ]
        self.zero_grad(set_to_none=True)

        cossims = [
            linear_feedback_cossim(gi, gj)
            for i, gi in enumerate(samples)
            for gj in samples[i + 1 :]
        ]

        return torch.stack(cossims).mean().item()

    def log_analytic(self, prefix: str = ""):
        with torch.enable_grad():
            analytic_val, analytic_svg = self.analytic_svg()
        self.zero_grad(set_to_none=True)

        true_val, true_svg = self.gold_standard
        self.log(prefix + "analytic_value", analytic_val)
        self.log(prefix + "analytic_svg_norm", linear_feedback_norm(analytic_svg))
        self.log(prefix + "analytic_abs_diff", torch.abs(analytic_val - true_val))
        self.log(
            prefix + "analytic_cossim", linear_feedback_cossim(analytic_svg, true_svg)
        )

    def log_gold_standard(self):
        """Logs gold standard value and gradient."""
        true_val, true_svg = self.gold_standard
        self.log("true_value", true_val)
        self.log("true_svg_norm", linear_feedback_norm(true_svg))


class RecurrentModel(LightningModel):
    # pylint:disable=too-many-ancestors
    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        obs, act, new_obs = (x.align_to("H", ..., "R") for x in (obs, act, new_obs))

        init_model = self.model.init
        trans_model = self.model.trans

        init_logp = init_model.log_prob(obs.select(dim="H", index=0))

        trans_logp = []
        obs_, _ = init_model.rsample(init_logp.shape)
        obs_ = obs_.rename(*init_logp.names, ...)
        for t in range(self.model.horizon):  # pylint:disable=invalid-name
            params = trans_model(obs_, act.select(dim="H", index=t))
            trans_logp += [
                trans_model.log_prob(new_obs.select(dim="H", index=t), params)
            ]

            obs_, _ = trans_model.rsample(params)

        trans_logp = nt.stack_horizon(*trans_logp).sum(dim="H")

        return (init_logp + trans_logp) / self.model.horizon


# noinspection PyTypeChecker
@nt.suppress_named_tensor_warning()
def test_lightning_model():
    import lqsvg

    from .worker import make_worker

    lqsvg.register_all()
    worker = make_worker(env_config=dict(n_state=2, n_ctrl=2, horizon=100, num_envs=1))
    model = LightningModel(worker.get_policy(), worker.env)

    def print_traj(traj):
        obs, act, rew, new_obs, logp = traj
        print(
            f"""\
        Obs: {obs.shape}, {obs.names}
        Act: {act.shape}, {act.names}
        Rew: {rew.shape}, {rew.names}
        New Obs: {new_obs.shape}, {new_obs.names}
        Logp: {logp.shape}, {logp.names}\
        """
        )

    monte_carlo = model.monte_carlo_svg
    true_mc = MonteCarloSVG(model.actor, model.mdp)

    print("Model sample:")
    print_traj(monte_carlo.rsample_trajectory([]))
    print("Batched model sample:")
    print_traj(monte_carlo.rsample_trajectory([10]))
    print("MDP sample:")
    print_traj(true_mc.rsample_trajectory([]))

    obs, act, _, new_obs, sample_logp = monte_carlo.rsample_trajectory([10])
    print(f"RSample logp: {sample_logp}, {sample_logp.shape}")
    traj_logp = model(obs, act, new_obs)
    print(f"Traj logp: {traj_logp}, {traj_logp.shape}")
    print("Model logp of MDP sample:")
    obs, act, _, new_obs, _ = true_mc.rsample_trajectory([10])
    traj_logp = model(obs, act, new_obs)
    print(traj_logp, traj_logp.shape)

    print("Monte Carlo value:", monte_carlo.value(samples=256))
    if model.analytic_svg is not None:
        print("Analytic value:", model.analytic_svg.value())
    print("True value:", model.gold_standard[0])


if __name__ == "__main__":
    test_lightning_model()
