"""Full policy optimization via Stochastic Value Gradients."""
# pylint:disable=missing-docstring
import itertools
import os
from collections import deque
from functools import partial
from typing import Callable, Generator, Optional, Sequence, Tuple

import click
import pytorch_lightning as pl
import torch
import yaml
from nnrl.nn.utils import update_polyak
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.logger import LoggerCallback, NoopLogger
from torch import Tensor, nn

from lqsvg import analysis, data, estimator, lightning
from lqsvg.envs import lqr
from lqsvg.random import RNG, make_rng
from lqsvg.torch import named as nt
from lqsvg.torch.nn import LQGModule, QuadQValue, QuadRewardModel, TVLinearPolicy
from lqsvg.torch.random import sample_with_replacement
from lqsvg.types import DeterministicPolicy, Transition

# isort: off
# pylint:disable=wrong-import-order
from critic import qval_constructor, td_modules
from model import make_model as dynamics_model
from model import surrogate_fn
from prediction_roundup import (
    dynamics_loss,
    reward_loss,
    td_mse,
    mage_loss,
    logging_setup,
)
from wandb_util import WANDB_DIR, calver, with_prefix


DynamicsBatch = Tuple[Tensor, Tensor, Tensor]
RewardBatch = Tuple[Tensor, Tensor, Tensor]
Replay = Sequence[data.Trajectory]


def make_model(
    lqg: LQGModule, policy: TVLinearPolicy, rng: RNG, config: dict
) -> nn.Module:
    # pylint:disable=unused-argument
    model = nn.Module()
    if config["perfect_model"]:
        model.dynamics = lqg.trans
        model.reward = lqg.reward
        model.qval = QuadQValue(
            n_tau=lqg.n_state + lqg.n_ctrl, horizon=lqg.horizon, rng=rng.numpy
        )
    else:
        model.dynamics = dynamics_model(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, config["dynamics"]
        )
        model.reward = QuadRewardModel(
            lqg.n_state, lqg.n_ctrl, lqg.horizon, stationary=True, rng=rng.numpy
        )
        qval_fn = qval_constructor(policy, config["qvalue"]["model"])
        model.qval, model.target_qval, model.target_vval = td_modules(
            policy, qval_fn, config["qvalue"]["model"], rng.numpy
        )
    return model


def env_model_trainer(model: nn.Module, config: dict) -> Callable[[RNG, Replay], dict]:
    if config["perfect_model"]:
        return lambda *_: {}

    pl_dynamics = lightning.Lightning(
        model.dynamics, dynamics_loss(model.dynamics), config
    )
    pl_reward = lightning.Lightning(model.reward, reward_loss(model.reward), config)

    @lightning.suppress_dataloader_warnings(num_workers=True)
    @lightning.suppress_datamodule_warnings()
    def train(rng: RNG, dataset: Replay) -> dict:
        obs, act, rew, _ = map(partial(torch.cat, dim="B"), zip(*dataset))
        obs, new_obs = data.obs_trajectory_to_transitions(obs)
        dynamics_dm = data.SequenceDataModule(
            obs, act, new_obs, spec=config["dynamics_dm"], rng=rng.torch
        )
        reward_dm = data.TensorDataModule(
            *data.merge_horizon_and_batch_dims(obs, act, rew),
            spec=config["reward_dm"],
            rng=rng.torch,
        )
        dynamics_info = lightning.train_lite(pl_dynamics, dynamics_dm, config)
        reward_info = lightning.train_lite(pl_reward, reward_dm, config)

        return {
            **with_prefix("dynamics/", dynamics_info),
            **with_prefix("reward/", reward_info),
        }

    return train


def critic_updater(
    model: nn.Module, policy: TVLinearPolicy, config: dict
) -> Callable[[RNG, Transition], dict]:
    if config["perfect_model"]:

        def match(*_) -> dict:
            model.qval.match_policy_(
                policy.standard_form(),
                model.dynamics.standard_form(),
                model.reward.standard_form(),
            )
            return {}

        return match

    q_optim = torch.optim.Adam(
        model.qval.parameters(), lr=config["qvalue"]["learning_rate"]
    )
    q_loss = (
        mage_loss(model, policy, model.dynamics, model.reward)
        if "mage" in config["strategy"]
        else td_mse(model)
    )
    batch_size = config["qvalue"]["batch_size"]

    def update(rng: RNG, transition: Transition) -> dict:
        obs, act, rew, new_obs = sample_with_replacement(
            transition, size=batch_size, dim="B", rng=rng.torch
        )

        q_optim.zero_grad()
        critic_loss = q_loss((obs, act, rew, new_obs))
        critic_loss.backward()
        q_optim.step()
        update_polyak(model.qval, model.target_qval, config["qvalue"]["polyak"])

        return {"critic_loss": critic_loss.item()}

    return update


def actor_obj(
    lqg: LQGModule, policy: TVLinearPolicy, model: nn.Module, config: dict
) -> Callable[[Tensor], Tensor]:
    if config["strategy"] == "perfect_grad":
        dynamics, cost, init = lqg.standard_form()
        return lambda _: estimator.analytic_value(
            policy.standard_form(), init, dynamics, cost
        )

    if "maac" in config["strategy"]:
        surrogate = surrogate_fn(model.dynamics, policy, model.reward, model.qval)
        return partial(surrogate, n_steps=config["pred_horizon"])

    return estimator.mfdpg_surrogate(policy, model.qval)


def policy_updater(
    lqg: LQGModule, policy: TVLinearPolicy, model: nn.Module, config: dict
) -> Callable[[RNG, Transition], dict]:
    # Ground-truth
    dynamics, cost, init = lqg.standard_form()
    with torch.no_grad():
        optimal = estimator.optimal_value(dynamics, cost, init)

    optim = torch.optim.Adam(policy.parameters(), lr=config["learning_rate"])
    obj_fn = actor_obj(lqg, policy, model, config)

    def update(rng: RNG, transition: Transition) -> dict:
        obs = sample_with_replacement(
            transition[0], size=config["svg_batch_size"], dim="B", rng=rng.torch
        )

        true_val, true_svg = estimator.analytic_svg(policy, init, dynamics, cost)

        optim.zero_grad()
        val = obj_fn(obs)
        val.neg().backward()
        grad_norm = nn.utils.clip_grad_norm_(
            policy.parameters(), config["clip_grad_norm"]
        )
        optim.step()
        svg = lqr.Linear(-policy.K.grad, -policy.k.grad)

        return {
            "surrogate_value": val.item(),
            "true_value": true_val.item(),
            "optimal_value": optimal.item(),
            "grad_acc": analysis.cosine_similarity(svg, true_svg).item(),
            "grad_norm": grad_norm.item(),
            "true_grad_norm": analysis.total_norm(true_svg).item(),
            "suboptimality_gap": analysis.relative_error(optimal, true_val).item(),
        }

    return update


@torch.no_grad()
def prepopulate_(
    dataset: deque,
    collect: Callable[[DeterministicPolicy, int], data.Trajectory],
    policy: DeterministicPolicy,
    config: dict,
):
    dataset.append(
        collect(policy, config["learning_starts"] - config["trajs_per_iter"])
    )


def collection_stats(
    iteration: int, trajectories: data.Trajectory, dataset: Replay
) -> dict:
    batched = tuple(map(partial(torch.cat, dim="B"), zip(*dataset)))
    obs, act, rew, logp = (t.align_to("B", "H", ...)[-100:] for t in batched)

    rets = rew.sum("H")
    log_likelihoods = logp.sum("H")
    episode_stats = {
        "obs_mean": obs.mean().item(),
        "obs_std": obs.std().item(),
        "act_mean": act.mean().item(),
        "act_std": act.std().item(),
        "episode_return_max": rets.max().item(),
        "episode_return_mean": rets.mean().item(),
        "episode_return_min": rets.min().item(),
        "episode_logp_mean": log_likelihoods.mean().item(),
    }

    actions = trajectories[1] if iteration > 0 else batched[1]
    # noinspection PyArgumentList
    timesteps_this_iter = actions.size("B") * actions.size("H")
    # noinspection PyArgumentList
    episodes_this_iter = actions.size("B")
    return {
        tune.result.TIMESTEPS_THIS_ITER: timesteps_this_iter,
        tune.result.EPISODES_THIS_ITER: episodes_this_iter,
        **with_prefix("collect/", episode_stats),
    }


def timesteps_in(trajectories: data.Trajectory) -> int:
    actions = trajectories[1]
    # noinspection PyArgumentList
    return actions.size("B") * actions.size("H")


def all_transitions(dataset: Replay) -> Transition:
    obs, act, rew, _ = map(partial(torch.cat, dim="B"), zip(*dataset))
    obs, new_obs = data.obs_trajectory_to_transitions(obs)
    obs, act, rew, new_obs = data.merge_horizon_and_batch_dims(obs, act, rew, new_obs)
    return obs, act, rew, new_obs


@torch.enable_grad()
def diagnostics(
    rng: RNG,
    transitions: Transition,
    lqg: LQGModule,
    policy: TVLinearPolicy,
    model: nn.Module,
) -> dict:
    # pylint:disable=too-many-locals
    obs, act, _, _ = sample_with_replacement(
        transitions, size=256, dim="B", rng=rng.torch
    )
    obs.requires_grad_()
    act.requires_grad_()

    # Q-value
    dynamics, cost, _ = lqg.standard_form()
    true_qval_fn = QuadQValue.from_policy(policy.standard_form(), dynamics, cost)

    qval = true_qval_fn(obs, act)
    grad_out = nt.unnamed(torch.ones_like(qval))
    ograd = torch.autograd.grad(qval, obs, grad_outputs=grad_out, retain_graph=True)
    agrad = torch.autograd.grad(qval, act, grad_outputs=grad_out)
    qval_ = model.qval(obs, act)
    ograd_ = torch.autograd.grad(qval_, obs, grad_outputs=grad_out, retain_graph=True)
    agrad_ = torch.autograd.grad(qval_, act, grad_outputs=grad_out)
    qval_logs = {
        "relative_err": analysis.relative_error(qval, qval_).mean().item(),
        "obs_grad_acc": analysis.cosine_similarity(ograd, ograd_, dim=-1).mean().item(),
        "act_grad_acc": analysis.cosine_similarity(agrad, agrad_, dim=-1).mean().item(),
    }

    # Reward
    reward = lqg.reward(obs, act)
    ograd = torch.autograd.grad(reward, obs, grad_outputs=grad_out, retain_graph=True)
    agrad = torch.autograd.grad(reward, act, grad_outputs=grad_out)
    reward_ = model.reward(obs, act)
    ograd_ = torch.autograd.grad(reward_, obs, grad_outputs=grad_out, retain_graph=True)
    agrad_ = torch.autograd.grad(reward_, act, grad_outputs=grad_out)
    reward_logs = {
        "relative_err": analysis.relative_error(reward, reward_).mean().item(),
        "obs_grad_acc": analysis.cosine_similarity(ograd, ograd_, dim=-1).mean().item(),
        "act_grad_acc": analysis.cosine_similarity(agrad, agrad_, dim=-1).mean().item(),
    }

    return {**with_prefix("qval/", qval_logs), **with_prefix("reward/", reward_logs)}


def policy_optimization_(
    lqg: LQGModule, policy: TVLinearPolicy, config: dict
) -> Generator[dict, None, None]:
    """Trains a policy via Stochastic Value Gradients."""
    rng = make_rng(config["seed"])
    model = make_model(lqg, policy, rng, config["model"])

    optimize_model_ = env_model_trainer(model, config["model"])
    update_critic_ = critic_updater(model, policy, config["model"])
    update_policy_ = policy_updater(lqg, policy, model, config)

    dataset = deque(maxlen=config["replay_size_trajs"])
    collect = data.environment_sampler(lqg)
    prepopulate_(dataset, collect, policy, config)

    for i in itertools.count():
        with torch.no_grad():
            trajectories = collect(policy, config["trajs_per_iter"])
        dataset.append(trajectories)

        metrics = {}
        metrics.update(optimize_model_(rng, dataset))
        transitions = all_transitions(dataset)
        for _ in range(timesteps_in(trajectories)):
            metrics.update(update_critic_(rng, transitions))
            metrics.update(update_policy_(rng, transitions))

        metrics.update(collection_stats(i, trajectories, dataset))
        metrics.update(diagnostics(rng, transitions, lqg, policy, model))
        yield metrics


def env_and_policy(config: dict) -> Tuple[LQGModule, TVLinearPolicy]:
    def create_env(conf: dict) -> LQGModule:
        generator = lqr.LQGGenerator(
            stationary=True,
            controllable=True,
            rng=conf["seed"],
            **conf["env_config"],
        )
        dynamics, cost, init = generator()
        return LQGModule.from_existing(dynamics, cost, init).requires_grad_(False)

    def init_policy(env: LQGModule, conf: dict) -> TVLinearPolicy:
        return TVLinearPolicy(env.n_state, env.n_ctrl, env.horizon).stabilize_(
            env.trans.standard_form(), rng=conf["seed"]
        )

    with nt.suppress_named_tensor_warning():
        lqg = create_env(config)
    policy = init_policy(lqg, config)
    return lqg, policy


class SVG(tune.Trainable):
    coroutine: Generator[dict, None, None]

    def setup(self, config: dict):
        lightning.suppress_info_logging()
        pl.seed_everything(config["seed"])
        lqg, policy = env_and_policy(config)
        config["model"]["strategy"] = config["strategy"]
        self.coroutine = policy_optimization_(lqg, policy, config)

    def step(self) -> dict:
        return next(self.coroutine)

    def cleanup(self):
        self.coroutine.close()

    def _create_logger(
        self,
        config: dict,
        logger_creator: Optional[Callable[[dict], LoggerCallback]] = None,
    ):
        # Override with no default logger creator.
        # Avoids creating junk in ~/ray_results
        if logger_creator is None:
            logger_creator = partial(NoopLogger, logdir=os.devnull)
        self._result_logger = logger_creator(config)
        self._logdir = self._result_logger.logdir


@click.group()
def main():
    pass


def base_config() -> dict:
    return {
        "seed": 124,
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
            "horizon": 100,
            "passive_eigval_range": (0.9, 1.1),
        },
        "learning_rate": 1e-4,
        "clip_grad_norm": 1_000,
        "svg_batch_size": 256,
        "strategy": "maac",
        "pred_horizon": 4,
        "replay_size_trajs": 2_000,
        "learning_starts": 20,
        "trajs_per_iter": 1,
        "model": {"perfect_model": True},
    }


@main.command()
@logging_setup()
def sweep():
    config = {
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
        },
        "seed": tune.grid_search(list(range(780, 800))),
        # "strategy": tune.grid_search(["maac", "mage", "maac+mage"]),
        "strategy": "maac",
        "learning_rate": 3e-4,
        "model": {
            "perfect_model": False,
            "learning_rate": 1e-3,
            "weight_decay": 0,
            "max_epochs": 40,
            "dynamics": {"type": "linear"},
            "qvalue": {
                "learning_rate": 3e-4,
                "batch_size": 128,
                "polyak": 0.995,
                "model": {"type": "quad"},
            },
            "dynamics_dm": {
                "train_batch_size": 128,
                "val_batch_size": 128,
                "seq_len": 8,
            },
            "reward_dm": {
                "train_batch_size": 64,
                "val_batch_size": 64,
            },
        },
    }
    config = tune.utils.merge_dicts(base_config(), config)

    logger = WandbLoggerCallback(
        name="ModelBasedControl",
        project="ch5",
        entity="angelovtt",
        tags=[calver(), "ModelBasedControl"],
        dir=WANDB_DIR,
    )
    tune.run(
        SVG,
        config=config,
        num_samples=1,
        stop={tune.result.TRAINING_ITERATION: 100},
        local_dir=WANDB_DIR,
        callbacks=[logger],
    )


@main.command()
def debug():
    config = {
        "env_config": {
            "n_state": 2,
            "n_ctrl": 2,
        },
        "seed": 780,
        "strategy": "maac+mage",
        "learning_rate": 3e-4,
        "model": {
            "perfect_model": False,
            "learning_rate": 1e-3,
            "weight_decay": 0,
            "max_epochs": 20,
            "dynamics": {"type": "linear"},
            "qvalue": {
                "learning_rate": 3e-4,
                "batch_size": 128,
                "polyak": 0.995,
                "model": {"type": "quad"},
            },
            "dynamics_dm": {
                "train_batch_size": 128,
                "val_batch_size": 128,
                "seq_len": 8,
            },
            "reward_dm": {
                "train_batch_size": 64,
                "val_batch_size": 64,
            },
        },
    }
    config = tune.utils.merge_dicts(base_config(), config)
    exp = SVG(tune.utils.merge_dicts(base_config(), config))
    print(yaml.dump({k: v for k, v in exp.train().items() if k != "config"}))


if __name__ == "__main__":
    main()
