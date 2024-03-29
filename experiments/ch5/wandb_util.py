# pylint:disable=missing-docstring
import os
from typing import Sequence, Tuple

import wandb
from wandb.sdk.wandb_run import Run

from lqsvg.envs.lqr.utils import iscontrollable, isstable, stationary_eigvals
from lqsvg.experiment.utils import calver
from lqsvg.torch.nn import LQGModule

WANDB_DIR = os.path.abspath("./results")


def env_info(lqg: LQGModule) -> dict:
    dynamics = lqg.trans.standard_form()
    eigvals = stationary_eigvals(dynamics)
    return {
        "stability": isstable(eigvals=eigvals),
        "controllability": iscontrollable(dynamics),
        "passive_eigvals": wandb.Histogram(eigvals),
    }


def extra_tags(config: dict) -> Tuple[str, ...]:
    extra = ()
    env_config = config.get("env_config")
    if env_config and env_config["n_state"] > env_config["n_ctrl"]:
        extra += ("underactuated",)
    return extra


def wandb_init(
    name: str,
    config: dict,
    project="ch5",
    entity="angelovtt",
    tags: Sequence[str] = (),
    reinit=True,
    **kwargs,
) -> Run:
    # pylint:disable=too-many-arguments
    return wandb.init(
        name=name,
        config=config,
        project=project,
        entity=entity,
        dir=WANDB_DIR,
        tags=(calver(),) + extra_tags(config) + tuple(tags),
        reinit=reinit,
        **kwargs,
    )


def with_prefix(prefix: str, dictionary: dict) -> dict:
    """Returns a dictionary copy with prefixed keys."""
    return {prefix + k: v for k, v in dictionary.items()}
