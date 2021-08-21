# pylint:disable=missing-docstring
import os
from typing import Sequence, Tuple

import wandb

from lqsvg.envs.lqr.modules import LQGModule
from lqsvg.envs.lqr.utils import iscontrollable, isstable, stationary_eigvals
from lqsvg.experiment.utils import calver

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
    if config["n_state"] > config["n_ctrl"]:
        extra += ("underactuated",)
    return extra


def wandb_init(
    name: str,
    config: dict,
    project="LQG-SVG",
    entity="angelovtt",
    tags: Sequence[str] = (),
    reinit=True,
    **kwargs,
) -> wandb.sdk.wandb_run.Run:
    # pylint:disable=too-many-arguments
    return wandb.init(
        name=name,
        config=config,
        project=project,
        entity=entity,
        dir=WANDB_DIR,
        tags=("ch5", calver()) + extra_tags(config) + tuple(tags),
        reinit=reinit,
        **kwargs,
    )
