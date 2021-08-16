"""Utilities for RLlib sample batches, warnings, and linear feedback policies."""
import datetime
import functools
import itertools
import logging
import operator
import os
import warnings
from contextlib import contextmanager
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
import torch
import wandb
from ray.rllib import SampleBatch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr

from .types import Directory


def directories_with_target_files(
    directories: List[str], istarget: Callable[[str], bool]
) -> List[Directory]:
    """Return directories that havy any file of interest."""
    # (path, subpath, files) for all dirs
    all_paths = itertools.chain(*map(os.walk, directories))
    # convert to directory type
    all_dirs = itertools.starmap(Directory, all_paths)
    # entries that have >0 files (may be unnecessary)
    nonempty = filter(operator.attrgetter("files"), all_dirs)
    # entries that have target files
    with_target_files = (x for x in nonempty if any(map(istarget, x.files)))
    return list(with_target_files)


def experiment_directories(rootdir: str) -> List[Directory]:
    """Return experiment directories."""
    return directories_with_target_files(
        [rootdir], lambda f: f.startswith("progress") and f.endswith(".csv")
    )


def crashed_experiments(rootdir: str) -> List[Directory]:
    """Return experiment directories that have crash logs."""
    exp_dirs = experiment_directories(rootdir)
    return [
        d
        for d in exp_dirs
        if any(map(functools.partial(operator.eq, "error.txt"), d.files))
    ]


def tagged_experiments_dataframe(tags: Iterable[str]) -> pd.DataFrame:
    """Retrieve data from experiments with given tags as a dataframe."""
    api = wandb.Api()
    runs = api.runs(
        "angelovtt/LQG-SVG", filters={"$and": [{"tags": tag} for tag in tags]}
    )
    dfs = (run.history() for run in runs)
    return pd.concat(dfs, ignore_index=True)


def suppress_lightning_info_logging():
    """Silences messages related to GPU/TPU availability."""
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3431
    logging.getLogger("lightning").setLevel(logging.WARNING)


def calver() -> str:
    """Return a standardized version number using CalVer."""
    today = datetime.date.today()
    return f"{today.month}.{today.day}.0"


def linear_feedback_cossim(linear_a: lqr.Linear, linear_b: lqr.Linear) -> Tensor:
    """Cosine similarity between the parameters of linear (affine) functions.

    Args:
        linear_a: tuple of affine function parameters
        linear_b: tuple of affine function parameters

    Returns:
        Tensor representing the (scalar) cosine similarity.
    """
    # pylint:disable=invalid-name
    Ka, ka = linear_a
    Kb, kb = linear_b
    dot_product = torch.sum(Ka * Kb) + torch.sum(ka * kb)
    norm_prod = linear_feedback_norm(linear_a) * linear_feedback_norm(linear_b)
    return dot_product / norm_prod


def linear_feedback_norm(linear: lqr.Linear) -> Tensor:
    """Norm of the parameters of a linear (affine) function.

    Uses the default norms for vectors and matrices chosen by PyTorch:
    frobenius for matrices and L2 for vectors.

    Equivalent to the norm of the flattened parameter vector.

    Args:
        linear: tuple of affine function parameters (weight matrix and bias
        column vector)

    Returns:
        Norm of the affine function's parameters
    """
    # pylint:disable=invalid-name
    K, k = linear
    K_norm = torch.linalg.norm(nt.unnamed(K), dim=(-2, -1))
    k_norm = torch.linalg.norm(nt.unnamed(k), dim=-1)
    # Following PyTorch's clip_grad_norm_ implementation
    # Reduce by horizon
    total_norm = torch.linalg.norm(torch.cat((K_norm, k_norm), dim=0), dim=0)
    return total_norm


def linear_feedback_distance(linear_a: lqr.Linear, linear_b: lqr.Linear) -> Tensor:
    """Distance between the parameters of linear (affine) functions.

    Args:
        linear_a: tuple of affine function parameters
        linear_b: tuple of affine function parameters

    Returns:
        Norm of the difference between the parameters
    """
    # pylint:disable=invalid-name
    Ka, ka = linear_a
    Kb, kb = linear_b
    return linear_feedback_norm(lqr.Linear(Ka - Kb, ka - kb))


@contextmanager
def suppress_dataloader_warnings(num_workers: bool = True, shuffle: bool = False):
    """Ignore PyTorch Lightning warnings regarding dataloaders.

    Args:
        num_workers: include number-of-workers warnings
        shuffle: include val/test dataloader shuffling warnings
    """
    suppress = functools.partial(
        warnings.filterwarnings,
        "ignore",
        module="pytorch_lightning.trainer.data_loading",
    )
    with warnings.catch_warnings():
        if num_workers:
            suppress(message=".*Consider increasing the value of the `num_workers`.*")
        if shuffle:
            suppress(message="Your .+_dataloader has `shuffle=True`")
        yield


def group_batch_episodes(samples: SampleBatch) -> SampleBatch:
    """Return the sample batch with rows grouped by episode id.

    Moreover, rows are sorted by timestep.

    Warning:
        Modifies the sample batch in-place
    """
    # Assume "t" is the timestep key in the sample batch
    sorted_timestep_idxs = np.argsort(samples["t"])
    for key, val in samples.items():
        samples[key] = val[sorted_timestep_idxs]

    # Stable sort is important so that we don't alter the order
    # of timesteps
    sorted_episode_idxs = np.argsort(samples[SampleBatch.EPS_ID], kind="stable")
    for key, val in samples.items():
        samples[key] = val[sorted_episode_idxs]

    return samples


def num_complete_episodes(samples: SampleBatch) -> int:
    """Return the number of complete episodes in a SampleBatch."""
    num_eps = len(np.unique(samples[SampleBatch.EPS_ID]))
    num_dones = np.sum(samples[SampleBatch.DONES]).item()
    assert (
        num_dones <= num_eps
    ), f"More done flags than episodes: dones={num_dones}, episodes={num_eps}"
    return num_dones
