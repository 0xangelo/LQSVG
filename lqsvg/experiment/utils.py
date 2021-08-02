"""Utilities for RLlib sample batches, warnings, and linear feedback policies."""
import datetime
import logging
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from ray.rllib import SampleBatch
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs import lqr


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
    return linear_feedback_norm((Ka - Kb, ka - kb))


@contextmanager
def suppress_dataloader_warning():
    """Ignore PyTorch Lightning warnings regarding num of dataloader workers."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Consider increasing the value of the `num_workers`.*",
            module="pytorch_lightning.trainer.data_loading",
        )
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
