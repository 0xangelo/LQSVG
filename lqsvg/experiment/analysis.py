"""Utilities for measuring gradient quality and optimization landscapes."""
from __future__ import annotations

import torch

from lqsvg.envs import lqr

from . import utils


def gradient_accuracy(svg_samples: list[lqr.Linear], target: lqr.Linear) -> float:
    """Compute the average cosine similarity with the target gradient."""
    cossims = [utils.linear_feedback_cossim(g, target) for g in svg_samples]
    return torch.stack(cossims).mean().item()


def empirical_variance(svg_samples: list[lqr.Linear]) -> float:
    """Compute the average pairwise cosine similarity between gradient samples."""
    # pylint:disable=invalid-name
    cossims = []
    for i, gi in enumerate(svg_samples):
        for gj in svg_samples[i + 1 :]:
            cossims += [utils.linear_feedback_cossim(gi, gj)]

    return torch.stack(cossims).mean().item()
