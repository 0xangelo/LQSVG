"""Utilities for random number generation."""
from typing import NamedTuple

import numpy as np
import torch as pt


class RNG(NamedTuple):
    """Container for Numpy and PyTorch generators."""

    numpy: np.random.Generator
    torch: pt.Generator


def make_rng(seed: int) -> RNG:
    """Initializes and packs default generators."""
    torch_generator = pt.Generator()
    torch_generator.manual_seed(seed)
    return RNG(numpy=np.random.default_rng(seed), torch=torch_generator)
