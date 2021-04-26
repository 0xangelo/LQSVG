from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from numpy.random import Generator

from lqsvg.experiment.analysis import optimization_surface
from lqsvg.testing.fixture import standard_fixture


def squared(x: np.ndarray) -> np.ndarray:
    return x ** 2


def dot_2d(x: np.ndarray) -> np.ndarray:
    return x.dot(np.array([1, 1]))


def dot_3d(x: np.ndarray) -> np.ndarray:
    return x.dot(np.array([1, 1, 1]))


FUNCTIONS = {1: squared, 2: dot_2d, 3: dot_3d}
dim = standard_fixture((1, 2, 3), "Dim")
max_scaling = standard_fixture((0.1, 1.0, 3.0), "MaxScaling")
steps = standard_fixture((2, 5, 10), "Steps")


@pytest.fixture
def rng(seed: int) -> Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def f_delta(dim: int) -> Callable[[np.ndarray], np.ndarray]:
    return FUNCTIONS[dim]


@pytest.fixture
def direction(rng: Generator, dim: int) -> np.ndarray:
    return rng.normal(size=(dim,))


def test_optimization_surface(
    f_delta: Callable[[np.ndarray], np.ndarray],
    direction: np.ndarray,
    max_scaling: float,
    steps: int,
    rng: Generator,
):
    X, Y, Z = optimization_surface(f_delta, direction, max_scaling, steps, rng=rng)

    arrs = (X, Y, Z)
    assert all(list(isinstance(x, np.ndarray) for x in arrs))
    assert all(list(np.all(np.isfinite(x)) for x in arrs))
    assert X.shape == Y.shape == Z.shape == (steps, steps)
