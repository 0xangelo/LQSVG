from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import torch
from numpy.random import Generator
from torch import nn

import lqsvg.torch.named as nt
from lqsvg.analysis import delta_to_return, optimization_surface
from lqsvg.envs import lqr
from lqsvg.envs.lqr.generators import LQGGenerator
from lqsvg.testing.fixture import standard_fixture
from lqsvg.torch.utils import tensors_to_vector


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


@pytest.fixture
def policy(n_state: int, n_ctrl: int, horizon: int) -> lqr.Linear:
    K = torch.Tensor(horizon, n_ctrl, n_state)
    k = torch.Tensor(horizon, n_ctrl)
    nn.init.xavier_uniform_(K)
    nn.init.constant_(k, 0)
    K, k = nt.horizon(nt.matrix(K), nt.vector(k))
    return K, k


@pytest.fixture
def dynamics(lqg_generator: LQGGenerator) -> lqr.LinSDynamics:
    with lqg_generator.config(passive_eigval_range=(0, 1)):
        return lqg_generator.make_dynamics()


@pytest.fixture
def cost(lqg_generator: LQGGenerator) -> lqr.QuadCost:
    return lqg_generator.make_cost()


@pytest.fixture
def init(lqg_generator: LQGGenerator) -> lqr.GaussInit:
    return lqg_generator.make_init()


def test_delta_to_return(
    policy: lqr.Linear,
    dynamics: lqr.LinSDynamics,
    cost: lqr.QuadCost,
    init: lqr.GaussInit,
):
    func = delta_to_return(policy, dynamics, cost, init)
    delta = tensors_to_vector(policy)
    delta = (delta + torch.rand_like(delta)).numpy()

    out = func(delta)
    assert np.ndim(out) == 0
    assert out.size == 1
    assert np.isfinite(out)

    # Should be repeatable
    assert np.allclose(out, func(delta))
