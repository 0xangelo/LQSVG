# pylint:disable=unsubscriptable-object
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from scipy.stats import norm as normal
from scipy.stats import ortho_group
from torch import Tensor

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.utils import random_matrix_from_eigs
from lqsvg.envs.lqr.utils import wrap_sample_shape_to_size
from lqsvg.torch.utils import default_generator_seed

from .utils import standard_fixture


@pytest.fixture(params=(0, 2))
def dim(request) -> int:
    return request.param


@pytest.fixture
def sampler(dim: int) -> callable[[int], np.ndarray]:
    call = normal.rvs if dim == 0 else partial(ortho_group.rvs, dim=3)

    def _sample(size: int) -> np.ndarray:
        return call(size=size)

    return _sample


def test_wrap_sample_shape_to_size(sampler: callable[[int], np.ndarray], dim: int):
    wrapped = wrap_sample_shape_to_size(sampler, dim)

    def prefix(arr: np.ndarray) -> tuple[int, ...]:
        return arr.shape[:-dim] if dim else arr.shape

    sample_shape = ()
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape

    sample_shape = (1,)
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape

    sample_shape = (2,)
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape

    sample_shape = (2, 1)
    sample = wrapped(sample_shape)
    assert prefix(sample) == sample_shape


vec_dim = standard_fixture((2, 3, 4), "VecDim")
batch_shape = standard_fixture(((), (1,), (2,), (2, 1)), "BatchShape")
seed = standard_fixture(range(10), "Seed")


@pytest.fixture()
def eigvals(vec_dim: int, batch_shape: tuple[int, ...], seed: int) -> Tensor:
    shape = batch_shape + (vec_dim,)
    with default_generator_seed(seed):
        return nt.vector(torch.empty(shape).uniform_(-1, 1))


def test_random_matrix_from_eigs(eigvals: Tensor, seed: int):
    mat = random_matrix_from_eigs(eigvals, rng=seed).numpy()
    eigvals_, _ = np.linalg.eig(mat)
    assert np.allclose(np.sort(eigvals_, axis=-1), np.sort(eigvals.numpy(), axis=-1))
