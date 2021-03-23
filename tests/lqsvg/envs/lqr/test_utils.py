# pylint:disable=unsubscriptable-object
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from scipy.stats import norm as normal
from scipy.stats import ortho_group

from lqsvg.envs.lqr.utils import wrap_sample_shape_to_size


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
