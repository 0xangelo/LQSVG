from typing import Any, Iterable

import pytest

import lqsvg.torch.named as nt
from lqsvg.envs.lqr.types import LinSDynamics, QuadCost


def standard_fixture(params: Iterable[Any], name: str) -> callable:
    @pytest.fixture(params=params, ids=lambda x: f"{name}:{x}")
    def func(request):
        return request.param

    return func


def allclose_dynamics(dyn1: LinSDynamics, dyn2: LinSDynamics) -> bool:
    equal = [nt.allclose(d1, d2) for d1, d2 in zip(dyn1, dyn2)]
    return all(equal)


def allclose_cost(cost1: QuadCost, cost2: QuadCost) -> bool:
    equal = [nt.allclose(c1, c2) for c1, c2 in zip(cost1, cost2)]
    return all(equal)
