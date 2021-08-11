"""Utilities for simple fixture creation."""
from typing import Any, Callable, Iterable

import pytest

# pylint:disable=missing-function-docstring


def std_id(name: str) -> Callable[[Any], str]:
    return lambda x: f"{name}:{x}"


def standard_fixture(params: Iterable[Any], name: str) -> Callable:
    @pytest.fixture(params=params, ids=std_id(name))
    def func(request):
        return request.param

    return func
