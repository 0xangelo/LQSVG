"""Utilities for simple fixture creation."""
from typing import Any, Iterable

import pytest

# pylint:disable=missing-function-docstring


def standard_fixture(params: Iterable[Any], name: str) -> callable:
    @pytest.fixture(params=params, ids=lambda x: f"{name}:{x}")
    def func(request):
        return request.param

    return func
