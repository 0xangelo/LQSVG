# pylint:disable=missing-docstring
from typing import Callable

from raylab.envs.termination import register as register_raylab

# For testing purposes
TERMINATIONS = {}


def register(*ids: str) -> Callable[[type], type]:
    raylab_callable = register_raylab(*ids)

    def librarian(cls):
        raylab_callable(cls)
        for id_ in ids:
            TERMINATIONS[id_] = cls

        return cls

    return librarian
