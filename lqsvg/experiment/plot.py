"""Utilities for plots in matplotlib."""
from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


def default_figsize(rows: int = 1, cols: int = 1) -> tuple[float, float]:
    """Default figure size for a given number of rows and columns."""
    default_width, default_height = mpl.rcParams["figure.figsize"]
    return rows * default_height, cols * default_width


def plot_surface(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, ax=None, invert_xaxis: bool = False
) -> Any:
    """Create a 3D surface plot given inputs and outputs.

    Args:
        X: x axis inputs to the functions as a meshgrid
        Y: y axis inputs to the functions as a meshgrid
        Z: z axis inputs to the functions as a meshgrid
        ax (Axes3DSubplot): optional axis to plot the surface on
        invert_xaxis: whether to invert the order of the x axis values
    """
    # pylint:disable=invalid-name
    if ax is None:
        ax = plt.axes(projection="3d")
    # noinspection PyUnresolvedReferences
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)  # pylint:disable=no-member
    if invert_xaxis:
        ax.invert_xaxis()
    return ax
