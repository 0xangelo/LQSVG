"""Utilities for plots in matplotlib.

Includes presets for LaTex based on:
https://jwalton.info/Embed-Publication-Matplotlib-Latex/
"""
from __future__ import annotations

import os.path as osp
from textwrap import dedent
from typing import Any, Optional, Union

import matplotlib as mpl
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

STYLE_PATH = osp.join(mpl.__path__[0], "mpl-data", "stylelib")


def create_latex_style():
    """Create latex style file if not present.

    Note:
        Requires restarting the program for Matplotlib to find the style
        file
    """
    texstyle_path = osp.join(STYLE_PATH, "tex.mplstyle")
    if osp.exists(texstyle_path):
        return

    with open(texstyle_path, "w") as file:
        file.write(
            dedent(
                """\
                text.usetex: True
                font.family: serif
                axes.labelsize: 10
                font.size: 10
                legend.fontsize: 8
                xtick.labelsize: 8
                ytick.labelsize: 8
                """
            )
        )


def available_styles():
    """List of available Matplotlib styles."""
    return plt.style.available


def save_pdf_tight(fig, path: str, **kwargs):
    """Save figure as PDF and remove excess whitespace."""
    return fig.savefig(path, format="pdf", bbox_inches="tight", **kwargs)


def latex_size(width: Union[str, float], fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "paper":
        width_pt = 347.12354
    elif width == "thesis":
        width_pt = 432.48189
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def default_figsize(rows: int = 1, cols: int = 1) -> tuple[float, float]:
    """Default figure size for a given number of rows and columns."""
    default_width, default_height = mpl.rcParams["figure.figsize"]
    return cols * default_width, rows * default_height


# noinspection PyPep8Naming
def plot_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    ax: Optional[Axes] = None,
    invert_xaxis: bool = False,
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
