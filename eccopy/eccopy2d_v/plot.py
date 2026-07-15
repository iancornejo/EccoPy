"""Vertical cross-section plotting for EccoPy-2D-V results."""

from __future__ import annotations
from typing import Optional
import numpy as np


def plot_result(result,
                dbz: np.ndarray,
                coords_z: np.ndarray,
                coords_x: np.ndarray,
                figsize=(12, 12),
                xlim: Optional[tuple] = None,
                ylim: Optional[tuple] = None,
                show: bool = False,
                outfile: Optional[str] = None):
    """
    3-panel figure: reflectivity, convectivity, echo type — for a single
    EccoPy-2D-V vertical cross-section.

    Parameters
    ----------
    result : Result2DV
        Output of eccopy2d_v.run().
    dbz : np.ndarray, shape (Z, X)
        The reflectivity input that was passed to eccopy2d_v.run().
    coords_z : np.ndarray, shape (Z,)
        Vertical coordinate values, km.
    coords_x : np.ndarray, shape (X,)
        Horizontal coordinate values, km.
    figsize : tuple
    xlim : (float, float) or None — horizontal axis limits [km]
    ylim : (float, float) or None — vertical axis limits [km]
    show : bool
    outfile : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from ..core.colormaps import (echo_type_cmap, echo_type_norm,
                                   basic_echo_type_cmap, basic_echo_type_norm,
                                   remap_echo_type, ECHO_TYPE_LABELS,
                                   BASIC_ECHO_TYPE_LABELS,
                                   convectivity_cmap, convectivity_norm)

    present = set(np.unique(
        result.echo_type[~np.isnan(result.echo_type)]
    ).astype(int).tolist())
    is_basic = present.issubset({1, 2, 3})
    cmap = basic_echo_type_cmap() if is_basic else echo_type_cmap()
    norm = basic_echo_type_norm() if is_basic else echo_type_norm()
    labels = BASIC_ECHO_TYPE_LABELS if is_basic else ECHO_TYPE_LABELS
    n_codes = 3 if is_basic else 9

    X, Z = np.meshgrid(coords_x, coords_z)

    x_lower = xlim[0] if xlim else float(np.nanmin(coords_x))
    x_upper = xlim[1] if xlim else float(np.nanmax(coords_x))
    y_lower = ylim[0] if ylim else float(np.nanmin(coords_z))
    y_upper = ylim[1] if ylim else float(np.nanmax(coords_z))

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    pc = ax.pcolormesh(X, Z, dbz, cmap="jet", vmin=-10, vmax=50, shading="auto")
    ax.set(xlim=(x_lower, x_upper), ylim=(y_lower, y_upper),
           xlabel="Distance (km)", ylabel="Altitude (km)",
           title="Reflectivity (dBZ)")
    plt.colorbar(pc, ax=ax)
    ax.grid(True)

    ax = axes[1]
    pc = ax.pcolormesh(X, Z, result.convectivity, cmap=convectivity_cmap(),
                       norm=convectivity_norm(), shading="auto")
    ax.set(xlim=(x_lower, x_upper), ylim=(y_lower, y_upper),
           xlabel="Distance (km)", ylabel="Altitude (km)", title="Convectivity")
    plt.colorbar(pc, ax=ax)
    ax.grid(True)

    ax = axes[2]
    pc = ax.pcolormesh(X, Z, remap_echo_type(result.echo_type),
                       cmap=cmap, norm=norm, shading="auto")
    cb = plt.colorbar(pc, ax=ax, ticks=range(1, n_codes + 1))
    cb.ax.set_yticklabels(labels)
    ax.set(xlim=(x_lower, x_upper), ylim=(y_lower, y_upper),
           xlabel="Distance (km)", ylabel="Altitude (km)", title="Echo Type")
    ax.grid(True)

    if outfile:
        fig.savefig(outfile, dpi=100, bbox_inches="tight")
    if show:
        plt.show()
    elif outfile:
        # Headless file render: close so figures don't accumulate when
        # plot_result is called in a loop. In an interactive/notebook
        # session (no outfile, show=False) the figure is returned OPEN so
        # it renders inline.
        plt.close(fig)
    return fig
