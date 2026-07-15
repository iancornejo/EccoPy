"""Plan-view plotting for EccoPy-2D-H (single-level / composite) results."""

from __future__ import annotations
from typing import Optional
import numpy as np


def plot_result(result,
                dbz: np.ndarray,
                coords_y: np.ndarray,
                coords_x: np.ndarray,
                figsize=(14, 5),
                show: bool = False,
                show_window: bool = True,
                outfile: Optional[str] = None):
    """
    3-panel figure: reflectivity, convectivity, echo type — for a single
    EccoPy-2D-H horizontal level.

    Parameters
    ----------
    result : Result2DH
        Output of eccopy2d_h.run().
    dbz : np.ndarray, shape (Y, X)
        The reflectivity input that was passed to eccopy2d_h.run().
    coords_y, coords_x : np.ndarray, shape (Y,), (X,)
        Coordinate axes, km.
    figsize : tuple
    show : bool
    show_window : bool
        If True (default), overlay the texture window footprint
        (``result.texture_radius``) as a dashed circle at the domain
        centre of each panel, so users can see the neighbourhood size
        relative to their features. Drawn only when the result carries a
        finite physical radius (a bare-pixel window on a unit-agnostic
        grid is skipped).
    outfile : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from ..core.colormaps import (basic_echo_type_cmap, basic_echo_type_norm,
                                   remap_echo_type, BASIC_ECHO_TYPE_LABELS,
                                   convectivity_cmap, convectivity_norm,
                                   draw_window_ring)

    Y, X = np.meshgrid(coords_y, coords_x, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    pc = ax.pcolormesh(X, Y, dbz, cmap="jet", vmin=-10, vmax=65, shading="auto")
    plt.colorbar(pc, ax=ax, label="dBZ")
    ax.set(title="Reflectivity", xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    ax = axes[1]
    pc = ax.pcolormesh(X, Y, result.convectivity, cmap=convectivity_cmap(),
                       norm=convectivity_norm(), shading="auto")
    plt.colorbar(pc, ax=ax)
    ax.set(title="Convectivity", xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    ax = axes[2]
    cmap = basic_echo_type_cmap()
    norm = basic_echo_type_norm()
    pc = ax.pcolormesh(X, Y, remap_echo_type(result.echo_type),
                       cmap=cmap, norm=norm, shading="auto")
    cb = plt.colorbar(pc, ax=ax, ticks=range(1, 4))
    cb.ax.set_yticklabels(BASIC_ECHO_TYPE_LABELS)
    ax.set(title="Echo Type", xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    if show_window:
        radius_km = getattr(result, "texture_radius", None)
        for ax in axes:
            draw_window_ring(ax, coords_x, coords_y, radius_km)

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
