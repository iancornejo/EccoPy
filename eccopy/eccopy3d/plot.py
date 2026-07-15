"""Plan-view plotting for EccoPy-3D results."""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def plot_result(result,
                dbz: np.ndarray,
                coords_y: np.ndarray,
                coords_x: np.ndarray,
                coords_z: np.ndarray,
                level_km: Optional[float] = None,
                figsize: Tuple = (14, 10),
                show: bool = False,
                show_window: bool = True,
                outfile: Optional[str] = None):
    """
    4-panel figure: column-max DBZ, column-max echo type (column dominant
    code), a single-level convectivity slice, and a single-level texture
    slice.

    Parameters
    ----------
    result : Result3D
        Output of eccopy3d.run().
    dbz : np.ndarray, shape (Z, Y, X)
        The reflectivity input that was passed to eccopy3d.run() (used
        here only for the column-max DBZ panel).
    coords_y, coords_x : np.ndarray, shape (Y,), (X,)
        Coordinate axes for plotting (not spacing — actual position
        values, e.g. the same 1-D arrays passed as coords_y/coords_x to
        eccopy3d.run() when those were 1-D).
    coords_z : np.ndarray, shape (Z,)
        Vertical coordinate values, km.
    level_km : float or None
        Altitude for the convectivity/texture slice panels; midpoint
        level used if None.
    figsize : tuple
    show : bool
    show_window : bool
        If True (default), overlay the texture window footprint
        (``result.texture_radius``) as a dashed circle at the domain
        centre of all four plan-view panels, so users can see the
        neighbourhood size relative to their features. Drawn only when the
        result carries a finite physical radius (a bare-pixel window on a
        unit-agnostic grid is skipped).
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
                                   convectivity_cmap, convectivity_norm,
                                   draw_window_ring)

    Y, X = np.meshgrid(coords_y, coords_x, indexing="ij")

    if level_km is None:
        level_km = coords_z[len(coords_z) // 2]
    iz = int(np.argmin(np.abs(coords_z - level_km)))

    col_max_dbz = np.nanmax(dbz, axis=0)            # (Y, X)

    # Column-dominant echo type: take the highest-priority code present
    # in each column (convective > mixed > stratiform), since echo_type
    # is shape (Z, Y, X).
    echo = result.echo_type
    col_echo = np.zeros(echo.shape[1:], dtype=echo.dtype)
    for code in sorted(np.unique(echo[echo > 0])):
        mask_col = np.any(echo == code, axis=0)
        col_echo[mask_col] = code

    present = set(np.unique(col_echo[col_echo > 0]).astype(int).tolist())
    is_basic = present.issubset({1, 2, 3})
    cmap = basic_echo_type_cmap() if is_basic else echo_type_cmap()
    norm = basic_echo_type_norm() if is_basic else echo_type_norm()
    labels = BASIC_ECHO_TYPE_LABELS if is_basic else ECHO_TYPE_LABELS
    n_codes = 3 if is_basic else 9

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle(f"EccoPy-3D — slice @ {coords_z[iz]:.1f} km", fontsize=13)

    ax = axes[0, 0]
    pc = ax.pcolormesh(X, Y, col_max_dbz, cmap="jet",
                       vmin=-10, vmax=65, shading="auto")
    plt.colorbar(pc, ax=ax, label="dBZ")
    ax.set(title="Column-max reflectivity", xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    pc = ax.pcolormesh(X, Y, remap_echo_type(col_echo),
                       cmap=cmap, norm=norm, shading="auto")
    cb = plt.colorbar(pc, ax=ax, ticks=range(1, n_codes + 1))
    cb.ax.set_yticklabels(labels, fontsize=7)
    ax.set(title="Echo type (column composite)", xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    ax = axes[1, 0]
    pc = ax.pcolormesh(X, Y, result.convectivity[iz], cmap=convectivity_cmap(),
                       norm=convectivity_norm(), shading="auto")
    plt.colorbar(pc, ax=ax)
    ax.set(title=f"Convectivity @ {coords_z[iz]:.1f} km",
           xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    ax = axes[1, 1]
    pc = ax.pcolormesh(X, Y, result.texture[iz], cmap="viridis",
                       vmin=0, shading="auto")
    plt.colorbar(pc, ax=ax, label="dBZ texture")
    ax.set(title=f"Texture @ {coords_z[iz]:.1f} km",
           xlabel="X (km)", ylabel="Y (km)")
    ax.set_aspect("equal")

    if show_window:
        radius_km = getattr(result, "texture_radius", None)
        for ax in axes.ravel():
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
