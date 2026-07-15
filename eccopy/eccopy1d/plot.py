"""Profile plotting for EccoPy-1D results."""

from __future__ import annotations
from typing import Optional
import numpy as np


def plot_result(result,
                dbz: np.ndarray,
                coords: np.ndarray,
                figsize=(11, 6),
                show: bool = False,
                outfile: Optional[str] = None):
    """
    3-panel figure: reflectivity, convectivity, echo type — for a single
    EccoPy-1D profile.

    Parameters
    ----------
    result : Result1D
        Output of eccopy1d.run().
    dbz : np.ndarray, shape (N,)
        The reflectivity input that was passed to eccopy1d.run().
    coords : np.ndarray, shape (N,)
        Position values along the profile (km or whatever unit `coords`
        was given in when calling run() with coord_mode="position").
    figsize : tuple
    show : bool
    outfile : str or None

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from ..core.colormaps import (basic_echo_type_cmap, basic_echo_type_norm,
                                   remap_echo_type, BASIC_ECHO_TYPE_LABELS,
                                   convectivity_cmap, convectivity_norm)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             constrained_layout=True)

    axes[0].plot(coords, dbz, color="k", lw=1)
    axes[0].set(ylabel="dBZ", title="Reflectivity")

    pc = axes[1].scatter(coords, np.zeros_like(coords), c=result.convectivity,
                         cmap=convectivity_cmap(), norm=convectivity_norm(), s=25)
    axes[1].set(title="Convectivity", yticks=[])
    plt.colorbar(pc, ax=axes[1], orientation="horizontal", pad=0.4, fraction=0.4)

    pc2 = axes[2].scatter(coords, np.zeros_like(coords),
                          c=remap_echo_type(result.echo_type),
                          cmap=basic_echo_type_cmap(), norm=basic_echo_type_norm(), s=25)
    cb = plt.colorbar(pc2, ax=axes[2], orientation="horizontal", pad=0.4, fraction=0.4,
                      ticks=range(1, 4))
    cb.ax.set_xticklabels(BASIC_ECHO_TYPE_LABELS)
    axes[2].set(title="Echo Type", xlabel="Position", yticks=[])

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
