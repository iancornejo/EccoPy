"""Colormaps and echo-type plotting utilities."""

from __future__ import annotations
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize

# Sub-classified echo-type code → 1-based plot index (9-colour map)
ECHO_TYPE_TO_IDX = {14: 1, 16: 2, 18: 3, 25: 4,
                    30: 5, 32: 6, 34: 7, 36: 8, 38: 9}

ECHO_TYPE_LABELS = [
    "Strat Low", "Strat Mid", "Strat High",
    "Mixed",
    "Conv", "Conv Elev", "Conv Shallow", "Conv Mid", "Conv Deep",
]

# Basic-only echo-type code (1=Strat, 2=Mixed, 3=Conv) → 1-based plot index
# (3-colour map), used by eccopy1d, eccopy2d_h, and eccopy2d_v/eccopy3d when
# no height/temp field is supplied.
BASIC_ECHO_TYPE_TO_IDX = {1: 1, 2: 2, 3: 3}
BASIC_ECHO_TYPE_LABELS = ["Stratiform", "Mixed", "Convective"]

_BASIC_ECHO_TYPE_COLORS = np.array([
    [0.38, 0.42, 0.96],  # 1  Stratiform
    [0.32, 0.78, 0.59],  # 2  Mixed
    [1.00, 0.00, 0.00],  # 3  Convective
])

_ECHO_TYPE_COLORS = np.array([
    [0,    0.10, 0.60],  # 1  Strat Low
    [0.38, 0.42, 0.96],  # 2  Strat Mid
    [0.65, 0.74, 0.86],  # 3  Strat High
    [0.32, 0.78, 0.59],  # 4  Mixed
    [1.00, 0.00, 0.00],  # 5  Conv
    [1.00, 0.00, 1.00],  # 6  Conv Elev
    [1.00, 1.00, 0.00],  # 7  Conv Shallow
    [0.99, 0.77, 0.22],  # 8  Conv Mid
    [0.70, 0.00, 0.00],  # 9  Conv Deep
])

_VEL_COLORS = np.array([
    [0,0,0.5312],[0,0,0.5625],[0,0,0.5938],[0,0,0.625],[0,0,0.6562],
    [0,0,0.6875],[0,0,0.7188],[0,0,0.75],[0,0,0.7812],[0,0,0.8125],
    [0,0,0.8438],[0,0,0.875],[0,0,0.9062],[0,0,0.9375],[0,0,0.9688],
    [0,0,1],[0,0.0312,1],[0,0.0625,1],[0,0.0938,1],[0,0.125,1],
    [0,0.1562,1],[0,0.1875,1],[0,0.2188,1],[0,0.25,1],[0,0.2812,1],
    [0,0.3125,1],[0,0.3438,1],[0,0.375,1],[0,0.4062,1],[0,0.4375,1],
    [0,0.4688,1],[0,0.5,1],[0,0.5312,1],[0,0.5625,1],[0,0.5938,1],
    [0,0.625,1],[0,0.6562,1],[0,0.6875,1],[0,0.7188,1],[0,0.75,1],
    [0,0.7812,1],[0,0.8125,1],[0,0.8438,1],[0,0.875,1],[0,0.9062,1],
    [0,0.9375,1],[0,0.9688,1],[0,1,1],[0.5,1,1],[0.8039,0.8039,0.8039],
    [1,1,0.5],[1,1,0],[1,0.9688,0],[1,0.9375,0],[1,0.9062,0],
    [1,0.875,0],[1,0.8438,0],[1,0.8125,0],[1,0.7812,0],[1,0.75,0],
    [1,0.7188,0],[1,0.6875,0],[1,0.6562,0],[1,0.625,0],[1,0.5938,0],
    [1,0.5625,0],[1,0.5312,0],[1,0.5,0],[1,0.4688,0],[1,0.4375,0],
    [1,0.4062,0],[1,0.375,0],[1,0.3438,0],[1,0.3125,0],[1,0.2812,0],
    [1,0.25,0],[1,0.2188,0],[1,0.1875,0],[1,0.1562,0],[1,0.125,0],
    [1,0.0938,0],[1,0.0625,0],[1,0.0312,0],[1,0,0],[0.9688,0,0],
    [0.9375,0,0],[0.9062,0,0],[0.875,0,0],[0.8438,0,0],[0.8125,0,0],
    [0.7812,0,0],[0.75,0,0],[0.7188,0,0],[0.6875,0,0],[0.6562,0,0],
    [0.625,0,0],[0.5938,0,0],[0.5625,0,0],[0.5312,0,0],
])


def echo_type_cmap() -> ListedColormap:
    """
    9-colour ListedColormap for sub-classified echo-type plotting.
    Use with vmin=0.5, vmax=9.5.
    """
    return ListedColormap(_ECHO_TYPE_COLORS, name="echo_type")


def echo_type_norm() -> BoundaryNorm:
    """BoundaryNorm that places each echo-type colour in the correct cell."""
    boundaries = np.arange(0.5, 10.5)
    return BoundaryNorm(boundaries, ncolors=9)


def basic_echo_type_cmap() -> ListedColormap:
    """3-colour ListedColormap for basic (Strat/Mixed/Conv) echo-type plotting."""
    return ListedColormap(_BASIC_ECHO_TYPE_COLORS, name="basic_echo_type")


def basic_echo_type_norm() -> BoundaryNorm:
    """BoundaryNorm for the 3-colour basic echo-type map."""
    boundaries = np.arange(0.5, 4.5)
    return BoundaryNorm(boundaries, ncolors=3)


def vel_cmap() -> ListedColormap:
    """HCR velocity colormap. Port of MATLAB velCols.m."""
    return ListedColormap(_VEL_COLORS, name="vel_cols")


def remap_echo_type(echo_type: np.ndarray) -> np.ndarray:
    """
    Remap echo-type integer codes to 1-based plot indices.

    Auto-detects whether `echo_type` contains basic codes (1, 2, 3 — no
    sub-classification, from eccopy1d / eccopy2d_h / eccopy2d_v|eccopy3d
    without height or temp) or sub-classified codes (14, 16, 18, 25, 30,
    32, 34, 36, 38), and remaps using the matching table. Use
    basic_echo_type_cmap()/basic_echo_type_norm() for the former, and
    echo_type_cmap()/echo_type_norm() for the latter.
    """
    present = set(np.unique(echo_type[~np.isnan(echo_type)]).astype(int).tolist())
    is_basic = present.issubset(set(BASIC_ECHO_TYPE_TO_IDX.keys()))

    table = BASIC_ECHO_TYPE_TO_IDX if is_basic else ECHO_TYPE_TO_IDX
    out = np.full_like(echo_type, np.nan, dtype=float)
    for code, idx in table.items():
        out[echo_type == code] = idx
    return out


# ---------------------------------------------------------------------------
# Convectivity colormap — continuous, 0-1
# ---------------------------------------------------------------------------
#
# Shares hue language with the classification colormaps above (blue for
# stratiform, teal for mixed, red for convective), but unlike those this
# map is continuous within each class and has HARD BREAKS at the two
# convectivity thresholds. Reading a convectivity panel therefore tells
# you the class directly -- blue pixels are stratiform, teal are mixed,
# red are convective -- while the ramp within each band still shows how
# marginal or how firmly-held that classification is.
#
# Blue band   : dark blue  -> light blue   (stratiform)
# Teal band   : dark teal  -> light teal   (mixed)
# Red band    : light red  -> dark red     (convective)
_CONVECTIVITY_BLUE = [(0.0, 0.0, 1.0), (0.5, 0.75, 1.0)]   # dark  -> light blue
_CONVECTIVITY_TEAL = [(0.0, 0.5, 0.5), (0.0, 1.0, 1.0)]    # dark  -> light teal
_CONVECTIVITY_RED  = [(1.0, 0.75, 0.75), (1.0, 0.0, 0.0)]  # light -> dark  red

_CONVECTIVITY_N = 1024


def _convectivity_stops(strat_mixed: float, mixed_conv: float, n: int):
    """
    Build the (position, color) stop list for convectivity_cmap().

    The duplicated positions are what create the hard breaks. Each break is
    nudged half a LUT bin BELOW its threshold, because the classifier is
    inclusive on the upper side (`conv >= strat_mixed` is Mixed, see
    core.classification.class_basic) whereas LinearSegmentedColormap resolves
    a duplicated stop to the lower band. Without the nudge, a convectivity of
    exactly `strat_mixed` renders stratiform-blue while classifying as mixed.
    """
    eps = 0.5 / n
    lo = strat_mixed - eps
    hi = mixed_conv - eps
    return [
        (0.0, _CONVECTIVITY_BLUE[0]),
        (lo,  _CONVECTIVITY_BLUE[1]),
        (lo,  _CONVECTIVITY_TEAL[0]),
        (hi,  _CONVECTIVITY_TEAL[1]),
        (hi,  _CONVECTIVITY_RED[0]),
        (1.0, _CONVECTIVITY_RED[1]),
    ]


def convectivity_cmap(strat_mixed: float = 0.4,
                      mixed_conv: float = 0.5) -> LinearSegmentedColormap:
    """
    Continuous colormap for convectivity (0-1), with hard breaks at the
    strat/mixed and mixed/conv thresholds. Use with convectivity_norm()
    (or vmin=0, vmax=1 directly).

    Parameters
    ----------
    strat_mixed, mixed_conv : float
        Convectivity thresholds the breaks are placed at. Defaults match
        ClassificationParams. If you run a classification with non-default
        thresholds, pass the same values here so the colors keep agreeing
        with the echo-type panel, e.g.::

            cmap = convectivity_cmap(params.strat_mixed, params.mixed_conv)
    """
    stops = _convectivity_stops(strat_mixed, mixed_conv, _CONVECTIVITY_N)
    return LinearSegmentedColormap.from_list("convectivity", stops,
                                             N=_CONVECTIVITY_N)


def convectivity_norm(vmin: float = 0.0, vmax: float = 1.0) -> Normalize:
    """Normalize for convectivity_cmap(). Convectivity is defined on [0, 1]."""
    return Normalize(vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Texture-window footprint overlay
# ---------------------------------------------------------------------------


def draw_window_ring(ax, coords_x, coords_y, radius_km,
                     center=None, label=True, color="black", alpha=0.9):
    """
    Draw the texture window's footprint as a circle in data coordinates,
    so users can see how large the texture neighbourhood is relative to
    the features in a plan-view (2D-H or 3D-slice) panel.

    The window is a circular neighbourhood of physical radius `radius_km`
    (``result.texture_radius``). On an isotropic plan-view grid drawn with
    equal aspect, that footprint is a true circle in km space, which is
    what this draws. On a strongly anisotropic grid (dx != dy) the actual
    per-axis pixel reach differs; the km circle is still the correct
    physical footprint, but bear in mind the pixel counts along X and Y
    won't match.

    Parameters
    ----------
    ax : matplotlib Axes
        A plan-view axis already drawn with equal aspect and in km.
    coords_x, coords_y : np.ndarray
        The 1-D coordinate axes (km) of the panel, used to place the ring
        at the domain centre by default.
    radius_km : float or None
        Physical window radius in km (``result.texture_radius``). If None,
        non-finite, or <= 0, nothing is drawn and None is returned (so a
        bare-pixel WindowSpec on a unit-agnostic grid degrades quietly).
    center : (x, y) tuple or None
        Ring centre in km. Domain centre if None.
    label : bool
        Annotate the ring with its radius.
    color, alpha : matplotlib color / float
        Ring styling.

    Returns
    -------
    matplotlib.patches.Circle or None
        The patch added to `ax`, or None if nothing was drawn.
    """
    import numpy as _np
    from matplotlib.patches import Circle as _Circle

    if radius_km is None:
        return None
    try:
        radius_km = float(radius_km)
    except (TypeError, ValueError):
        return None
    if not _np.isfinite(radius_km) or radius_km <= 0:
        return None

    coords_x = _np.asarray(coords_x, dtype=float)
    coords_y = _np.asarray(coords_y, dtype=float)
    if center is None:
        cx = float(_np.mean([coords_x[0], coords_x[-1]]))
        cy = float(_np.mean([coords_y[0], coords_y[-1]]))
    else:
        cx, cy = float(center[0]), float(center[1])

    ring = _Circle((cx, cy), radius_km, fill=False, ls="--", lw=1.6,
                   ec=color, alpha=alpha, zorder=5)
    ax.add_patch(ring)
    ax.plot([cx], [cy], marker="+", ms=8, mew=1.4, color=color,
            alpha=alpha, zorder=6)
    if label:
        ax.annotate(f"texture window\nr = {radius_km:.1f} km",
                    xy=(cx, cy + radius_km), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.75, lw=0.8),
                    zorder=7)
    return ring
