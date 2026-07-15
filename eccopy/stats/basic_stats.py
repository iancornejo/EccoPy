"""
Generic, data-agnostic statistics computed from EccoPy echo_type arrays.

Every function here operates on plain numpy arrays -- an `echo_type`
array (as returned by any of eccopy1d/eccopy2d_v/eccopy2d_h/eccopy3d's
`.echo_type`) and, where relevant, a co-located `height` or `spacing`
array of the SAME shape -- exactly like the rest of EccoPy. Nothing here
assumes a particular module produced the input; the same function works
on a (T,) eccopy1d result, a (Z, X) eccopy2d_v result, a (Y, X)
eccopy2d_h result, or a (Z, Y, X) eccopy3d result.

Basic vs sub-classified codes are auto-detected the same way
core.colormaps.remap_echo_type() does: if every non-NaN code present is
in {1, 2, 3}, the array is treated as basic; otherwise it is treated as
sub-classified (14/16/18/25/30/32/34/36/38). You can also pass
`codes=...` explicitly to any function to bypass auto-detection --
useful if you want, say, only "Convective Deep" (38) rather than every
convective sub-type.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Union

import numpy as np
from scipy.ndimage import label, generate_binary_structure

# Canonical code groupings. Mirrors eccopy.core.classification's
# CATEGORY_* constants (sub-classified) plus the basic 1/2/3 codes used
# by eccopy1d / eccopy2d_h / eccopy2d_v & eccopy3d without height+melt+temp.
BASIC_STRATIFORM_CODES    = frozenset({1})
BASIC_MIXED_CODES         = frozenset({2})
BASIC_CONVECTIVE_CODES    = frozenset({3})

SUB_STRATIFORM_CODES      = frozenset({14, 16, 18})
SUB_MIXED_CODES           = frozenset({25})
SUB_CONVECTIVE_CODES      = frozenset({30, 32, 34, 36, 38})

_CATEGORY_ALIASES = {"strat": "stratiform", "conv": "convective"}


def _is_basic(echo_type: np.ndarray) -> bool:
    """Auto-detect basic (1/2/3) vs sub-classified codes, like remap_echo_type()."""
    valid = echo_type[~np.isnan(echo_type)]
    if valid.size == 0:
        return True
    present = set(np.unique(valid).astype(int).tolist())
    return present.issubset({1, 2, 3})


def codes_for_category(echo_type: np.ndarray, category: str) -> frozenset:
    """
    Resolve a category name ('stratiform', 'mixed', 'convective' -- or
    the shorthands 'strat'/'conv') to the set of integer echo_type codes
    it corresponds to, auto-detecting whether `echo_type` holds basic or
    sub-classified codes.
    """
    category = _CATEGORY_ALIASES.get(category, category)
    is_basic = _is_basic(echo_type)
    table = {
        "stratiform": BASIC_STRATIFORM_CODES if is_basic else SUB_STRATIFORM_CODES,
        "mixed":      BASIC_MIXED_CODES      if is_basic else SUB_MIXED_CODES,
        "convective": BASIC_CONVECTIVE_CODES if is_basic else SUB_CONVECTIVE_CODES,
    }
    if category not in table:
        raise ValueError(
            f"category must be one of {sorted(table)} (or 'strat'/'conv'); got {category!r}"
        )
    return table[category]


def _mask_for(echo_type: np.ndarray, category: str,
             codes: Optional[Iterable[int]] = None) -> np.ndarray:
    use_codes = frozenset(codes) if codes is not None else codes_for_category(echo_type, category)
    return np.isin(echo_type, np.array(sorted(use_codes)))


# ---------------------------------------------------------------------------
# Coverage fractions
# ---------------------------------------------------------------------------

def echo_type_fractions(echo_type: np.ndarray) -> dict:
    """
    Fraction of valid (non-NaN) points in each category.

    Returns
    -------
    dict with keys 'stratiform', 'mixed', 'convective' (fractions, 0-1)
    and 'n_valid' (count of non-NaN points the fractions are computed
    over). Sub-classified arrays are collapsed to these three top-level
    categories -- use codes_for_category()/np.isin() directly if you
    want e.g. the Stratiform-Low fraction specifically.
    """
    echo_type = np.asarray(echo_type, dtype=float)
    valid = ~np.isnan(echo_type)
    n_valid = int(np.sum(valid))
    out = {"n_valid": n_valid}
    for cat in ("stratiform", "mixed", "convective"):
        mask = _mask_for(echo_type, cat)
        out[cat] = float(np.sum(mask)) / n_valid if n_valid > 0 else float("nan")
    return out


def convective_percentage(echo_type: np.ndarray,
                          codes: Optional[Iterable[int]] = None) -> float:
    """Percentage (0-100) of valid points classified Convective (any sub-type)."""
    echo_type = np.asarray(echo_type, dtype=float)
    n_valid = int(np.sum(~np.isnan(echo_type)))
    if n_valid == 0:
        return float("nan")
    mask = _mask_for(echo_type, "convective", codes)
    return 100.0 * float(np.sum(mask)) / n_valid


def stratiform_percentage(echo_type: np.ndarray,
                          codes: Optional[Iterable[int]] = None) -> float:
    """Percentage (0-100) of valid points classified Stratiform (any sub-type)."""
    echo_type = np.asarray(echo_type, dtype=float)
    n_valid = int(np.sum(~np.isnan(echo_type)))
    if n_valid == 0:
        return float("nan")
    mask = _mask_for(echo_type, "stratiform", codes)
    return 100.0 * float(np.sum(mask)) / n_valid


def mixed_percentage(echo_type: np.ndarray,
                     codes: Optional[Iterable[int]] = None) -> float:
    """Percentage (0-100) of valid points classified Mixed."""
    echo_type = np.asarray(echo_type, dtype=float)
    n_valid = int(np.sum(~np.isnan(echo_type)))
    if n_valid == 0:
        return float("nan")
    mask = _mask_for(echo_type, "mixed", codes)
    return 100.0 * float(np.sum(mask)) / n_valid


# ---------------------------------------------------------------------------
# Clumps (connected regions of a category within the echo_type array itself)
# ---------------------------------------------------------------------------

def _default_structure(ndim: int) -> np.ndarray:
    """
    Default connectivity used to label clumps, matching the conventions
    used elsewhere in EccoPy: 8-connectivity in 2-D (core.classification's
    _CONN8, matching MATLAB bwconncomp's default), 6-connectivity
    (face-adjacent only) in 3-D (eccopy3d.clumping's _STRUCT_6, matching
    LROSE ClumpingMgr), and simple adjacency in 1-D.
    """
    if ndim == 3:
        return generate_binary_structure(3, 1)   # face-connectivity (6-conn)
    if ndim == 2:
        return np.ones((3, 3), dtype=bool)         # 8-connectivity
    return generate_binary_structure(ndim, ndim)


def n_clumps(echo_type: np.ndarray, category: str = "convective",
            codes: Optional[Iterable[int]] = None,
            structure: Optional[np.ndarray] = None) -> int:
    """
    Number of spatially-connected clumps of `category` within the
    echo_type array.

    NOTE: this labels connectivity WITHIN the classified echo_type array
    itself, using a structuring element matching EccoPy's internal
    conventions (see _default_structure) -- it is independent of, and
    not guaranteed to numerically match, `Result3D.n_clumps` from
    eccopy3d.run(), which counts clumps found by the dual-threshold
    CONVECTIVITY-based clumping algorithm (a different, earlier stage of
    the pipeline that can split/merge differently once secondary
    thresholds and sub-clump area filters are applied -- see
    eccopy3d/clumping.py). Use eccopy3d's own `n_clumps` field if you
    need the exact count the classification algorithm itself used; use
    this function for a general-purpose clump count on any echo_type
    array, from any of the four modules.
    """
    echo_type = np.asarray(echo_type, dtype=float)
    mask = _mask_for(echo_type, category, codes)
    if not mask.any():
        return 0
    struct = structure if structure is not None else _default_structure(mask.ndim)
    _, n = label(mask, structure=struct)
    return int(n)


def clump_sizes(echo_type: np.ndarray, category: str = "convective",
                codes: Optional[Iterable[int]] = None,
                structure: Optional[np.ndarray] = None,
                spacing: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Size of each connected clump of `category`, largest first.

    Parameters
    ----------
    spacing : np.ndarray, same shape as echo_type, optional
        If given, each clump's size is its physical extent -- the sum of
        spacing**ndim over the clump's points (e.g. km^2 for a 2-D
        array with spacing in km, km^3 for 3-D) -- rather than a raw
        pixel count. Same idea as eccopy3d's own volume_km3 field, but
        computed generically here for any module's output.

    Returns
    -------
    np.ndarray, shape (n_clumps,), sorted descending.
    """
    echo_type = np.asarray(echo_type, dtype=float)
    mask = _mask_for(echo_type, category, codes)
    if not mask.any():
        return np.array([])
    struct = structure if structure is not None else _default_structure(mask.ndim)
    labeled, n = label(mask, structure=struct)
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        weights = spacing ** mask.ndim
    else:
        weights = np.ones_like(mask, dtype=float)
    sizes = np.array([float(np.sum(weights[labeled == cid])) for cid in range(1, n + 1)])
    return np.sort(sizes)[::-1]


# ---------------------------------------------------------------------------
# Height / depth (requires a co-located height field, e.g. from eccopy2d_v
# or eccopy3d -- meaningless for eccopy1d / eccopy2d_h, which have no
# vertical axis)
# ---------------------------------------------------------------------------

def echo_top_height(echo_type: np.ndarray, height: np.ndarray,
                    category: str = "convective",
                    codes: Optional[Iterable[int]] = None,
                    axis: int = 0) -> np.ndarray:
    """
    Highest `height` value at which `category` echo occurs, per column
    (i.e. collapsed along `axis`, the vertical axis -- 0 for both
    eccopy2d_v's (Z, X) and eccopy3d's (Z, Y, X) conventions).

    Returns
    -------
    np.ndarray, shape = echo_type.shape with `axis` removed. NaN in
    columns with no `category` echo at all.
    """
    echo_type = np.asarray(echo_type, dtype=float)
    height = np.asarray(height, dtype=float)
    mask = _mask_for(echo_type, category, codes)
    h = np.where(mask, height, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        out = np.nanmax(h, axis=axis)
    return out


def echo_base_height(echo_type: np.ndarray, height: np.ndarray,
                     category: str = "convective",
                     codes: Optional[Iterable[int]] = None,
                     axis: int = 0) -> np.ndarray:
    """Lowest `height` value at which `category` echo occurs, per column. See echo_top_height."""
    echo_type = np.asarray(echo_type, dtype=float)
    height = np.asarray(height, dtype=float)
    mask = _mask_for(echo_type, category, codes)
    h = np.where(mask, height, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        out = np.nanmin(h, axis=axis)
    return out


def echo_depth(echo_type: np.ndarray, height: np.ndarray,
               category: str = "convective",
               codes: Optional[Iterable[int]] = None,
               axis: int = 0) -> np.ndarray:
    """
    Vertical extent (top - base) of `category` echo, per column --
    echo_top_height() minus echo_base_height(). NaN in columns with no
    `category` echo.
    """
    top = echo_top_height(echo_type, height, category, codes, axis)
    base = echo_base_height(echo_type, height, category, codes, axis)
    return top - base


def convective_top_height(echo_type: np.ndarray, height: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shorthand for echo_top_height(..., category='convective')."""
    return echo_top_height(echo_type, height, category="convective", axis=axis)


def convective_base_height(echo_type: np.ndarray, height: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shorthand for echo_base_height(..., category='convective')."""
    return echo_base_height(echo_type, height, category="convective", axis=axis)


def convective_depth(echo_type: np.ndarray, height: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shorthand for echo_depth(..., category='convective')."""
    return echo_depth(echo_type, height, category="convective", axis=axis)


def stratiform_depth(echo_type: np.ndarray, height: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shorthand for echo_depth(..., category='stratiform')."""
    return echo_depth(echo_type, height, category="stratiform", axis=axis)


# ---------------------------------------------------------------------------
# One-call convenience summary
# ---------------------------------------------------------------------------

def summarize(echo_type: np.ndarray, height: Optional[np.ndarray] = None,
             spacing: Optional[np.ndarray] = None, axis: int = 0) -> dict:
    """
    Compute a standard bundle of statistics in one call: coverage
    fractions, convective/stratiform/mixed percentages, clump counts,
    and (if `height` is given) mean convective top/base/depth.

    Parameters
    ----------
    echo_type : np.ndarray
        Any shape -- (N,) from eccopy1d, (Z, X) from eccopy2d_v,
        (Y, X) from eccopy2d_h, or (Z, Y, X) from eccopy3d.
    height : np.ndarray, same shape as echo_type, optional
        Enables the height/depth statistics. Omit for eccopy1d /
        eccopy2d_h results, which have no vertical axis.
    spacing : np.ndarray, same shape as echo_type, optional
        Enables physical (not just pixel-count) clump sizes -- see
        clump_sizes().
    axis : int
        Vertical axis for height statistics (default 0, matching
        eccopy2d_v and eccopy3d's (Z, ...) convention).

    Returns
    -------
    dict
    """
    echo_type = np.asarray(echo_type, dtype=float)
    out = echo_type_fractions(echo_type)
    out["convective_pct"] = convective_percentage(echo_type)
    out["stratiform_pct"] = stratiform_percentage(echo_type)
    out["mixed_pct"] = mixed_percentage(echo_type)
    out["n_convective_clumps"] = n_clumps(echo_type, category="convective")
    out["n_stratiform_clumps"] = n_clumps(echo_type, category="stratiform")
    conv_sizes = clump_sizes(echo_type, category="convective", spacing=spacing)
    out["convective_clump_sizes"] = conv_sizes
    out["mean_convective_clump_size"] = (
        float(np.mean(conv_sizes)) if conv_sizes.size else float("nan")
    )

    if height is not None:
        height = np.asarray(height, dtype=float)
        top = convective_top_height(echo_type, height, axis=axis)
        base = convective_base_height(echo_type, height, axis=axis)
        depth = top - base
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN")
            out["mean_convective_top_height"] = float(np.nanmean(top))
            out["mean_convective_base_height"] = float(np.nanmean(base))
            out["mean_convective_depth"] = float(np.nanmean(depth))
            out["max_convective_top_height"] = float(np.nanmax(top)) if np.any(~np.isnan(top)) else float("nan")

    return out
