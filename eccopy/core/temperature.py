"""
Temperature field utilities.

isotherm_height()  — port of Ecco::_computeHts from Ecco.cc
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _isotherm_height_core(temp_3d: np.ndarray,
                          z_levels: np.ndarray,
                          target_temp_c: float) -> np.ndarray:
    """
    JIT-compiled core loop for isotherm_height(). Pure numeric logic
    only — see isotherm_height()'s docstring for the full explanation
    of what this computes and why it differs from the Ecco.cc reference.
    This function is intentionally a near-literal translation of the
    pure-Python version once was: same variable names, same control
    flow, just with type-stable operations Numba can compile.
    """
    nz, ny, nx = temp_3d.shape
    ht_grid = np.full((ny, nx), z_levels[0])

    for iy in range(ny):
        for ix in range(nx):
            # Find valid (non-NaN) level indices for this column.
            # (A manual scan, rather than np.where(...)[0], since this
            # is both what Numba compiles most efficiently and keeps
            # the logic explicit rather than relying on fancy indexing.)
            n_valid = 0
            for iz in range(nz):
                if not np.isnan(temp_3d[iz, iy, ix]):
                    n_valid += 1

            if n_valid == 0:
                ht_grid[iy, ix] = np.nan
                continue

            valid_iz = np.empty(n_valid, dtype=np.int64)
            k = 0
            for iz in range(nz):
                if not np.isnan(temp_3d[iz, iy, ix]):
                    valid_iz[k] = iz
                    k += 1

            bottom_temp = temp_3d[valid_iz[0], iy, ix]
            bottom_ht = z_levels[valid_iz[0]]
            top_temp = temp_3d[valid_iz[-1], iy, ix]
            top_ht = z_levels[valid_iz[-1]]

            ht_found = False

            # Walk consecutive VALID levels (skipping over any NaN gaps
            # between them), rather than only immediately adjacent indices.
            for j in range(1, n_valid):
                iz_below = valid_iz[j - 1]
                iz_above = valid_iz[j]
                t_below = temp_3d[iz_below, iy, ix]
                t_above = temp_3d[iz_above, iy, ix]

                crosses = ((t_below >= target_temp_c >= t_above) or
                           (t_below <= target_temp_c <= t_above))
                if crosses:
                    delta_t = t_above - t_below
                    delta_h = z_levels[iz_above] - z_levels[iz_below]
                    # Anchored at z_levels[iz_below] (the lower level) —
                    # see isotherm_height()'s docstring for why this
                    # differs from the Ecco.cc reference formula.
                    if delta_t == 0:
                        interp_ht = z_levels[iz_below]
                    else:
                        interp_ht = z_levels[iz_below] + (
                            (target_temp_c - t_below) / delta_t
                        ) * delta_h
                    ht_grid[iy, ix] = interp_ht
                    ht_found = True
                    break

            if not ht_found:
                if target_temp_c >= bottom_temp:
                    ht_grid[iy, ix] = bottom_ht
                elif target_temp_c <= top_temp:
                    ht_grid[iy, ix] = top_ht

    return ht_grid


def isotherm_height(temp_3d: np.ndarray,
                    z_levels: np.ndarray,
                    target_temp_c: float) -> np.ndarray:
    """
    For every (y, x) column, find the altitude [km] where temperature
    equals target_temp_c by linear interpolation between the nearest
    valid levels straddling the crossing.

    Direct port of Ecco::_computeHts() from Ecco.cc (lrose-core), with
    two corrections beyond the original:

    1. NaN gaps are skipped over when searching for a crossing, rather
       than only ever comparing immediately adjacent levels. For example,
       given column values [..., 1, NaN, -1, ...] searching for the 0°C
       isotherm, a human can immediately tell the crossing must be very
       close to the NaN level (we go from +1°C to -1°C across it) — but
       comparing only adjacent pairs never sees this, because the pair
       (1, NaN) and the pair (NaN, -1) are each skipped individually, and
       the search falls through to the bottom/top fallback instead of
       interpolating across the gap. This function instead tracks the
       last VALID level seen and compares it against the next VALID
       level found, using their true height difference (which may span
       more than one level if there are multiple consecutive NaNs) for
       the interpolation.

    2. The reference Ecco.cc interpolation line appears to anchor at the
       wrong end of the bracketing pair:

           interpHt = zProfile[iz] + ((tempC - tempBelow) / deltaTemp) * deltaHt;

       This anchors at zProfile[iz] — the UPPER level of the pair — but
       the standard linear-interpolation formula anchors at the LOWER
       level (zProfile[iz-1]). Verified independently three ways
       (symbolic algebra, first-principles re-derivation, and a numeric
       check against a known analytic profile): the C++ formula as
       written is off by exactly one level-spacing from the
       mathematically correct answer. This function uses the correct
       (lower-anchored) formula instead. This has NOT been confirmed
       against the live lrose-core GitHub master (only the snapshot
       provided), so it's possible this was already fixed upstream —
       worth rechecking against current lrose-core source before
       assuming either version is authoritative. If you ever need to
       reproduce Ecco.cc's behaviour bit-for-bit instead of the
       mathematically correct one, change `z_levels[iz_below]` to
       `z_levels[iz_above]` in the interpolation line below.

    Both corrections only matter when there's a real discrepancy to
    correct — on a clean, gapless input where the crossing happens to
    land close to a level (small deltaHt), the difference from #2 is
    small; it grows with level spacing.

    Handles:
      - Normal lapse-rate profiles and temperature inversions (both crossing
        directions).
      - Target temp below lowest model level → returns bottom height.
      - Target temp above highest model level → returns top height.
      - Missing (NaN) temperature values — skipped over when searching for
        a crossing (see above), not just skipped individually.

    Parameters
    ----------
    temp_3d : np.ndarray, shape (Z, Y, X)
        Temperature in °C. NaN for missing values. Matches the shape
        convention used throughout EccoPy (see eccopy3d.run()).
    z_levels : array-like, length Z
        Altitude of each level [km MSL], increasing upward.
    target_temp_c : float
        Target isotherm temperature [°C].

    Returns
    -------
    ht_grid : np.ndarray, shape (Y, X)
        Height of the isotherm [km MSL] at each (y, x) point.
        Falls back to lowest or highest valid level height when out of
        range, or when a column has no valid data at all (returns NaN
        in that case).
    """
    temp_3d = np.asarray(temp_3d, dtype=float)
    z_levels = np.asarray(z_levels, dtype=float)

    return _isotherm_height_core(temp_3d, z_levels, float(target_temp_c))


def melt_layer_from_temp(temp_2d: np.ndarray) -> np.ndarray:
    """
    Derive a simple melting-layer indicator from a 2-D temperature field.

    Mirrors the MATLAB run script logic:
        MELTING_LAYER = 20 where T <= 0°C (above/at freezing)
        MELTING_LAYER = 10 where T >  0°C (below freezing)

    Parameters
    ----------
    temp_2d : np.ndarray, shape (nHeight, nTime) or (ny, nx)
        Temperature [°C].

    Returns
    -------
    melt : np.ndarray, same shape
        Melting-layer indicator.
    """
    melt = np.full_like(temp_2d, np.nan, dtype=float)
    melt[temp_2d <= 0] = 20.0
    melt[temp_2d > 0] = 10.0
    melt[np.isnan(temp_2d)] = np.nan
    return melt


def broadcast_temp_field(temp: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Accept a temperature field given either as a full field matching the
    target shape, or as a single vertical profile (a sounding) to be
    mirrored identically across every horizontal point.

    A sounding gives temperature as a function of height only — one
    value per Z level, with no horizontal variation. This lets you pass
    a single representative profile (e.g. from a model sounding or
    reanalysis column) and have it apply uniformly across the whole
    horizontal domain, instead of having to manually broadcast it
    yourself.

    Parameters
    ----------
    temp : np.ndarray
        Either:
          - shape == target_shape  → returned unchanged (already a full field)
          - shape (Z,)              → a 1-D sounding, broadcast across all
                                       horizontal points to match target_shape
    target_shape : tuple
        The full field shape this temperature array should match —
        (Z, X) for EccoPy-2D-V, (Z, Y, X) for EccoPy-3D.

    Returns
    -------
    temp_full : np.ndarray, shape == target_shape

    Raises
    ------
    ValueError
        If `temp`'s shape is neither the target shape nor a 1-D profile
        of length target_shape[0] (the Z axis).
    """
    temp = np.asarray(temp, dtype=float)

    if temp.shape == target_shape:
        return temp

    nz = target_shape[0]
    if temp.ndim == 1 and temp.shape[0] == nz:
        # Sounding: one value per Z level, no horizontal variation.
        # Reshape to (Z, 1, 1, ...) so it broadcasts across every
        # remaining (horizontal) axis of target_shape.
        new_shape = (nz,) + (1,) * (len(target_shape) - 1)
        return np.broadcast_to(temp.reshape(new_shape), target_shape).copy()

    raise ValueError(
        f"temp has shape {temp.shape}, which matches neither the full "
        f"field shape {target_shape} nor a 1-D sounding of length "
        f"{nz} (target_shape[0], the Z axis)."
    )
