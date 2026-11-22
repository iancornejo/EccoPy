"""
Reflectivity texture calculations.

Two families of functions:

1-D sliding-window (EccoPy-1D / EccoPy-2D)
    refl_texture_1d() - port of f_reflTexture.m, generalised to accept a
    per-point pixel-radius array so the window size can vary along the
    array (e.g. when resolved from a physical-unit WindowSpec against a
    non-uniform spacing array).

2-D radial with planar detrend (EccoPy-3D / EccoPy-2D-H)
    refl_texture_2d() - port of ConvStratFinder::ComputeTexture::run(),
    generalised to accept per-point dy/dx spacing arrays instead of
    assuming a uniform grid. Since EccoPy 1.1 it resolves each distinct
    window once and reuses it ("Prepared 2-D windows" below); the 1.0
    cores it replaced are kept as the reference it is tested against.

Both carry the math of the LROSE-derived implementations unchanged; the
window/spacing resolution accepts per-point rather than globally uniform
spacing.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numba import njit

from ..params.window import WindowSpec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fillmissing_linear(arr: np.ndarray, axis: int) -> np.ndarray:
    """
    Fill NaN values by linear interpolation + nearest-edge extrapolation
    along the given axis. Mirrors MATLAB fillmissing(x,'linear','EndValues','nearest').
    """
    result = np.moveaxis(arr.copy(), axis, -1)
    shape = result.shape
    result = result.reshape(-1, shape[-1])
    x = np.arange(shape[-1], dtype=float)
    for i in range(result.shape[0]):
        row = result[i]
        nans = np.isnan(row)
        if nans.all() or not nans.any():
            continue
        good = ~nans
        result[i] = np.interp(x, x[good], row[good])
    result = result.reshape(shape)
    return np.moveaxis(result, -1, axis)


def _radius_field_along_axis(window: Union[WindowSpec, int],
                             spacing: Optional[np.ndarray],
                             n: int) -> np.ndarray:
    """
    Resolve a window into a 1-D array of per-point integer pixel radii,
    of length n, given a 1-D spacing array along the texture axis.
    """
    if isinstance(window, int):
        return np.full(n, window, dtype=int)
    if isinstance(window, WindowSpec) and window.is_pixel:
        # Bare-pixel WindowSpec never needs a spacing array, same as a
        # raw int radius - resolve it directly without requiring spacing.
        return window.pixel_radius_field(np.ones(n))
    if spacing is None:
        raise ValueError(
            "A physical-unit WindowSpec requires a spacing array; "
            "got spacing=None. Pass an integer pixel radius instead, "
            "or provide a spacing array."
        )
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape[-1] != n:
        raise ValueError(
            f"spacing array length ({spacing.shape[-1]}) along the texture "
            f"axis must match the data length ({n})."
        )
    return window.pixel_radius_field(spacing)


@njit(cache=True)
def _sliding_texture_core(data_padded: np.ndarray,
                           radius_field: np.ndarray,
                           pad: int,
                           nRows: int,
                           nCols: int) -> np.ndarray:
    """
    Core 1-D detrend + texture loop with a PER-POINT window radius.

    Replicates MATLAB f_reflTexture.m exactly at each point:
      - 1-D linear detrend per window
      - texture = sqrt(sample_std(detrended^2)), using N-1 normalisation
        to match MATLAB's std() default
      - missing values are dropped (nanstd), not substituted

    Parameters
    ----------
    data_padded : (nRows, nCols + 2*pad) array, pre-filled (no NaN gaps)
    radius_field : (nCols,) int array - per-point window half-width
    pad : int - maximum radius used when padding (>= max(radius_field))

    Implementation note
    --------------------
    This loops explicitly over (column, row, window-position) rather
    than using NumPy's row-vectorized axis=1 reductions (np.nansum,
    np.nanmean, np.nanstd, np.tile), because none of those are supported
    in Numba's nopython mode. The math is identical - every sum below is
    just the scalar equivalent of the array reduction it replaces - but
    expressed as accumulation loops so this function can be JIT-compiled.
    NaN values are skipped while accumulating sums (mirroring nansum's
    drop-rather-than-substitute behaviour), with a running valid-count
    used as the divisor in place of the window width.
    """
    texture = np.full((nRows, nCols), np.nan)

    for ii in range(nCols):
        r = int(radius_field[ii])
        if r < 1:
            continue
        lo = pad + ii - r
        hi = pad + ii + r + 1
        W = hi - lo
        if W < 2:
            continue

        for row in range(nRows):
            # --- accumulate sums for the linear fit, skipping NaNs ---
            # (x-coordinate is 1..W along the window, same as the
            # original np.arange(1, W+1) used for X)
            sumX = 0.0
            sumX2 = 0.0
            sumY = 0.0
            sumXY = 0.0
            n_valid = 0
            for jj in range(W):
                x = jj + 1.0   # 1-indexed, matching np.arange(1, W+1)
                sumX += x
                sumX2 += x * x
                y = data_padded[row, lo + jj]
                if not np.isnan(y):
                    sumY += y
                    sumXY += x * y
                    n_valid += 1

            denom = W * sumX2 - sumX * sumX
            if denom != 0.0:
                a = (sumY * sumX2 - sumX * sumXY) / denom
                b = (W * sumXY - sumX * sumY) / denom
            else:
                a = 0.0
                b = 0.0

            mean_block = sumY / n_valid if n_valid > 0 else np.nan

            # --- detrend, clip, and accumulate texture statistics ---
            # (sample variance of corrected^2, N-1 normalization, NaNs
            # dropped -- same as np.nanstd(corrected**2, ddof=1) on the
            # original block; see fix note below)
            sum_sq = 0.0
            sum_sq2 = 0.0
            n_corrected = 0
            for jj in range(W):
                y = data_padded[row, lo + jj]
                if np.isnan(y):
                    continue
                x = jj + 1.0
                new_y = a + b * x
                corrected = y - new_y + mean_block
                if corrected < 1.0:
                    corrected = 1.0
                csq = corrected * corrected
                sum_sq += csq
                sum_sq2 += csq * csq
                n_corrected += 1

            if n_corrected > 0:
                mean_sq = sum_sq / n_corrected
                # Sample variance (N-1 denominator), matching MATLAB's
                # std(dbzCorr.^2, [], 2, 'omitnan') -- the `[]` selects
                # MATLAB's DEFAULT normalization, which is N-1 (sample),
                # NOT N (population). The previous version of this block
                # computed population variance (N denominator), which is
                # NOT what MATLAB's std() actually does by default despite
                # the comment above claiming otherwise. Verified against
                # real MATLAB CONVECTIVITY output (SPOL case): N-1 gives
                # ~1e-17 mean absolute error (machine precision), N gave
                # ~1.5e-3 -- small in raw convectivity terms, but enough
                # to shift several threshold crossings and move downstream
                # classification agreement from 98.2% to 99.4%.
                if n_corrected > 1:
                    sum_sq_dev = sum_sq2 - n_corrected * mean_sq * mean_sq
                    var = sum_sq_dev / (n_corrected - 1)
                    if var < 0.0:
                        var = 0.0
                else:
                    var = 0.0   # matches MATLAB's std() of a single sample == 0
                # texture = sqrt(sample_std(corrected**2)), and
                # std = sqrt(var) -- so this is sqrt(sqrt(var)), a
                # FOURTH root, not sqrt(var). (Caught by exhaustive
                # cross-validation against the pre-Numba reference -
                # an earlier draft of this line wrote np.sqrt(var) and
                # silently produced the SQUARE of the correct answer
                # for most inputs; entry-point tests didn't catch it
                # because they only check that plausible-looking output
                # comes out, not exact values.)
                texture[row, ii] = np.sqrt(np.sqrt(var))
            # else: leave as NaN (matches np.nanstd on an all-NaN slice)

    return texture


# ---------------------------------------------------------------------------
# Public 1-D sliding-window texture (used by EccoPy-1D and EccoPy-2D)
# ---------------------------------------------------------------------------

def refl_texture_1d(dbz: np.ndarray,
                    window: Union[WindowSpec, int],
                    spacing: Optional[np.ndarray] = None,
                    dbz_base: float = 0.0,
                    min_valid_dbz: float = -np.inf,
                    kernel_mode: str = "uniform") -> np.ndarray:
    """
    Sliding 1-D reflectivity texture along the last axis.

    Port of MATLAB f_reflTexture.m, generalised to a per-point window
    radius so spacing may vary along the texture axis.

    When `kernel_mode="varying"`, each row resolves its own radius from
    its own spacing, so a genuinely row-varying (ndim > 1) `spacing`
    array is honoured per row - relevant to eccopy2d_v with a 2-D
    coords_x whose range-gate spacing changes with elevation angle.

    kernel_mode mirrors refl_texture_2d's parameter of the same name.
    f_reflTexture.m takes a single scalar `pixRad`, so "uniform" is the
    default here; "varying" has no upstream counterpart.

    Parameters
    ----------
    dbz : np.ndarray, shape (..., N)
        Reflectivity. Texture is computed along the LAST axis.
    window : WindowSpec or int
        Window radius. If a physical WindowSpec (e.g. WindowSpec((5,'km'))),
        `spacing` must be provided.
    spacing : np.ndarray, shape (N,) or (..., N), optional
        Point-to-point spacing along the last axis, in the base unit
        matching `window` (km for length, seconds for time - EccoPy's
        run() entry points handle this conversion; see eccopy1d.run()
        and `eccopy2d_v.run()` / `eccopy2d_h.run()`). Required if `window`
        is a physical WindowSpec. A `(N,)` array (or an `(..., N)` array
        whose leading dimensions are all size 1) is broadcast identically
        to every row. An `(..., N)` array whose leading dimensions match
        `dbz`'s (after flattening) supplies a genuinely DIFFERENT spacing
        -- and therefore, in kernel_mode="varying", a different resolved
        window radius -- to each row; anything in between raises a clear
        error rather than silently guessing. Ignored entirely (not even
        validated) if `window` is a bare pixel radius.
    dbz_base : float
        Subtracted from values before texture is computed.
    min_valid_dbz : float
        Values below this are treated as missing before texture is
        computed. Defaults to -inf (no gating), matching f_reflTexture.m,
        which has no such prefilter. See TextureParams.min_valid_dbz.
    kernel_mode : {"uniform", "varying"}
        "uniform" (default) - resolve ONE radius from the median of the
            (validated/broadcast) spacing array, applied everywhere.
            Matches per-point resolution exactly on uniform-spacing data.
        "varying" - resolve a distinct radius at every point (and, for
            multi-row input, every row) from its own local spacing. Not
            validated against any ground truth in configurations where
            it differs from "uniform" - see docstring above.

    Returns
    -------
    texture : np.ndarray, same shape as dbz
    """
    if kernel_mode not in ("uniform", "varying"):
        raise ValueError(
            f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}"
        )

    dbz = np.asarray(dbz, dtype=float)
    # Null out dbz below min_valid_dbz BEFORE any downstream computation,
    # mirroring refl_texture_2d(). Sub-threshold values left in place would
    # otherwise participate as valid neighbours in the window statistic and
    # inflate texture near the edges of valid-data regions.
    #
    # The default is -inf (no gating), NOT 0.0 as in the 2-D/3-D path:
    # min_valid_dbz ports ConvStratFinder::computeEchoType()'s prefilter,
    # which has no analog in f_reflTexture.m. EccoPy-2D-V agrees with real
    # MATLAB ECCO-V output at 99.4-100% without it; gating at 0 dBZ moves
    # that output by ~13%. Callers that want the C++ behaviour set it
    # explicitly. See TextureParams.min_valid_dbz.
    if min_valid_dbz > -np.inf:
        dbz = np.where(dbz < min_valid_dbz, np.nan, dbz)

    orig_shape = dbz.shape
    n = orig_shape[-1]

    if isinstance(window, int):
        window = WindowSpec(window)

    flat = dbz.reshape(-1, n)
    nRows = flat.shape[0]

    # Resolve a radius field, shape (nRows, n) -- per-point/per-row in
    # kernel_mode="varying", or a single global value broadcast to that
    # same shape in kernel_mode="uniform" (see docstring above).
    if window.is_pixel:
        radius_field_2d = np.full((nRows, n), int(window.size), dtype=int)
    else:
        if spacing is None:
            raise ValueError(
                "A physical-unit WindowSpec requires a spacing array; "
                "got spacing=None. Pass an integer pixel radius instead, "
                "or provide a spacing array."
            )
        spacing = np.asarray(spacing, dtype=float)
        if spacing.ndim == 1:
            if spacing.shape[0] != n:
                raise ValueError(
                    f"spacing array length ({spacing.shape[0]}) along the "
                    f"texture axis must match the data length ({n})."
                )
            spacing_rows = np.broadcast_to(spacing, (nRows, n))
        else:
            if spacing.shape[-1] != n:
                raise ValueError(
                    f"spacing array's last-axis length ({spacing.shape[-1]}) "
                    f"must match the data length ({n})."
                )
            spacing_rows = spacing.reshape(-1, n)
            if spacing_rows.shape[0] == 1:
                spacing_rows = np.broadcast_to(spacing_rows, (nRows, n))
            elif spacing_rows.shape[0] != nRows:
                raise ValueError(
                    f"spacing has {spacing_rows.shape[0]} row(s) after "
                    f"flattening its leading dimensions, but dbz has "
                    f"{nRows} row(s) after the same flattening; spacing "
                    f"must have either 1 row (shared across every row) or "
                    f"exactly {nRows} rows (one genuine spacing profile "
                    f"per row) -- anything else is ambiguous, so this "
                    f"raises rather than guessing which row(s) to use."
                )

        if kernel_mode == "uniform":
            # ONE representative radius for the whole array -- the
            # global median across every row and column of the
            # (validated/broadcast) spacing, matching refl_texture_2d's
            # "one kernel for the whole domain" convention exactly.
            median_spacing = np.array(float(np.nanmedian(spacing_rows)))
            r = int(window.pixel_radius_field(median_spacing))
            radius_field_2d = np.full((nRows, n), r, dtype=int)
        else:
            # WindowSpec.pixel_radius_field is elementwise, so it
            # resolves the WHOLE (nRows, n) spacing array to a (nRows, n)
            # radius field in one call -- each row/point gets its OWN
            # radius, resolved from its OWN spacing.
            radius_field_2d = window.pixel_radius_field(spacing_rows)

    pad = int(radius_field_2d.max()) if radius_field_2d.size else 0

    if pad > 0:
        padded = np.pad(flat, ((0, 0), (pad, pad)), mode="constant",
                        constant_values=np.nan)
    else:
        padded = flat.copy()

    padded = _fillmissing_linear(padded, axis=1)
    padded = padded - dbz_base

    # _sliding_texture_core (the validated, Numba-JIT'd core loop) takes
    # ONE radius_field shared by every row in a single call -- rather
    # than changing that core's signature (and re-validating it against
    # test_numba_cross_validation.py's frozen reference), row-batch here:
    # the common case (every row resolved to the SAME radius field --
    # always true in kernel_mode="uniform", and also true in "varying"
    # whenever spacing happens to be uniform across rows, or window is a
    # bare pixel radius) stays on the original single, fast call; only
    # genuinely row-varying resolved radii pay for a per-row Python loop.
    if nRows <= 1 or np.all(radius_field_2d == radius_field_2d[0]):
        texture = _sliding_texture_core(padded, radius_field_2d[0], pad, nRows, n)
    else:
        texture = np.full((nRows, n), np.nan)
        for row in range(nRows):
            texture[row:row + 1] = _sliding_texture_core(
                padded[row:row + 1], radius_field_2d[row], pad, 1, n

            )

    texture[np.isnan(flat)] = np.nan

    return texture.reshape(orig_shape)


@njit(cache=True)
def _sliding_texture_core_with_fit(data_padded: np.ndarray,
                                    radius_field: np.ndarray,
                                    pad: int,
                                    nRows: int,
                                    nCols: int):
    """
    DEBUG-ONLY variant of _sliding_texture_core that additionally records,
    at every point, the value the point's own local linear fit predicts
    AT THAT POINT (`fitted`), and the detrended/clipped value that value
    turns into before being squared into the texture statistic
    (`detrended`) -- i.e. the two intermediate fields requested for
    debugging "the fitted dbz prior to texture".

    This is a byte-for-byte duplicate of _sliding_texture_core's math
    (never edit one without the other) with two extra writes; kept as a
    separate function rather than adding an output-array flag to the
    validated core so the hot, non-debug path is never touched by this
    change. See test_debug.py's cross-validation test, which checks this
    function's `texture` output against _sliding_texture_core's output
    on the same input.
    """
    texture = np.full((nRows, nCols), np.nan)
    fitted = np.full((nRows, nCols), np.nan)
    detrended = np.full((nRows, nCols), np.nan)

    for ii in range(nCols):
        r = int(radius_field[ii])
        if r < 1:
            continue
        lo = pad + ii - r
        hi = pad + ii + r + 1
        W = hi - lo
        if W < 2:
            continue
        jj_center = r  # position of point `ii` within its own window

        for row in range(nRows):
            sumX = 0.0
            sumX2 = 0.0
            sumY = 0.0
            sumXY = 0.0
            n_valid = 0
            for jj in range(W):
                x = jj + 1.0
                sumX += x
                sumX2 += x * x
                y = data_padded[row, lo + jj]
                if not np.isnan(y):
                    sumY += y
                    sumXY += x * y
                    n_valid += 1

            denom = W * sumX2 - sumX * sumX
            if denom != 0.0:
                a = (sumY * sumX2 - sumX * sumXY) / denom
                b = (W * sumXY - sumX * sumY) / denom
            else:
                a = 0.0
                b = 0.0

            mean_block = sumY / n_valid if n_valid > 0 else np.nan

            x_center = jj_center + 1.0
            fitted_center = a + b * x_center
            fitted[row, ii] = fitted_center
            y_center = data_padded[row, pad + ii]
            if not np.isnan(y_center):
                dtr = y_center - fitted_center + mean_block
                if dtr < 1.0:
                    dtr = 1.0
                detrended[row, ii] = dtr

            sum_sq = 0.0
            sum_sq2 = 0.0
            n_corrected = 0
            for jj in range(W):
                y = data_padded[row, lo + jj]
                if np.isnan(y):
                    continue
                x = jj + 1.0
                new_y = a + b * x
                corrected = y - new_y + mean_block
                if corrected < 1.0:
                    corrected = 1.0
                csq = corrected * corrected
                sum_sq += csq
                sum_sq2 += csq * csq
                n_corrected += 1

            if n_corrected > 0:
                mean_sq = sum_sq / n_corrected
                if n_corrected > 1:
                    sum_sq_dev = sum_sq2 - n_corrected * mean_sq * mean_sq
                    var = sum_sq_dev / (n_corrected - 1)
                    if var < 0.0:
                        var = 0.0
                else:
                    var = 0.0
                texture[row, ii] = np.sqrt(np.sqrt(var))

    return texture, fitted, detrended


def refl_texture_1d_with_fit(dbz: np.ndarray,
                             window: Union[WindowSpec, int],
                             spacing: Optional[np.ndarray] = None,
                             dbz_base: float = 0.0,
                             min_valid_dbz: float = -np.inf,
                             kernel_mode: str = "uniform"):
    """
    DEBUG-ONLY entry point: same texture calculation as refl_texture_1d,
    but also returns the per-point fitted trend value and the detrended
    (clipped) value at each point -- i.e. "the fitted dbz prior to
    texture" -- so you can inspect why a point got the texture value it
    did without calling refl_texture_1d_debug() one index at a time.

    `texture` from this function is identical to refl_texture_1d(...)
    on the same inputs (see test_debug.py's cross-validation test).
    Not used by any production run() path unless return_intermediates=True
    is passed; costs nothing when that flag is left at its default False.

    Parameters
    ----------
    Same as refl_texture_1d.

    Returns
    -------
    texture, fitted_dbz, detrended_dbz : np.ndarray, each same shape as dbz
        fitted_dbz    -- value the local linear fit predicts at that point
        detrended_dbz -- fitted_dbz removed and re-centred on the window
                         mean, clipped to >= 1 (dbz_base already added
                         back) -- the value that gets squared into the
                         texture statistic. NaN wherever the original
                         input was NaN or the point falls in the un-
                         computed border (matching refl_texture_1d).
    """
    if kernel_mode not in ("uniform", "varying"):
        raise ValueError(
            f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}"
        )

    dbz = np.asarray(dbz, dtype=float)
    # Null out dbz below min_valid_dbz BEFORE any downstream computation,
    # mirroring refl_texture_2d(). Sub-threshold values left in place would
    # otherwise participate as valid neighbours in the window statistic and
    # inflate texture near the edges of valid-data regions.
    #
    # The default is -inf (no gating), NOT 0.0 as in the 2-D/3-D path:
    # min_valid_dbz ports ConvStratFinder::computeEchoType()'s prefilter,
    # which has no analog in f_reflTexture.m. EccoPy-2D-V agrees with real
    # MATLAB ECCO-V output at 99.4-100% without it; gating at 0 dBZ moves
    # that output by ~13%. Callers that want the C++ behaviour set it
    # explicitly. See TextureParams.min_valid_dbz.
    if min_valid_dbz > -np.inf:
        dbz = np.where(dbz < min_valid_dbz, np.nan, dbz)

    orig_shape = dbz.shape
    n = orig_shape[-1]

    if isinstance(window, int):
        window = WindowSpec(window)

    flat = dbz.reshape(-1, n)
    nRows = flat.shape[0]

    if window.is_pixel:
        radius_field_2d = np.full((nRows, n), int(window.size), dtype=int)
    else:
        if spacing is None:
            raise ValueError(
                "A physical-unit WindowSpec requires a spacing array; "
                "got spacing=None. Pass an integer pixel radius instead, "
                "or provide a spacing array."
            )
        spacing = np.asarray(spacing, dtype=float)
        if spacing.ndim == 1:
            if spacing.shape[0] != n:
                raise ValueError(
                    f"spacing array length ({spacing.shape[0]}) along the "
                    f"texture axis must match the data length ({n})."
                )
            spacing_rows = np.broadcast_to(spacing, (nRows, n))
        else:
            if spacing.shape[-1] != n:
                raise ValueError(
                    f"spacing array's last-axis length ({spacing.shape[-1]}) "
                    f"must match the data length ({n})."
                )
            spacing_rows = spacing.reshape(-1, n)
            if spacing_rows.shape[0] == 1:
                spacing_rows = np.broadcast_to(spacing_rows, (nRows, n))
            elif spacing_rows.shape[0] != nRows:
                raise ValueError(
                    f"spacing has {spacing_rows.shape[0]} row(s) after "
                    f"flattening its leading dimensions, but dbz has "
                    f"{nRows} row(s) after the same flattening; spacing "
                    f"must have either 1 row (shared across every row) or "
                    f"exactly {nRows} rows (one genuine spacing profile "
                    f"per row) -- anything else is ambiguous, so this "
                    f"raises rather than guessing which row(s) to use."
                )

        if kernel_mode == "uniform":
            median_spacing = np.array(float(np.nanmedian(spacing_rows)))
            r = int(window.pixel_radius_field(median_spacing))
            radius_field_2d = np.full((nRows, n), r, dtype=int)
        else:
            radius_field_2d = window.pixel_radius_field(spacing_rows)

    pad = int(radius_field_2d.max()) if radius_field_2d.size else 0

    if pad > 0:
        padded = np.pad(flat, ((0, 0), (pad, pad)), mode="constant",
                        constant_values=np.nan)
    else:
        padded = flat.copy()

    padded = _fillmissing_linear(padded, axis=1)
    padded = padded - dbz_base

    if nRows <= 1 or np.all(radius_field_2d == radius_field_2d[0]):
        texture, fitted, detrended = _sliding_texture_core_with_fit(
            padded, radius_field_2d[0], pad, nRows, n)
    else:
        texture = np.full((nRows, n), np.nan)
        fitted = np.full((nRows, n), np.nan)
        detrended = np.full((nRows, n), np.nan)
        for row in range(nRows):
            t, f, d = _sliding_texture_core_with_fit(
                padded[row:row + 1], radius_field_2d[row], pad, 1, n)
            texture[row:row + 1] = t
            fitted[row:row + 1] = f
            detrended[row:row + 1] = d

    nan_mask = np.isnan(flat)
    texture[nan_mask] = np.nan
    detrended[nan_mask] = np.nan

    return (texture.reshape(orig_shape), fitted.reshape(orig_shape) + dbz_base,
            detrended.reshape(orig_shape) + dbz_base)




# ---------------------------------------------------------------------------
# 2-D radial texture (used by EccoPy-3D)
# ---------------------------------------------------------------------------
#
# The kernel builders here are used throughout (including by
# eccopy.core.debug). The fraction/texture cores below them --
# _compute_fraction_active*, _compute_texture_one_level* -- are the
# EccoPy 1.0 cores. refl_texture_2d() no longer calls them (see "Prepared
# 2-D windows" further down); they are kept unchanged as the validated
# reference the 1.1 cores must reproduce bit for bit, and are themselves
# cross-validated against pure-Python references in
# tests/test_numba_cross_validation.py.

def _build_kernel_offsets_uniform(radius_km: float, dx_km: float, dy_km: float):
    """
    Pre-compute kernel offsets for a UNIFORM-spacing approximation,
    identical to ConvStratFinder::_computeKernels(). LROSE itself uses
    this approach even for lat/lon grids: a single dx_km/dy_km computed
    once at the domain's mean latitude (see refl_texture_2d's
    kernel_mode="uniform").

    Returns offsets as a list of (jdx, jdy, xx, yy) tuples. This is the
    interface used by eccopy.core.debug (which iterates the list
    directly); see _build_kernel_offsets_uniform_arrays() for the
    typed-array equivalent used internally by the JIT-compiled texture
    functions, which cannot accept Python lists of tuples.
    """
    r_km = radius_km
    ny_tex = int(np.floor(r_km / dy_km + 0.5))
    nx_tex = int(np.floor(r_km / dx_km + 0.5))
    while ny_tex < 2 or nx_tex < 2:
        r_km *= 1.1
        ny_tex = int(np.floor(r_km / dy_km + 0.5))
        nx_tex = int(np.floor(r_km / dx_km + 0.5))

    offsets = []
    for jdy in range(-ny_tex, ny_tex + 1):
        yy = jdy * dy_km
        for jdx in range(-nx_tex, nx_tex + 1):
            xx = jdx * dx_km
            if np.sqrt(yy * yy + xx * xx) <= r_km:
                offsets.append((jdx, jdy, xx, yy))
    return offsets, nx_tex, ny_tex


def _build_kernel_offsets_uniform_arrays(radius_km: float, dx_km: float, dy_km: float):
    """
    Same kernel as _build_kernel_offsets_uniform(), but returned as four
    typed NumPy arrays (jdx, jdy, xx, yy) instead of a list of tuples -
    the form the JIT-compiled texture functions need, since Numba's
    nopython mode cannot work with Python lists of heterogeneous tuples.
    """
    offsets, nx_tex, ny_tex = _build_kernel_offsets_uniform(radius_km, dx_km, dy_km)
    n = len(offsets)
    jdx_arr = np.empty(n, dtype=np.int64)
    jdy_arr = np.empty(n, dtype=np.int64)
    xx_arr = np.empty(n, dtype=np.float64)
    yy_arr = np.empty(n, dtype=np.float64)
    for k, (jdx, jdy, xx, yy) in enumerate(offsets):
        jdx_arr[k] = jdx
        jdy_arr[k] = jdy
        xx_arr[k] = xx
        yy_arr[k] = yy
    return jdx_arr, jdy_arr, xx_arr, yy_arr, nx_tex, ny_tex


@njit(cache=True)
def _max_half_width_core(radius_km: float, dx_km: np.ndarray, dy_km: np.ndarray):
    """
    JIT-compiled core for _max_half_width(). Uses an explicit scan for
    the minimum positive spacing rather than boolean-mask fancy indexing
    (dx_km[dx_km > 0]) - equivalent result, but the form Numba compiles
    most efficiently for this access pattern.
    """
    min_dy = np.inf
    found_dy = False
    for v in dy_km.flat:
        if v > 0 and v < min_dy:
            min_dy = v
            found_dy = True
    if not found_dy:
        min_dy = radius_km

    min_dx = np.inf
    found_dx = False
    for v in dx_km.flat:
        if v > 0 and v < min_dx:
            min_dx = v
            found_dx = True
    if not found_dx:
        min_dx = radius_km

    ny_tex = max(1, int(np.floor(radius_km / min_dy + 0.5)))
    nx_tex = max(1, int(np.floor(radius_km / min_dx + 0.5)))
    return nx_tex, ny_tex


def _max_half_width(radius_km: float, dx_km: np.ndarray, dy_km: np.ndarray):
    """
    Conservative (worst-case) half-width in grid cells needed to contain
    a `radius_km` circle anywhere on a varying-spacing grid - used to
    size the border that must be excluded from per-point kernel mode
    (the finest spacing anywhere in the grid determines the largest
    possible cell count).

    Uses the same floor(r/spacing + 0.5) convention as
    _build_kernel_offsets_uniform / _point_kernel_offsets, so that on a
    perfectly uniform grid this produces exactly the same margin as the
    uniform-kernel path (kernel_mode="uniform" and "varying" should
    agree exactly on a uniform grid).
    """
    return _max_half_width_core(radius_km, dx_km, dy_km)


def _point_kernel_offsets(radius_km: float, dx_km_pt: float, dy_km_pt: float):
    """
    Build kernel offsets for ONE point's local spacing (per-point mode).
    Returns a list of (jdx, jdy, xx, yy) tuples - the interface used by
    eccopy.core.debug; see _point_kernel_offsets_arrays() for the
    typed-array equivalent used internally by the JIT-compiled texture
    functions.
    """
    r_km = radius_km
    ny_tex = max(1, int(np.floor(r_km / dy_km_pt + 0.5)))
    nx_tex = max(1, int(np.floor(r_km / dx_km_pt + 0.5)))
    offsets = []
    for jdy in range(-ny_tex, ny_tex + 1):
        yy = jdy * dy_km_pt
        for jdx in range(-nx_tex, nx_tex + 1):
            xx = jdx * dx_km_pt
            if np.sqrt(yy * yy + xx * xx) <= r_km:
                offsets.append((jdx, jdy, xx, yy))
    return offsets, nx_tex, ny_tex


@njit(cache=True)
def _point_kernel_offsets_arrays(radius_km: float, dx_km_pt: float, dy_km_pt: float):
    """
    Same kernel as _point_kernel_offsets(), but returned as typed NumPy
    arrays instead of a list of tuples, and JIT-compiled directly since
    this gets called fresh at EVERY grid point in kernel_mode="varying"
    (unlike the uniform-grid kernel, which is built once and reused).
    """
    r_km = radius_km
    ny_tex = max(1, int(np.floor(r_km / dy_km_pt + 0.5)))
    nx_tex = max(1, int(np.floor(r_km / dx_km_pt + 0.5)))

    count = 0
    for jdy in range(-ny_tex, ny_tex + 1):
        yy = jdy * dy_km_pt
        for jdx in range(-nx_tex, nx_tex + 1):
            xx = jdx * dx_km_pt
            if np.sqrt(yy * yy + xx * xx) <= r_km:
                count += 1

    jdx_arr = np.empty(count, dtype=np.int64)
    jdy_arr = np.empty(count, dtype=np.int64)
    xx_arr = np.empty(count, dtype=np.float64)
    yy_arr = np.empty(count, dtype=np.float64)
    k = 0
    for jdy in range(-ny_tex, ny_tex + 1):
        yy = jdy * dy_km_pt
        for jdx in range(-nx_tex, nx_tex + 1):
            xx = jdx * dx_km_pt
            if np.sqrt(yy * yy + xx * xx) <= r_km:
                jdx_arr[k] = jdx
                jdy_arr[k] = jdy
                xx_arr[k] = xx
                yy_arr[k] = yy
                k += 1

    return jdx_arr, jdy_arr, xx_arr, yy_arr, nx_tex, ny_tex


@njit(cache=True)
def _compute_fraction_active_varying(dbz_col_max: np.ndarray,
                                      radius_km: float,
                                      dx_km: np.ndarray,
                                      dy_km: np.ndarray,
                                      min_valid_dbz: float,
                                      max_nx_tex: int,
                                      max_ny_tex: int) -> np.ndarray:
    """
    Coverage fraction array using a PER-POINT kernel, rebuilt from the
    local dx_km[y,x]/dy_km[y,x] at every grid point. Slower than the
    uniform-grid version but correct on grids where spacing varies
    significantly (e.g. large lat/lon domains).

    JIT-compiled, including the per-point kernel rebuild (via
    _point_kernel_offsets_arrays, itself @njit) - Numba supports calling
    one JIT-compiled function from another directly.
    """
    ny, nx = dbz_col_max.shape
    fraction = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(max_ny_tex, ny - max_ny_tex):
        for ix in range(max_nx_tex, nx - max_nx_tex):
            jdx_arr, jdy_arr, xx_arr, yy_arr, _, _ = _point_kernel_offsets_arrays(
                radius_km, dx_km[iy, ix], dy_km[iy, ix]
            )
            n_kernel = len(jdx_arr)
            if n_kernel == 0:
                continue
            count = 0
            for k in range(n_kernel):
                if dbz_col_max[iy + jdy_arr[k], ix + jdx_arr[k]] >= min_valid_dbz:
                    count += 1
            fraction[iy, ix] = count / n_kernel

    return fraction


@njit(cache=True)
def _compute_texture_one_level_varying(dbz_level: np.ndarray,
                                        fraction_active: np.ndarray,
                                        radius_km: float,
                                        dx_km: np.ndarray,
                                        dy_km: np.ndarray,
                                        base_dbz: float,
                                        min_frac_texture: float,
                                        min_frac_fit: float,
                                        max_nx_tex: int,
                                        max_ny_tex: int) -> np.ndarray:
    """
    Per-point-kernel version of _compute_texture_one_level: the circular
    kernel footprint is rebuilt from the LOCAL dx_km[y,x]/dy_km[y,x] at
    every point, so the physical kernel radius stays consistent even
    when grid spacing varies substantially across the domain (e.g. a
    lat/lon grid spanning a wide latitude range, where dx shrinks with
    cos(latitude) but dy stays roughly constant).

    This is NOT what the validated LROSE ConvStratFinder algorithm does
    (LROSE uses a single dx_km/dy_km for the whole grid, computed at the
    domain's mean latitude - see _build_kernel_offsets_uniform). Results
    from this mode have not been validated against LROSE output and may
    differ from it on grids where LROSE's single-kernel approximation
    breaks down.

    JIT-compiled; see _compute_texture_one_level_core()'s docstring for
    the same notes on the normal-equations plane fit replacing
    np.linalg.lstsq (not supported in Numba's nopython mode), which
    apply identically here.
    """
    ny, nx = dbz_level.shape
    texture = np.full((ny, nx), np.nan, dtype=np.float64)

    for iy in range(max_ny_tex, ny - max_ny_tex):
        for ix in range(max_nx_tex, nx - max_nx_tex):
            if fraction_active[iy, ix] < min_frac_texture:
                continue
            center_val = dbz_level[iy, ix]
            if np.isnan(center_val):
                continue

            jdx_arr, jdy_arr, xx_arr, yy_arr, _, _ = _point_kernel_offsets_arrays(
                radius_km, dx_km[iy, ix], dy_km[iy, ix]
            )
            n_kernel = len(jdx_arr)
            if n_kernel == 0:
                continue

            min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
            min_pts_fit = int(min_frac_fit * n_kernel + 0.5)

            vals = np.empty(n_kernel, dtype=np.float64)
            vxx = np.empty(n_kernel, dtype=np.float64)
            vyy = np.empty(n_kernel, dtype=np.float64)

            count = 0
            sum_dbz = 0.0
            for k in range(n_kernel):
                v = dbz_level[iy + jdy_arr[k], ix + jdx_arr[k]]
                if not np.isnan(v):
                    vals[count] = v
                    vxx[count] = xx_arr[k]
                    vyy[count] = yy_arr[k]
                    sum_dbz += v
                    count += 1

            if count < min_pts_texture:
                continue

            mean_dbz = sum_dbz / count
            if mean_dbz < 1.0:
                mean_dbz = 1.0

            if count >= min_pts_fit:
                Sx = 0.0; Sy = 0.0; Sxx = 0.0; Syy = 0.0; Sxy = 0.0
                Sz = 0.0; Sxz = 0.0; Syz = 0.0
                for i in range(count):
                    x = vxx[i]; y = vyy[i]; z = vals[i]
                    Sx += x; Sy += y
                    Sxx += x * x; Syy += y * y; Sxy += x * y
                    Sz += z; Sxz += x * z; Syz += y * z
                n = float(count)

                m00 = Sxx; m01 = Sxy; m02 = Sx
                m10 = Sxy; m11 = Syy; m12 = Sy
                m20 = Sx;  m21 = Sy;  m22 = n
                b0 = Sxz; b1 = Syz; b2 = Sz

                det = (m00 * (m11 * m22 - m12 * m21)
                       - m01 * (m10 * m22 - m12 * m20)
                       + m02 * (m10 * m21 - m11 * m20))

                if abs(det) > 1e-12:
                    aa = (b0 * (m11 * m22 - m12 * m21)
                          - m01 * (b1 * m22 - m12 * b2)
                          + m02 * (b1 * m21 - m11 * b2)) / det
                    bb = (m00 * (b1 * m22 - m12 * b2)
                          - b0 * (m10 * m22 - m12 * m20)
                          + m02 * (m10 * b2 - b1 * m20)) / det
                    for i in range(count):
                        vals[i] -= aa * vxx[i] + bb * vyy[i]

            sum_sq = 0.0
            sum_sq2 = 0.0
            nn = 0.0
            for i in range(count):
                val = vals[i] - base_dbz
                if val < 1.0:
                    val = 1.0
                dbz_sq = val * val
                sum_sq += dbz_sq
                sum_sq2 += dbz_sq * dbz_sq
                nn += 1.0

            n_missing = n_kernel - count
            if n_missing > 0:
                min_sq = mean_dbz * mean_dbz
                sum_sq += n_missing * min_sq
                sum_sq2 += n_missing * min_sq * min_sq
                nn += n_missing

            if nn < 1:
                continue

            mean_sq = sum_sq / nn
            var = sum_sq2 / nn - mean_sq * mean_sq
            if var < 0.0:
                var = 0.0
            tex = np.sqrt(np.sqrt(var))
            if not np.isnan(tex):
                texture[iy, ix] = tex

    return texture


@njit(cache=True)
def _compute_fraction_active_core(dbz_col_max: np.ndarray,
                                  jdx_arr: np.ndarray,
                                  jdy_arr: np.ndarray,
                                  nx_tex: int,
                                  ny_tex: int,
                                  min_valid_dbz: float) -> np.ndarray:
    """JIT-compiled core for _compute_fraction_active()."""
    ny, nx = dbz_col_max.shape
    n_kernel = len(jdx_arr)
    fraction = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny_tex, ny - ny_tex):
        for ix in range(nx_tex, nx - nx_tex):
            count = 0
            for k in range(n_kernel):
                v = dbz_col_max[iy + jdy_arr[k], ix + jdx_arr[k]]
                if v >= min_valid_dbz:
                    count += 1
            fraction[iy, ix] = count / n_kernel

    return fraction


def _compute_fraction_active(dbz_col_max: np.ndarray,
                              offsets: list,
                              nx_tex: int,
                              ny_tex: int,
                              min_valid_dbz: float) -> np.ndarray:
    """
    Coverage fraction array from column-max DBZ (uniform-grid kernel).

    `offsets` is a list of (jdx, jdy, xx, yy) tuples, matching
    _build_kernel_offsets_uniform()'s return value - kept as the public
    interface (used directly by eccopy.core.debug) while the actual
    computation happens in the JIT-compiled _compute_fraction_active_core().
    """
    n = len(offsets)
    jdx_arr = np.empty(n, dtype=np.int64)
    jdy_arr = np.empty(n, dtype=np.int64)
    for k, (jdx, jdy, _, _) in enumerate(offsets):
        jdx_arr[k] = jdx
        jdy_arr[k] = jdy
    return _compute_fraction_active_core(dbz_col_max, jdx_arr, jdy_arr,
                                         nx_tex, ny_tex, min_valid_dbz)


@njit(cache=True)
def _compute_texture_one_level_core(dbz_level: np.ndarray,
                                    fraction_active: np.ndarray,
                                    jdx_arr: np.ndarray,
                                    jdy_arr: np.ndarray,
                                    xx_arr: np.ndarray,
                                    yy_arr: np.ndarray,
                                    nx_tex: int,
                                    ny_tex: int,
                                    base_dbz: float,
                                    min_frac_texture: float,
                                    min_frac_fit: float) -> np.ndarray:
    """
    JIT-compiled core for _compute_texture_one_level().

    Implementation notes vs. the original:
      - The plane fit (z = a*x + b*y, fit by least squares over the
        valid kernel points) is solved via the 2x2 normal equations
        instead of np.linalg.lstsq, since lstsq is not supported in
        Numba's nopython mode. This is mathematically equivalent to
        lstsq EXCEPT when the system is singular/rank-deficient (e.g.
        all valid points collinear) - the original handled that via
        `except np.linalg.LinAlgError: pass` (silently skip detrending);
        this version checks the normal-equations determinant directly
        and does the same thing if it's too close to zero to invert
        reliably.
      - Per-point working buffers (vals/xx/yy) are pre-allocated to the
        kernel size and reused, rather than appending to Python lists
        (also unsupported in nopython mode for this access pattern).
    """
    ny, nx = dbz_level.shape
    n_kernel = len(jdx_arr)
    texture = np.full((ny, nx), np.nan, dtype=np.float64)

    min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
    min_pts_fit = int(min_frac_fit * n_kernel + 0.5)

    vals = np.empty(n_kernel, dtype=np.float64)
    vxx = np.empty(n_kernel, dtype=np.float64)
    vyy = np.empty(n_kernel, dtype=np.float64)

    for iy in range(ny_tex, ny - ny_tex):
        for ix in range(nx_tex, nx - nx_tex):
            if fraction_active[iy, ix] < min_frac_texture:
                continue
            center_val = dbz_level[iy, ix]
            if np.isnan(center_val):
                continue

            count = 0
            sum_dbz = 0.0
            for k in range(n_kernel):
                v = dbz_level[iy + jdy_arr[k], ix + jdx_arr[k]]
                if not np.isnan(v):
                    vals[count] = v
                    vxx[count] = xx_arr[k]
                    vyy[count] = yy_arr[k]
                    sum_dbz += v
                    count += 1

            if count < min_pts_texture:
                continue

            mean_dbz = sum_dbz / count
            if mean_dbz < 1.0:
                mean_dbz = 1.0

            if count >= min_pts_fit:
                # Plane fit z = a*x + b*y + c via 3x3 normal equations
                # (equivalent to np.linalg.lstsq for this 3-parameter
                # fit, except when singular - see docstring above).
                Sx = 0.0; Sy = 0.0; Sxx = 0.0; Syy = 0.0; Sxy = 0.0
                Sz = 0.0; Sxz = 0.0; Syz = 0.0
                for i in range(count):
                    x = vxx[i]; y = vyy[i]; z = vals[i]
                    Sx += x; Sy += y
                    Sxx += x * x; Syy += y * y; Sxy += x * y
                    Sz += z; Sxz += x * z; Syz += y * z
                n = float(count)

                m00 = Sxx; m01 = Sxy; m02 = Sx
                m10 = Sxy; m11 = Syy; m12 = Sy
                m20 = Sx;  m21 = Sy;  m22 = n
                b0 = Sxz; b1 = Syz; b2 = Sz

                det = (m00 * (m11 * m22 - m12 * m21)
                       - m01 * (m10 * m22 - m12 * m20)
                       + m02 * (m10 * m21 - m11 * m20))

                if abs(det) > 1e-12:
                    aa = (b0 * (m11 * m22 - m12 * m21)
                          - m01 * (b1 * m22 - m12 * b2)
                          + m02 * (b1 * m21 - m11 * b2)) / det
                    bb = (m00 * (b1 * m22 - m12 * b2)
                          - b0 * (m10 * m22 - m12 * m20)
                          + m02 * (m10 * b2 - b1 * m20)) / det
                    for i in range(count):
                        vals[i] -= aa * vxx[i] + bb * vyy[i]
                # else: singular system (e.g. all points collinear) -
                # skip detrending, same as the original's
                # `except np.linalg.LinAlgError: pass`.

            sum_sq = 0.0
            sum_sq2 = 0.0
            nn = 0.0
            for i in range(count):
                val = vals[i] - base_dbz
                if val < 1.0:
                    val = 1.0
                dbz_sq = val * val
                sum_sq += dbz_sq
                sum_sq2 += dbz_sq * dbz_sq
                nn += 1.0

            n_missing = n_kernel - count
            if n_missing > 0:
                min_sq = mean_dbz * mean_dbz
                sum_sq += n_missing * min_sq
                sum_sq2 += n_missing * min_sq * min_sq
                nn += n_missing

            if nn < 1:
                continue

            mean_sq = sum_sq / nn
            var = sum_sq2 / nn - mean_sq * mean_sq
            if var < 0.0:
                var = 0.0
            tex = np.sqrt(np.sqrt(var))
            if not np.isnan(tex):
                texture[iy, ix] = tex

    return texture


def _compute_texture_one_level(dbz_level: np.ndarray,
                                fraction_active: np.ndarray,
                                offsets: list,
                                nx_tex: int,
                                ny_tex: int,
                                base_dbz: float,
                                min_frac_texture: float,
                                min_frac_fit: float) -> np.ndarray:
    """
    2-D circular-neighbourhood texture with planar detrend, for one level.

    Exact port of ConvStratFinder::ComputeTexture::run() (see module
    docstring in the original validated implementation):
      1. Skip center if fractionActive < min_frac_texture or center missing.
      2. Collect valid kernel neighbours; fit a plane; subtract it.
      3. Clamp each value to >= 1; missing kernel points substituted
         with mean_dbz^2.
      4. texture = sqrt(sqrt(population_var(dbz^2)))

    `offsets` is a list of (jdx, jdy, xx, yy) tuples, matching
    _build_kernel_offsets_uniform()'s return value - kept as the public
    interface (used directly by eccopy.core.debug) while the actual
    computation happens in the JIT-compiled _compute_texture_one_level_core().
    """
    n = len(offsets)
    jdx_arr = np.empty(n, dtype=np.int64)
    jdy_arr = np.empty(n, dtype=np.int64)
    xx_arr = np.empty(n, dtype=np.float64)
    yy_arr = np.empty(n, dtype=np.float64)
    for k, (jdx, jdy, xx, yy) in enumerate(offsets):
        jdx_arr[k] = jdx
        jdy_arr[k] = jdy
        xx_arr[k] = xx
        yy_arr[k] = yy
    return _compute_texture_one_level_core(
        dbz_level, fraction_active, jdx_arr, jdy_arr, xx_arr, yy_arr,
        nx_tex, ny_tex, base_dbz, min_frac_texture, min_frac_fit,
    )


# ---------------------------------------------------------------------------
# Prepared 2-D windows (EccoPy 1.1): build each distinct window once
# ---------------------------------------------------------------------------
#
# The 1.0 cores above take ONE window for the whole grid ("uniform") or
# rebuild the window at EVERY point ("varying"). On a regular lat/lon grid
# the varying window depends only on the row, so the 1.0 varying path
# rebuilt the same couple of dozen windows tens of millions of times.
#
# _TextureKernels2D below stores each distinct window once, plus the
# window each row uses, and _fraction_active_rows()/_texture_rows() reuse
# it. Both kernel modes run through these cores: "uniform" is simply the
# case of one window shared by every row.
#
# Equivalence with the 1.0 cores is bit-for-bit, not approximate:
#
#   1. Which offsets a point's window contains is a monotone function of
#      that point's spacing. In _point_kernel_offsets_arrays() every test
#      -- floor(r/dx + 0.5), |jdx*dx|, sqrt(yy*yy + xx*xx) <= r -- is
#      monotone in dx and dy under IEEE arithmetic, so a larger spacing can
#      only remove offsets. Hence if the windows built from a row's
#      smallest and largest spacing are identical, every point in that
#      row has exactly that window, in the same order.
#   2. The plane-fit coordinates are still formed per point, as
#      jdx * dx[iy, ix] and jdy * dy[iy, ix] -- the same products the 1.0
#      cores formed -- and every sum is accumulated in the same order.
#
# A row whose smallest and largest spacing give different windows, or
# whose spacing is not finite and positive, is marked ROW_PER_POINT and
# keeps the 1.0 behaviour of rebuilding the window at each point. A grid
# whose spacing varies along x (not lat/lon) therefore falls back to the
# 1.0 arithmetic automatically.

ROW_SKIP = -1        # row is in the excluded border; nothing computed
ROW_PER_POINT = -2   # window rebuilt per point, exactly as in 1.0 "varying"


@dataclass(frozen=True, eq=False)
class _TextureKernels2D:
    """
    The windows refl_texture_2d() needs for one grid, resolved once.

    Kernel `k` occupies ``k_jdx[k_ptr[k]:k_ptr[k + 1]]`` (and the same
    slice of ``k_jdy``), in the order the 1.0 builders produce. Row `iy`
    uses kernel ``row_kernel[iy]``, or ROW_SKIP / ROW_PER_POINT. Texture
    and coverage are computed for rows with ``row_kernel >= 0`` or
    ROW_PER_POINT, and for columns ``ix0 <= ix < ix1``: the same region
    the 1.0 cores computed for the same kernel_mode.

    ``scale_y``/``scale_x`` are the spacings (km) the plane-fit
    coordinates are built from at each point: the median spacing
    broadcast over the grid in "uniform" mode, the local spacing in
    "varying" mode. Either may be a read-only broadcast view.
    """
    kernel_mode: str
    radius_km: float
    shape: tuple
    k_ptr: np.ndarray
    k_jdx: np.ndarray
    k_jdy: np.ndarray
    row_kernel: np.ndarray
    ix0: int
    ix1: int
    scale_y: np.ndarray
    scale_x: np.ndarray
    max_kernel: int

    @property
    def n_kernels(self) -> int:
        """Number of distinct windows stored."""
        return len(self.k_ptr) - 1

    @property
    def n_rows_per_point(self) -> int:
        """Rows that rebuild the window at every point (1.0 behaviour)."""
        return int(np.count_nonzero(self.row_kernel == ROW_PER_POINT))


def _nanmedian_all(a: np.ndarray) -> float:
    """
    ``float(np.nanmedian(a))`` over every element, without materialising
    a broadcast view.

    A stride-0 axis repeats every underlying value the same number of
    times, and repeating each value of a set m times leaves its median
    unchanged (the two middle positions of the repeated set fall on the
    same one or two underlying values), so the median is taken over the
    underlying values only. Arrays with no stride-0 axis are passed to
    np.nanmedian unchanged.
    """
    a = np.asarray(a)
    idx = tuple(0 if (stride == 0 and n > 1) else slice(None)
                for stride, n in zip(a.strides, a.shape))
    return float(np.nanmedian(a[idx]))


@njit(cache=True)
def _max_positive(arr: np.ndarray) -> float:
    """Largest strictly positive value in `arr`, or NaN if there is none."""
    best = -np.inf
    for v in arr.flat:
        if v > 0.0 and v > best:
            best = v
    if best == -np.inf:
        return np.nan
    return best


@njit(cache=True)
def _row_spacing_range(dy: np.ndarray, dx: np.ndarray,
                       iy0: int, iy1: int, ix0: int, ix1: int):
    """
    Per row iy0 <= iy < iy1, the smallest and largest dy and dx over
    columns ix0 <= ix < ix1, and whether every value there is finite and
    positive (`ok`). Rows that are not ok fall back to per-point windows.
    """
    n = iy1 - iy0
    ok = np.ones(n, dtype=np.bool_)
    dy_lo = np.empty(n, dtype=np.float64)
    dy_hi = np.empty(n, dtype=np.float64)
    dx_lo = np.empty(n, dtype=np.float64)
    dx_hi = np.empty(n, dtype=np.float64)
    for r in range(n):
        iy = iy0 + r
        ylo = np.inf
        yhi = -np.inf
        xlo = np.inf
        xhi = -np.inf
        good = True
        for ix in range(ix0, ix1):
            a = dy[iy, ix]
            b = dx[iy, ix]
            # NaN fails every comparison, so it is caught here too.
            if not (a > 0.0 and a < np.inf and b > 0.0 and b < np.inf):
                good = False
                break
            if a < ylo:
                ylo = a
            if a > yhi:
                yhi = a
            if b < xlo:
                xlo = b
            if b > xhi:
                xhi = b
        ok[r] = good
        dy_lo[r] = ylo
        dy_hi[r] = yhi
        dx_lo[r] = xlo
        dx_hi[r] = xhi
    return ok, dy_lo, dy_hi, dx_lo, dx_hi


def _resolve_radius_km(texture_radius) -> float:
    """refl_texture_2d's texture_radius (float km or length WindowSpec) in km."""
    if isinstance(texture_radius, (int, float)):
        return float(texture_radius)
    if isinstance(texture_radius, WindowSpec):
        if texture_radius.is_pixel:
            raise ValueError(
                "refl_texture_2d requires a physical (km) radius, not a "
                "bare pixel-count WindowSpec. Pass a float (km) or "
                "WindowSpec((radius, 'km'))."
            )
        # WindowSpec stores its physical size internally in base units
        # (metres for length); convert to km for this function's use.
        return texture_radius._base_value / 1000.0
    return float(texture_radius)


def _prepare_texture_2d(radius_km: float,
                        dy: np.ndarray,
                        dx: np.ndarray,
                        kernel_mode: str) -> _TextureKernels2D:
    """
    Resolve the windows refl_texture_2d() uses on a (ny, nx) grid.

    "uniform": one window from the median dy/dx, exactly as the 1.0
    uniform path built it, used by every row; the border is that window's
    own half-width. "varying": one window per row wherever a row's
    spacing admits only one window (see the section comment above), and
    ROW_PER_POINT elsewhere; the border is the worst-case half-width over
    the whole grid, as in the 1.0 varying path.

    Depends only on the grid, never on reflectivity, so it can be built
    once and reused for every frame on a fixed grid (see
    eccopy2d_h.prepare_grid()).
    """
    if kernel_mode not in ("uniform", "varying"):
        raise ValueError(
            f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}"
        )
    dy = np.asarray(dy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if dy.ndim != 2 or dy.shape != dx.shape:
        raise ValueError(
            f"dy and dx must be 2-D arrays of the same shape; got "
            f"dy.shape={dy.shape}, dx.shape={dx.shape}"
        )
    ny, nx = dy.shape

    if kernel_mode == "uniform":
        # Single representative dx_km/dy_km for the whole grid - matches
        # LROSE's own mean-latitude approach for lat/lon grids.
        dy_km = _nanmedian_all(dy)
        dx_km = _nanmedian_all(dx)
        jdx, jdy, _, _, nx_tex, ny_tex = _build_kernel_offsets_uniform_arrays(
            radius_km, dx_km, dy_km)
        row_kernel = np.full(ny, ROW_SKIP, dtype=np.int64)
        if ny - ny_tex > ny_tex:
            row_kernel[ny_tex:ny - ny_tex] = 0
        return _TextureKernels2D(
            kernel_mode="uniform", radius_km=float(radius_km), shape=(ny, nx),
            k_ptr=np.array([0, len(jdx)], dtype=np.int64),
            k_jdx=jdx, k_jdy=jdy, row_kernel=row_kernel,
            ix0=int(nx_tex), ix1=int(nx - nx_tex),
            scale_y=np.broadcast_to(np.float64(dy_km), (ny, nx)),
            scale_x=np.broadcast_to(np.float64(dx_km), (ny, nx)),
            max_kernel=int(len(jdx)),
        )

    # --- kernel_mode == "varying" ---
    max_nx_tex, max_ny_tex = _max_half_width(radius_km, dx, dy)

    # Warn if the texture radius is smaller than the local grid spacing
    # anywhere in the domain - the kernel degenerates to just the center
    # point there, making texture meaningless (always 0) at those
    # locations regardless of kernel_mode.
    coarsest_dx = _max_positive(dx)
    coarsest_dy = _max_positive(dy)
    if (np.isfinite(coarsest_dx) and coarsest_dx > radius_km) or \
       (np.isfinite(coarsest_dy) and coarsest_dy > radius_km):
        warnings.warn(
            f"texture_radius ({radius_km:.2f} km) is smaller than the "
            f"local grid spacing at some points (max dx={coarsest_dx:.2f} "
            f"km, max dy={coarsest_dy:.2f} km). At those points the "
            f"kernel degenerates to a single cell and texture will be "
            f"exactly 0 there (no meaningful neighbourhood exists). "
            f"Use a larger texture_radius or coarsen the grid for "
            f"physically meaningful results everywhere.",
            stacklevel=3,
        )

    iy0, iy1 = int(max_ny_tex), int(ny - max_ny_tex)
    ix0, ix1 = int(max_nx_tex), int(nx - max_nx_tex)
    row_kernel = np.full(ny, ROW_SKIP, dtype=np.int64)
    jdx_parts, jdy_parts = [], []
    seen = {}
    if iy1 > iy0 and ix1 > ix0:
        ok, dy_lo, dy_hi, dx_lo, dx_hi = _row_spacing_range(
            dy, dx, iy0, iy1, ix0, ix1)
        for r in range(iy1 - iy0):
            if not ok[r]:
                row_kernel[iy0 + r] = ROW_PER_POINT
                continue
            jdx_a, jdy_a, _, _, _, _ = _point_kernel_offsets_arrays(
                radius_km, dx_lo[r], dy_lo[r])
            jdx_b, jdy_b, _, _, _, _ = _point_kernel_offsets_arrays(
                radius_km, dx_hi[r], dy_hi[r])
            if not (np.array_equal(jdx_a, jdx_b) and np.array_equal(jdy_a, jdy_b)):
                row_kernel[iy0 + r] = ROW_PER_POINT
                continue
            key = (jdx_a.tobytes(), jdy_a.tobytes())
            k = seen.get(key)
            if k is None:
                k = len(jdx_parts)
                seen[key] = k
                jdx_parts.append(jdx_a)
                jdy_parts.append(jdy_a)
            row_kernel[iy0 + r] = k

    sizes = [len(p) for p in jdx_parts]
    k_ptr = np.zeros(len(sizes) + 1, dtype=np.int64)
    k_ptr[1:] = np.cumsum(sizes) if sizes else []
    k_jdx = np.concatenate(jdx_parts) if jdx_parts else np.empty(0, np.int64)
    k_jdy = np.concatenate(jdy_parts) if jdy_parts else np.empty(0, np.int64)
    # Per-point rows can need a larger window than any stored one; the
    # finest spacing anywhere bounds it.
    max_kernel = max(sizes) if sizes else 0
    if np.any(row_kernel == ROW_PER_POINT):
        max_kernel = max(max_kernel,
                         (2 * max_nx_tex + 1) * (2 * max_ny_tex + 1))
    return _TextureKernels2D(
        kernel_mode="varying", radius_km=float(radius_km), shape=(ny, nx),
        k_ptr=k_ptr, k_jdx=k_jdx.astype(np.int64), k_jdy=k_jdy.astype(np.int64),
        row_kernel=row_kernel, ix0=ix0, ix1=ix1,
        scale_y=dy, scale_x=dx, max_kernel=int(max_kernel),
    )


@njit(cache=True, nogil=True)
def _fraction_active_rows(dbz_col_max: np.ndarray,
                          min_valid_dbz: float,
                          k_ptr: np.ndarray, k_jdx: np.ndarray,
                          k_jdy: np.ndarray, row_kernel: np.ndarray,
                          scale_y: np.ndarray, scale_x: np.ndarray,
                          radius_km: float,
                          ix0: int, ix1: int,
                          row_start: int, row_stop: int,
                          fraction: np.ndarray) -> None:
    """
    Coverage fraction for rows row_start <= iy < row_stop, written into
    `fraction` in place. Same count and division as
    _compute_fraction_active_core() (a stored window) or
    _compute_fraction_active_varying() (a ROW_PER_POINT row).
    """
    for iy in range(row_start, row_stop):
        kb = row_kernel[iy]
        if kb == ROW_SKIP:
            continue
        if kb >= 0:
            # Slice the stored window rather than indexing k_jdx[s:e] with
            # a loop from s: a loop index that starts at 0 lets Numba drop
            # its negative-index check on every read (~2x faster here).
            jdx_arr = k_jdx[k_ptr[kb]:k_ptr[kb + 1]]
            jdy_arr = k_jdy[k_ptr[kb]:k_ptr[kb + 1]]
            n_kernel = len(jdx_arr)
            if n_kernel == 0:
                continue
            for ix in range(ix0, ix1):
                count = 0
                for k in range(n_kernel):
                    if dbz_col_max[iy + jdy_arr[k], ix + jdx_arr[k]] >= min_valid_dbz:
                        count += 1
                fraction[iy, ix] = count / n_kernel
        else:
            for ix in range(ix0, ix1):
                jdx_arr, jdy_arr, _, _, _, _ = _point_kernel_offsets_arrays(
                    radius_km, scale_x[iy, ix], scale_y[iy, ix])
                n_kernel = len(jdx_arr)
                if n_kernel == 0:
                    continue
                count = 0
                for k in range(n_kernel):
                    if dbz_col_max[iy + jdy_arr[k], ix + jdx_arr[k]] >= min_valid_dbz:
                        count += 1
                fraction[iy, ix] = count / n_kernel


@njit(cache=True, nogil=True)
def _texture_at_point(dbz_level, iy, ix, jdx, jdy, dxp, dyp,
                      base_dbz, min_pts_texture, min_pts_fit,
                      vals, vxx, vyy) -> float:
    """
    Texture at one point from the window (jdx, jdy), or NaN where the 1.0
    cores leave texture missing.

    Line-for-line the per-point body of _compute_texture_one_level_core()
    (see its docstring for the plane-fit notes): the same gather order,
    the same sums in the same order, the same clamps. The only difference
    is that the plane-fit coordinates are formed here, as jdx*dxp and
    jdy*dyp, instead of being read from a precomputed xx/yy array; they
    are the same products the 1.0 kernel builders formed.
    """
    n_kernel = len(jdx)
    count = 0
    sum_dbz = 0.0
    for k in range(n_kernel):
        v = dbz_level[iy + jdy[k], ix + jdx[k]]
        if not np.isnan(v):
            vals[count] = v
            vxx[count] = jdx[k] * dxp
            vyy[count] = jdy[k] * dyp
            sum_dbz += v
            count += 1

    if count < min_pts_texture:
        return np.nan

    mean_dbz = sum_dbz / count
    if mean_dbz < 1.0:
        mean_dbz = 1.0

    if count >= min_pts_fit:
        Sx = 0.0; Sy = 0.0; Sxx = 0.0; Syy = 0.0; Sxy = 0.0
        Sz = 0.0; Sxz = 0.0; Syz = 0.0
        for i in range(count):
            x = vxx[i]; y = vyy[i]; z = vals[i]
            Sx += x; Sy += y
            Sxx += x * x; Syy += y * y; Sxy += x * y
            Sz += z; Sxz += x * z; Syz += y * z
        n = float(count)

        m00 = Sxx; m01 = Sxy; m02 = Sx
        m10 = Sxy; m11 = Syy; m12 = Sy
        m20 = Sx;  m21 = Sy;  m22 = n
        b0 = Sxz; b1 = Syz; b2 = Sz

        det = (m00 * (m11 * m22 - m12 * m21)
               - m01 * (m10 * m22 - m12 * m20)
               + m02 * (m10 * m21 - m11 * m20))

        if abs(det) > 1e-12:
            aa = (b0 * (m11 * m22 - m12 * m21)
                  - m01 * (b1 * m22 - m12 * b2)
                  + m02 * (b1 * m21 - m11 * b2)) / det
            bb = (m00 * (b1 * m22 - m12 * b2)
                  - b0 * (m10 * m22 - m12 * m20)
                  + m02 * (m10 * b2 - b1 * m20)) / det
            for i in range(count):
                vals[i] -= aa * vxx[i] + bb * vyy[i]
        # else: singular system - skip detrending, as in 1.0.

    sum_sq = 0.0
    sum_sq2 = 0.0
    nn = 0.0
    for i in range(count):
        val = vals[i] - base_dbz
        if val < 1.0:
            val = 1.0
        dbz_sq = val * val
        sum_sq += dbz_sq
        sum_sq2 += dbz_sq * dbz_sq
        nn += 1.0

    n_missing = n_kernel - count
    if n_missing > 0:
        min_sq = mean_dbz * mean_dbz
        sum_sq += n_missing * min_sq
        sum_sq2 += n_missing * min_sq * min_sq
        nn += n_missing

    if nn < 1:
        return np.nan

    mean_sq = sum_sq / nn
    var = sum_sq2 / nn - mean_sq * mean_sq
    if var < 0.0:
        var = 0.0
    return np.sqrt(np.sqrt(var))


@njit(cache=True, nogil=True)
def _texture_rows(dbz_level: np.ndarray,
                  fraction_active: np.ndarray,
                  k_ptr: np.ndarray, k_jdx: np.ndarray,
                  k_jdy: np.ndarray, row_kernel: np.ndarray,
                  scale_y: np.ndarray, scale_x: np.ndarray,
                  radius_km: float,
                  base_dbz: float, min_frac_texture: float,
                  min_frac_fit: float,
                  ix0: int, ix1: int,
                  row_start: int, row_stop: int,
                  max_kernel: int,
                  texture: np.ndarray) -> None:
    """
    Texture for rows row_start <= iy < row_stop of one level, written into
    `texture` (pre-filled with NaN) in place. Per point, identical to
    _compute_texture_one_level_core() (a stored window) or
    _compute_texture_one_level_varying() (a ROW_PER_POINT row).
    """
    vals = np.empty(max(max_kernel, 1), dtype=np.float64)
    vxx = np.empty(max(max_kernel, 1), dtype=np.float64)
    vyy = np.empty(max(max_kernel, 1), dtype=np.float64)

    for iy in range(row_start, row_stop):
        kb = row_kernel[iy]
        if kb == ROW_SKIP:
            continue
        if kb >= 0:
            _texture_row_stored(
                dbz_level, fraction_active, iy,
                k_jdx[k_ptr[kb]:k_ptr[kb + 1]], k_jdy[k_ptr[kb]:k_ptr[kb + 1]],
                scale_y, scale_x, base_dbz, min_frac_texture, min_frac_fit,
                ix0, ix1, vals, vxx, vyy, texture)
        else:
            _texture_row_per_point(
                dbz_level, fraction_active, iy, radius_km,
                scale_y, scale_x, base_dbz, min_frac_texture, min_frac_fit,
                ix0, ix1, vals, vxx, vyy, texture)


# The two row bodies are separate functions so that each calls
# _texture_at_point() from exactly one place, which lets LLVM inline it;
# with two call sites in one function it did not, costing ~25%.

@njit(cache=True, nogil=True)
def _texture_row_stored(dbz_level, fraction_active, iy, jdx_arr, jdy_arr,
                        scale_y, scale_x, base_dbz, min_frac_texture,
                        min_frac_fit, ix0, ix1, vals, vxx, vyy, texture):
    """One row of _texture_rows() using a stored window."""
    n_kernel = len(jdx_arr)
    if n_kernel == 0:
        return
    min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
    min_pts_fit = int(min_frac_fit * n_kernel + 0.5)
    for ix in range(ix0, ix1):
        if fraction_active[iy, ix] < min_frac_texture:
            continue
        if np.isnan(dbz_level[iy, ix]):
            continue
        tex = _texture_at_point(
            dbz_level, iy, ix, jdx_arr, jdy_arr,
            scale_x[iy, ix], scale_y[iy, ix], base_dbz,
            min_pts_texture, min_pts_fit, vals, vxx, vyy)
        if not np.isnan(tex):
            texture[iy, ix] = tex


@njit(cache=True, nogil=True)
def _texture_row_per_point(dbz_level, fraction_active, iy, radius_km,
                           scale_y, scale_x, base_dbz, min_frac_texture,
                           min_frac_fit, ix0, ix1, vals, vxx, vyy, texture):
    """One ROW_PER_POINT row of _texture_rows(): the 1.0 varying body."""
    for ix in range(ix0, ix1):
        if fraction_active[iy, ix] < min_frac_texture:
            continue
        if np.isnan(dbz_level[iy, ix]):
            continue
        dxp = scale_x[iy, ix]
        dyp = scale_y[iy, ix]
        jdx_arr, jdy_arr, _, _, _, _ = _point_kernel_offsets_arrays(
            radius_km, dxp, dyp)
        n_kernel = len(jdx_arr)
        if n_kernel == 0:
            continue
        min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
        min_pts_fit = int(min_frac_fit * n_kernel + 0.5)
        tex = _texture_at_point(
            dbz_level, iy, ix, jdx_arr, jdy_arr,
            dxp, dyp, base_dbz, min_pts_texture, min_pts_fit,
            vals, vxx, vyy)
        if not np.isnan(tex):
            texture[iy, ix] = tex


def refl_texture_2d(dbz: np.ndarray,
                    texture_radius: Union["WindowSpec", float],
                    dy: np.ndarray,
                    dx: np.ndarray,
                    base_dbz: float = 0.0,
                    min_valid_dbz: float = 0.0,
                    min_frac_texture: float = 0.25,
                    min_frac_fit: float = 0.67,
                    n_threads: int = 1,
                    kernel_mode: str = "uniform") -> tuple:
    """
    2-D circular-neighbourhood texture with planar detrend, per horizontal
    level, for EccoPy-3D / EccoPy-2D-H.

    Port of ConvStratFinder::ComputeTexture::run() from lrose-core, with
    a choice of how to handle spatially-varying grid spacing (relevant
    for e.g. lat/lon grids, where dx shrinks with cos(latitude) but dy
    stays roughly constant):

    kernel_mode="uniform" (default)
        Builds ONE kernel from a single representative dx_km/dy_km
        (the median of the supplied dy/dx arrays), reused at every
        point. This is what the validated LROSE ConvStratFinder
        algorithm itself does for lat/lon grids - it computes one
        dxKm/dyKm pair at the domain's mean latitude and applies it
        everywhere, rather than adjusting per point. Fast (one kernel
        built once); matches validated behaviour; becomes increasingly
        approximate as the domain's latitude range grows (the kernel
        is the wrong physical size away from the representative
        latitude). Appropriate for radar-sized domains (up to a few
        hundred km) or any grid where spacing doesn't vary much.

    kernel_mode="varying"
        Resolves the kernel from the LOCAL dy[y,x]/dx[y,x] at every
        grid point, so the kernel's physical footprint stays correct
        everywhere regardless of domain size. This is NOT what LROSE
        does, and has not been validated against LROSE output - it is a
        genuine algorithmic extension for domains where the single-
        kernel approximation breaks down (e.g. continental or global
        lat/lon grids). Each distinct kernel is built once and reused:
        a row whose spacing admits only one kernel - every row of a
        regular lat/lon grid, whose spacing changes only with latitude -
        shares it, and costs about what "uniform" does. A row whose
        spacing changes the kernel along the row rebuilds it at every
        point, as EccoPy 1.0 did everywhere, which is much slower. The
        results are identical to 1.0's either way.

    Parameters
    ----------
    dbz : np.ndarray, shape (nz, ny, nx)
        Reflectivity volume. NaN where missing.
    texture_radius : WindowSpec or float (km)
    dy, dx : np.ndarray, shape (ny, nx)
        Local point-to-point spacing in km along y and x respectively
        (e.g. from eccopy.core.coords.latlon_to_xy_spacing, or derived
        from coordinate arrays via resolve_spacing()).
    base_dbz : float
    min_valid_dbz : float
    min_frac_texture : float
    min_frac_fit : float
    n_threads : int
        Threads for the computation. Rows are split into blocks that are
        processed in parallel, so a single level (nz=1) uses every
        thread too. Results are identical for any thread count.
    kernel_mode : {"uniform", "varying"}
        See above.

    Returns
    -------
    texture : np.ndarray, shape (nz, ny, nx)
    fraction_active : np.ndarray, shape (ny, nx)
    """
    if kernel_mode not in ("uniform", "varying"):
        raise ValueError(
            f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}"
        )

    dbz = np.asarray(dbz, dtype=float)
    if dbz.ndim != 3:
        raise ValueError(
            f"refl_texture_2d expects a 3-D (nz, ny, nx) dbz array; "
            f"got shape {dbz.shape}"
        )
    nz, ny, nx = dbz.shape
    dy = np.asarray(dy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if dy.shape != (ny, nx) or dx.shape != (ny, nx):
        raise ValueError(
            f"dy/dx must have shape (ny, nx) = {(ny, nx)}; "
            f"got dy.shape={dy.shape}, dx.shape={dx.shape}"
        )

    radius_km = _resolve_radius_km(texture_radius)
    kernels = _prepare_texture_2d(radius_km, dy, dx, kernel_mode)
    return _refl_texture_2d_prepared(
        dbz, kernels, base_dbz=base_dbz, min_valid_dbz=min_valid_dbz,
        min_frac_texture=min_frac_texture, min_frac_fit=min_frac_fit,
        n_threads=n_threads,
    )


def _refl_texture_2d_prepared(dbz: np.ndarray,
                              kernels: _TextureKernels2D,
                              base_dbz: float = 0.0,
                              min_valid_dbz: float = 0.0,
                              min_frac_texture: float = 0.25,
                              min_frac_fit: float = 0.67,
                              n_threads: int = 1) -> tuple:
    """
    refl_texture_2d() on windows already resolved by _prepare_texture_2d().

    `dbz` is (nz, ny, nx) with (ny, nx) == kernels.shape. Returns
    (texture, fraction_active) exactly as refl_texture_2d() does.
    """
    dbz = np.asarray(dbz, dtype=float)
    nz, ny, nx = dbz.shape
    if (ny, nx) != tuple(kernels.shape):
        raise ValueError(
            f"dbz grid {(ny, nx)} does not match the prepared grid "
            f"{tuple(kernels.shape)}"
        )

    # Null out dbz below min_valid_dbz before any downstream computation.
    # Port of ConvStratFinder::computeEchoType()'s prefiltering loop,
    # applied once to the whole volume before column-max, fraction_active
    # and per-level texture are computed from it. Gating only the
    # fraction_active coverage count would leave sub-threshold values
    # (e.g. -25 dBZ when min_valid_dbz=0) still participated as valid
    # neighbours in the texture plane-fit/variance calc, inflating texture
    # and convectivity near the edges of valid-data regions relative to
    # the C++ reference. Found via SPOL LOW truth comparison: ~191k pixels
    # where truth=Missing but EccoPy assigned a stratiform/mixed echo type.
    if min_valid_dbz > -np.inf:
        dbz = np.where(dbz < min_valid_dbz, np.nan, dbz)

    with warnings.catch_warnings():
        # An all-NaN column (no valid data at any level) is expected and
        # already handled below (replaced with -9999.0); suppress the
        # resulting "All-NaN slice encountered" RuntimeWarning from numpy.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        dbz_col_max = np.nanmax(dbz, axis=0)
    dbz_col_max = np.where(np.isnan(dbz_col_max), -9999.0, dbz_col_max)

    k = kernels
    fraction_active = np.zeros((ny, nx), dtype=np.float64)
    texture = np.full((nz, ny, nx), np.nan, dtype=np.float64)

    def _process_rows(rows):
        # Coverage then texture for one block of rows, every level. Texture
        # at (iy, ix) reads coverage only at (iy, ix) itself, so a block
        # never waits on another; each writes only its own rows.
        row_start, row_stop = rows
        _fraction_active_rows(
            dbz_col_max, float(min_valid_dbz), k.k_ptr, k.k_jdx, k.k_jdy,
            k.row_kernel, k.scale_y, k.scale_x, k.radius_km,
            k.ix0, k.ix1, row_start, row_stop, fraction_active,
        )
        for iz in range(nz):
            _texture_rows(
                dbz[iz], fraction_active, k.k_ptr, k.k_jdx, k.k_jdy,
                k.row_kernel, k.scale_y, k.scale_x, k.radius_km,
                float(base_dbz), float(min_frac_texture), float(min_frac_fit),
                k.ix0, k.ix1, row_start, row_stop, k.max_kernel, texture[iz],
            )

    n_threads = int(n_threads)
    if n_threads > 1 and ny > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(_process_rows, _row_blocks(ny, n_threads)))
    else:
        _process_rows((0, ny))

    return texture, fraction_active


def _row_blocks(ny: int, n_threads: int, per_thread: int = 8):
    """
    Split rows 0..ny into contiguous blocks for n_threads workers.

    Several blocks per thread, handed out as threads free up, keep the
    load even when echo (and so work) is concentrated in a few rows. The
    split never changes results: every point is computed independently,
    by the same arithmetic, whichever block it falls in.
    """
    n_blocks = max(1, min(ny, n_threads * per_thread))
    edges = np.linspace(0, ny, n_blocks + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]