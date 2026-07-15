"""
Debug / introspection helpers for EccoPy's texture calculations.

These functions are NOT used by the production pipeline (refl_texture_1d,
refl_texture_2d) — they exist purely so you can inspect the intermediate
quantities that the validated production code computes and discards:
the raw windowed data, the fitted trend, and the detrended residual.

Use these when you want to understand or sanity-check WHY a particular
point got the texture value it did, not as a substitute for the
production functions (which are faster and operate on the whole array
at once).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..params.window import WindowSpec
from .texture import _fillmissing_linear, _radius_field_along_axis, refl_texture_2d


@dataclass
class TextureDebug1D:
    """Intermediate quantities for one point's 1-D texture calculation."""
    index:           int                # index in the original (unpadded) array
    radius:          int                # window half-width used at this point (grid cells)
    window_indices:  np.ndarray         # original-array indices included in the window
    window_dbz:      np.ndarray         # raw dBZ values in the window (NaN-filled if gaps existed)
    fit_x:           np.ndarray         # 1..W, the x-coordinate used for the linear fit
    fit_slope:       float              # fitted line slope (b)
    fit_intercept:   float              # fitted line intercept (a)
    fitted_line:     np.ndarray         # a + b*x — the trend that gets removed
    mean_dbz:        float              # mean of window_dbz (added back after detrending)
    detrended:       np.ndarray         # window_dbz - fitted_line + mean_dbz, clipped to >= 1
    texture:         float              # sqrt(population_std(detrended**2))


def refl_texture_1d_debug(dbz: np.ndarray,
                          window: Union[WindowSpec, int],
                          index: int,
                          spacing: Optional[np.ndarray] = None,
                          dbz_base: float = 0.0) -> TextureDebug1D:
    """
    Run the EccoPy-1D texture calculation for ONE point, returning every
    intermediate quantity instead of just the final texture value.

    This reuses the exact same fill/pad/detrend math as refl_texture_1d
    (same NaN-filling, same window resolution from WindowSpec) so the
    `texture` value returned here will exactly match
    `refl_texture_1d(dbz, window, spacing)[index]`.

    Parameters
    ----------
    dbz : np.ndarray, shape (N,)
        Same input you would pass to refl_texture_1d.
    window : WindowSpec or int
        Same window you would pass to refl_texture_1d.
    index : int
        Which point (0-based, into the ORIGINAL unpadded array) to inspect.
    spacing : np.ndarray, optional
        Same spacing array you would pass to refl_texture_1d (already
        converted to the window's base unit — e.g. metres for a
        WindowSpec((_, 'km')) — exactly as eccopy1d.run() does internally;
        see that function if you want to replicate it from km coords).
    dbz_base : float
        Same dbz_base you would pass to refl_texture_1d.

    Returns
    -------
    TextureDebug1D
    """
    dbz = np.asarray(dbz, dtype=float)
    if dbz.ndim != 1:
        raise ValueError(f"refl_texture_1d_debug expects a 1-D array; got shape {dbz.shape}")
    n = dbz.shape[0]
    if not (0 <= index < n):
        raise ValueError(f"index {index} out of range for array of length {n}")

    radius_field = _radius_field_along_axis(window, spacing, n)
    r = int(radius_field[index])
    pad = int(radius_field.max()) if radius_field.size else 0

    # Production re-masks texture to NaN at any point where the ORIGINAL
    # input was NaN, even though the window around it gets gap-filled to
    # compute OTHER points' texture. Reproduce that here so this debug
    # function never disagrees with production at originally-missing points.
    original_was_nan = bool(np.isnan(dbz[index]))

    padded = np.pad(dbz[np.newaxis, :], ((0, 0), (pad, pad)),
                    mode="constant", constant_values=np.nan)
    padded = _fillmissing_linear(padded, axis=1)
    padded = padded - dbz_base

    if original_was_nan:
        filled_value = float(padded[0, pad + index])
        return TextureDebug1D(
            index=index, radius=r,
            window_indices=np.array([index]),
            window_dbz=np.array([dbz[index]]),   # NaN, the true raw value
            fit_x=np.array([1.0]), fit_slope=0.0, fit_intercept=filled_value,
            fitted_line=np.array([filled_value]), mean_dbz=filled_value,
            detrended=np.array([filled_value]),
            texture=float("nan"),   # matches production's final re-mask
        )

    if r < 1:
        return TextureDebug1D(
            index=index, radius=r,
            window_indices=np.array([index]),
            window_dbz=np.array([dbz[index]]),
            fit_x=np.array([1.0]), fit_slope=0.0, fit_intercept=float(dbz[index]),
            fitted_line=np.array([dbz[index]]), mean_dbz=float(dbz[index]),
            detrended=np.array([max(dbz[index] - dbz_base, 1.0)]),
            texture=0.0,
        )

    lo = pad + index - r
    hi = pad + index + r + 1
    block = padded[0, lo:hi]              # (W,) — the raw (gap-filled) window
    W = block.shape[0]

    window_indices = np.arange(index - r, index + r + 1)

    x = np.arange(1, W + 1, dtype=float)
    sumX = np.sum(x)
    sumY = np.nansum(block)
    sumXY = np.nansum(block * x)
    sumX2 = np.sum(x ** 2)
    denom = W * sumX2 - sumX ** 2

    if denom == 0:
        a, b = 0.0, 0.0
    else:
        a = (sumY * sumX2 - sumX * sumXY) / denom
        b = (W * sumXY - sumX * sumY) / denom

    fitted_line = a + b * x
    mean_block = np.nanmean(block)
    detrended = block - fitted_line + mean_block
    detrended = np.where(detrended < 1, 1.0, detrended)

    # Sample variance (N-1 denominator) of detrended**2, NaNs dropped --
    # matches refl_texture_1d's numba core EXACTLY (see that function's
    # "Sample variance (N-1 denominator)" comment for the validation
    # history: this was found against real MATLAB output to be N-1, not
    # N, moving downstream classification agreement from 98.2% to 99.4%).
    # This debug function previously used np.nanstd(..., ddof=0) --
    # population variance -- which was NEVER updated when that fix went
    # into the production core, silently breaking this function's core
    # promise of matching refl_texture_1d(...)[index] exactly. Caught by
    # re-running test_debug.py after the fact, not by the fix itself.
    finite = detrended[~np.isnan(detrended)] ** 2
    n_valid_pts = finite.size
    if n_valid_pts > 1:
        var = float(np.var(finite, ddof=1))
        if var < 0.0:
            var = 0.0
    elif n_valid_pts == 1:
        var = 0.0   # matches MATLAB's std() of a single sample == 0
    else:
        var = float("nan")
    texture = float(np.sqrt(np.sqrt(var)))

    return TextureDebug1D(
        index=index, radius=r,
        window_indices=window_indices,
        window_dbz=block.copy(),
        fit_x=x, fit_slope=float(b), fit_intercept=float(a),
        fitted_line=fitted_line,
        mean_dbz=float(mean_block),
        detrended=detrended,
        texture=texture,
    )


@dataclass
class TextureDebug2D:
    """Intermediate quantities for one point's 2-D radial texture calculation."""
    iy: int
    ix: int
    n_kernel:         int     # total points in the circular kernel footprint
    n_valid:          int     # non-NaN points actually used
    fraction_active:  float   # coverage fraction at this point
    kernel_dx:        np.ndarray  # x-offset (km) of each valid kernel point
    kernel_dy:        np.ndarray  # y-offset (km) of each valid kernel point
    kernel_dbz:       np.ndarray  # raw dBZ at each valid kernel point
    plane_fit_used:   bool        # whether enough points existed to fit a plane
    plane_a:          float      # fitted plane: value = a*dx + b*dy
    plane_b:          float
    detrended:        np.ndarray  # kernel_dbz with the fitted plane removed
    mean_dbz:         float       # mean dBZ used for missing-point substitution
    texture:          float       # sqrt(sqrt(population_variance(corrected**2)))


def refl_texture_2d_debug(dbz_level: np.ndarray,
                          texture_radius_km: float,
                          dy: np.ndarray,
                          dx: np.ndarray,
                          iy: int,
                          ix: int,
                          base_dbz: float = 0.0,
                          min_valid_dbz: float = 0.0,
                          min_frac_texture: float = 0.25,
                          min_frac_fit: float = 0.67,
                          kernel_mode: str = "uniform",
                          dbz_col_max: Optional[np.ndarray] = None) -> TextureDebug2D:
    """
    Run the EccoPy-3D / EccoPy-2D-H radial texture calculation for ONE
    point on ONE level, returning every intermediate quantity instead of
    just the final texture value.

    Parameters
    ----------
    dbz_level : np.ndarray, shape (ny, nx)
        A single horizontal level (e.g. dbz[iz] from a 3-D array, or the
        whole array for EccoPy-2D-H).
    texture_radius_km : float
        Same radius (in km) you would resolve from your WindowSpec.
    dy, dx : np.ndarray, shape (ny, nx)
        Same spacing arrays you would pass to refl_texture_2d.
    iy, ix : int
        Which point to inspect.
    base_dbz, min_frac_texture, min_frac_fit : float
        Same parameters you would pass to refl_texture_2d.
    min_valid_dbz : float
        Same parameter you would pass to refl_texture_2d. Used (together
        with dbz_col_max) to compute fraction_active exactly as
        production does — NOT simply "is this kernel point non-NaN".
    kernel_mode : {"uniform", "varying"}
        Whether to build the kernel from this point's local spacing
        ("varying") or from the grid's median spacing ("uniform") — see
        refl_texture_2d's kernel_mode documentation. Matching whichever
        mode you used in the real run will reproduce its exact kernel.
    dbz_col_max : np.ndarray, shape (ny, nx), optional
        The column-maximum reflectivity used to compute fraction_active.
        For genuinely single-level data (EccoPy-2D-H, or a 3-D volume
        with only one level), this equals dbz_level and may be omitted.
        For one level WITHIN a 3-D volume, fraction_active is computed
        from the FULL volume's column-max, not from this level alone —
        pass that column-max array here (np.nanmax(dbz_3d, axis=0)) to
        get a fraction_active that matches production exactly. If
        omitted, dbz_level is used as a stand-in, which is only exact
        for single-level data.

    Returns
    -------
    TextureDebug2D
        If (iy, ix) falls within the border excluded by production (width
        ny_tex/nx_tex cells from each edge — the kernel would extend past
        the array there), this returns texture=NaN with empty kernel
        arrays, matching production's behaviour exactly: that border is
        never computed regardless of whether enough valid data exists,
        because the kernel can't be built at all near the edge.
        fraction_active is reported as 0.0 at border points (not NaN),
        matching production's fraction_active array, which is
        initialised to all-zero and never written to within the border.
    """
    from .texture import _build_kernel_offsets_uniform, _point_kernel_offsets

    dbz_level = np.asarray(dbz_level, dtype=float)
    dy = np.asarray(dy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    ny, nx = dbz_level.shape

    if dbz_col_max is None:
        dbz_col_max = dbz_level
    else:
        dbz_col_max = np.asarray(dbz_col_max, dtype=float)
        if dbz_col_max.shape != (ny, nx):
            raise ValueError(
                f"dbz_col_max shape {dbz_col_max.shape} must match "
                f"dbz_level shape {(ny, nx)}"
            )

    if kernel_mode == "uniform":
        dy_km = float(np.nanmedian(dy))
        dx_km = float(np.nanmedian(dx))
        offsets, nx_tex, ny_tex = _build_kernel_offsets_uniform(texture_radius_km, dx_km, dy_km)
    elif kernel_mode == "varying":
        offsets, nx_tex, ny_tex = _point_kernel_offsets(texture_radius_km, dx[iy, ix], dy[iy, ix])
    else:
        raise ValueError(f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}")

    # The production functions (_compute_texture_one_level /
    # _compute_texture_one_level_varying) hard-exclude a border of width
    # ny_tex/nx_tex from computation entirely — points there are left NaN
    # even if the kernel COULD be clipped to fit, because the production
    # loop's range() never visits them. Reproduce that here so this debug
    # function never silently disagrees with production at edge points.
    if not (ny_tex <= iy < ny - ny_tex) or not (nx_tex <= ix < nx - nx_tex):
        return TextureDebug2D(
            iy=iy, ix=ix, n_kernel=len(offsets), n_valid=0,
            fraction_active=0.0,
            kernel_dx=np.array([]), kernel_dy=np.array([]), kernel_dbz=np.array([]),
            plane_fit_used=False, plane_a=0.0, plane_b=0.0,
            detrended=np.array([]), mean_dbz=float("nan"), texture=float("nan"),
        )

    n_kernel = len(offsets)

    # fraction_active is computed from dbz_col_max >= min_valid_dbz over
    # the kernel footprint — this is what production's _compute_fraction_
    # active() does, computed once from the column-max field (which, for
    # one level within a 3-D volume, is NOT the same as this level's own
    # valid-point count; see the dbz_col_max parameter note above).
    n_active = 0
    for (jdx, jdy, _, _) in offsets:
        py, px = iy + jdy, ix + jdx
        if 0 <= py < ny and 0 <= px < nx:
            cm = dbz_col_max[py, px]
            if not np.isnan(cm) and cm >= min_valid_dbz:
                n_active += 1
    fraction_active = n_active / n_kernel if n_kernel else 0.0

    # Production's FIRST gate: fraction_active < min_frac_texture -> skip
    # entirely (this happens BEFORE the center-NaN check).
    if fraction_active < min_frac_texture:
        return TextureDebug2D(
            iy=iy, ix=ix, n_kernel=n_kernel, n_valid=0,
            fraction_active=fraction_active,
            kernel_dx=np.array([]), kernel_dy=np.array([]), kernel_dbz=np.array([]),
            plane_fit_used=False, plane_a=0.0, plane_b=0.0,
            detrended=np.array([]), mean_dbz=float("nan"), texture=float("nan"),
        )

    # Production's SECOND gate: the center value itself is NaN -> skip.
    if np.isnan(dbz_level[iy, ix]):
        return TextureDebug2D(
            iy=iy, ix=ix, n_kernel=n_kernel, n_valid=0,
            fraction_active=fraction_active,
            kernel_dx=np.array([]), kernel_dy=np.array([]), kernel_dbz=np.array([]),
            plane_fit_used=False, plane_a=0.0, plane_b=0.0,
            detrended=np.array([]), mean_dbz=float("nan"), texture=float("nan"),
        )

    vals, xx_list, yy_list = [], [], []
    sum_dbz = 0.0
    for (jdx, jdy, xx, yy) in offsets:
        py, px = iy + jdy, ix + jdx
        if 0 <= py < ny and 0 <= px < nx:
            v = dbz_level[py, px]
            if not np.isnan(v):
                vals.append(v); xx_list.append(xx); yy_list.append(yy)
                sum_dbz += v

    n_valid = len(vals)

    # Production's THIRD gate: this level's own valid-kernel-point count
    # (NOT fraction_active) must meet min_pts_texture, computed from
    # min_frac_texture * n_kernel — same threshold value, different
    # variable than the fraction_active check above.
    min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
    if n_valid < min_pts_texture:
        return TextureDebug2D(
            iy=iy, ix=ix, n_kernel=n_kernel, n_valid=n_valid,
            fraction_active=fraction_active,
            kernel_dx=np.array([]), kernel_dy=np.array([]), kernel_dbz=np.array([]),
            plane_fit_used=False, plane_a=0.0, plane_b=0.0,
            detrended=np.array([]), mean_dbz=float("nan"), texture=float("nan"),
        )

    mean_dbz = max(sum_dbz / n_valid, 1.0)
    vals_arr = np.array(vals)
    xx_arr = np.array(xx_list)
    yy_arr = np.array(yy_list)

    min_pts_fit = int(min_frac_fit * n_kernel + 0.5)
    plane_fit_used = n_valid >= min_pts_fit
    aa = bb = 0.0
    detrended = vals_arr.copy()

    if plane_fit_used:
        A = np.column_stack([xx_arr, yy_arr, np.ones(n_valid)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, vals_arr, rcond=None)
            aa, bb = float(coeffs[0]), float(coeffs[1])
            detrended = vals_arr - (aa * xx_arr + bb * yy_arr)
        except np.linalg.LinAlgError:
            plane_fit_used = False

    clipped = np.maximum(detrended - base_dbz, 1.0)
    sum_sq = float(np.sum(clipped ** 2))
    sum_sq2 = float(np.sum(clipped ** 4))
    nn = float(n_valid)

    n_missing = n_kernel - n_valid
    if n_missing > 0:
        min_sq = mean_dbz * mean_dbz
        sum_sq += n_missing * min_sq
        sum_sq2 += n_missing * min_sq * min_sq
        nn += n_missing

    min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
    if n_valid < min_pts_texture or nn < 1:
        texture = float("nan")
    else:
        mean_sq = sum_sq / nn
        var = max(sum_sq2 / nn - mean_sq * mean_sq, 0.0)
        texture = float(np.sqrt(np.sqrt(var)))

    return TextureDebug2D(
        iy=iy, ix=ix, n_kernel=n_kernel, n_valid=n_valid,
        fraction_active=fraction_active,
        kernel_dx=xx_arr, kernel_dy=yy_arr, kernel_dbz=vals_arr,
        plane_fit_used=plane_fit_used, plane_a=aa, plane_b=bb,
        detrended=detrended, mean_dbz=mean_dbz, texture=texture,
    )


# ---------------------------------------------------------------------------
# Full-field intermediate output for the 2-D radial (plane-fit) texture path
# ---------------------------------------------------------------------------

def refl_texture_2d_field_debug(dbz_level: np.ndarray,
                                texture_radius_km: float,
                                dy: np.ndarray,
                                dx: np.ndarray,
                                base_dbz: float = 0.0,
                                min_valid_dbz: float = 0.0,
                                min_frac_texture: float = 0.25,
                                min_frac_fit: float = 0.67,
                                kernel_mode: str = "uniform",
                                dbz_col_max: Optional[np.ndarray] = None):
    """
    DEBUG-ONLY, whole-LEVEL counterpart of refl_texture_2d_debug(): loops
    refl_texture_2d_debug() over every (iy, ix) in `dbz_level` and
    assembles full-field arrays of the fitted plane's centre-point value
    ("fitted dbz prior to texture") and the detrended value that feeds
    the texture statistic, plus fraction_active -- for the 2-D radial
    plane-fit texture used by EccoPy-3D and EccoPy-2D-H.

    This is intentionally implemented by calling the already-validated
    per-point refl_texture_2d_debug() at every point rather than adding
    a new Numba code path (unlike the 1-D case's
    refl_texture_1d_with_fit(), which was cheap enough to duplicate as a
    fast core) -- correctness here leans entirely on
    refl_texture_2d_debug() already being cross-validated against
    refl_texture_2d() point-by-point (see test_debug.py). It is a plain
    Python double loop and is SLOW relative to refl_texture_2d itself
    (which is Numba-JIT'd) -- expect this to take much longer than the
    production texture call on the same array. It exists for debugging
    a specific level or region, not as something to run routinely on
    full 3-D volumes; see `eccopy3d.run(..., return_intermediates=True)`,
    which calls this only for the levels you ask it to.

    Parameters
    ----------
    Same as refl_texture_2d_debug, but dbz_level (and dbz_col_max, if
    given) is the FULL (ny, nx) level and no single (iy, ix) is picked.

    Returns
    -------
    fitted_dbz, detrended_dbz, fraction_active : np.ndarray, shape (ny, nx)
        fitted_dbz    -- the point's own value with the fitted plane's
                         local contribution removed (production's
                         `detrended` value evaluated at that point's own
                         kernel offset, i.e. the index of
                         (dx, dy) == (0, 0) within kernel_dx/kernel_dy)
                         subtracted from the point's own detrended value
                         -- NaN wherever refl_texture_2d_debug reports
                         texture=NaN (border, low coverage, or a NaN
                         centre pixel).
        detrended_dbz -- the centre pixel's own detrended value (same
                         condition as fitted_dbz).
        fraction_active : np.ndarray, shape (ny, nx)
            Always populated (0.0 in the border, matching production's
            fraction_active array -- see refl_texture_2d_debug's Returns
            docstring).
    """
    dbz_level = np.asarray(dbz_level, dtype=float)
    ny, nx = dbz_level.shape

    fitted_dbz = np.full((ny, nx), np.nan)
    detrended_dbz = np.full((ny, nx), np.nan)
    fraction_active = np.zeros((ny, nx))

    for iy in range(ny):
        for ix in range(nx):
            dbg = refl_texture_2d_debug(
                dbz_level, texture_radius_km, dy, dx, iy, ix,
                base_dbz=base_dbz, min_valid_dbz=min_valid_dbz,
                min_frac_texture=min_frac_texture, min_frac_fit=min_frac_fit,
                kernel_mode=kernel_mode, dbz_col_max=dbz_col_max,
            )
            fraction_active[iy, ix] = dbg.fraction_active
            if dbg.detrended.size == 0:
                continue
            center = np.where((dbg.kernel_dx == 0) & (dbg.kernel_dy == 0))[0]
            if center.size == 0:
                continue
            c = center[0]
            fitted_dbz[iy, ix] = dbg.kernel_dbz[c] - dbg.detrended[c]
            detrended_dbz[iy, ix] = dbg.detrended[c]

    return fitted_dbz, detrended_dbz, fraction_active
