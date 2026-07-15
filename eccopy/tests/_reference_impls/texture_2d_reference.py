"""
Frozen pre-Numba reference implementations of the 2D radial texture
functions, for cross-validation only. DO NOT use these in production —
they exist solely so the Numba-converted versions can be checked against
exactly what the code did before conversion.
"""
import numpy as np


def _build_kernel_offsets_uniform_reference(radius_km: float, dx_km: float, dy_km: float):
    """
    Pre-compute kernel offsets for a UNIFORM-spacing approximation,
    identical to ConvStratFinder::_computeKernels(). LROSE itself uses
    this approach even for lat/lon grids: a single dx_km/dy_km computed
    once at the domain's mean latitude (see refl_texture_2d's
    kernel_mode="uniform").
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


def _max_half_width_reference(radius_km: float, dx_km: np.ndarray, dy_km: np.ndarray):
    """
    Conservative (worst-case) half-width in grid cells needed to contain
    a `radius_km` circle anywhere on a varying-spacing grid — used to
    size the border that must be excluded from per-point kernel mode
    (the finest spacing anywhere in the grid determines the largest
    possible cell count).

    Uses the same floor(r/spacing + 0.5) convention as
    _build_kernel_offsets_uniform_reference / _point_kernel_offsets_reference, so that on a
    perfectly uniform grid this produces exactly the same margin as the
    uniform-kernel path (kernel_mode="uniform" and "varying" should
    agree exactly on a uniform grid).
    """
    min_dy = float(np.nanmin(dy_km[dy_km > 0])) if np.any(dy_km > 0) else radius_km
    min_dx = float(np.nanmin(dx_km[dx_km > 0])) if np.any(dx_km > 0) else radius_km
    ny_tex = max(1, int(np.floor(radius_km / min_dy + 0.5)))
    nx_tex = max(1, int(np.floor(radius_km / min_dx + 0.5)))
    return nx_tex, ny_tex


def _point_kernel_offsets_reference(radius_km: float, dx_km_pt: float, dy_km_pt: float):
    """Build kernel offsets for ONE point's local spacing (per-point mode)."""
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


def _compute_fraction_active_varying_reference(dbz_col_max: np.ndarray,
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
    """
    ny, nx = dbz_col_max.shape
    fraction = np.zeros((ny, nx), dtype=float)

    for iy in range(max_ny_tex, ny - max_ny_tex):
        for ix in range(max_nx_tex, nx - max_nx_tex):
            offsets, _, _ = _point_kernel_offsets_reference(
                radius_km, dx_km[iy, ix], dy_km[iy, ix]
            )
            n_kernel = len(offsets)
            if n_kernel == 0:
                continue
            count = 0
            for (jdx, jdy, xx, yy) in offsets:
                if dbz_col_max[iy + jdy, ix + jdx] >= min_valid_dbz:
                    count += 1
            fraction[iy, ix] = count / n_kernel

    return fraction


def _compute_texture_one_level_varying_reference(dbz_level: np.ndarray,
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
    Per-point-kernel version of _compute_texture_one_level_reference: the circular
    kernel footprint is rebuilt from the LOCAL dx_km[y,x]/dy_km[y,x] at
    every point, so the physical kernel radius stays consistent even
    when grid spacing varies substantially across the domain (e.g. a
    lat/lon grid spanning a wide latitude range, where dx shrinks with
    cos(latitude) but dy stays roughly constant).

    This is NOT what the validated LROSE ConvStratFinder algorithm does
    (LROSE uses a single dx_km/dy_km for the whole grid, computed at the
    domain's mean latitude — see _build_kernel_offsets_uniform_reference). Results
    from this mode have not been validated against LROSE output and may
    differ from it on grids where LROSE's single-kernel approximation
    breaks down.
    """
    ny, nx = dbz_level.shape
    texture = np.full((ny, nx), np.nan, dtype=float)

    for iy in range(max_ny_tex, ny - max_ny_tex):
        for ix in range(max_nx_tex, nx - max_nx_tex):
            if fraction_active[iy, ix] < min_frac_texture:
                continue
            center_val = dbz_level[iy, ix]
            if np.isnan(center_val):
                continue

            offsets, _, _ = _point_kernel_offsets_reference(
                radius_km, dx_km[iy, ix], dy_km[iy, ix]
            )
            n_kernel = len(offsets)
            if n_kernel == 0:
                continue

            min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
            min_pts_fit = int(min_frac_fit * n_kernel + 0.5)

            vals, xx_list, yy_list = [], [], []
            sum_dbz = 0.0
            count = 0
            for (jdx, jdy, xx, yy) in offsets:
                v = dbz_level[iy + jdy, ix + jdx]
                if not np.isnan(v):
                    vals.append(v); xx_list.append(xx); yy_list.append(yy)
                    sum_dbz += v; count += 1

            if count < min_pts_texture:
                continue

            mean_dbz = max(sum_dbz / count, 1.0)

            if count >= min_pts_fit:
                xx_arr = np.array(xx_list); yy_arr = np.array(yy_list)
                vv_arr = np.array(vals)
                A = np.column_stack([xx_arr, yy_arr, np.ones(count)])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, vv_arr, rcond=None)
                    aa, bb = coeffs[0], coeffs[1]
                    for ii in range(count):
                        vals[ii] -= aa * xx_list[ii] + bb * yy_list[ii]
                except np.linalg.LinAlgError:
                    pass

            nn = sum_sq = sum_sq2 = 0.0
            for v in vals:
                val = max(v - base_dbz, 1.0)
                dbz_sq = val * val
                sum_sq += dbz_sq; sum_sq2 += dbz_sq * dbz_sq; nn += 1.0

            n_missing = n_kernel - count
            if n_missing > 0:
                min_sq = mean_dbz * mean_dbz
                sum_sq += n_missing * min_sq
                sum_sq2 += n_missing * min_sq * min_sq
                nn += n_missing

            if nn < 1:
                continue

            mean_sq = sum_sq / nn
            var = max(sum_sq2 / nn - mean_sq * mean_sq, 0.0)
            tex = np.sqrt(np.sqrt(var))
            if not np.isnan(tex):
                texture[iy, ix] = tex

    return texture


def _compute_fraction_active_reference(dbz_col_max: np.ndarray,
                              offsets: list,
                              nx_tex: int,
                              ny_tex: int,
                              min_valid_dbz: float) -> np.ndarray:
    """Coverage fraction array from column-max DBZ (uniform-grid kernel)."""
    ny, nx = dbz_col_max.shape
    n_kernel = len(offsets)
    fraction = np.zeros((ny, nx), dtype=float)

    for iy in range(ny_tex, ny - ny_tex):
        for ix in range(nx_tex, nx - nx_tex):
            center = iy * nx + ix
            count = 0
            for (jdx, jdy, xx, yy) in offsets:
                jj = center + jdx + jdy * nx
                if dbz_col_max.flat[jj] >= min_valid_dbz:
                    count += 1
            fraction[iy, ix] = count / n_kernel

    return fraction


def _compute_texture_one_level_reference(dbz_level: np.ndarray,
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
    """
    ny, nx = dbz_level.shape
    n_kernel = len(offsets)
    texture = np.full((ny, nx), np.nan, dtype=float)

    min_pts_texture = int(min_frac_texture * n_kernel + 0.5)
    min_pts_fit = int(min_frac_fit * n_kernel + 0.5)

    for iy in range(ny_tex, ny - ny_tex):
        for ix in range(nx_tex, nx - nx_tex):
            if fraction_active[iy, ix] < min_frac_texture:
                continue
            center_val = dbz_level[iy, ix]
            if np.isnan(center_val):
                continue

            vals, xx_list, yy_list = [], [], []
            sum_dbz = 0.0
            count = 0
            for (jdx, jdy, xx, yy) in offsets:
                v = dbz_level[iy + jdy, ix + jdx]
                if not np.isnan(v):
                    vals.append(v); xx_list.append(xx); yy_list.append(yy)
                    sum_dbz += v; count += 1

            if count < min_pts_texture:
                continue

            mean_dbz = max(sum_dbz / count, 1.0)

            if count >= min_pts_fit:
                xx_arr = np.array(xx_list); yy_arr = np.array(yy_list)
                vv_arr = np.array(vals)
                A = np.column_stack([xx_arr, yy_arr, np.ones(count)])
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, vv_arr, rcond=None)
                    aa, bb = coeffs[0], coeffs[1]
                    for ii in range(count):
                        vals[ii] -= aa * xx_list[ii] + bb * yy_list[ii]
                except np.linalg.LinAlgError:
                    pass

            nn = sum_sq = sum_sq2 = 0.0
            for v in vals:
                val = max(v - base_dbz, 1.0)
                dbz_sq = val * val
                sum_sq += dbz_sq; sum_sq2 += dbz_sq * dbz_sq; nn += 1.0

            n_missing = n_kernel - count
            if n_missing > 0:
                min_sq = mean_dbz * mean_dbz
                sum_sq += n_missing * min_sq
                sum_sq2 += n_missing * min_sq * min_sq
                nn += n_missing

            if nn < 1:
                continue

            mean_sq = sum_sq / nn
            var = max(sum_sq2 / nn - mean_sq * mean_sq, 0.0)
            tex = np.sqrt(np.sqrt(var))
            if not np.isnan(tex):
                texture[iy, ix] = tex

    return texture


