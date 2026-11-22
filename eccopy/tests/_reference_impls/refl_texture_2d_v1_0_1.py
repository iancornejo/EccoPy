"""
Frozen copy of EccoPy 1.0.1's refl_texture_2d() driver, for cross-validation
only. DO NOT use in production.

EccoPy 1.1 replaced this driver with prepared windows reused across rows
(core.texture._prepare_texture_2d / _texture_rows). The 1.1 path is
required to reproduce this one bit for bit; the tests in
test_texture_prepared.py compare the two on random and real grids.

It calls the 1.0.1 Numba cores, which remain in eccopy.core.texture
unchanged and are themselves cross-validated against the frozen pure-Python
references in texture_2d_reference.py (see test_numba_cross_validation.py).
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np

from eccopy.core.texture import (
    _build_kernel_offsets_uniform_arrays,
    _compute_fraction_active_core,
    _compute_fraction_active_varying,
    _compute_texture_one_level_core,
    _compute_texture_one_level_varying,
    _max_half_width,
)


def refl_texture_2d_v1_0_1(dbz: np.ndarray,
                    texture_radius: Union["WindowSpec", float],
                    dy: np.ndarray,
                    dx: np.ndarray,
                    base_dbz: float = 0.0,
                    min_valid_dbz: float = 0.0,
                    min_frac_texture: float = 0.25,
                    min_frac_fit: float = 0.67,
                    n_threads: int = 1,
                    kernel_mode: str = "uniform") -> tuple:
    """Frozen copy of eccopy 1.0.1 refl_texture_2d(); see module docstring."""
    if kernel_mode not in ("uniform", "varying"):
        raise ValueError(
            f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}"
        )

    dbz = np.asarray(dbz, dtype=float)
    nz, ny, nx = dbz.shape
    dy = np.asarray(dy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if dy.shape != (ny, nx) or dx.shape != (ny, nx):
        raise ValueError(
            f"dy/dx must have shape (ny, nx) = {(ny, nx)}; "
            f"got dy.shape={dy.shape}, dx.shape={dx.shape}"
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

    # Resolve the texture radius in km.
    from eccopy.params.window import WindowSpec as _WS
    if isinstance(texture_radius, (int, float)):
        radius_km = float(texture_radius)
    elif isinstance(texture_radius, _WS):
        if texture_radius.is_pixel:
            raise ValueError(
                "refl_texture_2d requires a physical (km) radius, not a "
                "bare pixel-count WindowSpec. Pass a float (km) or "
                "WindowSpec((radius, 'km'))."
            )
        # WindowSpec stores its physical size internally in base units
        # (metres for length); convert to km for this function's use.
        radius_km = texture_radius._base_value / 1000.0
    else:
        radius_km = float(texture_radius)

    dbz_yx = np.transpose(dbz, (1, 2, 0))  # (ny, nx, nz)
    with warnings.catch_warnings():
        # An all-NaN column (no valid data at any level) is expected and
        # already handled below (replaced with -9999.0); suppress the
        # resulting "All-NaN slice encountered" RuntimeWarning from numpy.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        dbz_col_max = np.nanmax(dbz_yx, axis=2)
    dbz_col_max = np.where(np.isnan(dbz_col_max), -9999.0, dbz_col_max)

    if kernel_mode == "uniform":
        # Single representative dx_km/dy_km for the whole grid - matches
        # LROSE's own mean-latitude approach for lat/lon grids.
        dy_km = float(np.nanmedian(dy))
        dx_km = float(np.nanmedian(dx))

        # Built once as typed arrays (not the list-of-tuples form used
        # by eccopy.core.debug) and reused across every z-level below -
        # avoids rebuilding/reconverting the kernel per level.
        jdx_arr, jdy_arr, xx_arr, yy_arr, nx_tex, ny_tex = (
            _build_kernel_offsets_uniform_arrays(radius_km, dx_km, dy_km)
        )

        fraction_active = _compute_fraction_active_core(
            dbz_col_max, jdx_arr, jdy_arr, nx_tex, ny_tex, min_valid_dbz
        )

        def _process_level(iz):
            level = dbz_yx[:, :, iz].copy()
            t = _compute_texture_one_level_core(
                level, fraction_active, jdx_arr, jdy_arr, xx_arr, yy_arr,
                nx_tex, ny_tex, base_dbz, min_frac_texture, min_frac_fit
            )
            return iz, t

    else:  # kernel_mode == "varying"
        max_nx_tex, max_ny_tex = _max_half_width(radius_km, dx, dy)

        # Warn if the texture radius is smaller than the local grid
        # spacing anywhere in the domain - the kernel degenerates to just
        # the center point there, making texture meaningless (always 0)
        # at those locations regardless of kernel_mode.
        coarsest_dx = float(np.nanmax(dx[dx > 0])) if np.any(dx > 0) else np.nan
        coarsest_dy = float(np.nanmax(dy[dy > 0])) if np.any(dy > 0) else np.nan
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
                stacklevel=2,
            )

        fraction_active = _compute_fraction_active_varying(
            dbz_col_max, radius_km, dx, dy, min_valid_dbz,
            max_nx_tex, max_ny_tex,
        )

        def _process_level(iz):
            level = dbz_yx[:, :, iz].copy()
            t = _compute_texture_one_level_varying(
                level, fraction_active, radius_km, dx, dy,
                base_dbz, min_frac_texture, min_frac_fit,
                max_nx_tex, max_ny_tex,
            )
            return iz, t

    texture_yx = np.full((ny, nx, nz), np.nan, dtype=float)

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            for iz, t in ex.map(_process_level, range(nz)):
                texture_yx[:, :, iz] = t
    else:
        for iz in range(nz):
            _, t = _process_level(iz)
            texture_yx[:, :, iz] = t

    texture = np.transpose(texture_yx, (2, 0, 1))  # (nz, ny, nx)
    return texture, fraction_active