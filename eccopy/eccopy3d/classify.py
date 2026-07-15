"""
EccoPy-3D public entry point.

Input shapes:
    dbz     : (Z, Y, X)
    coords_z: (Z,) or (Z, Y, X)  — vertical positions/spacing (km)
    coords_y: (Y,) or (Z, Y, X)  — N-S positions/spacing (km)
    coords_x: (X,) or (Z, Y, X)  — E-W positions/spacing (km)
    height  : (Z, Y, X), optional — height field, km
    temp    : (Z, Y, X), optional — temperature field, °C

Output echo type codes:
    Without height/temp — basic clumping only:
        1 = Stratiform,  2 = Mixed,  3 = Convective
    With height or temp — sub-classified:
        14 = Stratiform Low,   16 = Stratiform Mid,  18 = Stratiform High
        25 = Mixed
        32 = Convective Elevated,  34 = Convective Shallow,
        36 = Convective Mid,       38 = Convective Deep

Typical usage
-------------
    from eccopy import eccopy3d
    from eccopy.params import WindowSpec

    result = eccopy3d.run(
        dbz,
        coords_z=z_km, coords_y=y_km, coords_x=x_km,
        height=height_km,
        window=WindowSpec((7, 'km')),
    )
    echo = result.echo_type    # shape (Z, Y, X)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..core.coords import resolve_spacing
from ..core.texture import refl_texture_2d
from ..core.convectivity import texture_to_convectivity_linear
from ..core.classification import set_echo_type_3d
from ..core.temperature import broadcast_temp_field
from ..eccopy3d.clumping import find_clumps_3d
from ..params.window import WindowSpec
from ..params import TextureParams, ClassificationParams, VerticalParams


@dataclass
class Result3D:
    """Output of eccopy3d.run()."""
    echo_type:       np.ndarray   # shape (Z, Y, X)
    convectivity:    np.ndarray   # shape (Z, Y, X)
    texture:         np.ndarray   # shape (Z, Y, X)
    fraction_active: np.ndarray   # shape (Y, X)
    n_clumps:        int
    # Physical radius (km) of the texture window actually used, resolved
    # from the `window` argument against the grid spacing. Available for
    # plotting the window footprint (see plot_result(show_window=True)).
    # None only for a bare-pixel window on a unit-agnostic grid.
    texture_radius:  Optional[float] = None
    # Populated ONLY when run(..., return_intermediates=True, levels=[...])
    # is passed; None otherwise (default, zero extra cost). Computed via
    # a slow, plain-Python per-point loop (core.debug.refl_texture_2d_
    # field_debug) -- see that function and run()'s `levels` parameter.
    fitted_dbz:    Optional[np.ndarray] = None   # shape (len(levels), Y, X)
    detrended_dbz: Optional[np.ndarray] = None   # shape (len(levels), Y, X)
    intermediate_levels: Optional[np.ndarray] = None   # the Z-indices fitted_dbz/detrended_dbz correspond to


def run(dbz: Union[np.ndarray, list],
        coords_z: Union[np.ndarray, list],
        coords_y: Union[np.ndarray, list],
        coords_x: Union[np.ndarray, list],
        height: Optional[Union[np.ndarray, list]] = None,
        temp: Optional[Union[np.ndarray, list]] = None,
        topo: Optional[Union[np.ndarray, list]] = None,
        terrain_ht: Optional[Union[np.ndarray, list]] = None,
        window: Union[WindowSpec, int, float] = WindowSpec((7, 'km')),
        coord_mode: str = "auto",
        kernel_mode: str = "uniform",
        texture_params: Optional[TextureParams] = None,
        class_params: Optional[ClassificationParams] = None,
        vert_params: Optional[VerticalParams] = None,
        n_threads: int = 1,
        return_intermediates: bool = False,
        levels: Optional[Union[list, np.ndarray]] = None) -> Result3D:
    """
    Run EccoPy-3D: 2D radial texture → convectivity → 3D clumping →
    echo type classification.

    Parameters
    ----------
    dbz : array-like, shape (Z, Y, X)
        Reflectivity, dBZ. NaN where missing.
    coords_z : array-like, shape (Z,) or (Z, Y, X)
        Vertical coordinate or spacing, km.
    coords_y : array-like, shape (Y,) or (Z, Y, X)
        N-S coordinate or spacing, km.
    coords_x : array-like, shape (X,) or (Z, Y, X)
        E-W coordinate or spacing, km.
    height : array-like, shape (Z, Y, X), optional
        Height field, km. AGL preferred; MSL acceptable if terrain flat.
        If provided, enables sub-classification into Low/Mid/High and
        Shallow/Mid/Deep/Elevated convective codes.
    temp : array-like, shape (Z, Y, X) or (Z,), optional
        Temperature field, °C. Used if `height` not provided. Accepts
        either a full (Z, Y, X) field, or a single vertical profile of
        shape (Z,) — e.g. a sounding — which is broadcast identically
        across every horizontal point.
    window : WindowSpec or float (km)
        Texture window radius. Must be a length unit.
    coord_mode : {'auto', 'position', 'spacing'}
        How to interpret coord arrays.
    kernel_mode : {'uniform', 'varying'}
        How to handle spatially-varying horizontal grid spacing (e.g.
        lat/lon grids, where dx shrinks with cos(latitude)):
          'uniform' (default) — one representative kernel for the whole
              grid, built from the median dy/dx at each level. Matches
              what the validated LROSE ConvStratFinder algorithm itself
              does for lat/lon grids (a single dx/dy computed at the
              domain's mean latitude). Fast; becomes approximate for
              domains spanning a wide latitude range.
          'varying' — rebuilds the kernel from the local spacing at
              every grid point, so the physical kernel size stays
              correct across large/non-uniform domains. This is NOT
              what LROSE does and is unvalidated against LROSE output;
              substantially slower. Recommended only for domains where
              'uniform' is known to break down (e.g. continental-scale
              lat/lon grids).
    texture_params, class_params, vert_params : optional param objects
    n_threads : int
        Parallel threads for per-level texture computation.
    return_intermediates : bool
        If True, also compute and attach `fitted_dbz` and `detrended_dbz`
        (the plane-fit value and detrended value at each point's own
        location, prior to texture) for the Z-levels listed in `levels`
        -- see core.debug.refl_texture_2d_field_debug(). This is SLOW (a
        plain Python double loop per level); default False skips it
        entirely at no cost. Ignored if `levels` is None/empty.
    levels : list or np.ndarray of int, optional
        Which Z-indices to compute fitted_dbz/detrended_dbz for when
        return_intermediates=True. Required (and must be non-empty) to
        actually get intermediates back -- there is no "all levels"
        default, since that would silently run the slow per-point loop
        over the entire volume. Ignored if return_intermediates=False.

    Returns
    -------
    Result3D
    """
    dbz = np.asarray(dbz, dtype=float)
    if dbz.ndim != 3:
        raise ValueError(
            f"eccopy3d expects a 3-D (Z, Y, X) dbz array; got shape {dbz.shape}"
        )
    nz, ny, nx = dbz.shape

    tp = texture_params or TextureParams()
    cp = class_params   or ClassificationParams()
    vp = vert_params    or VerticalParams()

    # --- resolve coords → per-point spacing (km) ---
    def _resolve_1d_or_nd(arr, axis, size, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != size:
                raise ValueError(f"{name} length {arr.shape[0]} != {size}")
            sp_1d = resolve_spacing(arr, axis=0, mode=coord_mode)
            return sp_1d
        return resolve_spacing(arr, axis=axis, mode=coord_mode)

    sp_z_1d = _resolve_1d_or_nd(coords_z, 0, nz, "coords_z")
    sp_y_1d = _resolve_1d_or_nd(coords_y, 0, ny, "coords_y")
    sp_x_1d = _resolve_1d_or_nd(coords_x, 0, nx, "coords_x")

    # Build 2D (Y, X) spacing arrays for texture calculation
    sp_y_2d = np.broadcast_to(sp_y_1d[:, np.newaxis], (ny, nx)).copy()
    sp_x_2d = np.broadcast_to(sp_x_1d[np.newaxis, :], (ny, nx)).copy()

    # --- dz for 3-D cell-volume / clump geometry: use the EXACT
    # ConvStratFinder/PjgGridGeom::dzKm() centered-difference convention
    # (iz=0: z[1]-z[0]; iz=last: z[last]-z[last-1]; else average of the
    # two neighbouring gaps) rather than resolve_spacing()'s simple
    # forward-difference. These are IDENTICAL for uniform Z grids (SPOL,
    # SEA) but diverge for non-uniform ones (e.g. WRF: 0.25 km near-
    # surface, coarsening aloft) -- a real bug found by comparing against
    # StormClump::computeGeom() in the C++ source. Scoped to this 3-D
    # volume-geometry path only; does NOT touch resolve_spacing() itself,
    # so the already-validated 2D-V texture-kernel spacing is unaffected.
    coords_z_arr = np.asarray(coords_z, dtype=float)
    dz_z_1d = sp_z_1d
    if coords_z_arr.ndim == 1 and coords_z_arr.shape[0] == nz and nz > 1:
        diffs = np.diff(coords_z_arr)
        finite = diffs[np.isfinite(diffs)]
        if finite.size and (np.all(finite > 0) or np.all(finite < 0)):
            dz_z_1d = np.empty(nz, dtype=float)
            dz_z_1d[0] = coords_z_arr[1] - coords_z_arr[0]
            dz_z_1d[-1] = coords_z_arr[-1] - coords_z_arr[-2]
            if nz > 2:
                dz_z_1d[1:-1] = (coords_z_arr[2:] - coords_z_arr[:-2]) / 2.0
            dz_z_1d = np.abs(dz_z_1d)

    # Build 3D (Z, Y, X) spacing array for clumping volume calculation.
    # Use geometric mean of the three axis spacings as a representative
    # per-cell size (assumes roughly cuboid cells; adequate for volume
    # filtering at the precision needed by the 30 km³ minimum).
    sp_z_3d = np.broadcast_to(
        dz_z_1d[:, np.newaxis, np.newaxis], (nz, ny, nx)
    ).copy()
    sp_y_3d = np.broadcast_to(
        sp_y_1d[np.newaxis, :, np.newaxis], (nz, ny, nx)
    ).copy()
    sp_x_3d = np.broadcast_to(
        sp_x_1d[np.newaxis, np.newaxis, :], (nz, ny, nx)
    ).copy()
    # Per-cell volume = dz * dy * dx; store geometric mean as "spacing"
    # so _compute_geom()'s  spacing**3  gives the correct cell volume.
    sp_3d = np.cbrt(sp_z_3d * sp_y_3d * sp_x_3d)

    # Resolve texture radius in km
    if isinstance(window, (int, float)):
        radius_km = float(window)
    elif isinstance(window, WindowSpec):
        if window.is_pixel:
            med_sp = float(np.nanmedian(np.stack([sp_y_2d, sp_x_2d])))
            radius_km = window.size * med_sp
        else:
            if window._base_kind != "length_m":
                raise ValueError(
                    "eccopy3d texture window must be a length unit (km); "
                    f"got unit kind '{window._base_kind}'."
                )
            radius_km = window._base_value / 1000.0
    else:
        radius_km = float(window)

    # 1. 2D radial texture per level
    texture, fraction_active = refl_texture_2d(
        dbz,
        texture_radius=radius_km,
        dy=sp_y_2d,
        dx=sp_x_2d,
        base_dbz=tp.dbz_base,
        min_valid_dbz=tp.min_valid_dbz,
        min_frac_texture=tp.min_frac_texture,
        min_frac_fit=tp.min_frac_fit,
        n_threads=n_threads,
        kernel_mode=kernel_mode,
    )

    # 2. Convectivity
    conv = texture_to_convectivity_linear(
        texture,
        upper_lim=tp.texture_limit_high,
    )

    # 3. 3D clumping
    height_arr = np.asarray(height, dtype=float) if height is not None else None
    temp_arr = broadcast_temp_field(temp, dbz.shape) if temp is not None else None
    topo_arr = np.asarray(topo, dtype=float) if topo is not None else None
    terrain_ht_arr = np.asarray(terrain_ht, dtype=float) if terrain_ht is not None else None

    clumps = find_clumps_3d(
        conv,
        spacing=sp_3d,
        min_conv=cp.min_convectivity_for_convective,
        min_vol_km3=cp.min_valid_volume_for_convective,
        height_km=height_arr,
        temp=temp_arr,
        topo_km=topo_arr,
        shallow_threshold_ht=vp.shallow_threshold_ht,
        deep_threshold_ht=vp.deep_threshold_ht,
        shallow_threshold_temp=vp.shallow_threshold_temp,
        deep_threshold_temp=vp.deep_threshold_temp,
        use_dual_thresholds=cp.use_dual_thresholds,
        secondary_threshold=cp.secondary_convectivity,
        all_subclumps_min_area_frac=cp.all_subclumps_min_area_frac,
        each_subclump_min_area_frac=cp.each_subclump_min_area_frac,
        each_subclump_min_area_km2=cp.each_subclump_min_area_km2,
        dx_km=float(np.median(sp_x_1d)),
        dy_km=float(np.median(sp_y_1d)),
        dz0_km=float(dz_z_1d[0]),
    )

    # 4. Assign echo type codes
    echo = set_echo_type_3d(
        conv,
        clumps=clumps,
        height_km=height_arr,
        temp=temp_arr,
        shallow_threshold_ht=vp.shallow_threshold_ht,
        deep_threshold_ht=vp.deep_threshold_ht,
        shallow_threshold_temp=vp.shallow_threshold_temp,
        deep_threshold_temp=vp.deep_threshold_temp,
        topo_km=topo_arr,
        terrain_ht_km=terrain_ht_arr,
        min_ht_agl_for_mid=cp.min_ht_km_agl_for_mid,
        min_ht_agl_for_deep=cp.min_ht_km_agl_for_deep,
        max_conv_for_strat=cp.max_convectivity_for_stratiform,
        min_conv_for_conv=cp.min_convectivity_for_convective,
        min_vol_km3=cp.min_valid_volume_for_convective,
        min_vert_extent_km=cp.min_vert_extent_for_convective,
        min_conv_frac_deep=cp.min_conv_fraction_for_deep,
        min_conv_frac_shallow=cp.min_conv_fraction_for_shallow,
        max_shallow_frac_elevated=cp.max_shallow_conv_fraction_for_elevated,
        max_deep_frac_elevated=cp.max_deep_conv_fraction_for_elevated,
        min_strat_frac_strat_below=cp.min_strat_fraction_for_strat_below,
    )

    fitted_dbz = detrended_dbz = intermediate_levels = None
    if return_intermediates and levels is not None and len(levels) > 0:
        from ..core.debug import refl_texture_2d_field_debug
        intermediate_levels = np.asarray(levels, dtype=int)
        fitted_list, detrended_list = [], []
        for iz in intermediate_levels:
            f, d, _ = refl_texture_2d_field_debug(
                dbz[iz], radius_km, dy=sp_y_2d, dx=sp_x_2d,
                base_dbz=tp.dbz_base, min_valid_dbz=tp.min_valid_dbz,
                min_frac_texture=tp.min_frac_texture, min_frac_fit=tp.min_frac_fit,
                kernel_mode=kernel_mode,
            )
            fitted_list.append(f)
            detrended_list.append(d)
        fitted_dbz = np.stack(fitted_list, axis=0)
        detrended_dbz = np.stack(detrended_list, axis=0)

    return Result3D(
        echo_type=echo,
        convectivity=conv,
        texture=texture,
        fraction_active=fraction_active,
        n_clumps=len(clumps),
        texture_radius=radius_km,
        fitted_dbz=fitted_dbz,
        detrended_dbz=detrended_dbz,
        intermediate_levels=intermediate_levels,
    )
