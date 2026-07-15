"""
EccoPy-2D-H (Horizontal single-level / composite) public entry point.

Input shapes:
    dbz     : (Y, X)         — single horizontal level (e.g. composite)
    coords_y: (Y,) or (Y, X)  — N-S coordinate or spacing (km)
    coords_x: (X,) or (Y, X)  — E-W coordinate or spacing (km)

No sub-classification into shallow/mid/deep is possible on a single
horizontal level. Output echo type codes:
    1 = Stratiform,  2 = Mixed,  3 = Convective

Uses the same 2D radial texture + planar-detrend algorithm as eccopy3d
(a circular neighbourhood kernel with a best-fit plane removed), as
validated against LROSE ConvStratFinder output.

Classification engine -- CHANGED this session
------------------------------------------------
Classification now runs through the SAME 2-D dual-threshold CLUMPING
architecture as EccoPy-3D (find_clumps_3d), not the older morphological
enlarge/close/fill/erode path (class_basic_isotropic). This is a
deliberate architectural change, not just an addition -- see
eccopy2d_h/clumping.py's module docstring for exactly what's reused from
the validated 3-D path (the Stage-2 splitting logic, which is already a
2-D computation there) versus what's new and unvalidated (area-based
filtering in place of volume, since a single level has no vertical
extent to draw a volume from).

class_basic_isotropic() itself still exists in core/classification.py
(now with the same border_value/sequential-closing fixes class_basic()
received) as a simpler, non-clumping alternative -- it is no longer
called by this module's default path, but nothing stops a caller from
using it directly against this module's `convectivity` output if the
clumping-based approach turns out not to be the right fit for a given
composite dataset.

Minimum clump area
--------------------
`min_convective_area`, if given (km²), drops any clump -- whether from
straightforward single-threshold labeling or a dual-threshold sub-split
-- smaller than this area, demoting its pixels to Mixed instead of
Convective. None (default) disables this floor: clumps are still found
and (if use_dual_thresholds) split, just never dropped for size alone.
Same opt-in convention as eccopy1d's min_convective_length, for the same
reason -- no validated default exists for composite data yet.

Typical usage
-------------
    from eccopy import eccopy2d_h
    from eccopy.params import WindowSpec

    result = eccopy2d_h.run(
        dbz_composite,
        coords_y=y_km, coords_x=x_km,
        window=WindowSpec((7, 'km')),
        min_convective_area=4.0,   # km^2, optional
    )
    echo = result.echo_type   # shape (Y, X), values in {1, 2, 3}
    n_clumps = result.n_clumps
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..core.coords import resolve_spacing
from ..core.texture import refl_texture_2d
from ..core.convectivity import texture_to_convectivity_linear
from ..core.classification import assign_echo_type_2d
from ..eccopy2d_h.clumping import find_clumps_2d
from ..params.window import WindowSpec
from ..params import TextureParams, ClassificationParams


@dataclass
class Result2DH:
    """Output of eccopy2d_h.run()."""
    echo_type:       np.ndarray   # shape (Y, X), int  {1=Strat, 2=Mixed, 3=Conv}
    convectivity:    np.ndarray   # shape (Y, X), float
    texture:         np.ndarray   # shape (Y, X), float
    fraction_active: np.ndarray   # shape (Y, X), float  (echo coverage fraction)
    n_clumps:        int
    # Physical radius (km) of the texture window actually used, resolved
    # from the `window` argument against the grid spacing. Available for
    # plotting the window footprint (see plot_result(show_window=True)).
    # None only for a bare-pixel window on a unit-agnostic grid.
    texture_radius:  Optional[float] = None
    # Populated ONLY when run(..., return_intermediates=True) is passed;
    # None otherwise (default, zero extra cost -- see run()'s docstring,
    # this is SLOW, a plain Python double loop over every grid point).
    fitted_dbz:    Optional[np.ndarray] = None   # shape (Y, X)
    detrended_dbz: Optional[np.ndarray] = None   # shape (Y, X)


def run(dbz: Union[np.ndarray, list],
        coords_y: Union[np.ndarray, list],
        coords_x: Union[np.ndarray, list],
        window: Union[WindowSpec, int, float] = WindowSpec((7, 'km')),
        coord_mode: str = "auto",
        kernel_mode: str = "uniform",
        texture_params: Optional[TextureParams] = None,
        class_params: Optional[ClassificationParams] = None,
        min_convective_area: Optional[float] = None,
        n_threads: int = 1,
        return_intermediates: bool = False) -> Result2DH:
    """
    Run EccoPy-2D-H: 2D radial texture → convectivity → 2D dual-threshold
    clumping → echo type classification.

    Parameters
    ----------
    dbz : array-like, shape (Y, X)
        Reflectivity, dBZ. NaN where missing.
    coords_y : array-like, shape (Y,) or (Y, X)
        N-S position or spacing, km.
    coords_x : array-like, shape (X,) or (Y, X)
        E-W position or spacing, km.
    window : WindowSpec or float (km)
        Texture window radius. Must be a length unit (km); bare pixel
        WindowSpec is also accepted.
    coord_mode : {'auto', 'position', 'spacing'}
    kernel_mode : {'uniform', 'varying'}
        How to handle spatially-varying grid spacing (e.g. lat/lon grids,
        where dx shrinks with cos(latitude)):
          'uniform' (default) — one representative kernel for the whole
              grid, built from the median dy/dx. Matches what the
              validated LROSE ConvStratFinder algorithm itself does for
              lat/lon grids (it uses a single dx/dy computed at the
              domain's mean latitude). Fast; becomes approximate for
              domains spanning a wide latitude range.
          'varying' — rebuilds the kernel from the local spacing at
              every grid point, so the physical kernel size stays
              correct across large/non-uniform domains. This is NOT
              what LROSE does and is unvalidated against LROSE output;
              substantially slower (no kernel reuse). Recommended only
              for domains where 'uniform' is known to break down (e.g.
              continental-scale lat/lon grids).
    texture_params : TextureParams, optional
    class_params : ClassificationParams, optional
        use_dual_thresholds, secondary_convectivity,
        all_subclumps_min_area_frac, each_subclump_min_area_frac, and
        each_subclump_min_area_km2 are reused directly from here for the
        2-D clumping step -- see eccopy2d_h/clumping.py's module
        docstring for how each_subclump_min_area_km2's meaning differs
        (genuine area) from its 3-D usage. CAUTION: because of that
        difference, a ClassificationParams instance tuned/validated
        against eccopy3d cases should NOT be reused as-is here -- the
        same each_subclump_min_area_km2 value gates a different physical
        quantity in each module. Tune it separately for eccopy2d_h.
    min_convective_area : float, optional
        Minimum clump area, km². Clumps (whether unsplit or emitted by a
        dual-threshold split) smaller than this are dropped -- their
        pixels become Mixed rather than Convective. None (default)
        disables this floor entirely; see module docstring.
    n_threads : int
        Number of parallel threads for the texture computation. Has no
        effect here since the input is a single level, but accepted for
        API consistency with eccopy3d.
    return_intermediates : bool
        If True, also compute and attach `fitted_dbz` and `detrended_dbz`
        (the plane-fit value and detrended value at each point's own
        location, prior to texture -- see
        core.debug.refl_texture_2d_field_debug()). This is SLOW: it is a
        plain Python double loop over every grid point (unlike
        eccopy1d/eccopy2d_v's return_intermediates, which reuses the
        fast Numba core) -- expect it to take much longer than the rest
        of run() combined on anything beyond a small grid. Default False
        skips it entirely at no cost.

    Returns
    -------
    Result2DH
    """
    dbz = np.asarray(dbz, dtype=float)
    if dbz.ndim != 2:
        raise ValueError(
            f"eccopy2d_h expects a 2-D (Y, X) dbz array; got shape {dbz.shape}"
        )
    ny, nx = dbz.shape

    tp = texture_params or TextureParams()
    cp = class_params   or ClassificationParams()

    # --- resolve coords → per-point spacing (km) ---
    coords_y = np.asarray(coords_y, dtype=float)
    coords_x = np.asarray(coords_x, dtype=float)

    if coords_y.ndim == 1:
        if coords_y.shape[0] != ny:
            raise ValueError(f"coords_y length {coords_y.shape[0]} != Y={ny}")
        sp_y_1d = resolve_spacing(coords_y, axis=0, mode=coord_mode)
        sp_y = np.broadcast_to(sp_y_1d[:, np.newaxis], (ny, nx)).copy()
    else:
        sp_y = resolve_spacing(coords_y, axis=0, mode=coord_mode)

    if coords_x.ndim == 1:
        if coords_x.shape[0] != nx:
            raise ValueError(f"coords_x length {coords_x.shape[0]} != X={nx}")
        sp_x_1d = resolve_spacing(coords_x, axis=0, mode=coord_mode)
        sp_x = np.broadcast_to(sp_x_1d[np.newaxis, :], (ny, nx)).copy()
    else:
        sp_x = resolve_spacing(coords_x, axis=1, mode=coord_mode)

    # refl_texture_2d expects shape (nz, ny, nx); wrap the single level
    dbz_3d = dbz[np.newaxis, :, :]   # (1, Y, X)

    # Resolve window radius in km
    if isinstance(window, (int, float)):
        radius_km = float(window)
    elif isinstance(window, WindowSpec):
        if window.is_pixel:
            # Convert pixel radius to km using median spacing
            med_sp = float(np.nanmedian(np.stack([sp_y, sp_x])))
            radius_km = window.size * med_sp
        else:
            if window.base_kind != "length_m":
                raise ValueError(
                    "eccopy2d_h texture window must be a length unit (km); "
                    f"got unit kind '{window.base_kind}'."
                )
            radius_km = window.base_value / 1000.0
    else:
        radius_km = float(window)

    # 1. 2D radial texture (one level)
    texture_3d, fraction_active = refl_texture_2d(
        dbz_3d,
        texture_radius=radius_km,
        dy=sp_y,
        dx=sp_x,
        base_dbz=tp.dbz_base,
        min_valid_dbz=tp.min_valid_dbz,
        min_frac_texture=tp.min_frac_texture,
        min_frac_fit=tp.min_frac_fit,
        n_threads=1,
        kernel_mode=kernel_mode,
    )
    texture = texture_3d[0]   # back to (Y, X)

    # 2. Convectivity
    conv = texture_to_convectivity_linear(
        texture,
        upper_lim=tp.texture_limit_high,
    )

    # 3. 2D dual-threshold clumping (see eccopy2d_h/clumping.py)
    clumps = find_clumps_2d(
        conv,
        spacing_y=sp_y,
        spacing_x=sp_x,
        min_conv=cp.min_convectivity_for_convective,
        min_area_km2=min_convective_area,
        use_dual_thresholds=cp.use_dual_thresholds,
        secondary_threshold=cp.secondary_convectivity,
        all_subclumps_min_area_frac=cp.all_subclumps_min_area_frac,
        each_subclump_min_area_frac=cp.each_subclump_min_area_frac,
        each_subclump_min_area_km2=cp.each_subclump_min_area_km2,
    )

    # 4. Assign echo type codes — basic 1/2/3 only, no sub-typing
    # (see core.classification.assign_echo_type_2d docstring).
    echo = assign_echo_type_2d(
        conv,
        clumps=clumps,
        max_conv_for_strat=cp.max_convectivity_for_stratiform,
    )

    fitted_dbz = detrended_dbz = None
    if return_intermediates:
        from ..core.debug import refl_texture_2d_field_debug
        fitted_dbz, detrended_dbz, _ = refl_texture_2d_field_debug(
            dbz, radius_km, dy=sp_y, dx=sp_x,
            base_dbz=tp.dbz_base, min_valid_dbz=tp.min_valid_dbz,
            min_frac_texture=tp.min_frac_texture, min_frac_fit=tp.min_frac_fit,
            kernel_mode=kernel_mode,
        )

    return Result2DH(
        echo_type=echo,
        convectivity=conv,
        texture=texture,
        fraction_active=fraction_active,
        n_clumps=len(clumps),
        texture_radius=radius_km,
        fitted_dbz=fitted_dbz,
        detrended_dbz=detrended_dbz,
    )
