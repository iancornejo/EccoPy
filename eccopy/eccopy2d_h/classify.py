"""
EccoPy-2D-H (Horizontal single-level / composite) public entry point.

Input shapes:
    dbz     : (Y, X)         - single horizontal level (e.g. composite)
    coords_y: (Y,) or (Y, X)  - N-S coordinate or spacing (km)
    coords_x: (X,) or (Y, X)  - E-W coordinate or spacing (km)

No sub-classification into shallow/mid/deep is possible on a single
horizontal level. Output echo type codes:
    1 = Stratiform,  2 = Mixed,  3 = Convective

Uses the same 2D radial texture + planar-detrend algorithm as eccopy3d
(a circular neighbourhood kernel with a best-fit plane removed), as
validated against LROSE ConvStratFinder output.

Classification engine
---------------------
Classification runs through the same 2-D dual-threshold clumping
architecture as EccoPy-3D (find_clumps_3d) rather than a morphological
enlarge/close/fill/erode path. See eccopy2d_h/clumping.py's module
docstring for what is shared with the 3-D path and where the two differ.

core/classification.py's class_basic_isotropic() remains available as a
simpler, non-clumping alternative. It is not called by this module, but
can be applied directly to the `convectivity` field this module returns.

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

Fixed grids
-----------
Everything run() derives from the coordinates -- the spacing, the texture
radius and the windows themselves -- depends only on the grid. A caller
classifying a stream of frames on one grid (an operational mosaic, a fixed
radar's Cartesian grid) can resolve it once with prepare_grid() and pass
the result to every run():

    grid = eccopy2d_h.prepare_grid(dy_km, dx_km, coord_mode="spacing",
                                   window=WindowSpec((7, 'km')),
                                   kernel_mode="varying")
    for dbz in frames:
        result = eccopy2d_h.run(dbz, grid=grid)

run(dbz, grid=prepare_grid(c_y, c_x, ...)) returns exactly what
run(dbz, c_y, c_x, ...) returns: the second form builds the same grid
internally on every call.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from ..core.coords import resolve_spacing
from ..core.texture import (_TextureKernels2D, _prepare_texture_2d,
                            _refl_texture_2d_prepared)
from ..core.convectivity import texture_to_convectivity_linear
from ..core.classification import assign_echo_type_2d
from ..eccopy2d_h.clumping import find_clumps_2d
from ..params.window import WindowSpec
from ..params import TextureParams, ClassificationParams

_DEFAULT_WINDOW = WindowSpec((7, 'km'))


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
    # plotting the window footprint (see core.colormaps.draw_window_ring()).
    # None only for a bare-pixel window on a unit-agnostic grid.
    texture_radius:  Optional[float] = None
    # Populated ONLY when run(..., return_intermediates=True) is passed;
    # None otherwise (default, zero extra cost -- see run()'s docstring,
    # this is SLOW, a plain Python double loop over every grid point).
    fitted_dbz:    Optional[np.ndarray] = None   # shape (Y, X)
    detrended_dbz: Optional[np.ndarray] = None   # shape (Y, X)


@dataclass(frozen=True, eq=False)
class Grid2DH:
    """
    A (Y, X) grid resolved once for eccopy2d_h.run(); see prepare_grid().

    Holds no reflectivity, so one Grid2DH serves every frame on the same
    grid. Treat it as read-only: its arrays are shared, not copied, and
    changing them after prepare_grid() returns invalidates the windows it
    resolved from them.
    """
    shape:          tuple          # (Y, X)
    spacing_y:      np.ndarray     # (Y, X) N-S spacing, km
    spacing_x:      np.ndarray     # (Y, X) E-W spacing, km
    window:         Union[WindowSpec, float]
    texture_radius: float          # km, resolved from `window`
    coord_mode:     str
    kernel_mode:    str
    _kernels:       _TextureKernels2D = field(repr=False)

    @property
    def n_windows(self) -> int:
        """Distinct texture windows the grid uses: 1 in "uniform" mode;
        in "varying" mode, one per distinct row window (23 on the full
        MRMS CONUS grid at 7 km). Rows whose window varies along the row
        are rebuilt per point and are not counted (see n_rows_per_point)."""
        return self._kernels.n_kernels

    @property
    def n_rows_per_point(self) -> int:
        """Rows whose spacing varies along x enough to change the window
        within the row; these rebuild the window at every point, as in
        EccoPy 1.0. Zero on regular lat/lon and Cartesian grids."""
        return self._kernels.n_rows_per_point


def _resolve_coords(coords, size, axis_2d, coord_mode, name):
    """One coordinate argument -> spacing, as run() always resolved it:
    a 1-D array along its own axis, a 2-D one along `axis_2d`. Returns
    (spacing, is_1d)."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != size:
            axis = "Y" if axis_2d == 0 else "X"
            raise ValueError(f"{name} length {arr.shape[0]} != {axis}={size}")
        return resolve_spacing(arr, axis=0, mode=coord_mode), True
    return resolve_spacing(arr, axis=axis_2d, mode=coord_mode), False


def prepare_grid(coords_y: Union[np.ndarray, list],
                 coords_x: Union[np.ndarray, list],
                 window: Union[WindowSpec, int, float] = _DEFAULT_WINDOW,
                 coord_mode: str = "auto",
                 kernel_mode: str = "uniform") -> Grid2DH:
    """
    Resolve a fixed (Y, X) grid once, for reuse by run(dbz, grid=...).

    Takes the grid arguments of run() -- with the same meanings and
    defaults -- and does every part of run() that depends only on them:
    spacing, the texture radius, and the texture windows. In
    kernel_mode="varying" that last part is the one worth saving: on a
    regular lat/lon grid the window depends only on the row, and
    prepare_grid() finds the handful of distinct windows once (see
    Grid2DH.n_windows).

    Parameters
    ----------
    coords_y, coords_x, window, coord_mode, kernel_mode
        As for run().

    Returns
    -------
    Grid2DH
        Pass it to run() as ``grid=``. Its shape is (len(coords_y),
        len(coords_x)) for 1-D coordinates, or the shape of a 2-D one.
    """
    if kernel_mode not in ("uniform", "varying"):
        raise ValueError(
            f"kernel_mode must be 'uniform' or 'varying', got {kernel_mode!r}"
        )
    cy = np.asarray(coords_y, dtype=float)
    cx = np.asarray(coords_x, dtype=float)
    if cy.ndim == 2:
        shape = cy.shape
    elif cx.ndim == 2:
        shape = cx.shape
    elif cy.ndim == 1 and cx.ndim == 1:
        shape = (cy.shape[0], cx.shape[0])
    else:
        raise ValueError(
            f"coords_y must be (Y,) or (Y, X) and coords_x (X,) or (Y, X); "
            f"got shapes {cy.shape} and {cx.shape}"
        )
    ny, nx = shape

    # --- resolve coords -> per-point spacing (km) ---
    sp_y, y_is_1d = _resolve_coords(cy, ny, 0, coord_mode, "coords_y")
    if y_is_1d:
        sp_y = np.broadcast_to(sp_y[:, np.newaxis], (ny, nx)).copy()
    sp_x, x_is_1d = _resolve_coords(cx, nx, 1, coord_mode, "coords_x")
    if x_is_1d:
        sp_x = np.broadcast_to(sp_x[np.newaxis, :], (ny, nx)).copy()
    if sp_y.shape != (ny, nx) or sp_x.shape != (ny, nx):
        raise ValueError(
            f"coords_y/coords_x describe different grids: spacing shapes "
            f"{sp_y.shape} and {sp_x.shape}"
        )

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

    kernels = _prepare_texture_2d(radius_km, sp_y, sp_x, kernel_mode)
    return Grid2DH(shape=(ny, nx), spacing_y=sp_y, spacing_x=sp_x,
                   window=window, texture_radius=radius_km,
                   coord_mode=coord_mode, kernel_mode=kernel_mode,
                   _kernels=kernels)


def run(dbz: Union[np.ndarray, list],
        coords_y: Optional[Union[np.ndarray, list]] = None,
        coords_x: Optional[Union[np.ndarray, list]] = None,
        window: Optional[Union[WindowSpec, int, float]] = None,
        coord_mode: Optional[str] = None,
        kernel_mode: Optional[str] = None,
        texture_params: Optional[TextureParams] = None,
        class_params: Optional[ClassificationParams] = None,
        min_convective_area: Optional[float] = None,
        n_threads: int = 1,
        return_intermediates: bool = False,
        grid: Optional[Grid2DH] = None) -> Result2DH:
    """
    Run EccoPy-2D-H: 2D radial texture → convectivity → 2D dual-threshold
    clumping → echo type classification.

    Parameters
    ----------
    dbz : array-like, shape (Y, X)
        Reflectivity, dBZ. NaN where missing.
    coords_y : array-like, shape (Y,) or (Y, X)
        N-S position or spacing, km. Required unless `grid` is given.
    coords_x : array-like, shape (X,) or (Y, X)
        E-W position or spacing, km. Required unless `grid` is given.
    window : WindowSpec or float (km), default WindowSpec((7, 'km'))
        Texture window radius. Must be a length unit (km); bare pixel
        WindowSpec is also accepted.
    coord_mode : {'auto', 'position', 'spacing'}, default 'auto'
    kernel_mode : {'uniform', 'varying'}, default 'uniform'
        How to handle spatially-varying grid spacing (e.g. lat/lon grids,
        where dx shrinks with cos(latitude)):
          'uniform' (default) - one representative kernel for the whole
              grid, built from the median dy/dx. Matches what the
              validated LROSE ConvStratFinder algorithm itself does for
              lat/lon grids (it uses a single dx/dy computed at the
              domain's mean latitude). Fast; becomes approximate for
              domains spanning a wide latitude range.
          'varying' - resolves the kernel from the local spacing at
              every grid point, so the physical kernel size stays
              correct across large/non-uniform domains. This is NOT
              what LROSE does and is unvalidated against LROSE output.
              Each distinct window is built once and reused (on a
              regular lat/lon grid the window depends only on the row),
              so on such grids it costs about the same as 'uniform';
              on a grid whose spacing varies along x the window is
              rebuilt at every point, as in EccoPy 1.0, which is much
              slower.
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
        Threads for the texture stage, the bulk of the run time. Rows
        are split across threads, so a single composite uses all of
        them. Results are identical for any thread count.
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

    if grid is None:
        if coords_y is None or coords_x is None:
            raise ValueError(
                "eccopy2d_h.run() needs coords_y and coords_x, or a grid "
                "from eccopy2d_h.prepare_grid()."
            )
        # 1-D coordinate lengths are checked against dbz here, where the
        # data shape is known; prepare_grid() takes its shape from them.
        for name, c, size, axis in (("coords_y", coords_y, ny, "Y"),
                                    ("coords_x", coords_x, nx, "X")):
            c = np.asarray(c)
            if c.ndim == 1 and c.shape[0] != size:
                raise ValueError(f"{name} length {c.shape[0]} != {axis}={size}")
        grid = prepare_grid(
            coords_y, coords_x,
            window=_DEFAULT_WINDOW if window is None else window,
            coord_mode="auto" if coord_mode is None else coord_mode,
            kernel_mode="uniform" if kernel_mode is None else kernel_mode,
        )
    else:
        if coords_y is not None or coords_x is not None:
            raise ValueError(
                "Pass either coords_y/coords_x or grid, not both: the grid "
                "already carries its coordinates."
            )
        given = [n for n, v in (("window", window), ("coord_mode", coord_mode),
                                ("kernel_mode", kernel_mode)) if v is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} cannot be passed with grid=: the grid "
                f"fixes them. Pass them to prepare_grid() instead."
            )
        if not isinstance(grid, Grid2DH):
            raise TypeError(
                f"grid must come from eccopy2d_h.prepare_grid(); got "
                f"{type(grid).__name__}"
            )
    if tuple(grid.shape) != (ny, nx):
        raise ValueError(
            f"dbz shape {(ny, nx)} does not match the grid's shape "
            f"{tuple(grid.shape)}"
        )
    sp_y, sp_x = grid.spacing_y, grid.spacing_x
    radius_km = grid.texture_radius

    # 1. 2D radial texture (one level; the texture stage takes (nz, ny, nx))
    texture_3d, fraction_active = _refl_texture_2d_prepared(
        dbz[np.newaxis, :, :],
        grid._kernels,
        base_dbz=tp.dbz_base,
        min_valid_dbz=tp.resolve_min_valid_dbz(0.0),
        min_frac_texture=tp.min_frac_texture,
        min_frac_fit=tp.min_frac_fit,
        n_threads=n_threads,
    )
    texture = texture_3d[0]   # back to (Y, X)

    # 2. Convectivity
    conv = texture_to_convectivity_linear(
        texture,
        upper_lim=tp.texture_limit_high,
        lower_lim=tp.texture_limit_low,
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

    # 4. Assign echo type codes - basic 1/2/3 only, no sub-typing
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
            base_dbz=tp.dbz_base, min_valid_dbz=tp.resolve_min_valid_dbz(0.0),
            min_frac_texture=tp.min_frac_texture, min_frac_fit=tp.min_frac_fit,
            kernel_mode=grid.kernel_mode,
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
