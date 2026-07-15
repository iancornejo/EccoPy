"""
EccoPy-1D public entry point.

Input shape: (T,) or (N,) — one-dimensional array along time or distance.
Output:      echo_type array of the same shape, with integer codes:
               1 = Stratiform
               2 = Mixed
               3 = Convective
             (No sub-classification is possible without a vertical axis.)

Classification engine
----------------------
The strat/mixed/conv classification itself is NOT a separate 1-D
implementation -- it reshapes `conv` to (1, N) and calls the exact same
class_basic() used (and validated against real MATLAB ECCO-V output) by
EccoPy-2D-V. This works correctly, not just "runs without erroring":
class_basic()'s disk-based morphology can only see a structuring
element's CENTER ROW when the input has one row (every other row offset
points off-array), and that center row is confirmed full-width at every
validated disk radius (3, 5, 15, 25) -- so the (1, N) reshape degenerates
exactly to 1-D horizontal dilation/erosion/closing at those radii, with
no separate 1-D port to maintain or independently validate. There is
currently no MATLAB or C++ "ECCO-1D" reference output to check the
end-to-end numbers against, unlike EccoPy-2D-V's SEA/SPOL cases -- see
core/classification.py's module docstring for the disk-mask verification
this claim rests on.

Unit-typed coordinates for time-series input
---------------------------------------------
`coords` is intentionally left as "whatever unit you have" (km, seconds,
raw grid index) -- EccoPy-1D does not assume a wind field or any other
conversion exists. If you have a fixed-point time series and want to
size the texture window in physical distance (Taylor's frozen-turbulence
hypothesis: distance = integral(|wind speed| dt)) rather than time, use
`eccopy.core.coords.time_to_distance_km()` to convert time + wind speed
into a distance array FIRST, then pass that as `coords` here with
coord_mode="position" -- exactly like any other pre-computed position
array. This mirrors how haversine-derived lat/lon spacing is handled
elsewhere in the package: conversion helpers are separate, composable
functions, not hidden inside run().

Minimum-length clump filter
-----------------------------
`min_convective_length`, if given, demotes any contiguous run of
Convective points shorter than the specified physical length (or pixel
count) to Mixed -- guarding against a brief, possibly-erroneous
high-convectivity blip getting labeled Convective just because it
survived class_basic()'s morphological cleanup. This has no MATLAB/C++
analogue to port; it is a 1-D-specific design addition modeled on
EccoPy-3D's clump volume filter (see core/classification.py's
filter_short_convective_runs_1d() docstring for the full rationale).

Typical usage
-------------
    import numpy as np
    from eccopy import eccopy1d
    from eccopy.params import WindowSpec

    dbz     = np.array([...])        # shape (T,)
    spacing = np.array([...])        # shape (T,) — time (s) or distance (km)

    result = eccopy1d.run(dbz, spacing, window=WindowSpec((5, 'km')))
    echo   = result.echo_type        # shape (T,), values in {1, 2, 3}

    # Time-series with wind-advected distance + a 5-minute minimum
    # convective-duration filter:
    from eccopy.core.coords import time_to_distance_km
    dist_km = time_to_distance_km(time_s, wind_speed_ms)
    result = eccopy1d.run(
        dbz, dist_km, coord_mode="position",
        window=WindowSpec((5, 'km')),
        min_convective_length=WindowSpec((5, 'min')),
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..core.coords import resolve_spacing
from ..core.texture import refl_texture_1d, refl_texture_1d_with_fit
from ..core.convectivity import texture_to_convectivity_linear
from ..core.classification import class_basic, filter_short_convective_runs_1d
from ..params.window import WindowSpec
from ..params import TextureParams, ClassificationParams


@dataclass
class Result1D:
    """Output of eccopy1d.run()."""
    echo_type:    np.ndarray   # shape (N,), int   {1=Strat, 2=Mixed, 3=Conv}
    convectivity: np.ndarray   # shape (N,), float [0, 1]
    texture:      np.ndarray   # shape (N,), float [0, ~30]
    # Populated ONLY when run(..., return_intermediates=True) is passed;
    # None otherwise (default, zero extra cost). See core.texture's
    # refl_texture_1d_with_fit() docstring for exactly what these are.
    fitted_dbz:    Optional[np.ndarray] = None   # shape (N,) — local linear-fit value at each point
    detrended_dbz: Optional[np.ndarray] = None   # shape (N,) — fit removed + re-centred, clipped >= 1


def _resolve_base_spacing(window: Union[WindowSpec, int, None],
                          spacing: np.ndarray) -> np.ndarray:
    """
    Convert a spacing array (in the caller's original coords unit --
    km for length axes, seconds for time axes) into the base unit a
    physical-size WindowSpec expects (metres / seconds). Pixel-based
    WindowSpecs (or a bare int) don't need conversion -- the caller-side
    logic that uses this treats a pixel WindowSpec as "count of points"
    directly and never multiplies by this function's output in that case.
    Shared between the texture window and min_convective_length so both
    interpret the same `coords` consistently.
    """
    if isinstance(window, WindowSpec) and not window.is_pixel:
        if window.base_kind == "length_m":
            return spacing * 1000.0   # km -> m
        return spacing                # already seconds
    return spacing


def run(dbz: Union[np.ndarray, list],
        coords: Union[np.ndarray, list],
        window: Union[WindowSpec, int] = WindowSpec((7, 'km')),
        coord_mode: str = "auto",
        texture_params: Optional[TextureParams] = None,
        class_params: Optional[ClassificationParams] = None,
        min_convective_length: Optional[Union[WindowSpec, int]] = None,
        kernel_mode: str = "uniform",
        return_intermediates: bool = False) -> Result1D:
    """
    Run EccoPy-1D: texture → convectivity → basic classification.

    Parameters
    ----------
    dbz : array-like, shape (N,)
        Reflectivity, dBZ. NaN where missing.
    coords : array-like, shape (N,)
        Either point positions (e.g. time in seconds, or distance in km)
        or pre-computed point-to-point spacing. Interpreted according to
        `coord_mode`. Units must match the WindowSpec unit. For
        wind-advected time series, see the module docstring's
        time_to_distance_km() example.
    window : WindowSpec or int
        Texture window half-width.
          WindowSpec(7)           — fixed 7-pixel radius
          WindowSpec((5, 'km'))   — 5 km radius (coords must be in km)
          WindowSpec((3, 'min'))  — 3-minute radius (coords must be in seconds)
    coord_mode : {'auto', 'position', 'spacing'}
        How to interpret `coords`:
          'position' — cumulative distance/time at each point; spacing is
                       derived as the difference between adjacent values.
          'spacing'  — already the point-to-point spacing.
          'auto'     — detect automatically (monotonic → position,
                       otherwise → spacing).
    texture_params : TextureParams, optional
    class_params : ClassificationParams, optional
    min_convective_length : WindowSpec or int, optional
        If given, any contiguous run of Convective points shorter than
        this length/duration (or point count, for a pixel WindowSpec) is
        demoted to Mixed. Must be in the same coordinate family as
        `coords`/`window` (a length WindowSpec if coords are distance, a
        time WindowSpec if coords are time) -- see module docstring and
        core.classification.filter_short_convective_runs_1d(). Default
        None disables the filter entirely (fully backward compatible).
    kernel_mode : {"uniform", "varying"}
        How to resolve the texture window's physical size into a pixel
        radius when `coords` gives non-uniform spacing (e.g. irregular
        sampling gaps in a time series):
          "uniform" (default) — one radius, resolved from the GLOBAL
              median spacing across the whole track, applied everywhere.
              Identical to "varying" whenever spacing is actually uniform
              (the common case) -- see core.texture.refl_texture_1d's
              kernel_mode docstring for why "uniform" is the default here
              (fidelity to the fact that per-point resolution has never
              been checked against ground truth in a case where it does
              something nontrivial, not a performance concern).
          "varying" — resolve a distinct radius at every point from its
              own local spacing. Physically more correct for genuinely
              irregular sampling, but unvalidated in that configuration.
    return_intermediates : bool
        If True, also compute and attach `fitted_dbz` and `detrended_dbz`
        to the returned Result1D (the per-point local-linear-fit value
        and the detrended/clipped value that feeds the texture
        statistic — see core.texture.refl_texture_1d_with_fit()). Costs
        a little extra Numba compute but no extra passes over the data;
        default False leaves both fields as None.

    Returns
    -------
    Result1D
    """
    dbz = np.asarray(dbz, dtype=float)
    if dbz.ndim != 1:
        raise ValueError(f"eccopy1d expects a 1-D dbz array; got shape {dbz.shape}")

    coords = np.asarray(coords, dtype=float)
    if coords.shape != dbz.shape:
        raise ValueError(
            f"coords shape {coords.shape} must match dbz shape {dbz.shape}"
        )

    tp = texture_params or TextureParams()
    cp = class_params or ClassificationParams()

    # Resolve coords → spacing (in km or seconds as appropriate)
    spacing = resolve_spacing(coords, axis=0, mode=coord_mode)

    # For WindowSpec with physical units, spacing must be in the same
    # base unit as the window (km→metres, min→seconds internally in
    # WindowSpec). Pass the spacing array unchanged — WindowSpec stores
    # its target size in metres/seconds and the user supplies coords in
    # km or seconds, so we multiply km→m here for length windows.
    spacing_for_window = _resolve_base_spacing(window, spacing)

    # 1. Texture
    fitted_dbz = detrended_dbz = None
    if return_intermediates:
        texture, fitted_dbz, detrended_dbz = refl_texture_1d_with_fit(
            dbz, window, spacing=spacing_for_window, dbz_base=tp.dbz_base,
            kernel_mode=kernel_mode,
        )
    else:
        texture = refl_texture_1d(
            dbz, window, spacing=spacing_for_window, dbz_base=tp.dbz_base,
            kernel_mode=kernel_mode,
        )

    # 2. Convectivity
    conv = texture_to_convectivity_linear(
        texture,
        upper_lim=tp.texture_limit_high,
    )

    # 3. Basic classification (strat/mixed/conv — no sub-types in 1D)
    # class_basic operates on >=2-D arrays (its morphological structuring
    # elements are 2-D). For genuinely 1-D EccoPy-1D input, treat the
    # array as a single row, run classification, and squeeze back.
    # See module docstring "Classification engine" note for why this is
    # a validated equivalence, not just a convenient reshape.
    echo = class_basic(
        conv[np.newaxis, :],
        strat_mixed=cp.max_convectivity_for_stratiform,
        mixed_conv=cp.min_convectivity_for_convective,
        melt=None, enlarge_mixed=cp.enlarge_mixed, enlarge_conv=cp.enlarge_conv,
    )[0]

    # 4. Optional minimum-convective-length clump filter (1-D-specific;
    # see module docstring "Minimum-length clump filter" and
    # core.classification.filter_short_convective_runs_1d()).
    if min_convective_length is not None:
        mcl = (min_convective_length if isinstance(min_convective_length, WindowSpec)
               else WindowSpec(min_convective_length))
        if mcl.is_pixel:
            min_len_base = float(mcl.size)
            spacing_base_len = np.ones_like(spacing)
        else:
            min_len_base = mcl.base_value
            spacing_base_len = _resolve_base_spacing(mcl, spacing)
        echo = filter_short_convective_runs_1d(echo, spacing_base_len, min_len_base)

    return Result1D(echo_type=echo, convectivity=conv, texture=texture,
                    fitted_dbz=fitted_dbz, detrended_dbz=detrended_dbz)
