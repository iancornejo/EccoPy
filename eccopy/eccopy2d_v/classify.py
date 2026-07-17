"""
EccoPy-2D-V (Vertical cross-section) public entry point.

Input shapes:
    dbz     : (Z, X)
    coords_z: (Z,) or (Z, X)  — vertical positions/spacing (km)
    coords_x: (X,) or (Z, X)  — horizontal positions/spacing (km)
    height  : (Z, X), optional — height field, km (MSL or AGL)
    melt    : (Z, X) or (Z,), optional — melting-layer field
    temp    : (Z, X) or (Z,), optional — temperature field, °C

Output echo type codes:
    Without height/melt/temp — basic only:
        1 = Stratiform,  2 = Mixed,  3 = Convective
    With height AND melt AND temp all three — sub-classified:
        14 = Stratiform Low,   16 = Stratiform Mid,  18 = Stratiform High
        25 = Mixed
        30 = Convective (near-aircraft override, rarely triggered)
        32 = Convective Elevated,  34 = Convective Shallow,
        36 = Convective Mid,       38 = Convective Deep

*** API CHANGE ***: sub-classification previously accepted "height OR
temp" as alternatives. It does not anymore. The real reference algorithm
(f_classSub.m) requires height (for AGL / near-surface tests), melt
(the primary signal for shallow/low classification), and temp (which
only distinguishes mid from deep/high) all together -- there is no
code path in the reference that uses height or temp alone. If any of
the three is missing, this function falls back to basic-only
classification (codes 1/2/3), the same as if none were given -- it does
NOT attempt a partial/best-effort sub-classification with whatever
subset is available, because that was found (this session) to silently
test a different, unvalidated algorithm.

melt is also now threaded into class_basic() (previously always called
with melt=None here), enabling MATLAB's "rain below melting layer"
correction check in basic classification -- NOTE this correction block
was found to be inert even on real, spatially-varying melt data during
validation and is a separate, still-open issue; passing melt through
does not fix it, but also should not make anything worse than the
previous melt=None behavior.

*** vert_params REMOVED from this function's signature ***: VerticalParams
is scoped to the 3-D path (set_echo_type_3d) only -- see its own
docstring. class_sub_2d() does not use height/temp thresholds at all
(the real algorithm hardcodes a melt threshold of 15 and a temperature
threshold of -25 deg C); the one parameter it does need beyond
height/melt/temp/topo, surf_alt_lim, already lives on ClassificationParams
and is read from there (cp.surf_alt_lim) instead.

Typical usage
-------------
    from eccopy import eccopy2d_v
    from eccopy.params import WindowSpec

    result = eccopy2d_v.run(
        dbz, coords_z=z_km, coords_x=x_km,
        height=height_km, melt=melt_field, temp=temp_c,
        window=WindowSpec((7, 'km')),
    )
    echo = result.echo_type   # shape (Z, X)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..core.coords import resolve_spacing
from ..core.texture import refl_texture_1d, refl_texture_1d_with_fit
from ..core.convectivity import texture_to_convectivity_linear
from ..core.classification import class_basic, class_sub_2d
from ..core.temperature import broadcast_temp_field
from ..params.window import WindowSpec
from ..params import TextureParams, ClassificationParams


@dataclass
class Result2DV:
    """Output of eccopy2d_v.run()."""
    echo_type:    np.ndarray   # shape (Z, X)
    convectivity: np.ndarray   # shape (Z, X)
    texture:      np.ndarray   # shape (Z, X)
    # Populated ONLY when run(..., return_intermediates=True) is passed;
    # None otherwise (default, zero extra cost).
    fitted_dbz:    Optional[np.ndarray] = None   # shape (Z, X) — local linear-fit value at each point
    detrended_dbz: Optional[np.ndarray] = None   # shape (Z, X) — fit removed + re-centred, clipped >= 1
    echo_basic:    Optional[np.ndarray] = None   # shape (Z, X) — basic 1/2/3 classification BEFORE sub-classification


def run(dbz: Union[np.ndarray, list],
        coords_z: Union[np.ndarray, list],
        coords_x: Union[np.ndarray, list],
        height: Optional[Union[np.ndarray, list]] = None,
        melt: Optional[Union[np.ndarray, list]] = None,
        temp: Optional[Union[np.ndarray, list]] = None,
        topo: Optional[Union[np.ndarray, list]] = None,
        window: Union[WindowSpec, int] = WindowSpec((7, 'km')),
        coord_mode: str = "auto",
        texture_params: Optional[TextureParams] = None,
        class_params: Optional[ClassificationParams] = None,
        kernel_mode: str = "uniform",
        remove_surface_echo: bool = False,
        return_intermediates: bool = False) -> Result2DV:
    """
    Run EccoPy-2D-V: texture → convectivity → classification.

    Sub-classification into low/mid/high stratiform and
    shallow/mid/deep/elevated convective is performed ONLY if `height`,
    `melt`, AND `temp` are ALL provided together -- see module docstring
    "API CHANGE" note. If any of the three is missing, only basic
    stratiform/mixed/convective codes are returned.

    Parameters
    ----------
    dbz : array-like, shape (Z, X)
        Reflectivity, dBZ. NaN where missing.
    coords_z : array-like, shape (Z,) or (Z, X)
        Vertical coordinate or spacing, km.
    coords_x : array-like, shape (X,) or (Z, X)
        Horizontal coordinate or spacing, km.
    height : array-like, shape (Z, X), optional
        Height field, km. AGL preferred (subtract `topo` yourself, or
        pass `topo` separately below, to get AGL from an MSL height
        field); MSL acceptable directly if terrain/beam-height is flat.
        Required (together with `melt` and `temp`) for sub-classification.
    melt : array-like, shape (Z, X) or (Z,), optional
        Melting-layer field. Accepts either a full (Z, X) field or a
        single vertical profile of shape (Z,), broadcast identically
        across every column (same convention as `temp`). This is the
        PRIMARY signal for shallow/low sub-classification -- it is not
        a fallback or secondary input. Also enables class_basic()'s
        rain-below-melting-layer correction check (see module docstring
        caveat: that check was found inert on real data this session).
        Required (together with `height` and `temp`) for
        sub-classification.
    temp : array-like, shape (Z, X) or (Z,), optional
        Temperature field, °C. Only distinguishes mid from deep/high
        sub-classification, given `melt` already places a point above
        the melting layer. Required (together with `height` and `melt`)
        for sub-classification.
    topo : array-like, shape (X,) or (Z, X), optional
        Terrain height (or, for a fixed-elevation-angle RHI, the
        earth-curvature beam-height correction at each range gate — see
        f_classSub.m's `topo` parameter, which this matches), km. Only
        used together with `height`: `height - topo` becomes the AGL
        height used throughout class_sub_2d. Ignored if `height` is not
        given.
    window : WindowSpec or int
        Texture window half-width along the X axis.
    coord_mode : {'auto', 'position', 'spacing'}
        How to interpret coords_z / coords_x.
    texture_params : TextureParams, optional
    class_params : ClassificationParams, optional
        surf_alt_lim has TWO independent roles, both driven by this one
        value (metres):
          1. class_sub_2d's near-surface convective test (always active
             when sub-classification runs) -- read here as cp.surf_alt_lim.
          2. Optional pre-texture surface-echo removal -- ONLY when
             `remove_surface_echo=True` (see below). Off by default.
        There is no separate vert_params argument; VerticalParams is scoped
        to the 3-D path only and does not apply to this function.
    kernel_mode : {"uniform", "varying"}
        How to resolve the texture window's physical size into a pixel
        radius when `coords_x` gives non-uniform spacing, INCLUDING
        spacing that genuinely differs by Z row (e.g. non-uniform
        range-gate spacing that changes with elevation angle -- the
        exact scenario the refl_texture_1d row-spacing fix addressed):
          "uniform" (default) — one radius for the WHOLE (Z, X) array,
              resolved from its global median spacing. Identical to
              "varying" whenever spacing is actually uniform (SEA/SPOL,
              and presumably most real single-elevation-angle data) --
              see core.texture.refl_texture_1d's kernel_mode docstring
              for why "uniform" is the default (fidelity to the fact
              that per-point/per-row resolution has never been checked
              against ground truth in a case where it does something
              nontrivial, not a performance concern).
          "varying" — resolve a distinct radius at every point, and at
              every Z row independently if coords_x is genuinely 2-D.
              Physically more correct for real non-uniform range-gate
              spacing, but unvalidated in that configuration. Note this
              collapses row-varying spacing to ONE number per row's
              median under "uniform" -- if you specifically want the
              row-by-row (but not point-by-point) behaviour, use
              "varying" together with a coords_x that is already
              constant within each row.
    remove_surface_echo : bool
        If True, set dbz to NaN wherever the vertical altitude coordinate
        (coords_z, interpreted as km positions) is below cp.surf_alt_lim,
        BEFORE texture is computed. This reproduces the driver-level
        surfAltLim masking in the real ECCO-V MATLAB scripts (e.g.
        `data.DBZ_F(data.Z.*1000 < surfAltLim) = nan;` in
        run_ecco_v_RHI_spol_gridded.m), used to suppress near-surface
        ground/ocean clutter. Masks on ALTITUDE (coords_z), not AGL --
        topo is not subtracted, matching MATLAB. The caller's dbz array is
        never mutated (a masked copy is used internally). Default False
        preserves the historical behavior of this function (no masking),
        which matches drivers like the SeaPol RHI that leave DBZ unmasked.
    return_intermediates : bool
        If True, also compute and attach `fitted_dbz`, `detrended_dbz`
        (the per-point local-linear-fit value and detrended/clipped
        value that feeds the texture statistic — see
        core.texture.refl_texture_1d_with_fit()) and `echo_basic` (the
        basic strat/mixed/conv classification BEFORE sub-classification
        is applied, even when height/melt/temp are all given and
        `echo_type` ends up sub-classified) to the returned Result2DV.
        Default False leaves all three fields as None.

    Returns
    -------
    Result2DV
    """
    dbz = np.asarray(dbz, dtype=float)
    if dbz.ndim != 2:
        raise ValueError(f"eccopy2d_v expects a 2-D (Z, X) dbz array; got shape {dbz.shape}")
    nz, nx = dbz.shape

    tp = texture_params or TextureParams()
    cp = class_params   or ClassificationParams()

    # --- resolve coords → per-point spacing (km) ---
    coords_z = np.asarray(coords_z, dtype=float)
    coords_x = np.asarray(coords_x, dtype=float)

    if coords_z.ndim == 1:
        if coords_z.shape[0] != nz:
            raise ValueError(f"coords_z length {coords_z.shape[0]} != Z={nz}")
    elif coords_z.shape != (nz, nx):
        raise ValueError(
            f"coords_z shape {coords_z.shape} must be (Z,) or (Z, X) = {(nz, nx)}"
        )

    if coords_x.ndim == 1:
        if coords_x.shape[0] != nx:
            raise ValueError(f"coords_x length {coords_x.shape[0]} != X={nx}")
        sp_x_1d = resolve_spacing(coords_x, axis=0, mode=coord_mode)
        sp_x = np.broadcast_to(sp_x_1d[np.newaxis, :], (nz, nx)).copy()
    else:
        sp_x = resolve_spacing(coords_x, axis=1, mode=coord_mode)

    # --- optional pre-texture surface-echo removal (surfAltLim masking) ---
    # Faithful port of the driver-level step in the real ECCO-V MATLAB
    # scripts, e.g. run_ecco_v_RHI_spol_gridded.m:
    #     data.DBZ_F(data.Z .* 1000 < surfAltLim) = nan;
    # applied to reflectivity BEFORE f_reflTexture. This is a SEPARATE role
    # from surf_alt_lim's use inside class_sub_2d (the near-surface
    # convective test); the MATLAB drivers apply this masking on the raw
    # vertical ALTITUDE coordinate (data.Z / data.asl, MSL), NOT on AGL
    # (Z - topo). We therefore mask on coords_z, interpreted as altitude
    # positions in km -- not on `height` and not topo-corrected.
    #
    # It is OPT-IN (default off) on purpose: not every reference driver
    # does it (e.g. the SeaPol RHI driver leaves DBZ unmasked and uses
    # surfAltLim only in class_sub_2d), so masking unconditionally would
    # diverge from those cases and silently change already-validated
    # behavior. Turn it on to reproduce the SPOL/APR3/CloudNet drivers.
    if remove_surface_echo:
        alt_km = coords_z if coords_z.ndim == 2 else coords_z[:, np.newaxis]
        # np.where returns a fresh array -- never mutate the caller's dbz.
        dbz = np.where(alt_km < (cp.surf_alt_lim / 1000.0), np.nan, dbz)

    # WindowSpec: texture slides along X axis; spacing for window resolution is sp_x
    if isinstance(window, WindowSpec) and not window.is_pixel:
        if window.base_kind == "length_m":
            sp_x_for_window = sp_x * 1000.0
        else:
            sp_x_for_window = sp_x
    else:
        sp_x_for_window = sp_x

    # 1. Texture — sliding window along X axis (last axis of 2D array)
    fitted_dbz = detrended_dbz = None
    if return_intermediates:
        texture, fitted_dbz, detrended_dbz = refl_texture_1d_with_fit(
            dbz, window, spacing=sp_x_for_window, dbz_base=tp.dbz_base,
            kernel_mode=kernel_mode,
        )
    else:
        texture = refl_texture_1d(
            dbz, window, spacing=sp_x_for_window, dbz_base=tp.dbz_base,
            kernel_mode=kernel_mode,
        )

    # 2. Convectivity
    conv = texture_to_convectivity_linear(
        texture,
        upper_lim=tp.texture_limit_high,
    )

    # Melt broadcast (same convention as temp: full field or (Z,) profile).
    # NOT converted/derived from anything else -- EccoPy is data-agnostic
    # and does not compute melt from temp internally; the caller supplies
    # it directly, same as temp.
    melt_arr = broadcast_temp_field(melt, dbz.shape) if melt is not None else None

    # 3. Classification — melt now threaded through instead of hardcoded
    # None (see module docstring caveat about the rain-correction check).
    echo_basic = class_basic(
        conv,
        strat_mixed=cp.max_convectivity_for_stratiform,
        mixed_conv=cp.min_convectivity_for_convective,
        melt=melt_arr, enlarge_mixed=cp.enlarge_mixed, enlarge_conv=cp.enlarge_conv,
    )

    # Sub-classification requires height AND melt AND temp together --
    # NOT any-one-of, see module docstring "API CHANGE". A partial subset
    # (e.g. height+temp but no melt) falls back to basic-only, same as
    # if none were given, rather than guessing at a reduced algorithm.
    if height is None or melt is None or temp is None:
        return Result2DV(echo_type=echo_basic, convectivity=conv, texture=texture,
                          fitted_dbz=fitted_dbz, detrended_dbz=detrended_dbz,
                          echo_basic=echo_basic if return_intermediates else None)

    # class_sub_2d (a near-literal port of f_classSub.m, which works in
    # metres throughout — e.g. `data.Z .* 1000`) expects height/topo in
    # METRES, but this function's own public contract (like the rest of
    # EccoPy) takes them in km — convert here, once, at the boundary.
    height_arr = np.asarray(height, dtype=float) * 1000.0
    topo_arr = np.asarray(topo, dtype=float) * 1000.0 if topo is not None else np.zeros((nz, nx))
    temp_arr = broadcast_temp_field(temp, dbz.shape)

    echo_sub = class_sub_2d(
        echo_basic,
        height=height_arr,
        topo=topo_arr,
        melt=melt_arr,
        temp=temp_arr,
        surf_alt_lim=cp.surf_alt_lim,
    )

    return Result2DV(echo_type=echo_sub, convectivity=conv, texture=texture,
                      fitted_dbz=fitted_dbz, detrended_dbz=detrended_dbz,
                      echo_basic=echo_basic if return_intermediates else None)