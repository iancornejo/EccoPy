"""
EccoPy-2D-V - classification of vertical cross-sections.

Operates on a (Z, X) slice: an RHI, a gridded cross-section, or a
column-versus-time curtain. Texture slides along the X axis only; the
vertical direction never enters the window.

Upstream analogues (lrose-ecco @ ad85c56):
    texture              f_reflTexture.m
    basic classification f_classBasic.m
    sub-classification   f_classSub.m
    driver               run_ecco_v_RHI_spol_gridded.m

Input shapes
------------
    dbz     : (Z, X)
    coords_z: (Z,) or (Z, X)   vertical positions/spacing (km)
    coords_x: (X,) or (Z, X)   horizontal positions/spacing (km)
    height  : (Z, X)           height field, km (MSL or AGL)
    melt    : (Z, X) or (Z,)   melting-layer field
    temp    : (Z, X) or (Z,)   temperature field, degC

Echo type codes
---------------
    Basic (height, melt or temp absent):
        1 Stratiform    2 Mixed    3 Convective

    Sub-classified (height, melt and temp all present):
        14 Stratiform Low     16 Stratiform Mid     18 Stratiform High
        25 Mixed
        30 Convective         32 Convective Elevated
        34 Convective Shallow 36 Convective Mid     38 Convective Deep

    Code 30 is reachable only by calling class_sub_2d() directly with its
    `elev`/`first_row` arguments, which correspond to f_classSub.m's
    airborne-radar branch. run() does not pass them.

Sub-classification inputs are all-or-nothing
--------------------------------------------
`height`, `melt` and `temp` must be supplied together. f_classSub.m uses
all three and has no path that takes a subset: melt drives the
shallow/low decision, temp separates mid from deep/high, and height
supplies the near-surface test. Given an incomplete subset this function
returns basic codes rather than running a reduced variant of the
algorithm.

VerticalParams does not apply here. Its thresholds are read only by the
3-D path; f_classSub.m hardcodes its melt threshold of 15 and its
temperature threshold of -25 degC. The one tunable it does consult,
surf_alt_lim, lives on ClassificationParams.

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
    # Populated only when run(..., return_intermediates=True).
    fitted_dbz:    Optional[np.ndarray] = None   # (Z, X) local linear fit at each point
    detrended_dbz: Optional[np.ndarray] = None   # (Z, X) fit removed, re-centred, clipped >= 1
    echo_basic:    Optional[np.ndarray] = None   # (Z, X) basic 1/2/3 codes, pre sub-classification


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
    Run EccoPy-2D-V: texture -> convectivity -> classification.

    Sub-classification runs only when `height`, `melt` and `temp` are all
    supplied; otherwise basic codes are returned. See the module
    docstring.

    Parameters
    ----------
    dbz : array-like, shape (Z, X)
        Reflectivity, dBZ. NaN where missing.
    coords_z : array-like, shape (Z,) or (Z, X)
        Vertical coordinate or spacing, km.
    coords_x : array-like, shape (X,) or (Z, X)
        Horizontal coordinate or spacing, km.
    height : array-like, shape (Z, X), optional
        Height field, km. AGL if `topo` is supplied, otherwise taken as
        given. Required with `melt` and `temp` for sub-classification.
    melt : array-like, shape (Z, X) or (Z,), optional
        Melting-layer field, either a full field or a single profile
        broadcast across every column. f_classSub.m thresholds it at 15;
        the SPOL driver supplies a binary field of 10 (above freezing)
        and 20 (below). Also enables f_classBasic.m's rain-below-melting-
        layer correction. Required with `height` and `temp`.
    temp : array-like, shape (Z, X) or (Z,), optional
        Temperature, degC. Separates mid from deep/high once `melt` has
        placed a point above the melting layer. Required with `height`
        and `melt`.
    topo : array-like, shape (X,) or (Z, X), optional
        Terrain height, km, or for a fixed-elevation RHI the
        earth-curvature beam-height correction per range gate -
        f_classSub.m's `topo` argument. Used only with `height`, where
        `height - topo` becomes the AGL height throughout class_sub_2d.
    window : WindowSpec or int
        Texture window half-width along X.
    coord_mode : {'auto', 'position', 'spacing'}
        How to interpret coords_z / coords_x.
    texture_params : TextureParams, optional
    class_params : ClassificationParams, optional
        `surf_alt_lim` (metres) drives two independent things: the
        near-surface convective test inside class_sub_2d, and - only when
        `remove_surface_echo=True` - the pre-texture masking below.
    kernel_mode : {"uniform", "varying"}
        How a physical window size becomes a pixel radius when `coords_x`
        spacing is non-uniform.

        "uniform" resolves one radius for the whole array from its global
        median spacing. This matches f_reflTexture.m, which takes a
        single `pixRad`, and is the default.

        "varying" resolves a radius per point, and per Z row when
        `coords_x` is 2-D. Physically closer for genuinely non-uniform
        range-gate spacing, but has no upstream counterpart. The two are
        identical when spacing is uniform.
    remove_surface_echo : bool
        Set dbz to NaN where the `coords_z` altitude is below
        `cp.surf_alt_lim`, before texture. Reproduces the driver-level
        masking in run_ecco_v_RHI_spol_gridded.m. Off by default because
        not every upstream driver applies it. The caller's array is not
        mutated.
    return_intermediates : bool
        Also attach `fitted_dbz`, `detrended_dbz` and `echo_basic` to the
        result. Default False leaves them None.

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

    # --- resolve coords to per-point spacing (km) ---
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

    # Pre-texture surface-echo removal, matching the driver-level step in
    # run_ecco_v_RHI_spol_gridded.m:
    #     data.DBZ_F(data.Z .* 1000 < surfAltLim) = nan;
    # Masking is on altitude (coords_z), not AGL: the driver does not
    # subtract topo here. This is a distinct role from surf_alt_lim's use
    # inside class_sub_2d. Opt-in because upstream drivers disagree - the
    # SeaPol RHI driver leaves reflectivity unmasked and applies
    # surfAltLim only in f_classSub.m.
    if remove_surface_echo:
        alt_km = coords_z if coords_z.ndim == 2 else coords_z[:, np.newaxis]
        # np.where returns a new array; the caller's dbz is not mutated.
        dbz = np.where(alt_km < (cp.surf_alt_lim / 1000.0), np.nan, dbz)

    # Texture slides along X, so the window resolves against sp_x.
    if isinstance(window, WindowSpec) and not window.is_pixel:
        if window.base_kind == "length_m":
            sp_x_for_window = sp_x * 1000.0
        else:
            sp_x_for_window = sp_x
    else:
        sp_x_for_window = sp_x

    # 1. Texture - f_reflTexture.m, sliding along the last axis.
    # min_valid_dbz resolves to no gating here: it ports a
    # ConvStratFinder prefilter that f_reflTexture.m does not have. An
    # explicit value is still honoured.
    min_valid = tp.resolve_min_valid_dbz(-np.inf)

    fitted_dbz = detrended_dbz = None
    if return_intermediates:
        texture, fitted_dbz, detrended_dbz = refl_texture_1d_with_fit(
            dbz, window, spacing=sp_x_for_window, dbz_base=tp.dbz_base,
            min_valid_dbz=min_valid, kernel_mode=kernel_mode,
        )
    else:
        texture = refl_texture_1d(
            dbz, window, spacing=sp_x_for_window, dbz_base=tp.dbz_base,
            min_valid_dbz=min_valid, kernel_mode=kernel_mode,
        )

    # 2. Convectivity
    conv = texture_to_convectivity_linear(
        texture,
        upper_lim=tp.texture_limit_high,
        lower_lim=tp.texture_limit_low,
    )

    # Melt broadcasts like temp: a full field or a (Z,) profile. It is
    # taken as given - melt is never derived from temp here. Use
    # core.temperature.melt_layer_from_temp() to build one.
    melt_arr = broadcast_temp_field(melt, dbz.shape) if melt is not None else None

    # 3. Basic classification - f_classBasic.m. Passing melt enables its
    # rain-below-melting-layer correction.
    echo_basic = class_basic(
        conv,
        strat_mixed=cp.max_convectivity_for_stratiform,
        mixed_conv=cp.min_convectivity_for_convective,
        melt=melt_arr, enlarge_mixed=cp.enlarge_mixed, enlarge_conv=cp.enlarge_conv,
    )

    # Sub-classification needs all three inputs; a partial subset returns
    # basic codes. See the module docstring.
    if height is None or melt is None or temp is None:
        return Result2DV(echo_type=echo_basic, convectivity=conv, texture=texture,
                          fitted_dbz=fitted_dbz, detrended_dbz=detrended_dbz,
                          echo_basic=echo_basic if return_intermediates else None)

    # f_classSub.m works in metres throughout (`data.Z .* 1000`), so
    # class_sub_2d does too. EccoPy's public contract is km; convert once
    # here, at the boundary.
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