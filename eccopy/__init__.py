"""
EccoPy — Radar Echo Classification (Python port of ECCO / ConvStratFinder).

Modules
-------
eccopy1d    : 1-D (time or distance), basic strat/mixed/conv.
eccopy2d_v  : 2-D vertical cross-section (Z, X), with optional sub-classification.
eccopy2d_h  : 2-D horizontal composite (Y, X), basic strat/mixed/conv.
eccopy3d    : 3-D volume (Z, Y, X), with optional sub-classification.

Helper
------
eccopy.core.coords.haversine_distance()   — great-circle distance between two points.
eccopy.core.coords.latlon_to_xy_spacing() — convert lat/lon grid to dx/dy spacing arrays.
eccopy.core.coords.resolve_spacing()      — convert position arrays to spacing arrays.

Quick start
-----------
    from eccopy import eccopy3d
    from eccopy.params import WindowSpec

    result = eccopy3d.run(
        dbz,                          # (Z, Y, X) numpy array
        coords_z=z_km,                # (Z,) heights in km
        coords_y=y_km,                # (Y,) N-S positions in km
        coords_x=x_km,                # (X,) E-W positions in km
        height=height_km,             # (Z, Y, X) height field, optional
        window=WindowSpec((7, 'km')), # 7 km texture radius
    )
    echo = result.echo_type           # (Z, Y, X), echo type codes
"""

from . import eccopy1d, eccopy2d_v, eccopy2d_h, eccopy3d
from . import stats
from .core.coords import haversine_distance, latlon_to_xy_spacing, resolve_spacing

__all__ = [
    "eccopy1d", "eccopy2d_v", "eccopy2d_h", "eccopy3d", "stats",
    "haversine_distance", "latlon_to_xy_spacing", "resolve_spacing",
]
