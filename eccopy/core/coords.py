"""
Coordinate helpers for EccoPy.

EccoPy is data-agnostic: it never reads files and never assumes a
particular coordinate system. Distances between grid points must be
supplied explicitly, either as:

  - Position/coordinate arrays (e.g. x_km giving each point's location),
    from which local point-to-point spacing is derived, or
  - Pre-computed spacing/delta arrays (the distance between adjacent
    points directly), or
  - Latitude/longitude grids, converted to local x/y distance arrays
    via haversine() below.

This module provides:
  haversine_distance()   - great-circle distance between two lat/lon points
  latlon_to_xy_spacing() - convert a lat/lon grid into local dx/dy spacing
                           arrays (same shape as the input grid)
  latlon_grid_spacing()  - the same for a regular lat/lon grid given as
                           1-D lat/lon, as memory-free broadcast views
  resolve_spacing()       - internal: turn either a position array or a
                           pre-computed spacing array into a spacing array
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float,
                       radius_km: float = EARTH_RADIUS_KM) -> float:
    """
    Great-circle distance between two points on a sphere.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float, degrees
    radius_km : float, default Earth's mean radius

    Returns
    -------
    distance_km : float
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return radius_km * c


def latlon_to_xy_spacing(lat: np.ndarray,
                         lon: np.ndarray,
                         radius_km: float = EARTH_RADIUS_KM
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a lat/lon grid into local x/y point-to-point spacing arrays
    using the haversine formula.

    Parameters
    ----------
    lat, lon : np.ndarray, same shape, degrees
        Latitude/longitude at every grid point. Any number of dimensions
        is supported as long as the LAST axis is "x" (e.g. longitude
        varies fastest) and the second-to-last axis is "y", matching the
        convention used elsewhere in EccoPy (..., Y, X).

    Returns
    -------
    dy_km, dx_km : np.ndarray, same shape as lat/lon
        Local spacing to the next point along the y-axis / x-axis.
        The last row/column repeats the previous spacing value (so the
        output shape matches the input shape exactly).

    Notes
    -----
    This computes the distance from each point to its neighbour at
    index+1 along the relevant axis - i.e. dx_km[..., i] is the distance
    from point i to point i+1 (and the final column duplicates the
    second-to-last value so the array shape is preserved).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.shape != lon.shape:
        raise ValueError(f"lat and lon must have the same shape, got "
                         f"{lat.shape} and {lon.shape}")

    # --- x-spacing: distance along the last axis ---
    lat_a = lat[..., :-1]
    lon_a = lon[..., :-1]
    lat_b = lat[..., 1:]
    lon_b = lon[..., 1:]
    dx_inner = haversine_distance(lat_a, lon_a, lat_b, lon_b, radius_km)
    dx_km = np.concatenate([dx_inner, dx_inner[..., -1:]], axis=-1)

    # --- y-spacing: distance along the second-to-last axis ---
    if lat.ndim < 2:
        dy_km = np.zeros_like(dx_km)
    else:
        lat_a = lat[..., :-1, :]
        lon_a = lon[..., :-1, :]
        lat_b = lat[..., 1:, :]
        lon_b = lon[..., 1:, :]
        dy_inner = haversine_distance(lat_a, lon_a, lat_b, lon_b, radius_km)
        dy_km = np.concatenate([dy_inner, dy_inner[..., -1:, :]], axis=-2)

    return dy_km, dx_km


def latlon_grid_spacing(lat: np.ndarray,
                        lon: np.ndarray,
                        radius_km: float = EARTH_RADIUS_KM
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spacing of a regular latitude/longitude grid, as read-only (Y, X)
    views that cost almost no memory.

    On a grid made of every combination of a 1-D `lat` and a uniformly
    spaced 1-D `lon` -- MRMS, and most model output on a lat/lon mesh --
    both spacings depend only on the row. latlon_to_xy_spacing() on the
    meshgrid returns them as two full float64 arrays (about 400 MB
    together for the 3500 x 7000 MRMS CONUS grid); this returns per-row
    values broadcast to (Y, X) instead. Pass them to eccopy2d_h with
    coord_mode="spacing".

    Parameters
    ----------
    lat : np.ndarray, shape (Y,)
        Latitude of each row, degrees, monotonic (either direction).
    lon : np.ndarray, shape (X,)
        Longitude of each column, degrees, uniformly spaced (to a relative
        tolerance of 1e-6).
    radius_km : float, default Earth's mean radius

    Returns
    -------
    dy_km, dx_km : np.ndarray, shape (Y, X), read-only broadcast views
        The same haversine distances latlon_to_xy_spacing() computes, taken
        along the first column: dy_km[i] from row i to row i+1, dx_km[i]
        from lon[0] to lon[1] at row i, with the last row repeating the
        previous dy as there. latlon_to_xy_spacing()'s dx differs along a
        row only by rounding (about 1e-12, relative).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            f"lat and lon must be 1-D (one value per row / column); got "
            f"shapes {lat.shape} and {lon.shape}. For 2-D fields use "
            f"latlon_to_xy_spacing()."
        )
    if lon.size < 2:
        raise ValueError("lon needs at least 2 values to define a spacing")
    dlon = np.diff(lon)
    if not np.allclose(dlon, dlon[0], rtol=1e-6, atol=0.0):
        raise ValueError(
            "lon is not uniformly spaced, so dx varies along each row; use "
            "latlon_to_xy_spacing() on the meshgrid instead."
        )
    dx_row = haversine_distance(lat, lon[0], lat, lon[1], radius_km)
    if lat.size < 2:
        dy_row = np.zeros_like(dx_row)
    else:
        dy_inner = haversine_distance(lat[:-1], lon[0], lat[1:], lon[0], radius_km)
        dy_row = np.concatenate([dy_inner, dy_inner[-1:]])
    shape = (lat.size, lon.size)
    return (np.broadcast_to(dy_row[:, np.newaxis], shape),
            np.broadcast_to(dx_row[:, np.newaxis], shape))


def time_to_distance_km(time: np.ndarray,
                        wind_speed_ms: np.ndarray,
                        mode: str = "auto") -> np.ndarray:
    """
    Convert a 1-D time series into a pseudo-spatial "position" coordinate
    via Taylor's frozen-turbulence hypothesis: distance = integral(|speed|
    dt). This is the standard way to turn a fixed-point time series (e.g.
    a vertically-pointing or scanning radar sampling a field advected past
    it by the wind) into something a spatial texture/classification window
    can be sized against in km, instead of only being able to size a
    window in raw time.

    This is a stand-alone, composable helper -- like haversine_distance /
    latlon_to_xy_spacing above, it is NOT called automatically by
    eccopy1d.run(). EccoPy is data-agnostic and does not assume a wind
    field exists; a caller with wind speed available derives a distance
    coordinate here FIRST, then passes the result into eccopy1d.run() as
    `coords` (with `coord_mode="position"`) exactly as they would a
    directly-measured distance array. A caller without wind data simply
    keeps using `coords` as time or grid position, unchanged -- this
    function is purely additive.

    Parameters
    ----------
    time : array-like, shape (N,)
        Either cumulative time (seconds) at each sample, or already-
        computed point-to-point time spacing -- interpreted according to
        `mode`, same convention as resolve_spacing().
    wind_speed_ms : array-like, shape (N,), or scalar
        Wind speed, m/s, at each sample (or a single constant speed
        broadcast to every sample). Only magnitude matters -- sign/
        direction is not used, since this produces a monotonically
        non-decreasing along-track distance, not a signed displacement.
        NaN propagates: a missing wind sample makes every subsequent
        cumulative distance value NaN (see Notes).
    mode : {"auto", "position", "spacing"}
        How to interpret `time` -- passed straight through to
        resolve_spacing().

    Returns
    -------
    distance_km : np.ndarray, shape (N,)
        Cumulative along-track distance, km, starting at 0 for the first
        sample. Monotonically non-decreasing (strictly increasing
        wherever wind_speed_ms != 0). Intended to be passed directly as
        `coords` to eccopy1d.run() with `coord_mode="position"`.

    Notes
    -----
    Because NaN propagates through np.cumsum, a single missing wind
    sample poisons every distance value after it. If wind data has gaps,
    fill them (e.g. interpolate) before calling this function -- EccoPy
    does not do this for you, matching its data-agnostic, no-hidden-
    imputation design elsewhere (see refl_texture_1d's NaN handling for
    the one deliberate exception, which is documented there).

    Also note "auto" mode's monotonicity heuristic in resolve_spacing()
    can misfire on a `time` array that is already point-to-point spacing
    with irregular (but always-positive) gaps -- pass mode="position" or
    mode="spacing" explicitly if there's any doubt.
    """
    dt_s = resolve_spacing(np.asarray(time, dtype=float), axis=0, mode=mode)
    speed = np.broadcast_to(
        np.abs(np.asarray(wind_speed_ms, dtype=float)), dt_s.shape
    )
    dx_km = speed * dt_s / 1000.0
    # dx_km[i] is the distance covered going FROM point i TO point i+1
    # (same point-to-point convention as resolve_spacing's spacing
    # output), so the cumulative position at point i is the running sum
    # of all PRIOR segments, with the first point pinned at 0.
    distance_km = np.concatenate(([0.0], np.cumsum(dx_km[:-1])))
    return distance_km


def resolve_spacing(coord_or_spacing: np.ndarray,
                    axis: int,
                    mode: str = "auto") -> np.ndarray:
    """
    Turn either a position/coordinate array or a pre-computed spacing
    array into a spacing array (distance to the next point along `axis`).

    Parameters
    ----------
    coord_or_spacing : np.ndarray
        Either point positions (e.g. cumulative distance / coordinate
        value at each grid point) or already-computed point-to-point
        spacing.
    axis : int
        Axis along which spacing is measured.
    mode : {"auto", "position", "spacing"}
        "position": treat input as coordinate values; spacing is
            np.diff along axis (with the last point repeating the final
            spacing value so the shape is preserved).
        "spacing": treat input as already being point-to-point spacing;
            returned unchanged.
        "auto" (default): treat as "position" if values are monotonically
            increasing or decreasing along `axis` (typical of coordinate
            arrays); otherwise treat as "spacing" directly. This heuristic
            covers the common cases but "position"/"spacing" can be
            specified explicitly to avoid ambiguity.

    Returns
    -------
    spacing : np.ndarray, same shape as coord_or_spacing
    """
    arr = np.asarray(coord_or_spacing, dtype=float)

    if mode not in ("auto", "position", "spacing"):
        raise ValueError(f"mode must be 'auto', 'position', or 'spacing', got {mode!r}")

    if mode == "spacing":
        return arr

    if mode == "auto":
        diffs = np.diff(arr, axis=axis)
        finite = diffs[np.isfinite(diffs)]
        if finite.size == 0:
            mode = "spacing"
        else:
            all_pos = np.all(finite > 0)
            all_neg = np.all(finite < 0)
            mode = "position" if (all_pos or all_neg) else "spacing"

    if mode == "spacing":
        return arr

    # mode == "position": derive spacing via diff, pad last slice
    spacing = np.abs(np.diff(arr, axis=axis))
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, 1)
    spacing = np.pad(spacing, pad_widths, mode="edge")
    return spacing
