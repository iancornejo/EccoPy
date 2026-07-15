"""Tests for eccopy.core.coords — haversine and spacing-resolution helpers."""

import numpy as np
import pytest

from eccopy.core.coords import (
    haversine_distance, latlon_to_xy_spacing, resolve_spacing, time_to_distance_km,
)


def test_haversine_known_distance():
    # 1 degree of longitude at the equator is ~111.2 km
    d = haversine_distance(0, 0, 0, 1)
    assert abs(d - 111.2) < 0.5


def test_haversine_latitude_scaling():
    # 1 degree of longitude at 60N should be about half the equatorial value
    d_eq = haversine_distance(0, 0, 0, 1)
    d_60 = haversine_distance(60, 0, 60, 1)
    assert abs(d_60 / d_eq - 0.5) < 0.02


def test_haversine_zero_distance():
    d = haversine_distance(45, 10, 45, 10)
    assert d == 0.0


def test_haversine_array_input():
    lat1 = np.array([0, 10, 20])
    lon1 = np.array([0, 0, 0])
    lat2 = np.array([0, 10, 20])
    lon2 = np.array([1, 1, 1])
    d = haversine_distance(lat1, lon1, lat2, lon2)
    assert d.shape == (3,)
    assert np.all(d > 0)


def test_latlon_to_xy_spacing_shape_preserved():
    lat = np.tile(np.linspace(20, 25, 5)[:, None], (1, 6))
    lon = np.tile(np.linspace(120, 125, 6)[None, :], (5, 1))
    dy, dx = latlon_to_xy_spacing(lat, lon)
    assert dy.shape == lat.shape
    assert dx.shape == lat.shape


def test_latlon_to_xy_spacing_positive():
    lat = np.tile(np.linspace(20, 25, 5)[:, None], (1, 6))
    lon = np.tile(np.linspace(120, 125, 6)[None, :], (5, 1))
    dy, dx = latlon_to_xy_spacing(lat, lon)
    assert np.all(dy > 0)
    assert np.all(dx > 0)


def test_latlon_mismatched_shapes_raises():
    lat = np.zeros((5, 5))
    lon = np.zeros((5, 6))
    with pytest.raises(ValueError):
        latlon_to_xy_spacing(lat, lon)


def test_resolve_spacing_position_mode():
    pos = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    sp = resolve_spacing(pos, axis=0, mode="position")
    np.testing.assert_allclose(sp, [1.0, 1.0, 2.0, 4.0, 4.0])


def test_resolve_spacing_spacing_mode_passthrough():
    sp_in = np.array([1.0, 1.0, 2.0, 4.0, 4.0])
    sp_out = resolve_spacing(sp_in, axis=0, mode="spacing")
    np.testing.assert_array_equal(sp_in, sp_out)


def test_resolve_spacing_auto_detects_position():
    pos = np.array([0.0, 1.0, 2.0, 4.0, 8.0])  # monotonic increasing
    sp = resolve_spacing(pos, axis=0, mode="auto")
    np.testing.assert_allclose(sp, [1.0, 1.0, 2.0, 4.0, 4.0])


def test_resolve_spacing_auto_detects_spacing():
    sp_in = np.array([1.0, 2.0, 1.0, 3.0, 1.0])  # not monotonic
    sp_out = resolve_spacing(sp_in, axis=0, mode="auto")
    np.testing.assert_array_equal(sp_in, sp_out)


def test_resolve_spacing_invalid_mode():
    with pytest.raises(ValueError):
        resolve_spacing(np.zeros(5), axis=0, mode="bogus")


# ---------------------------------------------------------------------------
# time_to_distance_km — wind-advected (Taylor's hypothesis) distance
# ---------------------------------------------------------------------------

def test_time_to_distance_constant_wind():
    time_s = np.arange(0, 100, 10.0)  # 10 points, 10s cadence
    dist_km = time_to_distance_km(time_s, wind_speed_ms=20.0)
    assert dist_km[0] == 0.0
    # 9 segments * 10s * 20 m/s = 1800 m = 1.8 km total
    assert abs(dist_km[-1] - 1.8) < 1e-9


def test_time_to_distance_scalar_broadcast_matches_array():
    time_s = np.arange(0, 50, 5.0)
    d_scalar = time_to_distance_km(time_s, 12.0)
    d_array = time_to_distance_km(time_s, np.full_like(time_s, 12.0))
    np.testing.assert_allclose(d_scalar, d_array)


def test_time_to_distance_negative_speed_uses_magnitude():
    time_s = np.arange(0, 50, 5.0)
    d_pos = time_to_distance_km(time_s, 12.0)
    d_neg = time_to_distance_km(time_s, -12.0)
    np.testing.assert_allclose(d_pos, d_neg)


def test_time_to_distance_monotonic_nondecreasing():
    time_s = np.arange(0, 200, 5.0)
    wind = 5 + 3 * np.sin(time_s / 30)  # variable, always positive
    dist_km = time_to_distance_km(time_s, wind)
    assert np.all(np.diff(dist_km) >= -1e-12)


def test_time_to_distance_zero_wind_is_flat():
    time_s = np.arange(0, 50, 5.0)
    dist_km = time_to_distance_km(time_s, 0.0)
    assert np.all(dist_km == 0.0)
