"""Tests for eccopy1d.run() — array-based 1-D classification."""

import numpy as np
import pytest

from eccopy import eccopy1d
from eccopy.params import WindowSpec


def _synthetic_profile(n=300, conv_center=200, seed=1):
    rng = np.random.default_rng(seed)
    x_km = np.linspace(0, 150, n)
    dbz = 18 + 2 * np.sin(x_km / 20) + rng.normal(0, 0.5, n)
    dbz[conv_center - 8: conv_center + 8] += 30 * np.exp(
        -0.5 * ((np.arange(-8, 8)) / 3) ** 2
    )
    return dbz, x_km


def test_basic_shapes():
    dbz, x_km = _synthetic_profile()
    r = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")))
    assert r.echo_type.shape == dbz.shape
    assert r.convectivity.shape == dbz.shape
    assert r.texture.shape == dbz.shape


def test_basic_classes_present():
    dbz, x_km = _synthetic_profile()
    r = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    # Only basic codes are possible for 1-D — no sub-classification
    assert codes.issubset({1, 2, 3})
    assert 1 in codes  # smooth background should be stratiform
    assert 3 in codes  # embedded spike should be convective


def test_convective_peak_detected():
    dbz, x_km = _synthetic_profile(conv_center=200)
    r = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")))
    assert r.echo_type[200] == 3
    assert r.convectivity[200] > r.convectivity[10]


def test_nan_propagation():
    dbz, x_km = _synthetic_profile()
    dbz[50:60] = np.nan
    r = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")))
    assert np.all(np.isnan(r.echo_type[50:60]))


def test_pixel_radius_window():
    dbz, x_km = _synthetic_profile()
    r = eccopy1d.run(dbz, coords=x_km, window=WindowSpec(7))
    assert r.echo_type.shape == dbz.shape


def test_coord_mode_spacing():
    dbz, x_km = _synthetic_profile()
    spacing = np.full_like(x_km, x_km[1] - x_km[0])
    r_pos = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                         coord_mode="position")
    r_sp = eccopy1d.run(dbz, coords=spacing, window=WindowSpec((5, "km")),
                        coord_mode="spacing")
    np.testing.assert_allclose(r_pos.texture, r_sp.texture, equal_nan=True)


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        eccopy1d.run(np.zeros((5, 5)), coords=np.zeros((5, 5)))


def test_rejects_mismatched_coords_shape():
    with pytest.raises(ValueError):
        eccopy1d.run(np.zeros(10), coords=np.zeros(5))


def test_list_input_accepted():
    dbz_list = [10.0, 12.0, 14.0, np.nan, 16.0, 18.0, 20.0]
    coords_list = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    r = eccopy1d.run(dbz_list, coords=coords_list, window=WindowSpec(2))
    assert r.echo_type.shape == (7,)


# ---------------------------------------------------------------------------
# Wind-advected distance coordinate (time_to_distance_km)
# ---------------------------------------------------------------------------

def test_wind_advected_distance_feeds_run():
    from eccopy.core.coords import time_to_distance_km

    n = 300
    time_s = np.arange(n) * 10.0  # 10-second cadence
    dist_km = time_to_distance_km(time_s, wind_speed_ms=15.0)

    rng = np.random.default_rng(2)
    dbz = 18 + 2 * np.sin(dist_km / 20) + rng.normal(0, 0.5, n)
    dbz[150 - 8: 150 + 8] += 30 * np.exp(-0.5 * ((np.arange(-8, 8)) / 3) ** 2)

    r = eccopy1d.run(dbz, dist_km, coord_mode="position",
                     window=WindowSpec((5, "km")))
    assert r.echo_type.shape == dbz.shape
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    assert codes.issubset({1, 2, 3})


def test_wind_advected_distance_variable_speed_monotonic():
    from eccopy.core.coords import time_to_distance_km

    time_s = np.arange(100) * 5.0
    wind = 8 + 3 * np.sin(np.arange(100) / 15)  # variable, always positive
    dist_km = time_to_distance_km(time_s, wind)
    assert dist_km[0] == 0.0
    assert np.all(np.diff(dist_km) >= 0)


# ---------------------------------------------------------------------------
# Minimum-convective-length clump filter
# ---------------------------------------------------------------------------

def test_min_convective_length_suppresses_brief_blip():
    rng = np.random.default_rng(3)
    n = 400
    x_km = np.linspace(0, 200, n)  # 0.5 km spacing
    dbz = 18 + 2 * np.sin(x_km / 20) + rng.normal(0, 0.5, n)
    dbz[150] += 35                          # single-sample erroneous spike
    dbz[250:290] += rng.normal(0, 8, 40)    # sustained ~20 km convective stretch

    win = WindowSpec(1)  # minimal-radius pixel window -> minimal texture smearing
    r_nofilter = eccopy1d.run(dbz, x_km, window=win)
    r_filtered = eccopy1d.run(dbz, x_km, window=win, min_convective_length=15)

    # Without a filter, the blip registers as Convective.
    assert 3 in r_nofilter.echo_type[145:156]
    # With the filter, the blip is demoted (no longer Convective there)...
    assert 3 not in r_filtered.echo_type[145:156]
    # ...but the sustained, genuinely-wide convective stretch survives.
    assert 3 in r_filtered.echo_type[260:280]


def test_min_convective_length_disabled_by_default():
    dbz, x_km = _synthetic_profile()
    r_default = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")))
    r_explicit_none = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                                   min_convective_length=None)
    np.testing.assert_array_equal(r_default.echo_type, r_explicit_none.echo_type)


def test_min_convective_length_pixel_count_form():
    # A bare int is accepted (interpreted as minimum point count, not physical size).
    dbz, x_km = _synthetic_profile()
    r = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                     min_convective_length=3)
    assert r.echo_type.shape == dbz.shape


# ---------------------------------------------------------------------------
# kernel_mode
# ---------------------------------------------------------------------------

def test_kernel_mode_uniform_is_default():
    dbz, x_km = _synthetic_profile()
    r_default = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")))
    r_explicit = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                              kernel_mode="uniform")
    np.testing.assert_array_equal(r_default.echo_type, r_explicit.echo_type)
    np.testing.assert_allclose(r_default.texture, r_explicit.texture, equal_nan=True)


def test_kernel_mode_uniform_matches_varying_on_uniform_track():
    """_synthetic_profile() has uniform point spacing -- uniform and
    varying kernel_mode must agree exactly there."""
    dbz, x_km = _synthetic_profile()
    r_uniform = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                             kernel_mode="uniform")
    r_varying = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                             kernel_mode="varying")
    np.testing.assert_allclose(r_uniform.texture, r_varying.texture, equal_nan=True)


def test_kernel_mode_varying_differs_on_irregular_spacing():
    n = 200
    rng = np.random.default_rng(5)
    # Irregular sampling gaps -- e.g. a time series with dropped samples.
    steps = rng.choice([1.0, 1.0, 1.0, 5.0], size=n)  # occasional big gaps
    x_km = np.cumsum(steps)
    dbz = 18 + 2 * np.sin(x_km / 20) + rng.normal(0, 0.5, n)

    r_uniform = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                             kernel_mode="uniform")
    r_varying = eccopy1d.run(dbz, coords=x_km, window=WindowSpec((5, "km")),
                             kernel_mode="varying")
    assert not np.allclose(r_uniform.texture, r_varying.texture, equal_nan=True)


def test_return_intermediates_default_off_and_matches_when_on():
    dbz, coords = _synthetic_profile()
    r_default = eccopy1d.run(dbz, coords, window=WindowSpec((5, "km")))
    r_debug = eccopy1d.run(dbz, coords, window=WindowSpec((5, "km")), return_intermediates=True)

    assert r_default.fitted_dbz is None
    assert r_default.detrended_dbz is None
    assert r_debug.fitted_dbz is not None
    assert r_debug.fitted_dbz.shape == dbz.shape
    assert r_debug.detrended_dbz.shape == dbz.shape

    np.testing.assert_allclose(r_default.texture, r_debug.texture, equal_nan=True)
    np.testing.assert_allclose(r_default.convectivity, r_debug.convectivity, equal_nan=True)
    np.testing.assert_array_equal(r_default.echo_type, r_debug.echo_type)
