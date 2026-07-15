"""Tests for eccopy2d_h.run() — array-based 2-D horizontal/composite."""

import numpy as np
import pytest

from eccopy import eccopy2d_h
from eccopy.params import WindowSpec


def _synthetic_composite(ny=60, nx=80, seed=3):
    rng = np.random.default_rng(seed)
    y_km = np.linspace(-30, 30, ny)
    x_km = np.linspace(-40, 40, nx)
    dbz = 15 + rng.normal(0, 0.5, (ny, nx))
    yy, xx = np.meshgrid(y_km, x_km, indexing="ij")
    cell = np.exp(-((yy - 5) ** 2 + (xx - 10) ** 2) / 30)
    dbz += 30 * cell
    return dbz, y_km, x_km


def test_basic_shapes():
    dbz, y_km, x_km = _synthetic_composite()
    r = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                       window=WindowSpec((7, "km")))
    assert r.echo_type.shape == dbz.shape
    assert r.convectivity.shape == dbz.shape
    assert r.texture.shape == dbz.shape
    assert r.fraction_active.shape == dbz.shape


def test_only_basic_codes():
    dbz, y_km, x_km = _synthetic_composite()
    r = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                       window=WindowSpec((7, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    assert codes.issubset({1, 2, 3})


def test_convective_cell_detected():
    dbz, y_km, x_km = _synthetic_composite()
    r = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                       window=WindowSpec((7, "km")))
    # The embedded cell is near y=5, x=10
    iy = int(np.argmin(np.abs(y_km - 5)))
    ix = int(np.argmin(np.abs(x_km - 10)))
    assert r.echo_type[iy, ix] in (2, 3)


def test_no_signature_accepts_height_or_temp():
    """eccopy2d_h.run() should not accept height/temp kwargs (no vertical axis)."""
    dbz, y_km, x_km = _synthetic_composite()
    import inspect
    sig = inspect.signature(eccopy2d_h.run)
    assert "height" not in sig.parameters
    assert "temp" not in sig.parameters


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        eccopy2d_h.run(np.zeros(10), coords_y=np.zeros(5), coords_x=np.zeros(2))


def _latlon_grid(ny=80, nx=100, lat_range=(20, 50), lon_range=(-15, 15)):
    from eccopy.core.coords import latlon_to_xy_spacing
    lat_1d = np.linspace(*lat_range, ny)
    lon_1d = np.linspace(*lon_range, nx)
    lat = np.tile(lat_1d[:, None], (1, nx))
    lon = np.tile(lon_1d[None, :], (ny, 1))
    return latlon_to_xy_spacing(lat, lon)


def test_kernel_mode_uniform_is_default():
    dbz, y_km, x_km = _synthetic_composite()
    r_default = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                               window=WindowSpec((7, "km")))
    r_explicit = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                                window=WindowSpec((7, "km")), kernel_mode="uniform")
    np.testing.assert_array_equal(r_default.texture, r_explicit.texture)


def test_kernel_mode_varying_runs():
    dy_km, dx_km = _latlon_grid(lat_range=(20, 45))
    ny, nx = dy_km.shape
    rng = np.random.default_rng(9)
    dbz = 15 + rng.normal(0, 0.5, (ny, nx))
    dbz[35:45, 45:55] += 30
    r = eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km, coord_mode="spacing",
                       window=WindowSpec((40, "km")), kernel_mode="varying")
    assert r.echo_type.shape == dbz.shape
    assert not np.all(np.isnan(r.texture))


def test_kernel_mode_uniform_and_varying_differ_on_nonuniform_grid():
    dy_km, dx_km = _latlon_grid(lat_range=(20, 45))
    ny, nx = dy_km.shape
    rng = np.random.default_rng(9)
    dbz = 15 + rng.normal(0, 0.5, (ny, nx))
    dbz[35:45, 45:55] += 30
    r_u = eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km, coord_mode="spacing",
                         window=WindowSpec((40, "km")), kernel_mode="uniform")
    r_v = eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km, coord_mode="spacing",
                         window=WindowSpec((40, "km")), kernel_mode="varying")
    # dx varies meaningfully with latitude on this grid, so the two modes
    # should not produce identical texture fields.
    assert not np.allclose(r_u.texture, r_v.texture, equal_nan=True)


def test_kernel_mode_invalid_raises():
    dbz, y_km, x_km = _synthetic_composite()
    with pytest.raises(ValueError):
        eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                       window=WindowSpec((7, "km")), kernel_mode="bogus")


def test_kernel_mode_varying_warns_on_degenerate_kernel():
    # Texture radius much smaller than grid spacing -> kernel collapses
    # to a single cell -> should warn.
    dy_km, dx_km = _latlon_grid(ny=40, nx=50, lat_range=(20, 70))
    ny, nx = dy_km.shape
    rng = np.random.default_rng(9)
    dbz = 15 + rng.normal(0, 0.5, (ny, nx))
    with pytest.warns(UserWarning):
        eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km, coord_mode="spacing",
                       window=WindowSpec((1, "km")), kernel_mode="varying")


# ---------------------------------------------------------------------------
# Clumping integration (n_clumps, min_convective_area)
# ---------------------------------------------------------------------------

def test_n_clumps_present_and_nonnegative():
    dbz, y_km, x_km = _synthetic_composite()
    r = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                       window=WindowSpec((7, "km")))
    assert isinstance(r.n_clumps, int)
    assert r.n_clumps >= 0


def test_min_convective_area_none_is_default_and_unfiltered():
    dbz, y_km, x_km = _synthetic_composite()
    r_default = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                               window=WindowSpec((7, "km")))
    r_explicit = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                                window=WindowSpec((7, "km")),
                                min_convective_area=None)
    np.testing.assert_array_equal(r_default.echo_type, r_explicit.echo_type)


def test_min_convective_area_can_only_shrink_convective_footprint():
    dbz, y_km, x_km = _synthetic_composite()
    r_nofilter = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                                window=WindowSpec((7, "km")))
    r_filtered = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                                window=WindowSpec((7, "km")),
                                min_convective_area=1e6)  # absurdly large -> drops everything
    assert np.sum(r_filtered.echo_type == 3) <= np.sum(r_nofilter.echo_type == 3)
    assert r_filtered.n_clumps == 0


def test_use_dual_thresholds_toggle_runs():
    from eccopy.params import ClassificationParams
    dbz, y_km, x_km = _synthetic_composite()
    cp = ClassificationParams(use_dual_thresholds=False)
    r = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                       window=WindowSpec((7, "km")), class_params=cp)
    assert r.echo_type.shape == dbz.shape


def test_return_intermediates_default_off_and_matches_when_on():
    # Small grid -- the debug path here is an explicit slow per-point loop.
    dbz, y_km, x_km = _synthetic_composite(ny=14, nx=14)
    r_default = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                               window=WindowSpec((7, "km")))
    r_debug = eccopy2d_h.run(dbz, coords_y=y_km, coords_x=x_km,
                             window=WindowSpec((7, "km")), return_intermediates=True)

    assert r_default.fitted_dbz is None
    assert r_default.detrended_dbz is None
    assert r_debug.fitted_dbz is not None
    assert r_debug.fitted_dbz.shape == dbz.shape
    assert r_debug.detrended_dbz.shape == dbz.shape

    np.testing.assert_allclose(r_default.texture, r_debug.texture, equal_nan=True)
    np.testing.assert_array_equal(r_default.echo_type, r_debug.echo_type)
    assert r_default.n_clumps == r_debug.n_clumps
