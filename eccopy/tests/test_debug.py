"""
Tests for eccopy.core.debug — the introspection/debug functions must
exactly reproduce the production texture functions' output at every
point, including edge cases (NaN gaps, array borders, varying spacing).

These tests intentionally stress several tricky agreement points that
were bugs during development:
  - the center point itself being NaN (must report texture=NaN, but a
    correctly-computed fraction_active — these are two separate checks
    in production, not one)
  - points within the border excluded by production's kernel half-width
    (must report texture=NaN AND fraction_active=0.0, matching
    production's array initialisation, not NaN)
  - non-uniform spacing (kernel_mode="varying")
"""

import numpy as np
import pytest

from eccopy.core.texture import refl_texture_1d, refl_texture_2d
from eccopy.core.debug import refl_texture_1d_debug, refl_texture_2d_debug
from eccopy.params import WindowSpec


# ---------------------------------------------------------------------------
# 1-D debug exact-agreement tests
# ---------------------------------------------------------------------------

def test_1d_debug_matches_production_simple():
    dbz = np.array([18.]*10 + [19,22,28,35,42,45,42,35,28,22,19] + [18.]*10)
    window = WindowSpec((3, "km"))
    spacing_m = np.full(dbz.shape, 1000.0)
    texture_full = refl_texture_1d(dbz, window, spacing=spacing_m)
    for i in range(len(dbz)):
        dbg = refl_texture_1d_debug(dbz, window, index=i, spacing=spacing_m)
        prod = texture_full[i]
        assert np.isclose(dbg.texture, prod, atol=1e-9) or (np.isnan(dbg.texture) and np.isnan(prod))


def test_1d_debug_matches_production_with_nan_gap():
    rng = np.random.default_rng(7)
    n = 80
    dbz = 18 + rng.normal(0, 3, n)
    dbz[40:50] += 20
    dbz[15:18] = np.nan
    window = WindowSpec((4, "km"))
    spacing_m = np.full(n, 1000.0)
    texture_full = refl_texture_1d(dbz, window, spacing=spacing_m)
    for i in range(n):
        dbg = refl_texture_1d_debug(dbz, window, index=i, spacing=spacing_m)
        prod = texture_full[i]
        assert np.isclose(dbg.texture, prod, atol=1e-9) or (np.isnan(dbg.texture) and np.isnan(prod))


def test_1d_debug_at_nan_point_reports_nan_texture():
    dbz = np.array([18.0, 19.0, np.nan, 21.0, 18.0, 18.0, 18.0])
    window = WindowSpec(1)
    dbg = refl_texture_1d_debug(dbz, window, index=2)
    assert np.isnan(dbg.texture)
    assert np.isnan(dbg.window_dbz[0])  # raw value preserved as NaN


def test_1d_debug_random_stress():
    total = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = rng.integers(30, 120)
        dbz = 15 + rng.normal(0, 4, n)
        if rng.random() < 0.5:
            gap_start = rng.integers(0, n - 5)
            dbz[gap_start:gap_start + 3] = np.nan
        win_km = rng.uniform(2, 8)
        window = WindowSpec((win_km, "km"))
        spacing_m = np.full(n, rng.uniform(500, 2000))
        texture_full = refl_texture_1d(dbz, window, spacing=spacing_m)
        for i in range(n):
            dbg = refl_texture_1d_debug(dbz, window, index=i, spacing=spacing_m)
            prod = texture_full[i]
            assert np.isclose(dbg.texture, prod, atol=1e-6) or (np.isnan(dbg.texture) and np.isnan(prod))
            total += 1
    assert total > 500  # sanity: stress test actually ran a meaningful number of points


# ---------------------------------------------------------------------------
# 2-D debug exact-agreement tests
# ---------------------------------------------------------------------------

def test_2d_debug_matches_production_simple():
    rng = np.random.default_rng(3)
    ny, nx = 25, 25
    dbz_level = 18 + rng.normal(0, 2, (ny, nx))
    dbz_level[10:15, 10:15] += 25
    dbz_3d = dbz_level[np.newaxis, :, :]
    dy = np.full((ny, nx), 1.0)
    dx = np.full((ny, nx), 1.0)
    texture_full, frac_full = refl_texture_2d(dbz_3d, 5.0, dy, dx)
    for iy in range(ny):
        for ix in range(nx):
            dbg = refl_texture_2d_debug(dbz_level, 5.0, dy, dx, iy, ix)
            prod_tex = texture_full[0, iy, ix]
            assert np.isclose(dbg.texture, prod_tex, atol=1e-6) or (np.isnan(dbg.texture) and np.isnan(prod_tex))
            assert np.isclose(dbg.fraction_active, frac_full[iy, ix], atol=1e-9)


def test_2d_debug_center_nan_reports_nan_texture_but_real_fraction():
    """A center point that is itself NaN must report texture=NaN, but
    fraction_active should still reflect real kernel coverage (these are
    two separate checks in production, not one)."""
    rng = np.random.default_rng(7)
    ny, nx = 33, 27
    dbz_level = 15 + rng.normal(0, 3, (ny, nx))
    dbz_level[11:13, 16:18] = np.nan
    dbz_3d = dbz_level[np.newaxis, :, :]
    radius_km = 6.47
    dy = np.full((ny, nx), 1.718)
    dx = np.full((ny, nx), 1.179)

    texture_full, frac_full = refl_texture_2d(dbz_3d, radius_km, dy, dx)
    dbg = refl_texture_2d_debug(dbz_level, radius_km, dy, dx, 11, 16)

    assert np.isnan(texture_full[0, 11, 16])
    assert np.isnan(dbg.texture)
    assert frac_full[11, 16] > 0.9  # high coverage despite center being NaN
    assert np.isclose(dbg.fraction_active, frac_full[11, 16], atol=1e-9)


def test_2d_debug_border_reports_zero_fraction_not_nan():
    """Points within the kernel half-width of the array edge are never
    computed by production; fraction_active there is 0.0 (array init
    value), not NaN."""
    rng = np.random.default_rng(1)
    ny, nx = 25, 25
    dbz_level = 15 + rng.normal(0, 2, (ny, nx))
    dbz_3d = dbz_level[np.newaxis, :, :]
    dy = np.full((ny, nx), 1.0)
    dx = np.full((ny, nx), 1.0)
    texture_full, frac_full = refl_texture_2d(dbz_3d, 5.0, dy, dx)

    dbg = refl_texture_2d_debug(dbz_level, 5.0, dy, dx, 0, 0)
    assert np.isnan(dbg.texture)
    assert np.isnan(texture_full[0, 0, 0])
    assert dbg.fraction_active == 0.0
    assert frac_full[0, 0] == 0.0


def test_2d_debug_random_stress():
    total = 0
    for seed in range(8):
        rng = np.random.default_rng(seed)
        ny, nx = rng.integers(15, 35), rng.integers(15, 35)
        dbz_level = 15 + rng.normal(0, 3, (ny, nx))
        if rng.random() < 0.5:
            gy, gx = rng.integers(0, ny - 3), rng.integers(0, nx - 3)
            dbz_level[gy:gy + 2, gx:gx + 2] = np.nan
        dbz_3d = dbz_level[np.newaxis, :, :]
        radius_km = rng.uniform(3, 8)
        dy = np.full((ny, nx), rng.uniform(0.5, 2.0))
        dx = np.full((ny, nx), rng.uniform(0.5, 2.0))
        texture_full, frac_full = refl_texture_2d(dbz_3d, radius_km, dy, dx)
        for iy in range(ny):
            for ix in range(nx):
                dbg = refl_texture_2d_debug(dbz_level, radius_km, dy, dx, iy, ix)
                prod_tex = texture_full[0, iy, ix]
                assert (np.isclose(dbg.texture, prod_tex, atol=1e-6)
                       or (np.isnan(dbg.texture) and np.isnan(prod_tex)))
                assert np.isclose(dbg.fraction_active, frac_full[iy, ix], atol=1e-9)
                total += 1
    assert total > 3000


def test_2d_debug_kernel_mode_varying():
    rng = np.random.default_rng(2)
    ny, nx = 30, 30
    dbz_level = 15 + rng.normal(0, 2, (ny, nx))
    dbz_3d = dbz_level[np.newaxis, :, :]
    dy = np.full((ny, nx), 1.5)
    dx = np.full((ny, nx), 0.8)
    texture_full, frac_full = refl_texture_2d(dbz_3d, 6.0, dy, dx, kernel_mode="varying")
    for iy in [10, 15, 20]:
        for ix in [10, 15, 20]:
            dbg = refl_texture_2d_debug(dbz_level, 6.0, dy, dx, iy, ix, kernel_mode="varying")
            prod_tex = texture_full[0, iy, ix]
            assert (np.isclose(dbg.texture, prod_tex, atol=1e-6)
                   or (np.isnan(dbg.texture) and np.isnan(prod_tex)))


def test_2d_debug_invalid_kernel_mode_raises():
    dbz_level = np.zeros((20, 20))
    dy = np.full((20, 20), 1.0)
    dx = np.full((20, 20), 1.0)
    with pytest.raises(ValueError):
        refl_texture_2d_debug(dbz_level, 5.0, dy, dx, 10, 10, kernel_mode="bogus")


# ---------------------------------------------------------------------------
# refl_texture_1d_with_fit -- full-field fitted/detrended debug output
# (item 1 of the v0.1 checklist: "output intermediate fields for debugging")
# ---------------------------------------------------------------------------

from eccopy.core.texture import refl_texture_1d_with_fit


def test_refl_texture_1d_with_fit_matches_production_texture():
    np.random.seed(0)
    dbz = np.random.uniform(0, 45, size=(4, 70))
    dbz[2, 20:25] = np.nan
    window = WindowSpec((4, "km"))
    spacing_m = np.full(dbz.shape, 1000.0)

    texture_prod = refl_texture_1d(dbz, window, spacing=spacing_m)
    texture_dbg, fitted, detrended = refl_texture_1d_with_fit(dbz, window, spacing=spacing_m)

    assert np.allclose(texture_prod, texture_dbg, equal_nan=True)
    assert fitted.shape == dbz.shape
    assert detrended.shape == dbz.shape
    # NaN pattern for detrended must match production's texture NaN pattern
    # (both come from the same original-NaN + border exclusion logic).
    assert np.array_equal(np.isnan(texture_prod), np.isnan(detrended))


def test_refl_texture_1d_with_fit_pixel_window():
    dbz = np.array([18.0] * 10 + [19, 22, 28, 35, 42, 45, 42, 35, 28, 22, 19] + [18.0] * 10)
    texture_prod = refl_texture_1d(dbz, window=3)
    texture_dbg, fitted, detrended = refl_texture_1d_with_fit(dbz, window=3)
    assert np.allclose(texture_prod, texture_dbg, equal_nan=True)
    assert fitted.shape == dbz.shape
