"""Tests for eccopy.core.texture — refl_texture_2d kernel_mode handling."""

import numpy as np
import pytest

from eccopy.core.texture import (
    refl_texture_2d, _point_kernel_offsets, _build_kernel_offsets_uniform,
    _max_half_width,
)


def test_uniform_and_point_kernel_agree_on_uniform_grid():
    """On a truly uniform grid, the per-point kernel at any location
    should match the single uniform-grid kernel exactly."""
    dx_km = dy_km = 1.0
    radius_km = 7.0
    offsets_uniform, nx_u, ny_u = _build_kernel_offsets_uniform(radius_km, dx_km, dy_km)
    offsets_point, nx_p, ny_p = _point_kernel_offsets(radius_km, dx_km, dy_km)
    assert nx_u == nx_p
    assert ny_u == ny_p
    assert sorted(offsets_uniform) == sorted(offsets_point)


def test_point_kernel_shrinks_with_finer_spacing():
    """Finer local spacing should produce a kernel with MORE points for
    the same physical radius."""
    radius_km = 10.0
    offsets_coarse, _, _ = _point_kernel_offsets(radius_km, 2.0, 2.0)
    offsets_fine, _, _ = _point_kernel_offsets(radius_km, 0.5, 0.5)
    assert len(offsets_fine) > len(offsets_coarse)


def test_point_kernel_degenerates_gracefully():
    """When spacing far exceeds the radius, the kernel should still
    return at least the center point, not crash or return empty."""
    offsets, nx_tex, ny_tex = _point_kernel_offsets(1.0, 50.0, 50.0)
    assert len(offsets) >= 1
    assert (0, 0, 0.0, 0.0) in offsets


def test_max_half_width_uses_finest_spacing():
    dx = np.array([[1.0, 1.0], [5.0, 5.0]])
    dy = np.array([[1.0, 1.0], [5.0, 5.0]])
    nx_tex, ny_tex = _max_half_width(10.0, dx, dy)
    # Finest spacing (1.0) should drive a larger half-width than coarsest (5.0)
    nx_tex_coarse, ny_tex_coarse = _max_half_width(10.0, np.full((2, 2), 5.0),
                                                    np.full((2, 2), 5.0))
    assert nx_tex >= nx_tex_coarse


def test_refl_texture_2d_kernel_mode_uniform_matches_legacy_default():
    rng = np.random.default_rng(0)
    nz, ny, nx = 1, 30, 30
    dbz = 15 + rng.normal(0, 1, (nz, ny, nx))
    dy = np.full((ny, nx), 1.0)
    dx = np.full((ny, nx), 1.0)
    tex_default, frac_default = refl_texture_2d(dbz, 5.0, dy, dx)
    tex_explicit, frac_explicit = refl_texture_2d(dbz, 5.0, dy, dx, kernel_mode="uniform")
    np.testing.assert_array_equal(tex_default, tex_explicit)
    np.testing.assert_array_equal(frac_default, frac_explicit)


def test_refl_texture_2d_uniform_grid_modes_agree():
    """On a perfectly uniform grid, 'uniform' and 'varying' modes should
    produce identical (or near-identical) results, since every point's
    local kernel is the same as the single global kernel."""
    rng = np.random.default_rng(1)
    nz, ny, nx = 1, 25, 25
    dbz = 15 + rng.normal(0, 1, (nz, ny, nx))
    dy = np.full((ny, nx), 1.0)
    dx = np.full((ny, nx), 1.0)
    tex_u, _ = refl_texture_2d(dbz, 5.0, dy, dx, kernel_mode="uniform")
    tex_v, _ = refl_texture_2d(dbz, 5.0, dy, dx, kernel_mode="varying")
    np.testing.assert_allclose(tex_u, tex_v, equal_nan=True, atol=1e-10)


def test_refl_texture_2d_invalid_kernel_mode():
    dbz = np.zeros((1, 10, 10))
    dy = np.full((10, 10), 1.0)
    dx = np.full((10, 10), 1.0)
    with pytest.raises(ValueError):
        refl_texture_2d(dbz, 5.0, dy, dx, kernel_mode="nonsense")
