"""
Tests for EccoPy 1.1's memory reductions.

Default behaviour must be unchanged, bit for bit:
  - float32 reflectivity is used as is instead of being copied to float64;
  - spacing derived from 1-D coordinates is a broadcast view, not a copy;
  - values below min_valid_dbz are skipped inside the texture cores
    instead of in a NaN-masked copy of the input;
  - convectivity is built in place; clump area is formed per bounding box.

Opt-in dtype=np.float32 changes only the rounding of the stored outputs.
"""

import warnings

import numpy as np
import pytest

from eccopy import eccopy2d_h, latlon_grid_spacing
from eccopy.core.classification import assign_echo_type_2d
from eccopy.core.convectivity import texture_to_convectivity_linear
from eccopy.core.coords import latlon_to_xy_spacing
from eccopy.core.texture import refl_texture_2d
from eccopy.eccopy2d_h.clumping import find_clumps_2d

from ._reference_impls.refl_texture_2d_v1_0_1 import refl_texture_2d_v1_0_1
from .test_texture_prepared import _latlon_case, _random_case, _same, _WINDOW

_FIELDS = ("echo_type", "convectivity", "texture", "fraction_active")


@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_float32_input_gives_identical_float64_output(kernel_mode):
    """float32 values promote to float64 exactly, so passing them as
    float32 (no copy) or as float64 must give the same bits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(8):
            dbz, dy, dx, radius, kw = _random_case(seed, "latlon")
            d32 = dbz.astype(np.float32)
            t32, f32 = refl_texture_2d(d32, radius, dy, dx, kernel_mode=kernel_mode, **kw)
            t64, f64 = refl_texture_2d_v1_0_1(d32.astype(np.float64), radius, dy, dx,
                                              kernel_mode=kernel_mode, **kw)
            assert t32.dtype == np.float64
            assert _same(t32, t64) and _same(f32, f64)


def test_min_valid_dbz_below_fill_value_matches_1_0_1():
    """1.0 masked values below min_valid_dbz, then filled empty columns
    with -9999 before counting coverage. The cores now test values in
    place; check the corner where the fill itself passes the threshold."""
    rng = np.random.default_rng(3)
    dbz = rng.normal(20, 10, (3, 30, 40))
    dbz[:, 5:12, 5:15] = -20000.0           # below the threshold and the fill
    dbz[rng.random(dbz.shape) < 0.3] = np.nan
    dy = np.full((30, 40), 1.0)
    dx = np.full((30, 40), 1.0)
    for mv in (-10000.0, -9999.0, -np.inf, 0.0, np.nan):
        for mode in ("uniform", "varying"):
            new = refl_texture_2d(dbz, 4.0, dy, dx, min_valid_dbz=mv, kernel_mode=mode)
            ref = refl_texture_2d_v1_0_1(dbz, 4.0, dy, dx, min_valid_dbz=mv, kernel_mode=mode)
            assert _same(new[0], ref[0]) and _same(new[1], ref[1]), (mv, mode)


def test_texture_stage_does_not_modify_input():
    dbz, dy, dx, radius, kw = _random_case(1, "latlon")
    before = dbz.copy()
    refl_texture_2d(dbz, radius, dy, dx, kernel_mode="varying", **kw)
    assert _same(dbz, before)


def test_prepare_grid_keeps_1d_spacing_as_views():
    grid = eccopy2d_h.prepare_grid(np.arange(500) * 1.0, np.arange(800) * 0.8)
    assert grid.spacing_y.strides[1] == 0
    assert grid.spacing_x.strides[0] == 0
    assert not grid.spacing_y.flags.writeable


@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_broadcast_spacing_matches_materialised(kernel_mode):
    dy, dx, frames = _latlon_case()
    dy_v = np.broadcast_to(dy[:, :1], dy.shape)
    dx_v = np.broadcast_to(dx[:, :1], dx.shape)
    a = eccopy2d_h.run(frames[0], dy_v, dx_v, coord_mode="spacing",
                       window=_WINDOW, kernel_mode=kernel_mode)
    b = eccopy2d_h.run(frames[0], np.array(dy_v), np.array(dx_v),
                       coord_mode="spacing", window=_WINDOW,
                       kernel_mode=kernel_mode)
    for f in _FIELDS:
        assert _same(getattr(a, f), getattr(b, f)), f


def test_latlon_grid_spacing_matches_latlon_to_xy_spacing():
    lat = 54.995 - 0.01 * np.arange(300)
    lon = -129.995 + 0.01 * np.arange(400)
    dy_v, dx_v = latlon_grid_spacing(lat, lon)
    dy, dx = latlon_to_xy_spacing(*np.meshgrid(lat, lon, indexing="ij"))
    assert dy_v.shape == dx_v.shape == (300, 400)
    assert not dy_v.flags.writeable and dx_v.strides[1] == 0
    # The same haversine distances, taken along the first column...
    np.testing.assert_array_equal(dy_v[:, 0], dy[:, 0])
    np.testing.assert_array_equal(dx_v[:, 0], dx[:, 0])
    # ...and latlon_to_xy_spacing's dx varies along a row only by rounding.
    np.testing.assert_allclose(dx_v, dx, rtol=1e-10)
    np.testing.assert_array_equal(dy_v, dy)


def test_latlon_grid_spacing_rejects_irregular_or_2d():
    with pytest.raises(ValueError, match="uniformly spaced"):
        latlon_grid_spacing(np.linspace(30, 40, 10), np.array([0.0, 0.1, 0.3]))
    with pytest.raises(ValueError, match="1-D"):
        latlon_grid_spacing(np.zeros((3, 3)), np.zeros((3, 3)))


def test_convectivity_in_place_matches_formula():
    rng = np.random.default_rng(0)
    tex = rng.uniform(-5, 40, (50, 60))
    tex[rng.random(tex.shape) < 0.2] = np.nan
    for lo, hi in ((0.0, 30.0), (3.0, 25.0)):
        old = np.clip((tex - lo) / (hi - lo), 0.0, 1.0)
        if lo > 0.0:
            old = np.where(tex < lo, np.nan, old)
        assert _same(texture_to_convectivity_linear(tex, hi, lo), old)
    c32 = texture_to_convectivity_linear(tex.astype(np.float32), 30.0)
    assert c32.dtype == np.float32


def test_clump_area_per_bbox_matches_full_array():
    dy, dx, frames = _latlon_case()
    r = eccopy2d_h.run(frames[2], dy, dx, coord_mode="spacing", window=_WINDOW)
    dy_v = np.broadcast_to(dy[:, :1], dy.shape)
    dx_v = np.broadcast_to(dx[:, :1], dx.shape)
    a = find_clumps_2d(r.convectivity, dy_v, dx_v, min_conv=0.5)
    b = find_clumps_2d(r.convectivity, np.array(dy_v), np.array(dx_v), min_conv=0.5)
    assert len(a) == len(b) > 0
    for ca, cb in zip(a, b):
        assert ca["area_km2"] == cb["area_km2"]
        assert all(np.array_equal(i, j) for i, j in zip(ca["index"], cb["index"]))


def test_assign_echo_type_2d_dtype():
    conv = np.array([[0.1, 0.45, np.nan], [0.9, 0.2, 0.41]])
    e64 = assign_echo_type_2d(conv, [], 0.4)
    e32 = assign_echo_type_2d(conv.astype(np.float32), [], 0.4, dtype=np.float32)
    assert e64.dtype == np.float64 and e32.dtype == np.float32
    assert _same(e64, e32.astype(np.float64))


@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_float32_output_mode(kernel_mode):
    dy, dx, frames = _latlon_case()
    grid = eccopy2d_h.prepare_grid(dy, dx, coord_mode="spacing", window=_WINDOW,
                                   kernel_mode=kernel_mode)
    for dbz in frames:
        a = eccopy2d_h.run(dbz, grid=grid)
        b = eccopy2d_h.run(dbz, grid=grid, dtype=np.float32)
        for f in _FIELDS:
            assert getattr(b, f).dtype == np.float32, f
        # Only the final rounding of each stored value changes...
        np.testing.assert_allclose(b.texture, a.texture, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(b.convectivity, a.convectivity, rtol=0, atol=1e-6)
        np.testing.assert_allclose(b.fraction_active, a.fraction_active, rtol=1e-6)
        # ...and the set of points with a texture is decided in float64.
        assert np.array_equal(np.isnan(a.texture), np.isnan(b.texture))
        assert np.array_equal(a.echo_type, b.echo_type.astype(np.float64), equal_nan=True)
        assert a.n_clumps == b.n_clumps


def test_dtype_rejects_non_float():
    dy, dx, frames = _latlon_case()
    with pytest.raises(ValueError, match="float32 or float64"):
        eccopy2d_h.run(frames[0], dy, dx, coord_mode="spacing", window=_WINDOW,
                       dtype=np.int16)
