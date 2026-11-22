"""
Tests for EccoPy 1.1's prepared 2-D texture windows (core.texture.
_prepare_texture_2d, _fraction_active_rows, _texture_rows) and for
eccopy2d_h.prepare_grid() / run(grid=...).

The 1.1 texture path must reproduce EccoPy 1.0.1 bit for bit, in both
kernel modes, so every comparison here uses exact equality against the
frozen 1.0.1 driver (_reference_impls/refl_texture_2d_v1_0_1.py), which
calls the 1.0.1 Numba cores that are themselves cross-validated against
pure-Python references in test_numba_cross_validation.py.
"""

import warnings

import numpy as np
import pytest

from eccopy import eccopy2d_h
from eccopy.core.texture import (
    ROW_PER_POINT, ROW_SKIP, _nanmedian_all, _point_kernel_offsets_arrays,
    _prepare_texture_2d, refl_texture_2d,
)
from eccopy.params import TextureParams, WindowSpec

from ._reference_impls.refl_texture_2d_v1_0_1 import refl_texture_2d_v1_0_1


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _random_case(seed, kind):
    rng = np.random.default_rng(seed)
    nz = int(rng.integers(1, 4))
    ny, nx = int(rng.integers(20, 60)), int(rng.integers(20, 60))
    dbz = rng.normal(20, 10, (nz, ny, nx))
    if rng.random() < 0.7:
        dbz[rng.random(dbz.shape) < rng.uniform(0, 0.6)] = np.nan
    if rng.random() < 0.3:
        dbz = np.round(dbz * 2) / 2          # 0.5 dB quantisation, as MRMS
    if kind == "latlon":
        lat = np.linspace(rng.uniform(20, 50), rng.uniform(55, 65), ny)
        dy = np.full((ny, nx), 1.112)
        dx = (1.112 * np.cos(np.radians(lat)))[:, None] * np.ones((1, nx))
        # The along-row rounding noise latlon_to_xy_spacing() leaves in dx.
        dx = dx * (1 + 1e-12 * rng.standard_normal(dx.shape))
    elif kind == "uniform":
        dy = np.full((ny, nx), rng.uniform(0.3, 2.0))
        dx = np.full((ny, nx), rng.uniform(0.3, 2.0))
    elif kind == "xvary":      # not lat/lon: spacing varies along x
        dx = np.ones((ny, 1)) * rng.uniform(0.4, 2.0, (1, nx))
        dy = rng.uniform(0.4, 2.0, (ny, nx))
    else:                      # arbitrary per-point spacing
        dy = rng.uniform(0.4, 2.0, (ny, nx))
        dx = rng.uniform(0.4, 2.0, (ny, nx))
    kw = dict(base_dbz=float(rng.choice([0.0, 5.0])),
              min_valid_dbz=float(rng.choice([0.0, -10.0, 5.0, -np.inf])),
              min_frac_texture=float(rng.uniform(0.1, 0.4)),
              min_frac_fit=float(rng.uniform(0.5, 0.8)))
    return dbz, dy, dx, float(rng.uniform(2.5, 10.0)), kw


@pytest.mark.parametrize("kind", ["latlon", "uniform", "xvary", "random"])
@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_matches_1_0_1_bit_for_bit(kind, kernel_mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(12):
            dbz, dy, dx, radius, kw = _random_case(seed, kind)
            tex, frac = refl_texture_2d(dbz, radius, dy, dx,
                                        kernel_mode=kernel_mode, **kw)
            tex_ref, frac_ref = refl_texture_2d_v1_0_1(
                dbz, radius, dy, dx, kernel_mode=kernel_mode, **kw)
            assert _same(tex, tex_ref), (kind, kernel_mode, seed)
            assert _same(frac, frac_ref), (kind, kernel_mode, seed)


def test_latlon_grid_resolves_to_row_windows():
    """A lat/lon grid needs one window per distinct row spacing, never a
    per-point rebuild, and far fewer windows than rows."""
    dbz, dy, dx, _, _ = _random_case(0, "latlon")
    k = _prepare_texture_2d(7.0, dy, dx, "varying")
    interior = k.row_kernel[k.row_kernel != ROW_SKIP]
    assert k.n_rows_per_point == 0
    assert np.all(interior >= 0)
    assert 1 <= k.n_kernels < len(interior)


def test_xvarying_grid_falls_back_to_per_point():
    """Spacing varying along x changes the window within a row, so every
    computed row keeps the 1.0 per-point rebuild."""
    _, dy, dx, _, _ = _random_case(2, "xvary")
    k = _prepare_texture_2d(2.0, dy, dx, "varying")
    interior = k.row_kernel[k.row_kernel != ROW_SKIP]
    assert len(interior) > 0
    assert np.all(interior == ROW_PER_POINT)
    assert k.n_kernels == 0


def test_uniform_mode_is_one_window():
    _, dy, dx, _, _ = _random_case(0, "latlon")
    k = _prepare_texture_2d(7.0, dy, dx, "uniform")
    assert k.n_kernels == 1
    assert k.n_rows_per_point == 0


def test_window_is_monotone_in_spacing():
    """The equivalence argument rests on this: a larger spacing can only
    remove offsets from a point's window, never add or reorder them."""
    rng = np.random.default_rng(5)
    for _ in range(400):
        r = rng.uniform(1.0, 12.0)
        dx, dy = rng.uniform(0.2, 3.0, 2)
        dx2 = dx * (1 + rng.uniform(0, 0.3))
        dy2 = dy * (1 + rng.uniform(0, 0.3))
        a = _point_kernel_offsets_arrays(r, dx, dy)
        b = _point_kernel_offsets_arrays(r, dx2, dy2)
        small = set(zip(b[0].tolist(), b[1].tolist()))
        large = list(zip(a[0].tolist(), a[1].tolist()))
        assert small <= set(large)
        # ...and the surviving offsets keep their relative order.
        assert [o for o in large if o in small] == list(zip(b[0].tolist(), b[1].tolist()))


@pytest.mark.parametrize("shape", [(7, 9), (8, 10), (7, 10), (8, 9), (1, 5), (6, 1)])
def test_nanmedian_all_matches_materialised(shape):
    rng = np.random.default_rng(sum(shape))
    col = rng.uniform(0.5, 1.5, shape[0])
    row = rng.uniform(0.5, 1.5, shape[1])
    col[0] = np.nan
    for view in (np.broadcast_to(col[:, None], shape),
                 np.broadcast_to(row[None, :], shape),
                 np.broadcast_to(np.float64(0.7), shape),
                 rng.uniform(0.5, 1.5, shape)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # all-NaN slices
            got = _nanmedian_all(view)
            want = float(np.nanmedian(np.array(view)))
        assert got == want or (np.isnan(got) and np.isnan(want))


# ---------------------------------------------------------------------------
# eccopy2d_h.prepare_grid / run(grid=...)
# ---------------------------------------------------------------------------

# A coarse (0.25 deg) grid spanning 20 deg of latitude, used with a 100 km
# window: small enough for a fast test, wide enough in latitude that the
# varying window genuinely changes from row to row.
_WINDOW = WindowSpec((100, "km"))


def _latlon_case(ny=80, nx=100, seed=4):
    from eccopy.core.coords import latlon_to_xy_spacing
    lat = 50.0 - 0.25 * np.arange(ny)
    lon = -100.0 + 0.25 * np.arange(nx)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dy, dx = latlon_to_xy_spacing(LAT, LON)
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(3):
        d = 15 + rng.normal(0, 2, (ny, nx))
        cy, cx = rng.integers(15, ny - 15), rng.integers(15, nx - 15)
        yy, xx = np.ogrid[:ny, :nx]
        d += 35 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0)
        d[rng.random(d.shape) < 0.2] = np.nan
        frames.append(d)
    return dy, dx, frames


_FIELDS = ("echo_type", "convectivity", "texture", "fraction_active")


@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_run_with_grid_matches_run_with_coords(kernel_mode):
    dy, dx, frames = _latlon_case()
    grid = eccopy2d_h.prepare_grid(dy, dx, coord_mode="spacing",
                                   window=_WINDOW, kernel_mode=kernel_mode)
    for dbz in frames:            # one grid, reused for every frame
        a = eccopy2d_h.run(dbz, grid=grid, min_convective_area=2.0)
        b = eccopy2d_h.run(dbz, coords_y=dy, coords_x=dx, coord_mode="spacing",
                           window=_WINDOW, kernel_mode=kernel_mode,
                           min_convective_area=2.0)
        for f in _FIELDS:
            assert _same(getattr(a, f), getattr(b, f)), f
        assert a.n_clumps == b.n_clumps
        assert a.texture_radius == b.texture_radius


def test_run_varying_matches_1_0_1_texture_on_latlon_grid():
    dy, dx, frames = _latlon_case()
    grid = eccopy2d_h.prepare_grid(dy, dx, coord_mode="spacing",
                                   window=_WINDOW, kernel_mode="varying")
    assert grid.n_rows_per_point == 0
    assert grid.n_windows > 1          # the window really does vary by row
    tex_ref, frac_ref = refl_texture_2d_v1_0_1(
        frames[0][np.newaxis], 100.0, dy, dx, kernel_mode="varying")
    r = eccopy2d_h.run(frames[0], grid=grid)
    assert _same(r.texture, tex_ref[0])
    assert _same(r.fraction_active, frac_ref)


def test_prepare_grid_1d_coords_shape_and_defaults():
    y_km = np.arange(40) * 1.0
    x_km = np.arange(55) * 0.8
    grid = eccopy2d_h.prepare_grid(y_km, x_km)
    assert grid.shape == (40, 55)
    assert grid.kernel_mode == "uniform"
    assert grid.coord_mode == "auto"
    assert grid.texture_radius == 7.0
    assert grid.n_windows == 1


def test_run_grid_rejects_conflicting_arguments():
    dy, dx, frames = _latlon_case()
    grid = eccopy2d_h.prepare_grid(dy, dx, coord_mode="spacing", window=_WINDOW)
    with pytest.raises(ValueError, match="not both"):
        eccopy2d_h.run(frames[0], dy, dx, grid=grid)
    for kw in ({"window": WindowSpec((7, "km"))}, {"kernel_mode": "uniform"},
               {"coord_mode": "spacing"}):
        with pytest.raises(ValueError, match="prepare_grid"):
            eccopy2d_h.run(frames[0], grid=grid, **kw)
    with pytest.raises(ValueError, match="does not match"):
        eccopy2d_h.run(frames[0][:-1], grid=grid)
    with pytest.raises(TypeError):
        eccopy2d_h.run(frames[0], grid="not a grid")


def test_run_without_coords_or_grid_raises():
    with pytest.raises(ValueError, match="coords_y and coords_x"):
        eccopy2d_h.run(np.zeros((10, 10)))


def test_grid_texture_params_still_per_call():
    """Texture parameters are per call, not part of the grid."""
    dy, dx, frames = _latlon_case()
    grid = eccopy2d_h.prepare_grid(dy, dx, coord_mode="spacing", window=_WINDOW)
    tp = TextureParams(min_valid_dbz=10.0, dbz_base=3.0)
    a = eccopy2d_h.run(frames[1], grid=grid, texture_params=tp)
    b = eccopy2d_h.run(frames[1], coords_y=dy, coords_x=dx, window=_WINDOW,
                       coord_mode="spacing", texture_params=tp)
    assert _same(a.convectivity, b.convectivity)
