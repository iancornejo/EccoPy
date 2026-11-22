"""
Tests for row-parallel texture (EccoPy 1.1): n_threads splits rows into
blocks processed concurrently, in eccopy2d_h as well as eccopy3d.

Every point is computed independently by the same arithmetic whichever
block it lands in, so results must be identical -- bit for bit -- at any
thread count.
"""

import warnings

import numpy as np
import pytest

from eccopy import eccopy2d_h, eccopy3d
from eccopy.core import texture as T
from eccopy.core.texture import _row_blocks, refl_texture_2d

from ._reference_impls.refl_texture_2d_v1_0_1 import refl_texture_2d_v1_0_1
from .test_texture_prepared import _latlon_case, _random_case, _same, _WINDOW


@pytest.mark.parametrize("ny,n_threads", [(1, 4), (5, 4), (37, 2), (100, 3), (3500, 4)])
def test_row_blocks_cover_every_row_once(ny, n_threads):
    blocks = _row_blocks(ny, n_threads)
    rows = [r for a, b in blocks for r in range(a, b)]
    assert rows == list(range(ny))
    assert all(b > a for a, b in blocks)


@pytest.mark.parametrize("kind", ["latlon", "xvary", "random"])
@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_any_thread_count_matches_1_0_1(kind, kernel_mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(6):
            dbz, dy, dx, radius, kw = _random_case(seed, kind)
            tex_ref, frac_ref = refl_texture_2d_v1_0_1(
                dbz, radius, dy, dx, kernel_mode=kernel_mode, **kw)
            for n_threads in (2, 3, 8):
                tex, frac = refl_texture_2d(dbz, radius, dy, dx,
                                            kernel_mode=kernel_mode,
                                            n_threads=n_threads, **kw)
                assert _same(tex, tex_ref), (kind, kernel_mode, seed, n_threads)
                assert _same(frac, frac_ref), (kind, kernel_mode, seed, n_threads)


@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
def test_eccopy2d_h_n_threads_identical(kernel_mode):
    dy, dx, frames = _latlon_case()
    grid = eccopy2d_h.prepare_grid(dy, dx, coord_mode="spacing",
                                   window=_WINDOW, kernel_mode=kernel_mode)
    one = eccopy2d_h.run(frames[0], grid=grid, n_threads=1)
    for n_threads in (2, 4):
        many = eccopy2d_h.run(frames[0], grid=grid, n_threads=n_threads)
        for f in ("echo_type", "convectivity", "texture", "fraction_active"):
            assert _same(getattr(one, f), getattr(many, f)), (f, n_threads)
        assert one.n_clumps == many.n_clumps


def test_eccopy3d_n_threads_identical():
    rng = np.random.default_rng(11)
    nz, ny, nx = 6, 40, 50
    dbz = 20 + 8 * rng.standard_normal((nz, ny, nx))
    dbz[:, 10:20, 15:30] += 25
    dbz[rng.random(dbz.shape) < 0.3] = np.nan
    z = np.arange(nz) * 1.0
    y = np.arange(ny) * 1.0
    x = np.arange(nx) * 1.0
    height = np.broadcast_to(z[:, None, None], dbz.shape)
    a = eccopy3d.run(dbz, z, y, x, height=height, n_threads=1)
    b = eccopy3d.run(dbz, z, y, x, height=height, n_threads=3)
    for f in ("echo_type", "convectivity", "texture", "fraction_active"):
        assert _same(getattr(a, f), getattr(b, f)), f


def test_cores_release_the_gil():
    """The threaded driver relies on these entry points running without
    the GIL; without it, n_threads silently does nothing (as in 1.0, where
    the per-level threads all waited on the GIL)."""
    for core in (T._fraction_active_rows, T._texture_rows):
        assert core.targetoptions.get("nogil") is True, core.__name__
