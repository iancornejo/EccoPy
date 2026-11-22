"""
Tests for EccoPy 1.1's echo-aware coverage count (_fraction_active_rows):
coverage is counted from per-row prefix sums of the valid-cell mask, two
reads per window row, and rows whose window reaches no valid cell are
skipped. The count is the same integer as 1.0's cell-by-cell count, so
coverage -- and therefore texture -- must match 1.0.1 bit for bit.
"""

import numpy as np
import pytest

from eccopy.core import texture as T
from eccopy.core.texture import (_build_kernel_offsets_uniform_arrays,
                                 _kernel_runs, _prepare_texture_2d,
                                 refl_texture_2d)

from ._reference_impls.refl_texture_2d_v1_0_1 import refl_texture_2d_v1_0_1
from .test_texture_prepared import _same


@pytest.mark.parametrize("radius,dx,dy", [(7.0, 0.8, 1.11), (3.0, 1.0, 1.0),
                                          (12.0, 0.4, 2.0), (1.0, 50.0, 50.0)])
def test_runs_reconstruct_the_window(radius, dx, dy):
    jdx, jdy, _, _, _, _ = _build_kernel_offsets_uniform_arrays(radius, dx, dy)
    k_ptr = np.array([0, len(jdx)], dtype=np.int64)
    run_ptr, run_dy, run_lo, run_hi, k_runs = _kernel_runs(k_ptr, jdx, jdy)
    assert k_runs[0]
    cells = [(int(x), int(y)) for y, lo, hi in zip(run_dy, run_lo, run_hi)
             for x in range(lo, hi + 1)]
    assert cells == list(zip(jdx.tolist(), jdy.tolist()))


def test_non_contiguous_window_has_no_runs():
    jdx = np.array([-2, 0, 2, -1, 0, 1], dtype=np.int64)     # gap in row 0
    jdy = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    k_ptr = np.array([0, 6], dtype=np.int64)
    _, _, _, _, k_runs = _kernel_runs(k_ptr, jdx, jdy)
    assert not k_runs[0]


def _sparse_echo(ny=120, nx=150, seed=0):
    """Mostly empty ground with a few isolated cells and one wide band --
    the case the row skip is for."""
    rng = np.random.default_rng(seed)
    dbz = np.full((1, ny, nx), np.nan)
    for _ in range(6):
        cy, cx = rng.integers(10, ny - 10), rng.integers(10, nx - 10)
        dbz[0, cy - 3:cy + 4, cx - 4:cx + 5] = rng.normal(30, 8, (7, 9))
    dbz[0, 60:63, :] = rng.normal(15, 5, (3, nx))
    dbz[0, 0, 0] = 40.0                                     # corner, in the border
    return dbz


@pytest.mark.parametrize("kernel_mode", ["uniform", "varying"])
@pytest.mark.parametrize("n_threads", [1, 3])
def test_sparse_echo_matches_1_0_1(kernel_mode, n_threads):
    dbz = _sparse_echo()
    ny, nx = dbz.shape[1:]
    lat = np.linspace(50, 25, ny)
    dy = np.full((ny, nx), 1.11)
    dx = (1.11 * np.cos(np.radians(lat)))[:, None] * np.ones((1, nx))
    for mv in (0.0, 10.0, -np.inf):
        new = refl_texture_2d(dbz, 5.0, dy, dx, min_valid_dbz=mv,
                              kernel_mode=kernel_mode, n_threads=n_threads)
        ref = refl_texture_2d_v1_0_1(dbz, 5.0, dy, dx, min_valid_dbz=mv,
                                     kernel_mode=kernel_mode)
        assert _same(new[0], ref[0]) and _same(new[1], ref[1]), mv


def test_cell_by_cell_fallback_matches_runs():
    """Force every stored window down the cell-by-cell path and check it
    gives the same coverage as the prefix-sum path."""
    dbz = _sparse_echo(seed=3)[0]
    ny, nx = dbz.shape
    k = _prepare_texture_2d(6.0, np.full((ny, nx), 1.0), np.full((ny, nx), 0.9), "uniform")
    fast = np.zeros((ny, nx))
    slow = np.zeros((ny, nx))
    args = (dbz, 0.0, k.k_ptr, k.k_jdx, k.k_jdy, k.row_kernel,
            k.run_ptr, k.run_dy, k.run_lo, k.run_hi)
    tail = (k.halo_y, k.scale_y, k.scale_x, k.radius_km, k.ix0, k.ix1, 0, ny)
    T._fraction_active_rows(*args, k.k_runs, *tail, fast)
    T._fraction_active_rows(*args, np.zeros_like(k.k_runs), *tail, slow)
    assert np.array_equal(fast, slow)
    assert fast.max() > 0
