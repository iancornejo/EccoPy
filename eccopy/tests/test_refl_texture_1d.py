"""Tests for eccopy.core.texture.refl_texture_1d() -- specifically its
per-row spacing/window-radius resolution (see core/texture.py's
"FIXED this session" docstring note). A prior version silently resolved
the window radius from only row 0's spacing whenever a genuinely
row-varying (ndim > 1) spacing array was passed."""

import numpy as np
import pytest

from eccopy.core.texture import refl_texture_1d
from eccopy.params import WindowSpec


def test_uniform_spacing_across_rows_unaffected():
    """The common case (spacing shared across every row) must still work
    exactly as before -- this is the fast, single-call path."""
    rng = np.random.default_rng(1)
    nz, nx = 4, 60
    dbz2d = 15 + rng.normal(0, 3, (nz, nx))
    win = WindowSpec((5, "km"))

    sp_1d = np.full(nx, 1.0) * 1000.0
    sp_2d = np.broadcast_to(sp_1d, (nz, nx))

    tex_shared = refl_texture_1d(dbz2d, win, spacing=sp_1d)
    tex_broadcast = refl_texture_1d(dbz2d, win, spacing=sp_2d)
    np.testing.assert_allclose(tex_shared, tex_broadcast, equal_nan=True)

    for row in range(nz):
        tex_row_alone = refl_texture_1d(dbz2d[row], win, spacing=sp_1d)
        np.testing.assert_allclose(tex_shared[row], tex_row_alone, equal_nan=True)


def test_row_varying_spacing_uses_each_rows_own_spacing():
    """The actual bug fix from last session: a row with different spacing
    than row 0 must get its OWN resolved window radius, not row 0's --
    only true in kernel_mode='varying' (the default 'uniform' mode
    deliberately collapses row-varying spacing to one global-median
    radius, see test_uniform_kernel_mode_is_default_and_collapses_to_
    global_median below)."""
    rng = np.random.default_rng(1)
    nz, nx = 3, 60
    dbz2d = 15 + rng.normal(0, 3, (nz, nx))
    win = WindowSpec((5, "km"))

    sp_row0 = np.full(nx, 0.5) * 1000.0   # fine spacing
    sp_row2 = np.full(nx, 2.0) * 1000.0   # coarse spacing -- different window radius
    sp_2d = np.stack([sp_row0, sp_row0, sp_row2])

    tex_2d = refl_texture_1d(dbz2d, win, spacing=sp_2d, kernel_mode="varying")
    tex_row2_correct = refl_texture_1d(dbz2d[2], win, spacing=sp_row2, kernel_mode="varying")
    tex_row2_wrong = refl_texture_1d(dbz2d[2], win, spacing=sp_row0, kernel_mode="varying")

    np.testing.assert_allclose(tex_2d[2], tex_row2_correct, equal_nan=True)
    assert not np.allclose(tex_2d[2], tex_row2_wrong, equal_nan=True)

    # Rows 0 and 1 share spacing with each other -- should match directly.
    tex_row0_alone = refl_texture_1d(dbz2d[0], win, spacing=sp_row0, kernel_mode="varying")
    np.testing.assert_allclose(tex_2d[0], tex_row0_alone, equal_nan=True)


# ---------------------------------------------------------------------------
# kernel_mode itself
# ---------------------------------------------------------------------------

def test_uniform_kernel_mode_is_default_and_collapses_to_global_median():
    """'uniform' (the default) resolves ONE radius from the global median
    spacing across the whole array and applies it everywhere -- so a
    row-varying spacing array is deliberately flattened to a single
    number, unlike 'varying'."""
    rng = np.random.default_rng(1)
    nz, nx = 3, 60
    dbz2d = 15 + rng.normal(0, 3, (nz, nx))
    win = WindowSpec((5, "km"))

    sp_row0 = np.full(nx, 0.5) * 1000.0
    sp_row2 = np.full(nx, 2.0) * 1000.0
    sp_2d = np.stack([sp_row0, sp_row0, sp_row2])

    tex_default = refl_texture_1d(dbz2d, win, spacing=sp_2d)
    tex_explicit_uniform = refl_texture_1d(dbz2d, win, spacing=sp_2d, kernel_mode="uniform")
    np.testing.assert_allclose(tex_default, tex_explicit_uniform, equal_nan=True)

    # All three rows get the SAME resolved radius under 'uniform' -- so
    # rows 0 and 1 (which share spacing with each other) should differ
    # from what 'varying' would give row 2 (whose own spacing differs).
    tex_varying = refl_texture_1d(dbz2d, win, spacing=sp_2d, kernel_mode="varying")
    assert not np.allclose(tex_default[2], tex_varying[2], equal_nan=True)


def test_uniform_kernel_mode_identical_to_varying_on_truly_uniform_spacing():
    """On genuinely uniform-spacing data (the validated SEA/SPOL case
    shape), 'uniform' and 'varying' must agree exactly -- the median of a
    constant array is that constant."""
    rng = np.random.default_rng(9)
    nz, nx = 5, 80
    dbz2d = 15 + rng.normal(0, 3, (nz, nx))
    win = WindowSpec((6, "km"))
    sp_uniform = np.full((nz, nx), 1.2) * 1000.0

    tex_uniform = refl_texture_1d(dbz2d, win, spacing=sp_uniform, kernel_mode="uniform")
    tex_varying = refl_texture_1d(dbz2d, win, spacing=sp_uniform, kernel_mode="varying")
    np.testing.assert_allclose(tex_uniform, tex_varying, equal_nan=True)


def test_kernel_mode_invalid_raises():
    dbz = np.zeros(20) + 15
    with pytest.raises(ValueError):
        refl_texture_1d(dbz, WindowSpec((3, "km")), spacing=np.full(20, 1000.0),
                        kernel_mode="bogus")


def test_kernel_mode_irrelevant_for_pixel_window():
    """A bare pixel-radius window doesn't resolve anything from spacing,
    so kernel_mode shouldn't matter at all."""
    rng = np.random.default_rng(4)
    dbz = 15 + rng.normal(0, 2, 40)
    tex_uniform = refl_texture_1d(dbz, WindowSpec(4), kernel_mode="uniform")
    tex_varying = refl_texture_1d(dbz, WindowSpec(4), kernel_mode="varying")
    np.testing.assert_allclose(tex_uniform, tex_varying, equal_nan=True)


def test_single_shared_row_spacing_broadcasts():
    """spacing with a single leading row (shape (1, N)) should broadcast
    to every row, same as a plain (N,) array."""
    rng = np.random.default_rng(2)
    nz, nx = 5, 40
    dbz2d = 15 + rng.normal(0, 2, (nz, nx))
    win = WindowSpec((3, "km"))
    sp_1d = np.full(nx, 1.5) * 1000.0

    tex_1row = refl_texture_1d(dbz2d, win, spacing=sp_1d[np.newaxis, :])
    tex_flat = refl_texture_1d(dbz2d, win, spacing=sp_1d)
    np.testing.assert_allclose(tex_1row, tex_flat, equal_nan=True)


def test_mismatched_row_count_raises_clear_error():
    rng = np.random.default_rng(3)
    nz, nx = 4, 30
    dbz2d = 15 + rng.normal(0, 2, (nz, nx))
    win = WindowSpec((3, "km"))
    # 2 rows of spacing for 4 rows of data -- ambiguous, must raise.
    sp_wrong = np.full((2, nx), 1.0) * 1000.0
    with pytest.raises(ValueError, match="row"):
        refl_texture_1d(dbz2d, win, spacing=sp_wrong)


def test_pixel_window_ignores_spacing_shape_entirely():
    """A bare pixel-radius window never needs spacing -- row-varying
    spacing arrays shouldn't matter (or even need to be supplied)."""
    rng = np.random.default_rng(4)
    nz, nx = 3, 30
    dbz2d = 15 + rng.normal(0, 2, (nz, nx))
    tex_no_spacing = refl_texture_1d(dbz2d, WindowSpec(4))
    tex_with_bogus_spacing = refl_texture_1d(
        dbz2d, WindowSpec(4),
        spacing=np.stack([np.full(nx, 1.0), np.full(nx, 99.0), np.full(nx, 0.001)]) * 1000.0
    )
    np.testing.assert_allclose(tex_no_spacing, tex_with_bogus_spacing, equal_nan=True)


def test_bare_int_window_still_accepted():
    rng = np.random.default_rng(5)
    dbz = 15 + rng.normal(0, 2, 50)
    tex = refl_texture_1d(dbz, 5)
    assert tex.shape == dbz.shape
