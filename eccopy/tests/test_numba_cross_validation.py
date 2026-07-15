"""
Cross-validation tests: Numba-accelerated functions vs frozen pure-Python
reference implementations.

These reference implementations (in eccopy/tests/_reference_impls/) are
DELIBERATELY FROZEN COPIES of what the production code looked like before
each Numba conversion — not re-derivations, not simplifications. The goal
is to catch any discrepancy introduced during conversion, the same way
exhaustive cross-validation against production caught real bugs while
building the debug module (see test_debug.py).

NUMERICAL NOTE on near-zero-variance cases (see test_sliding_texture_
core_matches_reference below): the underlying texture formula uses a
numerically unstable one-pass variance calculation (E[X^2] - E[X]^2),
inherited from the original algorithm — present in BOTH the reference
and the Numba version, not introduced by the conversion. When the true
variance is extremely small (detrended values that are nearly but not
exactly identical), floating-point rounding differences between the
scalar-loop (Numba) and array-reduction (NumPy) accumulation paths can
cause the catastrophic-cancellation error to come out differently in
each, producing visibly different "noise floor" texture values (e.g.
1e-7 vs 1e-3) even though both are physically meaningless relative to
the function's actual 0-30 dB operating range. Comparisons below use a
tolerance (atol=0.01) that reflects this — tight enough to catch any
real algorithmic divergence, loose enough not to flag two numerically-
negligible-but-differently-rounded near-zero results as a bug.
"""

import numpy as np
import warnings

from eccopy.core.texture import (_sliding_texture_core, _radius_field_along_axis,
                                  _fillmissing_linear,
                                  _compute_fraction_active_core,
                                  _compute_texture_one_level_core,
                                  _build_kernel_offsets_uniform_arrays,
                                  _compute_fraction_active_varying,
                                  _compute_texture_one_level_varying,
                                  _max_half_width,
                                  refl_texture_2d)
from eccopy.core.temperature import isotherm_height
from eccopy.params import WindowSpec

from ._reference_impls.sliding_texture_reference import sliding_texture_core_reference
from ._reference_impls.isotherm_height_reference import isotherm_height_reference
from ._reference_impls.texture_2d_reference import (
    _build_kernel_offsets_uniform_reference,
    _compute_fraction_active_reference,
    _compute_texture_one_level_reference,
    _compute_fraction_active_varying_reference,
    _compute_texture_one_level_varying_reference,
    _max_half_width_reference,
)


# ---------------------------------------------------------------------------
# _sliding_texture_core (Numba) vs frozen pure-Python reference
# ---------------------------------------------------------------------------

def test_sliding_texture_core_matches_reference_random_stress():
    total_checked = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nanstd-on-all-nan, expected/benign
        for seed in range(50):
            rng = np.random.default_rng(seed)
            nRows = rng.integers(1, 10)
            nCols = rng.integers(5, 80)
            max_radius = rng.integers(1, 15)
            pad = max_radius

            data_padded = rng.normal(15, 5, (nRows, nCols + 2 * pad))
            if rng.random() < 0.6:
                nan_mask = rng.random(data_padded.shape) < 0.1
                data_padded[nan_mask] = np.nan

            radius_field = rng.integers(0, max_radius + 1, nCols)

            result_new = _sliding_texture_core(data_padded, radius_field, pad, nRows, nCols)
            result_ref = sliding_texture_core_reference(data_padded, radius_field, pad, nRows, nCols)

            # atol=0.01: see module docstring re: near-zero-variance
            # catastrophic-cancellation noise floor.
            assert np.allclose(result_new, result_ref, atol=0.01, equal_nan=True), (
                f"seed={seed}: max diff "
                f"{np.nanmax(np.abs(result_new - result_ref))}"
            )
            total_checked += nRows * nCols

    assert total_checked > 5000


def test_sliding_texture_core_matches_reference_via_public_api():
    """Same check, but going through refl_texture_1d's public API end to
    end (covers the padding/fill-missing/spacing-resolution glue code
    too, not just the core loop in isolation)."""
    from eccopy.core.texture import refl_texture_1d

    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = rng.integers(30, 200)
        dbz = 15 + rng.normal(0, 4, n)
        if rng.random() < 0.5:
            gap_start = rng.integers(0, n - 5)
            dbz[gap_start:gap_start + 3] = np.nan
        win_km = rng.uniform(2, 10)
        window = WindowSpec((win_km, "km"))
        spacing_m = np.full(n, rng.uniform(500, 2000))

        result_new = refl_texture_1d(dbz, window, spacing=spacing_m)

        radius_field = _radius_field_along_axis(window, spacing_m, n)
        pad = int(radius_field.max())
        flat = dbz.reshape(1, n)
        padded = np.pad(flat, ((0, 0), (pad, pad)), mode="constant", constant_values=np.nan)
        padded = _fillmissing_linear(padded, axis=1)
        result_ref = sliding_texture_core_reference(padded, radius_field, pad, 1, n)[0]
        result_ref[np.isnan(dbz)] = np.nan

        assert np.allclose(result_new, result_ref, atol=0.01, equal_nan=True), (
            f"seed={seed}: max diff "
            f"{np.nanmax(np.abs(result_new - result_ref))}"
        )


# ---------------------------------------------------------------------------
# isotherm_height (Numba) vs frozen pure-Python reference
# ---------------------------------------------------------------------------

def test_isotherm_height_matches_reference_random_stress():
    total_checked = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        nz = rng.integers(3, 30)
        ny = rng.integers(1, 15)
        nx = rng.integers(1, 15)
        z_levels = np.sort(rng.uniform(0, 15, nz))

        mode = seed % 4
        if mode == 0:
            surface = rng.uniform(15, 30, (ny, nx))
            lapse = rng.uniform(4, 9)
            temp_3d = surface[None, :, :] - lapse * z_levels[:, None, None]
            temp_3d += rng.normal(0, 0.3, temp_3d.shape)
        elif mode == 1:
            surface = rng.uniform(15, 30, (ny, nx))
            lapse = rng.uniform(4, 9)
            temp_3d = surface[None, :, :] - lapse * z_levels[:, None, None]
            nan_mask = rng.random(temp_3d.shape) < 0.2
            temp_3d[nan_mask] = np.nan
        elif mode == 2:
            temp_3d = rng.uniform(-60, 30, (nz, ny, nx))
            nan_mask = rng.random(temp_3d.shape) < 0.15
            temp_3d[nan_mask] = np.nan
        else:
            surface = rng.uniform(15, 30, (ny, nx))
            lapse = rng.uniform(4, 9)
            temp_3d = surface[None, :, :] - lapse * z_levels[:, None, None]
            all_nan_cols = rng.random((ny, nx)) < 0.1
            temp_3d[:, all_nan_cols] = np.nan

        target = rng.uniform(-50, 35)

        result_new = isotherm_height(temp_3d, z_levels, target)
        result_ref = isotherm_height_reference(temp_3d, z_levels, target)

        assert np.allclose(result_new, result_ref, atol=1e-9, equal_nan=True), (
            f"seed={seed} mode={mode}: max diff "
            f"{np.nanmax(np.abs(result_new - result_ref))}"
        )
        total_checked += ny * nx

    assert total_checked > 1000


# ---------------------------------------------------------------------------
# 2-D radial texture, uniform kernel mode (Numba) vs frozen reference
# ---------------------------------------------------------------------------

def test_texture_2d_uniform_matches_reference_random_stress():
    total_checked = 0
    total_tex_mismatch = 0
    total_frac_mismatch = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(40):
            rng = np.random.default_rng(seed)
            ny, nx = rng.integers(15, 40), rng.integers(15, 40)
            dbz_level = 15 + rng.normal(0, 4, (ny, nx))
            if rng.random() < 0.5:
                gy, gx = rng.integers(0, ny - 3), rng.integers(0, nx - 3)
                gh, gw = rng.integers(1, 4), rng.integers(1, 4)
                dbz_level[gy:gy + gh, gx:gx + gw] = np.nan
            radius_km = rng.uniform(3, 12)
            dx_km = rng.uniform(0.3, 2.5)
            dy_km = rng.uniform(0.3, 2.5)
            min_valid_dbz = rng.choice([0.0, -10.0, 5.0])
            min_frac_texture = rng.uniform(0.1, 0.4)
            min_frac_fit = rng.uniform(0.5, 0.8)
            base_dbz = rng.choice([0.0, 5.0])

            offsets_ref, nxt_ref, nyt_ref = _build_kernel_offsets_uniform_reference(
                radius_km, dx_km, dy_km)
            frac_ref = _compute_fraction_active_reference(
                dbz_level, offsets_ref, nxt_ref, nyt_ref, min_valid_dbz)
            tex_ref = _compute_texture_one_level_reference(
                dbz_level, frac_ref, offsets_ref, nxt_ref, nyt_ref,
                base_dbz, min_frac_texture, min_frac_fit)

            jdx, jdy, xx, yy, nxt_new, nyt_new = _build_kernel_offsets_uniform_arrays(
                radius_km, dx_km, dy_km)
            frac_new = _compute_fraction_active_core(
                dbz_level, jdx, jdy, nxt_new, nyt_new, min_valid_dbz)
            tex_new = _compute_texture_one_level_core(
                dbz_level, frac_new, jdx, jdy, xx, yy,
                nxt_new, nyt_new, base_dbz, min_frac_texture, min_frac_fit)

            assert (nxt_ref, nyt_ref) == (nxt_new, nyt_new), (
                f"seed={seed}: kernel half-width mismatch"
            )

            total_checked += ny * nx
            if not np.allclose(frac_ref, frac_new, atol=1e-9):
                total_frac_mismatch += 1
            # atol=0.01: see module docstring re: near-zero-variance
            # catastrophic-cancellation noise floor.
            if not np.allclose(tex_ref, tex_new, atol=0.01, equal_nan=True):
                total_tex_mismatch += 1

    assert total_frac_mismatch == 0
    assert total_tex_mismatch == 0
    assert total_checked > 10000


def test_texture_2d_uniform_via_public_api_matches_reference():
    """Same check, end to end through refl_texture_2d's public API."""

    def reference_refl_texture_2d(dbz, radius_km, dy, dx, base_dbz=0.0,
                                   min_valid_dbz=0.0, min_frac_texture=0.25,
                                   min_frac_fit=0.67):
        nz, ny, nx = dbz.shape
        # Null dbz below min_valid_dbz BEFORE any downstream computation --
        # matches the fix applied to production refl_texture_2d() (see its
        # docstring / README "Validation status"): min_valid_dbz must gate
        # the actual texture-contributing values, not just the fraction_
        # active coverage count.
        dbz = np.where(dbz < min_valid_dbz, np.nan, dbz)
        dy_km = float(np.nanmedian(dy))
        dx_km = float(np.nanmedian(dx))
        offsets, nx_tex, ny_tex = _build_kernel_offsets_uniform_reference(
            radius_km, dx_km, dy_km)
        dbz_yx = np.transpose(dbz, (1, 2, 0))
        dbz_col_max = np.nanmax(dbz_yx, axis=2)
        dbz_col_max = np.where(np.isnan(dbz_col_max), -9999.0, dbz_col_max)
        fraction_active = _compute_fraction_active_reference(
            dbz_col_max, offsets, nx_tex, ny_tex, min_valid_dbz)
        texture_yx = np.full((ny, nx, nz), np.nan)
        for iz in range(nz):
            level = dbz_yx[:, :, iz].copy()
            texture_yx[:, :, iz] = _compute_texture_one_level_reference(
                level, fraction_active, offsets, nx_tex, ny_tex,
                base_dbz, min_frac_texture, min_frac_fit)
        return np.transpose(texture_yx, (2, 0, 1)), fraction_active

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(5):
            rng = np.random.default_rng(seed)
            nz, ny, nx = rng.integers(3, 8), rng.integers(20, 40), rng.integers(20, 40)
            dbz = 15 + rng.normal(0, 4, (nz, ny, nx))
            dy = np.full((ny, nx), 1.0)
            dx = np.full((ny, nx), 1.0)

            tex_ref, frac_ref = reference_refl_texture_2d(dbz, 7.0, dy, dx)
            tex_new, frac_new = refl_texture_2d(dbz, 7.0, dy, dx, kernel_mode="uniform")

            assert np.allclose(tex_ref, tex_new, atol=0.01, equal_nan=True)
            assert np.allclose(frac_ref, frac_new, atol=1e-9)


# ---------------------------------------------------------------------------
# 2-D radial texture, varying kernel mode (Numba) vs frozen reference
# ---------------------------------------------------------------------------

def test_texture_2d_varying_matches_reference_random_stress():
    total_checked = 0
    total_tex_mismatch = 0
    total_frac_mismatch = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(30):
            rng = np.random.default_rng(seed)
            ny, nx = rng.integers(20, 45), rng.integers(20, 45)
            dbz_level = 15 + rng.normal(0, 4, (ny, nx))
            if rng.random() < 0.5:
                gy, gx = rng.integers(0, ny - 3), rng.integers(0, nx - 3)
                dbz_level[gy:gy + 2, gx:gx + 2] = np.nan

            radius_km = rng.uniform(3, 10)
            dx_km = rng.uniform(0.5, 2.0, (ny, nx))
            dy_km = rng.uniform(0.5, 2.0, (ny, nx))
            min_valid_dbz = rng.choice([0.0, -10.0])
            min_frac_texture = rng.uniform(0.1, 0.4)
            min_frac_fit = rng.uniform(0.5, 0.8)
            base_dbz = 0.0

            max_nx_tex, max_ny_tex = _max_half_width_reference(radius_km, dx_km, dy_km)
            assert (max_nx_tex, max_ny_tex) == _max_half_width(radius_km, dx_km, dy_km)

            frac_ref = _compute_fraction_active_varying_reference(
                dbz_level, radius_km, dx_km, dy_km, min_valid_dbz, max_nx_tex, max_ny_tex)
            tex_ref = _compute_texture_one_level_varying_reference(
                dbz_level, frac_ref, radius_km, dx_km, dy_km,
                base_dbz, min_frac_texture, min_frac_fit, max_nx_tex, max_ny_tex)

            frac_new = _compute_fraction_active_varying(
                dbz_level, radius_km, dx_km, dy_km, min_valid_dbz, max_nx_tex, max_ny_tex)
            tex_new = _compute_texture_one_level_varying(
                dbz_level, frac_new, radius_km, dx_km, dy_km,
                base_dbz, min_frac_texture, min_frac_fit, max_nx_tex, max_ny_tex)

            total_checked += ny * nx
            if not np.allclose(frac_ref, frac_new, atol=1e-9):
                total_frac_mismatch += 1
            if not np.allclose(tex_ref, tex_new, atol=0.01, equal_nan=True):
                total_tex_mismatch += 1

    assert total_frac_mismatch == 0
    assert total_tex_mismatch == 0
    assert total_checked > 10000
