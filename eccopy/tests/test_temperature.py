"""
Tests for eccopy.core.temperature.

isotherm_height() previously had zero test coverage, which is how both
a shape-convention bug (it used the old (nx,ny,nz) ordering instead of
the package-wide (Z,Y,X) convention) and a NaN-gap interpolation bug
(adjacent-pair-only crossing search couldn't see a crossing straddled by
a NaN gap) went unnoticed. These tests are deliberately thorough.
"""

import numpy as np
import pytest

from eccopy.core.temperature import isotherm_height, broadcast_temp_field


# ---------------------------------------------------------------------------
# isotherm_height — shape convention
# ---------------------------------------------------------------------------

def test_isotherm_height_shape_is_z_y_x_in_y_x_out():
    nz, ny, nx = 12, 5, 6
    temp_3d = np.random.uniform(-40, 20, (nz, ny, nx))
    z_km = np.linspace(0, 12, nz)
    ht = isotherm_height(temp_3d, z_km, target_temp_c=0.0)
    assert ht.shape == (ny, nx)


def test_isotherm_height_matches_analytic_constant_lapse_rate():
    nz, ny, nx = 20, 5, 6
    z_km = np.linspace(0, 15, nz)
    lapse_rate = 6.5
    surface_temp = 25.0
    temp_3d = np.full((nz, ny, nx), np.nan)
    for iz in range(nz):
        temp_3d[iz, :, :] = surface_temp - lapse_rate * z_km[iz]

    expected_z = surface_temp / lapse_rate
    ht = isotherm_height(temp_3d, z_km, target_temp_c=0.0)
    assert np.allclose(ht, expected_z, atol=0.5)


def test_isotherm_height_preserves_horizontal_variation():
    """Each (y, x) column has a genuinely different surface temperature;
    the result must vary correspondingly and NOT be accidentally
    transposed between y and x."""
    nz, ny, nx = 15, 4, 4
    z_km = np.linspace(0, 12, nz)
    surface_temp_grid = np.array([
        [20, 22, 24, 26],
        [21, 23, 25, 27],
        [22, 24, 26, 28],
        [23, 25, 27, 29],
    ], dtype=float)
    temp_3d = np.full((nz, ny, nx), np.nan)
    for iz in range(nz):
        temp_3d[iz] = surface_temp_grid - 6.5 * z_km[iz]

    expected = surface_temp_grid / 6.5
    ht = isotherm_height(temp_3d, z_km, target_temp_c=0.0)
    assert np.allclose(ht, expected, atol=0.3)


# ---------------------------------------------------------------------------
# isotherm_height — fallback behaviour (target outside the profile's range)
# ---------------------------------------------------------------------------

def test_isotherm_height_target_warmer_than_whole_column_returns_bottom():
    nz, ny, nx = 15, 2, 2
    z_km = np.linspace(0, 12, nz)
    temp_3d = np.full((nz, ny, nx), np.nan)
    for iz in range(nz):
        temp_3d[iz] = 25.0 - 6.5 * z_km[iz]   # top level still only -53C
    ht = isotherm_height(temp_3d, z_km, target_temp_c=30.0)  # warmer than surface
    assert np.all(ht == z_km[0])


def test_isotherm_height_target_colder_than_whole_column_returns_top():
    nz, ny, nx = 15, 2, 2
    z_km = np.linspace(0, 12, nz)
    temp_3d = np.full((nz, ny, nx), np.nan)
    for iz in range(nz):
        temp_3d[iz] = 25.0 - 6.5 * z_km[iz]
    ht = isotherm_height(temp_3d, z_km, target_temp_c=-60.0)  # colder than top
    assert np.all(ht == z_km[-1])


def test_isotherm_height_all_nan_column_returns_nan():
    nz, ny, nx = 10, 2, 2
    z_km = np.linspace(0, 9, nz)
    temp_3d = np.full((nz, ny, nx), np.nan)
    ht = isotherm_height(temp_3d, z_km, target_temp_c=0.0)
    assert np.all(np.isnan(ht))


# ---------------------------------------------------------------------------
# isotherm_height — NaN-gap crossing search
# ---------------------------------------------------------------------------

def test_isotherm_height_finds_crossing_across_single_nan_gap():
    """The case from discussion: [1, NaN, -1] searching for the 0C
    isotherm. A human can tell the crossing is near the NaN level (goes
    from +1 to -1 across it); the search must not fall through to the
    bottom/top fallback just because the immediately-adjacent pairs are
    each individually NaN-contaminated."""
    z_km = np.array([0.0, 1.0, 2.0])
    temp_3d = np.array([1.0, np.nan, -1.0]).reshape(3, 1, 1)
    ht = isotherm_height(temp_3d, z_km, target_temp_c=0.0)
    # Symmetric crossing over a 2km span centered at z=1 -> midpoint
    assert np.isclose(ht[0, 0], 1.0)


def test_isotherm_height_finds_crossing_across_multi_level_nan_gap():
    z_km = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    temp_3d = np.array([10.0, np.nan, np.nan, np.nan, -10.0]).reshape(5, 1, 1)
    ht = isotherm_height(temp_3d, z_km, target_temp_c=0.0)
    assert np.isclose(ht[0, 0], 2.0)


def test_isotherm_height_gap_not_at_crossing_unaffected():
    """A NaN gap that does NOT straddle the true crossing should not
    change the result at all."""
    nz, ny, nx = 15, 4, 4
    z_km = np.linspace(0, 12, nz)
    surface_temp_grid = np.array([
        [20, 22, 24, 26], [21, 23, 25, 27], [22, 24, 26, 28], [23, 25, 27, 29],
    ], dtype=float)
    temp_3d = np.full((nz, ny, nx), np.nan)
    for iz in range(nz):
        temp_3d[iz] = surface_temp_grid - 6.5 * z_km[iz]
    expected = surface_temp_grid / 6.5

    temp_3d_gap = temp_3d.copy()
    # Punch a gap near the top of the profile, far from any 0C crossing
    temp_3d_gap[-3:, 1, 1] = np.nan

    ht_gap = isotherm_height(temp_3d_gap, z_km, target_temp_c=0.0)
    assert np.isclose(ht_gap[1, 1], expected[1, 1], atol=0.3)


def test_isotherm_height_gap_straddling_crossing_recovers_correct_height():
    """The original regression case: a gap that DOES straddle the true
    crossing must still resolve to (approximately) the correct height,
    not silently fall back to the bottom-level default."""
    nz, ny, nx = 15, 4, 4
    z_km = np.linspace(0, 12, nz)
    surface_temp_grid = np.array([
        [20, 22, 24, 26], [21, 23, 25, 27], [22, 24, 26, 28], [23, 25, 27, 29],
    ], dtype=float)
    temp_3d = np.full((nz, ny, nx), np.nan)
    for iz in range(nz):
        temp_3d[iz] = surface_temp_grid - 6.5 * z_km[iz]
    expected = surface_temp_grid / 6.5

    temp_3d_gap = temp_3d.copy()
    temp_3d_gap[5:8, 2, 2] = np.nan  # straddles the ~4km crossing at (2,2)

    ht_gap = isotherm_height(temp_3d_gap, z_km, target_temp_c=0.0)
    level_spacing = z_km[1] - z_km[0]
    assert abs(ht_gap[2, 2] - expected[2, 2]) < level_spacing
    # Specifically must NOT have fallen back to the bottom level
    assert not np.isclose(ht_gap[2, 2], z_km[0])


# ---------------------------------------------------------------------------
# broadcast_temp_field
# ---------------------------------------------------------------------------

def test_broadcast_temp_field_full_field_passthrough():
    target_shape = (10, 20, 30)
    temp = np.random.uniform(-40, 20, target_shape)
    result = broadcast_temp_field(temp, target_shape)
    np.testing.assert_array_equal(result, temp)


def test_broadcast_temp_field_sounding_3d():
    nz, ny, nx = 10, 20, 30
    sounding = np.linspace(20, -40, nz)
    result = broadcast_temp_field(sounding, (nz, ny, nx))
    assert result.shape == (nz, ny, nx)
    for iz in range(nz):
        assert np.all(result[iz] == sounding[iz])


def test_broadcast_temp_field_sounding_2d():
    nz, nx = 15, 50
    sounding = np.linspace(25, -50, nz)
    result = broadcast_temp_field(sounding, (nz, nx))
    assert result.shape == (nz, nx)
    for iz in range(nz):
        assert np.all(result[iz] == sounding[iz])


def test_broadcast_temp_field_matches_manual_broadcast():
    nz, ny, nx = 8, 15, 20
    sounding = np.linspace(20, -50, nz)
    result = broadcast_temp_field(sounding, (nz, ny, nx))
    manual = np.broadcast_to(sounding[:, None, None], (nz, ny, nx))
    np.testing.assert_array_equal(result, manual)


def test_broadcast_temp_field_wrong_shape_raises():
    with pytest.raises(ValueError):
        broadcast_temp_field(np.zeros(13), (10, 20, 30))


def test_broadcast_temp_field_wrong_ndim_raises():
    with pytest.raises(ValueError):
        broadcast_temp_field(np.zeros((10, 5)), (10, 20, 30))
