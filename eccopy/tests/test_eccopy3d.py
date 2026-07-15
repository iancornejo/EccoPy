"""Tests for eccopy3d.run() — array-based 3-D classification."""

import numpy as np
import pytest

from eccopy import eccopy3d
from eccopy.params import WindowSpec, ClassificationParams


def _synthetic_volume(nz=10, ny=20, nx=30, seed=4):
    rng = np.random.default_rng(seed)
    z_km = np.linspace(0.5, 9.5, nz)
    y_km = np.linspace(-10, 10, ny)
    x_km = np.linspace(-15, 15, nx)
    dbz = 15 - 0.8 * z_km[:, None, None] + rng.normal(0, 0.5, (nz, ny, nx))
    # Convective tower
    for iz in range(nz):
        weight = np.exp(-((iz - 3) ** 2) / 10)
        dbz[iz, 8:13, 12:17] += 30 * weight
    return dbz, z_km, y_km, x_km


def test_basic_mode_shapes():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     window=WindowSpec((5, "km")))
    assert r.echo_type.shape == dbz.shape
    assert r.convectivity.shape == dbz.shape
    assert r.texture.shape == dbz.shape


def test_basic_codes_only_without_height_or_temp():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[r.echo_type > 0]).astype(int))
    # Without height/temp, stratiform sub-types collapse to MIXED (25);
    # clump-derived convective sub-types can still appear.
    assert codes.issubset({25, 32, 34, 36, 38})


def test_height_enables_full_subclassification():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    height = np.broadcast_to(z_km[:, None, None], dbz.shape).copy()
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     height=height, window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[r.echo_type > 0]).astype(int))
    assert codes.issubset({14, 16, 18, 25, 32, 34, 36, 38})


def test_temp_enables_full_subclassification():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    height = np.broadcast_to(z_km[:, None, None], dbz.shape)
    temp = 20 - 6.5 * height
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     temp=temp, window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[r.echo_type > 0]).astype(int))
    assert codes.issubset({14, 16, 18, 25, 32, 34, 36, 38})


def test_temp_as_sounding_matches_manually_broadcast_field():
    """A (Z,) sounding should give identical results to manually
    broadcasting that same profile across the whole horizontal domain."""
    dbz, z_km, y_km, x_km = _synthetic_volume()
    sounding = 20.0 - 6.5 * z_km

    r_sounding = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                              temp=sounding, window=WindowSpec((5, "km")))

    full_field = np.broadcast_to(sounding[:, None, None], dbz.shape).copy()
    r_full = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                          temp=full_field, window=WindowSpec((5, "km")))

    np.testing.assert_array_equal(r_sounding.echo_type, r_full.echo_type)
    np.testing.assert_allclose(r_sounding.convectivity, r_full.convectivity,
                               equal_nan=True)


def test_temp_sounding_enables_full_subclassification():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    sounding = 20.0 - 6.5 * z_km
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     temp=sounding, window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[r.echo_type > 0]).astype(int))
    assert codes.issubset({14, 16, 18, 25, 32, 34, 36, 38})


def test_temp_wrong_shape_raises_clear_error():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    bad_temp = np.zeros(len(z_km) + 5)  # neither (Z,) nor (Z, Y, X)
    with pytest.raises(ValueError):
        eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                    temp=bad_temp, window=WindowSpec((5, "km")))


def test_clumping_finds_at_least_one_clump():
    dbz, z_km, y_km, x_km = _synthetic_volume()
    cp = ClassificationParams(min_valid_volume_for_convective=1.0)
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     window=WindowSpec((5, "km")), class_params=cp)
    assert r.n_clumps >= 1


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        eccopy3d.run(np.zeros((5, 5)), coords_z=np.zeros(5),
                     coords_y=np.zeros(5), coords_x=np.zeros(5))


# ---------------------------------------------------------------------
# Regression tests for real bugs found comparing against LROSE-ECCO
# truth output (SPOL/WRF/SEA cases) -- see README "Validation status"
# and eccopy3d/clumping.py's module docstring for full details.
# ---------------------------------------------------------------------

def _two_storm_volume(nz=8, ny=30, nx=30, seed=7):
    """Two separate convective cores joined by a weak bridge that only
    clears the PRIMARY convectivity threshold, not the secondary one --
    the real dual-threshold algorithm should split this into 2 clumps;
    naive single-threshold labeling at the secondary threshold merges
    them into 1 (or, if the bridge doesn't even reach threshold, may
    lose the connection and structure of the test entirely)."""
    rng = np.random.default_rng(seed)
    z_km = np.linspace(0.5, 7.5, nz)
    y_km = np.linspace(-15, 15, ny)
    x_km = np.linspace(-15, 15, nx)
    dbz = 10 + rng.normal(0, 0.3, (nz, ny, nx))
    for iz in range(nz):
        weight = np.exp(-((iz - 2) ** 2) / 6)
        # two strong, well-separated cores
        dbz[iz, 5:10, 5:10] += 35 * weight
        dbz[iz, 20:25, 20:25] += 35 * weight
        # weak connecting bridge between them -- moderate reflectivity,
        # enough to raise convectivity above a lenient primary threshold
        # but well below the cores themselves
        dbz[iz, 12:18, 12:18] += 12 * weight
    return dbz, z_km, y_km, x_km


def test_dual_threshold_clumping_splits_separate_storms():
    """Real bug (found via SPOL LOW truth comparison): the previous
    clumping implementation did single-stage labeling directly at the
    secondary threshold, discarding the real two-stage envelope/subclump-
    split algorithm entirely -- this collapsed 96.5% of true convective
    pixels into a single Mixed blob. This test checks the qualitative
    behaviour the fix restores: two clearly separate storm cores, weakly
    bridged only at a lenient primary threshold, should be found as 2+
    clumps once past that primary threshold -- not silently merged into
    one via the bridge, and not lost by clumping at too strict a
    threshold to see the bridge at all."""
    dbz, z_km, y_km, x_km = _two_storm_volume()
    height = np.broadcast_to(z_km[:, None, None], dbz.shape).copy()
    cp = ClassificationParams(
        min_convectivity_for_convective=0.3,   # lenient enough to include the bridge
        secondary_convectivity=0.65,
        use_dual_thresholds=True,
        min_valid_volume_for_convective=1.0,
        each_subclump_min_area_frac=0.02,
        each_subclump_min_area_km2=0.5,
        all_subclumps_min_area_frac=0.1,
    )
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     height=height, window=WindowSpec((4, "km")),
                     class_params=cp)
    assert r.n_clumps >= 2


def test_min_valid_dbz_nulls_subthreshold_before_texture():
    """Real bug (found via SPOL LOW truth comparison, closed a 191,000-
    pixel Missing-boundary mismatch): min_valid_dbz previously only
    gated the fraction_active coverage count -- sub-threshold dBZ values
    still participated as valid neighbours in the texture plane-fit,
    inflating texture/convectivity near the edges of valid-data regions.
    Fixed by nulling dbz < min_valid_dbz to NaN before ANY texture
    computation. This test checks that a field with real sub-threshold
    noise gives IDENTICAL output to the same field with that noise
    replaced by NaN outright -- if min_valid_dbz genuinely nulls
    sub-threshold values upstream of texture, these two inputs are
    indistinguishable to the rest of the pipeline."""
    dbz, z_km, y_km, x_km = _synthetic_volume()
    min_valid_dbz = 0.0

    # explicitly inject a block of sub-threshold "clear air" noise near
    # the edge of the storm, rather than relying on natural noise to
    # produce sub-threshold points -- guarantees the test actually
    # exercises the fix regardless of the base fixture's value range
    rng = np.random.default_rng(99)
    dbz_with_subthreshold_noise = dbz.copy()
    inject_region = np.zeros_like(dbz, dtype=bool)
    inject_region[:, 2:6, 2:8] = True
    dbz_with_subthreshold_noise[inject_region] = rng.uniform(
        min_valid_dbz - 10, min_valid_dbz - 0.1, size=np.sum(inject_region)
    )
    below = dbz_with_subthreshold_noise < min_valid_dbz
    assert np.any(below), "test fixture has no sub-threshold points to exercise"

    dbz_with_nan = dbz.copy()
    dbz_with_nan[below] = np.nan

    from eccopy.params import TextureParams
    tp = TextureParams(min_valid_dbz=min_valid_dbz)

    r_noise = eccopy3d.run(dbz_with_subthreshold_noise, coords_z=z_km,
                           coords_y=y_km, coords_x=x_km,
                           window=WindowSpec((5, "km")), texture_params=tp)
    r_nan = eccopy3d.run(dbz_with_nan, coords_z=z_km, coords_y=y_km,
                         coords_x=x_km, window=WindowSpec((5, "km")),
                         texture_params=tp)

    np.testing.assert_allclose(r_noise.texture, r_nan.texture, equal_nan=True)
    np.testing.assert_allclose(r_noise.convectivity, r_nan.convectivity,
                               equal_nan=True)
    np.testing.assert_array_equal(r_noise.echo_type, r_nan.echo_type)


def test_return_intermediates_default_off_and_matches_when_on():
    dbz, z_km, y_km, x_km = _synthetic_volume(nz=6, ny=12, nx=14)
    r_default = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                             window=WindowSpec((5, "km")))
    r_debug = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                           window=WindowSpec((5, "km")),
                           return_intermediates=True, levels=[2, 3])

    assert r_default.fitted_dbz is None
    assert r_default.intermediate_levels is None
    assert r_debug.fitted_dbz is not None
    assert r_debug.fitted_dbz.shape == (2, dbz.shape[1], dbz.shape[2])
    np.testing.assert_array_equal(r_debug.intermediate_levels, [2, 3])

    np.testing.assert_allclose(r_default.texture, r_debug.texture, equal_nan=True)
    np.testing.assert_array_equal(r_default.echo_type, r_debug.echo_type)
    assert r_default.n_clumps == r_debug.n_clumps


def test_return_intermediates_without_levels_stays_none():
    dbz, z_km, y_km, x_km = _synthetic_volume(nz=6, ny=8, nx=8)
    r = eccopy3d.run(dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
                     window=WindowSpec((5, "km")), return_intermediates=True)
    assert r.fitted_dbz is None
    assert r.intermediate_levels is None
