"""Tests for eccopy2d_v.run() — array-based 2-D vertical cross-section."""

import numpy as np
import pytest

from eccopy import eccopy2d_v
from eccopy.params import WindowSpec


def _synthetic_section(nz=20, nx=100, seed=2):
    rng = np.random.default_rng(seed)
    z_km = np.linspace(0.5, 12, nz)
    x_km = np.linspace(-75, 75, nx)
    base = 15 - 0.5 * z_km[:, None]
    dbz = base + rng.normal(0, 0.5, (nz, nx))
    # Embed a convective tower spanning multiple z-levels
    cx = 50
    for iz in range(nz):
        dbz[iz, cx - 4: cx + 4] += 25 * np.exp(-((iz - 3) ** 2) / 20)
    return dbz, z_km, x_km


def test_basic_mode_shapes():
    dbz, z_km, x_km = _synthetic_section()
    r = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                       window=WindowSpec((5, "km")))
    assert r.echo_type.shape == dbz.shape
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    assert codes.issubset({1, 2, 3})


def test_height_enables_subclassification():
    """height alone no longer triggers sub-classification -- the current
    API requires height, melt, AND temp together (see eccopy2d_v/
    classify.py module docstring 'API CHANGE'). height-only should fall
    back to basic-only codes; see test_all_three_required_together below
    for the actual sub-classification trigger."""
    dbz, z_km, x_km = _synthetic_section()
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()
    r = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km, height=height,
                       window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    assert codes.issubset({1, 2, 3})


def test_temp_enables_subclassification():
    """temp alone no longer triggers sub-classification -- see
    test_height_enables_subclassification above."""
    dbz, z_km, x_km = _synthetic_section()
    height = np.broadcast_to(z_km[:, None], dbz.shape)
    temp = 20 - 6.5 * height
    r = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km, temp=temp,
                       window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    assert codes.issubset({1, 2, 3})


def test_height_melt_and_temp_together_enable_subclassification():
    """The actual current trigger for sub-classification: height, melt,
    AND temp must ALL be provided together (see module docstring 'API
    CHANGE' -- height/temp alone used to be accepted as alternatives,
    but the real f_classSub.m reference requires all three)."""
    dbz, z_km, x_km = _synthetic_section()
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()
    temp = 20 - 6.5 * height
    melt = np.broadcast_to((10.0 * z_km)[:, None], dbz.shape).copy()
    r = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                       height=height, melt=melt, temp=temp,
                       window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    sub_codes = {14, 16, 18, 25, 32, 34, 36, 38}
    assert codes.issubset(sub_codes)
    assert not codes & {1, 2, 3}  # basic codes should not appear


def test_partial_args_fall_back_to_basic():
    """Any ONE of height/melt/temp missing -> basic-only codes, not a
    partial/best-effort sub-classification -- see module docstring
    'does NOT attempt a partial/best-effort sub-classification'."""
    dbz, z_km, x_km = _synthetic_section()
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()
    temp = 20 - 6.5 * height
    melt = np.broadcast_to((10.0 * z_km)[:, None], dbz.shape).copy()

    r_no_melt = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                               height=height, temp=temp,
                               window=WindowSpec((5, "km")))
    r_no_height = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                                 melt=melt, temp=temp,
                                 window=WindowSpec((5, "km")))
    r_no_temp = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                               height=height, melt=melt,
                               window=WindowSpec((5, "km")))
    for r in (r_no_melt, r_no_height, r_no_temp):
        codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
        assert codes.issubset({1, 2, 3})


def test_temp_as_sounding_matches_manually_broadcast_field():
    """A (Z,) sounding should give identical results to manually
    broadcasting that same profile across every column."""
    dbz, z_km, x_km = _synthetic_section()
    sounding = 20.0 - 6.5 * z_km   # shape (nz,)

    r_sounding = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                                temp=sounding, window=WindowSpec((5, "km")))

    full_field = np.broadcast_to(sounding[:, None], dbz.shape).copy()
    r_full = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                            temp=full_field, window=WindowSpec((5, "km")))

    np.testing.assert_array_equal(r_sounding.echo_type, r_full.echo_type)
    np.testing.assert_allclose(r_sounding.convectivity, r_full.convectivity,
                               equal_nan=True)


def test_temp_sounding_enables_subclassification():
    dbz, z_km, x_km = _synthetic_section()
    sounding = 20.0 - 6.5 * z_km
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()
    melt = np.broadcast_to((10.0 * z_km)[:, None], dbz.shape).copy()
    r = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                       height=height, melt=melt, temp=sounding,
                       window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    sub_codes = {14, 16, 18, 25, 32, 34, 36, 38}
    assert codes.issubset(sub_codes)


def test_temp_wrong_shape_raises_clear_error():
    """temp's shape is only validated once it's actually used -- i.e.
    together with height and melt (see the early-return gate in
    eccopy2d_v/classify.py). An unused temp (height/melt missing) is
    never touched, so this must supply all three to actually exercise
    the validation."""
    dbz, z_km, x_km = _synthetic_section()
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()
    melt = np.broadcast_to((10.0 * z_km)[:, None], dbz.shape).copy()
    bad_temp = np.zeros(len(z_km) + 5)  # neither (Z,) nor (Z, X)
    with pytest.raises(ValueError):
        eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                       height=height, melt=melt, temp=bad_temp,
                       window=WindowSpec((5, "km")))


def test_temp_wrong_shape_silently_unused_without_height_and_melt():
    """CAUTION -- documents current behaviour, not necessarily desired
    behaviour: if height or melt is missing, temp is never touched at
    all (see the early-return gate), so a malformed temp shape does NOT
    raise here, unlike the all-three-present case above. This is a
    'fail slow, not fail fast' gap worth being aware of when debugging
    a silently-basic-only result -- a shape typo in temp alone won't
    tell you why sub-classification didn't trigger."""
    dbz, z_km, x_km = _synthetic_section()
    bad_temp = np.zeros(len(z_km) + 5)  # neither (Z,) nor (Z, X)
    r = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km, temp=bad_temp,
                       window=WindowSpec((5, "km")))
    codes = set(np.unique(r.echo_type[~np.isnan(r.echo_type)]).astype(int))
    assert codes.issubset({1, 2, 3})


def test_height_priority_over_temp():
    """If both height and temp are given, height should take priority."""
    dbz, z_km, x_km = _synthetic_section()
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()
    # Deliberately wrong/conflicting temp field
    bad_temp = np.full(dbz.shape, -999.0)
    r_height_only = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                                   height=height, window=WindowSpec((5, "km")))
    r_both = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                            height=height, temp=bad_temp,
                            window=WindowSpec((5, "km")))
    np.testing.assert_array_equal(r_height_only.echo_type, r_both.echo_type)


def test_2d_spacing_field_accepted():
    dbz, z_km, x_km = _synthetic_section()
    sp_z_2d = np.broadcast_to((np.diff(z_km, append=z_km[-1]))[:, None], dbz.shape)
    sp_x_2d = np.broadcast_to((np.diff(x_km, append=x_km[-1]))[None, :], dbz.shape)
    r = eccopy2d_v.run(dbz, coords_z=sp_z_2d, coords_x=sp_x_2d,
                       window=WindowSpec((5, "km")), coord_mode="spacing")
    assert r.echo_type.shape == dbz.shape


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        eccopy2d_v.run(np.zeros(10), coords_z=np.zeros(5), coords_x=np.zeros(2))


# ---------------------------------------------------------------------------
# Row-varying coords_x (non-uniform range-gate spacing per Z level)
# ---------------------------------------------------------------------------

def test_row_varying_coords_x_uses_each_rows_own_spacing():
    """A genuinely 2-D coords_x (spacing that differs by Z level) must
    resolve each row's own window radius when kernel_mode='varying' --
    see core/texture.py's refl_texture_1d fix. Regression test for the
    bug found/fixed last session: row 0's spacing used to be silently
    applied to every row. Not the default any more (see
    test_kernel_mode_uniform_is_default_for_2d_v below) -- 'varying'
    must be requested explicitly."""
    nz, nx = 3, 60
    rng = np.random.default_rng(7)
    dbz = 15 + rng.normal(0, 2, (nz, nx))

    x_fine = np.arange(nx) * 0.5     # 0.5 km spacing
    x_coarse = np.arange(nx) * 2.0   # 2.0 km spacing
    coords_x_2d = np.stack([x_fine, x_fine, x_coarse])
    coords_z = np.array([1.0, 2.0, 3.0])

    r_2d = eccopy2d_v.run(dbz, coords_z=coords_z, coords_x=coords_x_2d,
                          window=WindowSpec((5, "km")), kernel_mode="varying")

    # Row 2 alone, run through eccopy1d with its OWN true spacing (also
    # 'varying', since eccopy1d's per-point resolution is what's being
    # matched here), should match row 2 of the 2D-V texture exactly.
    from eccopy import eccopy1d
    r_row2_alone = eccopy1d.run(dbz[2], coords=x_coarse,
                                window=WindowSpec((5, "km")), kernel_mode="varying")
    np.testing.assert_allclose(r_2d.texture[2], r_row2_alone.texture, equal_nan=True)

    # And it should NOT match what row 0's (finer) spacing would have
    # produced for the same data -- confirms 'varying' actually changed
    # behavior, not just that both code paths agree by coincidence.
    r_row2_wrong_spacing = eccopy1d.run(dbz[2], coords=x_fine,
                                        window=WindowSpec((5, "km")), kernel_mode="varying")
    assert not np.allclose(r_2d.texture[2], r_row2_wrong_spacing.texture, equal_nan=True)


def test_kernel_mode_uniform_is_default_for_2d_v():
    """The default ('uniform') collapses row-varying coords_x to ONE
    global-median radius for the whole (Z, X) array -- explicitly NOT
    the per-row behaviour exercised above."""
    nz, nx = 3, 60
    rng = np.random.default_rng(7)
    dbz = 15 + rng.normal(0, 2, (nz, nx))
    x_fine = np.arange(nx) * 0.5
    x_coarse = np.arange(nx) * 2.0
    coords_x_2d = np.stack([x_fine, x_fine, x_coarse])
    coords_z = np.array([1.0, 2.0, 3.0])

    r_default = eccopy2d_v.run(dbz, coords_z=coords_z, coords_x=coords_x_2d,
                               window=WindowSpec((5, "km")))
    r_explicit_uniform = eccopy2d_v.run(dbz, coords_z=coords_z, coords_x=coords_x_2d,
                                        window=WindowSpec((5, "km")), kernel_mode="uniform")
    np.testing.assert_allclose(r_default.texture, r_explicit_uniform.texture, equal_nan=True)

    r_varying = eccopy2d_v.run(dbz, coords_z=coords_z, coords_x=coords_x_2d,
                               window=WindowSpec((5, "km")), kernel_mode="varying")
    assert not np.allclose(r_default.texture[2], r_varying.texture[2], equal_nan=True)


def test_kernel_mode_uniform_matches_varying_on_truly_uniform_grid():
    """On a genuinely uniform-spacing grid (the SEA/SPOL-shaped case),
    'uniform' and 'varying' must agree exactly."""
    dbz, z_km, x_km = _synthetic_section()
    r_uniform = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                               window=WindowSpec((5, "km")), kernel_mode="uniform")
    r_varying = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                               window=WindowSpec((5, "km")), kernel_mode="varying")
    np.testing.assert_allclose(r_uniform.texture, r_varying.texture, equal_nan=True)


def test_uniform_coords_x_1d_still_matches_eccopy1d():
    """Baseline: the common case (1-D coords_x, shared across all rows)
    should still match a standalone eccopy1d run row-for-row."""
    from eccopy import eccopy1d
    nz, nx = 4, 80
    rng = np.random.default_rng(8)
    dbz = 15 + rng.normal(0, 2, (nz, nx))
    x_km = np.arange(nx) * 1.0
    coords_z = np.arange(nz) * 1.0

    r_2d = eccopy2d_v.run(dbz, coords_z=coords_z, coords_x=x_km,
                          window=WindowSpec((5, "km")))
    for row in range(nz):
        r_row = eccopy1d.run(dbz[row], coords=x_km, window=WindowSpec((5, "km")))
        np.testing.assert_allclose(r_2d.texture[row], r_row.texture, equal_nan=True)


def test_return_intermediates_default_off_and_matches_when_on():
    """return_intermediates=False (default) must leave the debug fields
    None and not change texture/convectivity/echo_type at all; =True
    must add fitted_dbz/detrended_dbz/echo_basic without changing them
    either."""
    dbz, z_km, x_km = _synthetic_section()
    r_default = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                               window=WindowSpec((5, "km")))
    r_debug = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                             window=WindowSpec((5, "km")), return_intermediates=True)

    assert r_default.fitted_dbz is None
    assert r_default.detrended_dbz is None
    assert r_default.echo_basic is None

    assert r_debug.fitted_dbz is not None
    assert r_debug.fitted_dbz.shape == dbz.shape
    assert r_debug.echo_basic is not None
    assert r_debug.echo_basic.shape == dbz.shape

    np.testing.assert_allclose(r_default.texture, r_debug.texture, equal_nan=True)
    np.testing.assert_allclose(r_default.convectivity, r_debug.convectivity, equal_nan=True)
    np.testing.assert_array_equal(r_default.echo_type, r_debug.echo_type)


def test_remove_surface_echo_matches_manual_masking():
    """remove_surface_echo=True must reproduce the driver-level surfAltLim
    masking from the real ECCO-V MATLAB scripts:
        data.DBZ_F(data.Z .* 1000 < surfAltLim) = nan;   (before texture)
    which is exactly the manual `dbz[z_km < surf_alt_lim/1000] = nan`
    workaround, and must NOT mutate the caller's dbz array."""
    from eccopy.params import ClassificationParams

    dbz, z_km, x_km = _synthetic_section(nz=24)
    # Ensure some rows fall below the limit.
    z_km = np.linspace(0.05, 12, dbz.shape[0])
    cp = ClassificationParams(surf_alt_lim=200.0)   # 0.2 km
    below = z_km < 0.2
    assert below.any(), "test grid must include rows below surf_alt_lim"

    dbz_caller = dbz.copy()
    r_flag = eccopy2d_v.run(dbz_caller, coords_z=z_km, coords_x=x_km,
                            window=WindowSpec(7), class_params=cp,
                            remove_surface_echo=True)

    # caller's dbz must be untouched
    assert np.array_equal(dbz_caller, dbz)

    dbz_manual = dbz.copy()
    dbz_manual[below] = np.nan
    r_manual = eccopy2d_v.run(dbz_manual, coords_z=z_km, coords_x=x_km,
                              window=WindowSpec(7), class_params=cp,
                              remove_surface_echo=False)

    def _key(a):
        return np.nan_to_num(a, nan=-1.0)

    assert np.array_equal(_key(r_flag.echo_type), _key(r_manual.echo_type))
    assert np.all(np.isnan(r_flag.echo_type[below]))


def test_remove_surface_echo_default_off_is_row_local():
    """Default (off) leaves dbz unmasked; turning it on only affects rows
    at/below the limit -- rows above must be bit-identical (2-D-V texture
    slides along X, so masking a bottom row cannot contaminate rows above)."""
    from eccopy.params import ClassificationParams

    dbz, _, x_km = _synthetic_section(nz=24)
    z_km = np.linspace(0.05, 12, dbz.shape[0])
    cp = ClassificationParams(surf_alt_lim=200.0)
    below = z_km < 0.2

    r_off = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                           window=WindowSpec(7), class_params=cp,
                           remove_surface_echo=False)
    r_on = eccopy2d_v.run(dbz, coords_z=z_km, coords_x=x_km,
                          window=WindowSpec(7), class_params=cp,
                          remove_surface_echo=True)

    def _key(a):
        return np.nan_to_num(a, nan=-1.0)

    assert not np.array_equal(_key(r_off.echo_type), _key(r_on.echo_type))
    assert np.array_equal(_key(r_off.echo_type[~below]),
                          _key(r_on.echo_type[~below]))
