"""Tests for eccopy.core.classification.filter_short_convective_runs_1d()
in isolation from the rest of the eccopy1d pipeline (no texture/class_basic
involved) — see test_eccopy1d.py for the end-to-end version."""
import pytest

import numpy as np

from eccopy.core.classification import (
    filter_short_convective_runs_1d, CATEGORY_MIXED,
)


def test_short_run_demoted_to_mixed():
    echo = np.array([1, 1, 3, 3, 1, 1], dtype=float)
    spacing = np.full(6, 1.0)  # 1 unit per point
    out = filter_short_convective_runs_1d(echo, spacing, min_length_base=5.0)
    assert np.all(out[2:4] == CATEGORY_MIXED)
    assert out[0] == 1 and out[1] == 1


def test_long_run_survives():
    echo = np.array([1, 3, 3, 3, 3, 3, 3, 1], dtype=float)
    spacing = np.full(8, 1.0)
    out = filter_short_convective_runs_1d(echo, spacing, min_length_base=5.0)
    assert np.all(out[1:7] == 3)


def test_disabled_when_min_length_none_or_zero():
    echo = np.array([1, 3, 1], dtype=float)
    spacing = np.full(3, 1.0)
    out_none = filter_short_convective_runs_1d(echo, spacing, min_length_base=None)
    out_zero = filter_short_convective_runs_1d(echo, spacing, min_length_base=0.0)
    np.testing.assert_array_equal(out_none, echo)
    np.testing.assert_array_equal(out_zero, echo)


def test_multiple_runs_filtered_independently():
    # Run A: short (demoted). Run B: long (survives).
    echo = np.array([3, 1, 1, 3, 3, 3, 3, 3, 1], dtype=float)
    spacing = np.full(9, 1.0)
    out = filter_short_convective_runs_1d(echo, spacing, min_length_base=4.0)
    assert out[0] == CATEGORY_MIXED
    assert np.all(out[3:8] == 3)


def test_variable_spacing_respected():
    # A 2-point run with wide spacing should survive; an equal-length run
    # with narrow spacing should not.
    echo = np.array([3, 3, 1, 3, 3], dtype=float)
    spacing = np.array([5.0, 5.0, 1.0, 0.5, 0.5])
    out = filter_short_convective_runs_1d(echo, spacing, min_length_base=6.0)
    assert np.all(out[0:2] == 3)          # 5+5=10 >= 6 -> survives
    assert np.all(out[3:5] == CATEGORY_MIXED)  # 0.5+0.5=1 < 6 -> demoted


def test_nan_untouched():
    echo = np.array([np.nan, 3, 3, np.nan], dtype=float)
    spacing = np.full(4, 1.0)
    out = filter_short_convective_runs_1d(echo, spacing, min_length_base=10.0)
    assert np.isnan(out[0]) and np.isnan(out[3])


# ---------------------------------------------------------------------------
# class_basic_isotropic() -- border_value / sequential-close fix sanity
# ---------------------------------------------------------------------------

def test_class_basic_isotropic_runs_at_validated_radius():
    from eccopy.core.classification import class_basic_isotropic
    conv = np.full((40, 40), 0.1)
    conv[15:25, 15:25] = 0.9
    result = class_basic_isotropic(conv, strat_mixed=0.4, mixed_conv=0.5,
                                   enlarge_mixed=5, enlarge_conv=5)
    assert result.shape == conv.shape
    assert set(np.unique(result[~np.isnan(result)])).issubset({1, 2, 3})
    assert 3 in result  # the embedded high-convectivity block should register


def test_class_basic_isotropic_arbitrary_radius_now_supported():
    # Radii without a bundled MATLAB .mat used to raise FileNotFoundError;
    # eccopy.core.disk now generates the strel/decomposition on demand
    # (bit-exact to MATLAB -- see test_disk_generator.py), so any radius
    # works. enlarge=8 needs disk r8 + decomposition r24/r40, none of
    # which were ever exported.
    from eccopy.core.classification import class_basic_isotropic
    conv = np.full((60, 60), 0.1)
    conv[25:35, 25:35] = 0.9
    result = class_basic_isotropic(conv, strat_mixed=0.4, mixed_conv=0.5,
                                   enlarge_mixed=8, enlarge_conv=8)
    assert result.shape == conv.shape
    assert set(np.unique(result[~np.isnan(result)])).issubset({1, 2, 3})
    assert 3 in result


# ---------------------------------------------------------------------------
# available_enlarge_radii_px() / resolve_enlarge_radius_px()
# ---------------------------------------------------------------------------

def test_available_enlarge_radii_px_lists_bundled_fixtures():
    # No longer a hard limit -- just which radii have a ground-truth fixture.
    from eccopy.core.classification import available_enlarge_radii_px
    assert available_enlarge_radii_px() == [3, 5, 15, 25]


def test_resolve_enlarge_radius_px_exact_match_no_warning():
    from eccopy.core.classification import resolve_enlarge_radius_px
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here should fail the test
        r = resolve_enlarge_radius_px(target_km=5.0, representative_spacing_km=1.0)
    assert r == 5


def test_resolve_enlarge_radius_px_resolves_exact_radius_no_snapping():
    # Any radius is now generatable, so resolution is exact (rounded), not
    # snapped to a bundled set -- and never warns.
    from eccopy.core.classification import resolve_enlarge_radius_px
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 6 km at 1 km/px -> 6 px exactly (old behaviour snapped to 5)
        assert resolve_enlarge_radius_px(target_km=6.0, representative_spacing_km=1.0) == 6
        # 20 km at 1 km/px -> 20 px exactly (old behaviour snapped to 15/25)
        assert resolve_enlarge_radius_px(target_km=20.0, representative_spacing_km=1.0) == 20
        # far-from-any-fixture target that used to warn now resolves cleanly
        assert resolve_enlarge_radius_px(target_km=100.0, representative_spacing_km=1.0) == 100


def test_resolve_enlarge_radius_px_rounds_and_clamps():
    from eccopy.core.classification import resolve_enlarge_radius_px
    assert resolve_enlarge_radius_px(target_km=5.4, representative_spacing_km=1.0) == 5
    assert resolve_enlarge_radius_px(target_km=5.6, representative_spacing_km=1.0) == 6
    # sub-pixel target clamps to a minimum radius of 1
    assert resolve_enlarge_radius_px(target_km=0.1, representative_spacing_km=1.0) == 1


def test_resolve_enlarge_radius_px_invalid_spacing_raises():
    from eccopy.core.classification import resolve_enlarge_radius_px
    with pytest.raises(ValueError):
        resolve_enlarge_radius_px(target_km=5.0, representative_spacing_km=0.0)
    with pytest.raises(ValueError):
        resolve_enlarge_radius_px(target_km=5.0, representative_spacing_km=-2.0)
    with pytest.raises(ValueError):
        resolve_enlarge_radius_px(target_km=5.0, representative_spacing_km=float("nan"))


def test_resolve_enlarge_radius_px_respects_representative_spacing():
    from eccopy.core.classification import resolve_enlarge_radius_px
    # Same target_km, different spacing -> different pixel-space target -> different result.
    r_fine = resolve_enlarge_radius_px(target_km=5.0, representative_spacing_km=0.5)   # 10 px
    r_coarse = resolve_enlarge_radius_px(target_km=5.0, representative_spacing_km=2.5)  # 2 px
    assert r_fine == 10 and r_coarse == 2 and r_fine != r_coarse
