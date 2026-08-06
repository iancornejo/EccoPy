"""
stats must treat BOTH missing-data conventions as missing.

eccopy2d_h / eccopy2d_v return float arrays using NaN for no-echo;
eccopy3d returns int16 using CATEGORY_MISSING (0). Testing only for NaN
counted every 3-D sentinel voxel as valid, inflating the denominator of
every fraction and understating all three categories.
"""
import numpy as np

from eccopy import stats


def _float_field():
    """2-D style: NaN marks no echo."""
    a = np.full((10, 10), np.nan)
    a[:4] = 14.0     # stratiform low
    a[4:6] = 25.0    # mixed
    a[6:8] = 38.0    # convective deep
    return a         # 80 classified of 100


def _int16_field():
    """3-D style: 0 marks no echo."""
    a = np.zeros((10, 10), dtype=np.int16)
    a[:4] = 14
    a[4:6] = 25
    a[6:8] = 38
    return a         # 80 classified of 100


def test_int16_sentinel_is_excluded_from_n_valid():
    assert stats.echo_type_fractions(_int16_field())["n_valid"] == 80


def test_nan_is_excluded_from_n_valid():
    assert stats.echo_type_fractions(_float_field())["n_valid"] == 80


def test_both_conventions_agree():
    """The same classification expressed either way must give one answer."""
    a = stats.echo_type_fractions(_float_field())
    b = stats.echo_type_fractions(_int16_field())
    assert a == b


def test_fractions_sum_to_one_for_int16():
    f = stats.echo_type_fractions(_int16_field())
    total = f["stratiform"] + f["mixed"] + f["convective"]
    assert abs(total - 1.0) < 1e-12


def test_percentages_use_the_classified_denominator():
    field = _int16_field()
    # 40 of 80 classified points are stratiform, not 40 of 100.
    assert abs(stats.stratiform_percentage(field) - 50.0) < 1e-9
    assert abs(stats.mixed_percentage(field) - 25.0) < 1e-9
    assert abs(stats.convective_percentage(field) - 25.0) < 1e-9


def test_all_sentinel_field_reports_no_valid_points():
    f = stats.echo_type_fractions(np.zeros((5, 5), dtype=np.int16))
    assert f["n_valid"] == 0
    assert np.isnan(f["stratiform"])


def test_basic_codes_still_detected_with_int16_sentinel():
    """A 1/2/3 array padded with 0 must not be mistaken for sub-classified."""
    a = np.zeros((6, 6), dtype=np.int16)
    a[:2] = 1
    a[2:4] = 2
    a[4:5] = 3
    assert stats.codes_for_category(a, "stratiform") == frozenset({1})
    assert stats.echo_type_fractions(a)["n_valid"] == 30
