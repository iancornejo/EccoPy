"""Tests for eccopy.stats -- generic post-classification statistics."""

import numpy as np
import pytest

from eccopy import stats


def test_echo_type_fractions_basic_codes():
    echo = np.array([1, 1, 2, 3, 3, 3, np.nan])
    out = stats.echo_type_fractions(echo)
    assert out["n_valid"] == 6
    assert np.isclose(out["stratiform"], 2 / 6)
    assert np.isclose(out["mixed"], 1 / 6)
    assert np.isclose(out["convective"], 3 / 6)


def test_echo_type_fractions_sub_codes():
    echo = np.array([14, 16, 18, 25, 32, 34, 36, 38])
    out = stats.echo_type_fractions(echo)
    assert np.isclose(out["stratiform"], 3 / 8)
    assert np.isclose(out["mixed"], 1 / 8)
    assert np.isclose(out["convective"], 4 / 8)


def test_percentages_sum_to_100_when_fully_classified():
    echo = np.array([1, 2, 3, 1, 2, 3])
    total = (stats.convective_percentage(echo) + stats.stratiform_percentage(echo)
             + stats.mixed_percentage(echo))
    assert np.isclose(total, 100.0)


def test_percentages_all_nan_returns_nan():
    echo = np.full((3, 3), np.nan)
    assert np.isnan(stats.convective_percentage(echo))


def test_n_clumps_counts_disconnected_regions_2d():
    echo = np.ones((10, 10))  # all stratiform
    echo[2:4, 2:4] = 3        # one convective blob
    echo[7:9, 7:9] = 3        # a second, disconnected, convective blob
    assert stats.n_clumps(echo, category="convective") == 2


def test_n_clumps_connected_region_is_one_clump():
    echo = np.ones((10, 10))
    echo[2:8, 2:4] = 3   # single connected convective region
    assert stats.n_clumps(echo, category="convective") == 1


def test_n_clumps_zero_when_category_absent():
    echo = np.ones((5, 5))
    assert stats.n_clumps(echo, category="convective") == 0


def test_clump_sizes_pixel_count_descending():
    echo = np.ones((10, 10))
    echo[0:1, 0:4] = 3   # size 4
    echo[5:9, 5:9] = 3   # size 16
    sizes = stats.clump_sizes(echo, category="convective")
    assert list(sizes) == [16.0, 4.0]


def test_clump_sizes_physical_with_spacing():
    echo = np.ones((4, 4))
    echo[0:2, 0:2] = 3   # 4 pixels
    spacing = np.full((4, 4), 2.0)   # 2 km spacing -> 4 km^2 per pixel
    sizes = stats.clump_sizes(echo, category="convective", spacing=spacing)
    assert np.isclose(sizes[0], 4 * (2.0 ** 2))


def test_convective_depth_basic_column():
    # (Z, X) = (5, 1): convective from z-index 1 to 3
    echo = np.array([[1], [3], [3], [3], [1]])
    height = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    depth = stats.convective_depth(echo, height, axis=0)
    assert depth.shape == (1,)
    assert np.isclose(depth[0], 2.0)   # top=3.0, base=1.0


def test_convective_top_base_height_nan_when_no_convective():
    echo = np.array([[1], [1], [1]])
    height = np.array([[0.0], [1.0], [2.0]])
    top = stats.convective_top_height(echo, height, axis=0)
    base = stats.convective_base_height(echo, height, axis=0)
    assert np.isnan(top[0])
    assert np.isnan(base[0])


def test_summarize_returns_expected_keys():
    echo = np.ones((6, 6))
    echo[1:3, 1:3] = 3
    out = stats.summarize(echo)
    for key in ("stratiform", "mixed", "convective", "convective_pct",
                "n_convective_clumps", "convective_clump_sizes"):
        assert key in out


def test_summarize_with_height_adds_depth_stats():
    echo = np.array([[1, 1], [3, 3], [3, 3], [1, 1]])
    height = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    out = stats.summarize(echo, height=height, axis=0)
    assert np.isclose(out["mean_convective_depth"], 1.0)
    assert np.isclose(out["mean_convective_top_height"], 2.0)
    assert np.isclose(out["mean_convective_base_height"], 1.0)


def test_codes_for_category_explicit_lookup():
    assert stats.codes_for_category(np.array([1, 2, 3]), "convective") == frozenset({3})
    assert stats.codes_for_category(np.array([14, 25, 32]), "convective") == frozenset(
        {30, 32, 34, 36, 38}
    )


def test_codes_for_category_invalid_raises():
    with pytest.raises(ValueError):
        stats.codes_for_category(np.array([1, 2, 3]), "bogus")
