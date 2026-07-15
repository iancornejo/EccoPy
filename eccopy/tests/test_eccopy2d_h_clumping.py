"""Tests for eccopy.eccopy2d_h.clumping.find_clumps_2d() in isolation
from the rest of the eccopy2d_h pipeline (no texture step involved) —
see test_eccopy2d_h.py for the end-to-end version."""

import numpy as np

from eccopy.eccopy2d_h.clumping import find_clumps_2d


def _two_bridged_peaks(ny=20, nx=40):
    """Two well-separated high-convectivity cores connected by a bridge
    that clears the PRIMARY threshold but not the SECONDARY one."""
    conv = np.full((ny, nx), 0.1)
    conv[8:12, 5:9] = 0.9
    conv[8:12, 30:34] = 0.9
    conv[9:11, 9:30] = 0.55
    return conv


def test_single_isolated_clump_found():
    conv = np.full((10, 10), 0.1)
    conv[3:7, 3:7] = 0.9
    sp = np.ones((10, 10))
    clumps = find_clumps_2d(conv, sp, sp, min_conv=0.5, use_dual_thresholds=False)
    assert len(clumps) == 1
    assert clumps[0]['n_pts_total'] == 16
    assert clumps[0]['area_km2'] == 16.0


def test_no_clumps_below_threshold():
    conv = np.full((10, 10), 0.1)
    sp = np.ones((10, 10))
    clumps = find_clumps_2d(conv, sp, sp, min_conv=0.5, use_dual_thresholds=False)
    assert clumps == []


def test_dual_threshold_splits_bridged_peaks():
    conv = _two_bridged_peaks()
    sp = np.ones_like(conv)
    split = find_clumps_2d(conv, sp, sp, min_conv=0.5,
                           use_dual_thresholds=True, secondary_threshold=0.65,
                           all_subclumps_min_area_frac=0.1,
                           each_subclump_min_area_frac=0.02,
                           each_subclump_min_area_km2=1.0)
    assert len(split) == 2


def test_single_threshold_does_not_split():
    conv = _two_bridged_peaks()
    sp = np.ones_like(conv)
    nosplit = find_clumps_2d(conv, sp, sp, min_conv=0.5, use_dual_thresholds=False)
    assert len(nosplit) == 1


def test_split_pieces_partition_the_primary_clump():
    conv = _two_bridged_peaks()
    sp = np.ones_like(conv)
    split = find_clumps_2d(conv, sp, sp, min_conv=0.5,
                           use_dual_thresholds=True, secondary_threshold=0.65,
                           all_subclumps_min_area_frac=0.1,
                           each_subclump_min_area_frac=0.02,
                           each_subclump_min_area_km2=1.0)
    nosplit = find_clumps_2d(conv, sp, sp, min_conv=0.5, use_dual_thresholds=False)
    total_split_pts = sum(c['n_pts_total'] for c in split)
    assert total_split_pts == nosplit[0]['n_pts_total']


def test_min_area_drops_small_clump_entirely():
    conv = np.full((20, 20), 0.1)
    conv[5:7, 5:7] = 0.9   # small: 4 points
    conv[10:16, 10:16] = 0.9   # large: 36 points
    sp = np.ones((20, 20))
    clumps = find_clumps_2d(conv, sp, sp, min_conv=0.5, min_area_km2=10.0,
                            use_dual_thresholds=False)
    assert len(clumps) == 1
    assert clumps[0]['area_km2'] == 36.0


def test_min_area_none_disables_filter():
    conv = np.full((20, 20), 0.1)
    conv[5:7, 5:7] = 0.9   # small: 4 points, area 4 km^2
    sp = np.ones((20, 20))
    clumps = find_clumps_2d(conv, sp, sp, min_conv=0.5, min_area_km2=None,
                            use_dual_thresholds=False)
    assert len(clumps) == 1
    assert clumps[0]['area_km2'] == 4.0


def test_variable_spacing_area_computed_correctly():
    conv = np.full((4, 4), 0.9)
    sp_y = np.full((4, 4), 2.0)
    sp_x = np.full((4, 4), 0.5)
    clumps = find_clumps_2d(conv, sp_y, sp_x, min_conv=0.5, use_dual_thresholds=False)
    assert len(clumps) == 1
    # 16 points * (2.0 * 0.5) km^2 each = 16.0 km^2
    assert clumps[0]['area_km2'] == 16.0


def test_nan_excluded_from_clumps():
    conv = np.full((10, 10), 0.9)
    conv[3, 3] = np.nan
    sp = np.ones((10, 10))
    clumps = find_clumps_2d(conv, sp, sp, min_conv=0.5, use_dual_thresholds=False)
    assert len(clumps) == 1
    assert clumps[0]['n_pts_total'] == 99
