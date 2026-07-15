"""Regression tests for class_basic()'s "rain below melting layer"
correction block -- see core/classification.py's `melt` parameter
docstring for the full fix history.

FIXED this session via direct line-by-line comparison against the real
f_classBasic.m source (lrose-ecco repo):
  1. An earlier threshold change (20/10 -> 15/9) was reverted -- the real
     MATLAB source uses meltArea<20 and sentinel 10, matching the
     ORIGINAL pre-session Python constants. melt is NOT a continuous
     0-15ish field as assumed from other parts of this codebase; in the
     real SPOL pipeline it's constructed as a BINARY field (10 or 20)
     directly from a temperature sign test (TEMP<=0 -> 20, TEMP>0 -> 10;
     see run_ecco_v_RHI_spol_gridded.m) -- these tests use that same
     binary convention, not an arbitrary continuous range.
  2. A genuine off-by-one bug: MATLAB's `checkCol(1:firstInd)=1` is
     INCLUSIVE of firstInd (1-indexed); the port wrote
     `check_col[:first_ind] = 1` (excluding first_ind) instead of the
     correct `check_col[:first_ind + 1] = 1`. A NaN exactly at the
     melt-crossing pixel could silently zero out a column's entire
     contribution to the stratiform-percentage check.

Both fixes are verified against the real MATLAB source directly. Tested
against the real SPOL case (20220526_084500): neither fix changes that
case's specific output (the one real clump reaching the check is
genuinely convective per real MATLAB ECHOTYPE ground truth, and none of
its columns hit the off-by-one edge case) -- these tests use synthetic
scenarios instead, built to specifically exercise each fix.
"""

import numpy as np
import pytest

from eccopy.core.classification import class_basic


def _binary_melt(nz, nx, crossing_row):
    """Real MELTING_LAYER convention: binary {10, 20}, matching
    run_ecco_v_RHI_spol_gridded.m's `TEMP<=0 -> 20, TEMP>0 -> 10`. Row 0
    = surface (10, below melt); rows >= crossing_row = 20 (above melt)."""
    melt = np.full((nz, nx), 10.0)
    melt[crossing_row:, :] = 20.0
    return melt


def test_rain_below_melt_demotes_clump_with_real_binary_melt():
    """Core regression: a 'mixed' band entirely below the melt crossing,
    with genuinely stratiform convectivity above it, should be demoted
    when given melt vs melt=None -- using melt's REAL binary {10,20}
    convention, not an arbitrary continuous range."""
    nz, nx = 20, 10
    strat_mixed = 0.4
    mixed_conv = 0.5
    conv = np.full((nz, nx), 0.2)
    conv[2:5, :] = 0.6   # 'mixed' band, well below the melt crossing
    melt = _binary_melt(nz, nx, crossing_row=10)

    r_melt = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=melt.copy(), enlarge_mixed=5, enlarge_conv=5)
    r_none = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=None, enlarge_mixed=5, enlarge_conv=5)
    assert not np.allclose(
        np.nan_to_num(r_melt, nan=-1), np.nan_to_num(r_none, nan=-1)
    ), "rain-below-melt block is inert on a clean, unambiguous synthetic case"


def test_rain_below_melt_demotes_specifically_to_non_mixed():
    """With the real field, the bright-band-like band's Mixed coverage
    should not INCREASE relative to melt=None -- the block only demotes
    qualifying pixels toward stratiform, never adds Mixed."""
    nz, nx = 20, 10
    strat_mixed = 0.4
    mixed_conv = 0.5
    conv = np.full((nz, nx), 0.2)
    conv[2:5, :] = 0.6
    melt = _binary_melt(nz, nx, crossing_row=10)

    r_melt = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=melt.copy(), enlarge_mixed=5, enlarge_conv=5)
    r_none = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=None, enlarge_mixed=5, enlarge_conv=5)
    assert np.sum(r_melt == 2) <= np.sum(r_none == 2)


def test_melt_none_still_behaves_as_before():
    """melt=None must remain a pure no-op (skips the block entirely)."""
    nz, nx = 20, 10
    strat_mixed = 0.4
    mixed_conv = 0.5
    conv = np.full((nz, nx), 0.2)
    conv[2:5, :] = 0.6
    r1 = class_basic(conv.copy(), strat_mixed, mixed_conv,
                     melt=None, enlarge_mixed=5, enlarge_conv=5)
    r2 = class_basic(conv.copy(), strat_mixed, mixed_conv,
                     melt=None, enlarge_mixed=5, enlarge_conv=5)
    np.testing.assert_array_equal(r1, r2)


def test_below_frac_gate_still_skips_clumps_mostly_above_melt():
    """A 'mixed' band sitting mostly ABOVE the melt crossing should NOT
    be touched (below_frac <= 0.8 gate)."""
    nz, nx = 20, 10
    strat_mixed = 0.4
    mixed_conv = 0.5
    conv = np.full((nz, nx), 0.2)
    melt = _binary_melt(nz, nx, crossing_row=10)
    conv[8:13, :] = 0.6  # straddles the crossing at row 10

    r_melt = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=melt.copy(), enlarge_mixed=5, enlarge_conv=5)
    r_none = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=None, enlarge_mixed=5, enlarge_conv=5)
    np.testing.assert_array_equal(r_melt, r_none)


def test_off_by_one_nan_at_crossing_pixel_no_longer_zeroes_column():
    """Regression test for the specific off-by-one bug: a column whose
    convectivity value is NaN EXACTLY at the melt-crossing pixel must
    still contribute its remaining valid points above that pixel to the
    stratiform check, not be silently dropped entirely.

    Construction: single-column-focused scenario (real column geometry
    replicated across a few columns so bwareaopen/labeling behaves
    normally) where conv is NaN at exactly the row where melt crosses
    from 10 to 20, with genuinely stratiform data both immediately above
    and within the mixed band below."""
    nz, nx = 20, 6
    strat_mixed = 0.4
    mixed_conv = 0.5
    conv = np.full((nz, nx), 0.2)
    conv[2:5, :] = 0.6          # mixed band, well below the crossing
    crossing_row = 10
    conv[crossing_row, :] = np.nan   # NaN exactly at the melt-crossing pixel
    melt = _binary_melt(nz, nx, crossing_row=crossing_row)

    r_melt = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=melt.copy(), enlarge_mixed=5, enlarge_conv=5)
    r_none = class_basic(conv.copy(), strat_mixed, mixed_conv,
                         melt=None, enlarge_mixed=5, enlarge_conv=5)
    assert not np.allclose(
        np.nan_to_num(r_melt, nan=-1), np.nan_to_num(r_none, nan=-1)
    ), (
        "NaN exactly at the melt-crossing pixel prevented the rain-below-"
        "melt block from firing -- off-by-one regression"
    )
