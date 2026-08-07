"""
Basic and sub-classification functions.

class_basic()       - port of f_classBasic.m  (EccoPy-1D / EccoPy-2D)
class_sub_2d()      - port of f_classSub.m    (EccoPy-1D / EccoPy-2D)
set_echo_type_3d()  - port of ConvStratFinder::_setEchoType3D() + StormClump

Upstream analogues:
    lrose-ecco @ ad85c56   f_classBasic.m, f_classSub.m
    lrose-core @ b29264bf  ConvStratFinder.cc

Echo type codes (matching f_classSub.m and the C++ enum values)
---------------------------------------------------------------
14  CATEGORY_STRATIFORM_LOW
16  CATEGORY_STRATIFORM_MID
18  CATEGORY_STRATIFORM_HIGH
25  CATEGORY_MIXED
30  CATEGORY_CONVECTIVE (airborne-radar branch; see class_sub_2d)
32  CATEGORY_CONVECTIVE_ELEVATED
34  CATEGORY_CONVECTIVE_SHALLOW
36  CATEGORY_CONVECTIVE_MID
38  CATEGORY_CONVECTIVE_DEEP
 0  CATEGORY_MISSING

Morphology conventions
----------------------
Two details of this file's morphological logic differ from a naive
translation and are load-bearing:

  - `binary_erosion` / `binary_closing` are called with `border_value=1`.
    SciPy defaults to 0, which erodes inward from every array edge and
    removes near-surface regions that MATLAB's imclose retains.

  - Closing runs through `_sequential_close()`, which applies MATLAB's
    own strel decomposition primitives (from `getsequence()`) one at a
    time, each as edge-pad, dilate, erode, crop - see `_edge_pad_close()`.
    A single dilate-then-erode with the full disk gives a different
    result.

`class_basic_isotropic()` uses `scipy.ndimage.binary_closing()` directly
rather than `_sequential_close()`, so the second point above does not
apply to it.

Structuring elements
--------------------
`_disk()` and `_sequential_close()` obtain their `strel('disk', r)`
neighbourhoods and `getsequence()` decompositions from `eccopy.core.disk`,
which reproduces MATLAB's default n=4 periodic-line octagon (Adams 1993)
in pure Python at any radius.

The octagon is not a Euclidean circle, and the difference is not
cosmetic: `strel('disk', 25)` has a 49x49 bounding box, not 51x51, so a
literal `x^2 + y^2 <= r^2` mask over-reaches MATLAB's shape at every
radius. `eccopy.core.disk` produces the octagon; Euclidean (n=0) is
available there as an explicit opt-in.

Because shapes are generated on demand, `enlarge_mixed` / `enlarge_conv`
are not restricted to any pre-exported set of radii. The `.mat` files
bundled under `core/data/` are optional: `_disk()` and `_load_decomp()`
use one as an override when present - byte-identical to the generator -
and otherwise generate. They serve as regression fixtures; see
`tests/test_disk_generator.py`.

Data-file paths are anchored to this module's location
(`Path(__file__).parent / "data" / ...`), not the caller's working
directory. The fixtures ship via `pyproject.toml` package-data and
`MANIFEST.in`.

enlarge_mixed / enlarge_conv are pixel counts
---------------------------------------------
Both default to 5, matching f_classBasic.m, and are read as gate/pixel
counts exactly as the MATLAB source does - not as physical distances.

On a non-uniform grid (an RHI whose range-gate spacing changes with
elevation angle, or any axis whose spacing varies by row) the same
`enlarge_conv=5` therefore represents a different physical radius at
different rows. That is the upstream behaviour. A spacing-aware cleanup
would need a distinct disk at every local spacing, which no upstream
source specifies.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import (
    label, binary_dilation, binary_erosion,
    binary_fill_holes,
)

# Integer echo-type codes
CATEGORY_MISSING             = 0
CATEGORY_STRATIFORM_LOW      = 14
CATEGORY_STRATIFORM_MID      = 16
CATEGORY_STRATIFORM_HIGH     = 18
CATEGORY_MIXED               = 25
CATEGORY_CONVECTIVE          = 30   # near-aircraft override, both branches
CATEGORY_CONVECTIVE_ELEVATED = 32
CATEGORY_CONVECTIVE_SHALLOW  = 34
CATEGORY_CONVECTIVE_MID      = 36
CATEGORY_CONVECTIVE_DEEP     = 38

_CONN8 = np.ones((3, 3), dtype=bool)  # MATLAB bwconncomp's default 2-D connectivity;
                                       # scipy's label() default is 4-connectivity and
                                       # will silently disagree with MATLAB unless this
                                       # structure is passed explicitly every time.

# ---------------------------------------------------------------------------
# Structuring elements -- loaded from MATLAB, not approximated.
# See module docstring "PACKAGING REQUIREMENT" above.
# ---------------------------------------------------------------------------

_DISK_STREL_DIR = str(Path(__file__).parent / "data" / "disk_strels")
_DISK_DECOMP_DIR = str(Path(__file__).parent / "data" / "disk_decomp")
# Anchored to __file__, not the caller's working directory: a relative
# path would resolve only when the process happened to be started beside
# a disk_strels/ folder.
#
# This assumes the data files live at <package_dir>/core/data/disk_strels/
# and <package_dir>/core/data/disk_decomp/ (i.e. a `data/` folder next to
# this classification.py file). Adjust the two lines above if the actual
# package layout puts the data files elsewhere -- e.g. if data files are
# meant to live at the top-level package root rather than inside core/,
# use Path(__file__).parent.parent / "data" / ... instead. Whoever wires
# this into the actual package's setup.py/pyproject.toml packaging
# (package_data / include_package_data) needs to make sure these two
# folders actually ship with an installed copy of eccopy, not just exist
# in a development checkout.

_disk_cache = {}
_decomp_cache = {}


def _disk(radius: int) -> np.ndarray:
    """Exact MATLAB strel('disk', radius).Neighborhood (the n=4 octagon),
    for any radius. This is MATLAB's octagon, not a Euclidean circle -
    see the module docstring.

    Generated in pure Python by eccopy.core.disk, bit-exact to MATLAB -
    see tests/test_disk_generator.py. A bundled .mat for this radius, if
    present, is used as-is; it is byte-identical to the generator, so the
    validated
    radii changes."""
    from .disk import disk_neighborhood
    r = int(radius)
    if r not in _disk_cache:
        mat = Path(f"{_DISK_STREL_DIR}/disk_strel_r{r}.mat")
        if mat.exists():
            _disk_cache[r] = loadmat(str(mat))["nhood"].astype(bool)
        else:
            _disk_cache[r] = disk_neighborhood(r, n=4)
    return _disk_cache[r]


def _load_decomp(radius: int):
    """MATLAB's getsequence(strel('disk', radius)) primitives, for any
    radius. Generated in pure Python by eccopy.core.disk, bit-exact to
    MATLAB - see tests/test_disk_generator.py. A bundled .mat for this
    radius, if present, is used as-is."""
    from .disk import disk_decomposition
    r = int(radius)
    if r not in _decomp_cache:
        info_mat = Path(f"{_DISK_DECOMP_DIR}/disk_decomp_r{r}_info.mat")
        if info_mat.exists():
            n_steps = int(loadmat(str(info_mat))["n_steps"][0][0])
            _decomp_cache[r] = [
                loadmat(f"{_DISK_DECOMP_DIR}/disk_decomp_r{r}_step{k}.mat")["nhood_k"].astype(bool)
                for k in range(1, n_steps + 1)
            ]
        else:
            _decomp_cache[r] = disk_decomposition(r, n=4)
    return _decomp_cache[r]


def _line_h(length: int) -> np.ndarray:
    return np.ones((1, int(length)), dtype=bool)


# ---------------------------------------------------------------------------
# Physical-radius -> pixel-radius convenience for enlarge_mixed/enlarge_conv
# (see module docstring "PIXEL COUNTS, NOT PHYSICAL UNITS" section).
# ---------------------------------------------------------------------------

def available_enlarge_radii_px() -> list:
    """
    List the pixel radii that have a bundled MATLAB-exported disk-mask
    fixture. NOTE: this is NO LONGER a limit on which radii _disk() can
    produce -- eccopy.core.disk generates any radius on demand (see module
    docstring "DISK STRUCTURING ELEMENTS"). This function now just reports
    which radii have exact-ground-truth .mat fixtures bundled (useful for
    tests / provenance), queried live from the data directory so it can't
    drift out of sync with whatever .mat files ship with a given install.

    Returns
    -------
    list of int, sorted ascending. Empty if the data directory is
    missing or contains no recognisable disk_strel_r{R}.mat files.
    """
    import re
    d = Path(_DISK_STREL_DIR)
    if not d.exists():
        return []
    radii = []
    for f in d.glob("disk_strel_r*.mat"):
        m = re.match(r"disk_strel_r(\d+)\.mat$", f.name)
        if m:
            radii.append(int(m.group(1)))
    return sorted(radii)


def resolve_enlarge_radius_px(target_km: float,
                              representative_spacing_km: float) -> int:
    """
    Convert a target PHYSICAL enlarge radius (km) into a pixel radius --
    a one-shot, UNIFORM-GRID-ONLY convenience. See module docstring
    "PIXEL COUNTS, NOT PHYSICAL UNITS" for why this does NOT have a
    genuinely spacing-aware ("varying", in refl_texture_1d's kernel_mode
    sense) counterpart: that would be a modelling decision, not a shape
    lookup.

    Since eccopy.core.disk generates the structuring element at any
    radius, this now resolves to the EXACT rounded pixel radius --
    round(target_km / representative_spacing_km) -- rather than snapping
    to a small set of pre-exported masks. No approximation, no warning.

    Parameters
    ----------
    target_km : float
        Desired physical enlarge radius, km.
    representative_spacing_km : float
        ONE spacing value for the whole array (e.g.
        `float(np.nanmedian(spacing_km_array))`), matching the "uniform"
        convention already used elsewhere in EccoPy (see
        refl_texture_1d's kernel_mode). On a genuinely non-uniform grid,
        the returned pixel radius represents a DIFFERENT physical size at
        points whose true local spacing differs from this representative
        value -- the same caveat class_basic()'s pixel-based radii always
        carried, just made explicit here instead of silent.
    Returns
    -------
    int
        round(target_km / representative_spacing_km), clamped to a minimum
        of 1.

    Raises
    ------
    ValueError
        If representative_spacing_km is not a positive, finite number.
    """
    if not np.isfinite(representative_spacing_km) or representative_spacing_km <= 0:
        raise ValueError(
            f"representative_spacing_km must be a positive, finite number; "
            f"got {representative_spacing_km!r}."
        )
    return max(1, int(round(target_km / representative_spacing_km)))


def _edge_pad_close(arr: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Single closing step matching MATLAB's real imclose mechanism,
    found by direct comparison against MATLAB ground truth on real SPOL
    data: edge-replicate the INPUT (not the already-dilated array)
    before dilating, then standard dilate+erode with border_value=0,
    then crop back to the original shape.

    Do not substitute scipy.ndimage.binary_closing(): it applies one
    global border_value across a single dilate-erode pair, which diverges
    from MATLAB by thousands of pixels on real cases.
    """
    h0 = se.shape[0] // 2 + 1
    h1 = se.shape[1] // 2 + 1
    padded = np.pad(arr, ((h0, h0), (h1, h1)), mode='edge')
    d = binary_dilation(padded, structure=se, border_value=0)
    e = binary_erosion(d, structure=se, border_value=0)
    return e[h0:-h0, h1:-h1]


def _sequential_close(arr: np.ndarray, radius: int) -> np.ndarray:
    """Closing by a large disk, applied as MATLAB really does it: through
    the disk's real decomposition primitives, one at a time, each via
    _edge_pad_close(). This is NOT mathematically equivalent to a single
    dilate-then-erode with the full disk mask near array boundaries --
    that equivalence only holds in the interior, far from any edge. See
    module docstring for validated accuracy."""
    steps = _load_decomp(radius)
    current = arr
    for s in steps:
        current = _edge_pad_close(current, s)
    return current


# ---------------------------------------------------------------------------
# Basic classification (EccoPy-1D / EccoPy-2D)
# ---------------------------------------------------------------------------

def class_basic_isotropic(conv: np.ndarray,
                          strat_mixed: float,
                          mixed_conv: float,
                          enlarge_mixed: int = 0,
                          enlarge_conv: int = 0) -> np.ndarray:
    """
    Basic convective / mixed / stratiform classification for plan-view
    (horizontal) data. Retained for backward compatibility and as a
    simpler alternative to the clumping-based path now used by default
    in eccopy2d_h.run() (see eccopy2d_h/clumping.py) -- this function is
    a straight morphological threshold-and-clean approach with no
    connectivity/area accounting, analogous to class_basic() but with
    isotropic (disk-only) structuring since both axes are spatial here.

    Carries the same two morphological conventions as class_basic():
    border_value=1 on erosion, and MATLAB's exact disk shape. Both are
    properties of the operation rather than of any one dataset.
    Uses the same disk decomposition as class_basic()
    (radius = enlarge*3 / enlarge*5), generated by eccopy.core.disk.

    Parameters
    ----------
    conv : np.ndarray, shape (Y, X)
        Convectivity field.
    strat_mixed, mixed_conv : float
        Convectivity thresholds.
    enlarge_mixed, enlarge_conv : int
        Disk radii for morphological cleanup. 0 disables cleanup beyond
        basic thresholding (still removes single-pixel speckle via a
        minimal disk-3 close+erode pass).

    Returns
    -------
    result : 1=stratiform, 2=mixed, 3=convective, NaN=no data
    """
    conv = np.asarray(conv, dtype=float).copy()
    result = np.full(conv.shape, np.nan)

    mask_mixed = conv >= strat_mixed
    if not mask_mixed.any():
        result[~np.isnan(conv)] = 1
        return result

    mixed_large = binary_dilation(mask_mixed, structure=_disk(max(enlarge_mixed, 1)))
    mixed_large = _sequential_close(mixed_large, max(enlarge_mixed, 1) * 3)
    mixed_large[np.isnan(conv)] = 0
    mixed_large = binary_fill_holes(mixed_large)
    # border_value=1 (not scipy's default 0) -- see docstring above.
    mixed_large_e = binary_erosion(mixed_large, structure=_disk(3), border_value=1)
    if not mixed_large_e.any() and mixed_large.any():
        mixed_large_e = mixed_large
    mixed_large = binary_dilation(mixed_large_e, structure=_disk(3))

    mask_conv = conv >= mixed_conv
    conv_large = binary_dilation(mask_conv, structure=_disk(max(enlarge_conv, 1)))
    conv_large = _sequential_close(conv_large, max(enlarge_conv, 1) * 5)
    conv_large[np.isnan(conv)] = 0
    conv_large = binary_fill_holes(conv_large)
    conv_large_e = binary_erosion(conv_large, structure=_disk(3), border_value=1)
    if not conv_large_e.any() and conv_large.any():
        conv_large_e = conv_large
    conv_large = binary_dilation(conv_large_e, structure=_disk(3))

    result[conv_large] = 3
    result[np.isnan(result) & mixed_large] = 2
    result[np.isnan(result)] = 1
    result[np.isnan(conv)] = np.nan
    return result


def class_basic(conv: np.ndarray,
                strat_mixed: float,
                mixed_conv: float,
                melt: Optional[np.ndarray] = None,
                enlarge_mixed: int = 0,
                enlarge_conv: int = 0) -> np.ndarray:
    """
    Basic convective / mixed / stratiform classification.
    Port of MATLAB f_classBasic.m -- see module docstring for the
    validation history of this specific function's morphological steps.

    Parameters
    ----------
    conv : np.ndarray
        Convectivity field, any shape with at least 2 dimensions.
    strat_mixed, mixed_conv : float
        Convectivity thresholds.
    melt : np.ndarray, optional
        Melting-layer height/flag field, same shape as conv. If provided,
        enables the "rain below melting layer" override check.
        Thresholded at `meltArea < 20` with a sentinel of 10,
        matching f_classBasic.m.
    Returns
    -------
    result : 1=stratiform, 2=mixed, 3=convective, NaN=no data
    """
    conv = conv.astype(float).copy()
    result = np.full(conv.shape, np.nan)

    mask_mixed_orig = conv >= strat_mixed
    if not mask_mixed_orig.any():
        result[~np.isnan(conv)] = 1
        return result

    mask_mixed = mask_mixed_orig.copy()
    conv[(mask_mixed == 0) & mask_mixed_orig] = 0

    # Rain-below-melt check.
    # MATLAB's `checkCol(1:firstInd)=1` is inclusive of firstInd, so
    # the Python slice must be `check_col[:first_ind + 1]`. Excluding
    # it drops the whole column's contribution whenever the
    # convectivity value at the melt-crossing pixel is NaN.
    if melt is not None:
        melt = melt.astype(float).copy()
        labeled_mixed, n_mixed = label(mask_mixed, structure=_CONN8)
        for ii in range(1, n_mixed + 1):
            pix_flat = np.where((labeled_mixed == ii).ravel())[0]
            melt_area = melt.ravel()[pix_flat]
            below_frac = np.sum(melt_area < 20) / max(len(pix_flat), 1)
            if below_frac <= 0.8:
                continue

            rows, cols = np.unravel_index(pix_flat, conv.shape)
            ucols = np.unique(cols)
            conv_cols = conv[:, ucols]
            melt_cols = melt[:, ucols]
            this_mat = np.zeros(conv.shape)
            this_mat.ravel()[pix_flat] = 1
            this_cols = this_mat[:, ucols]

            conv_cols = conv_cols[::-1, :]
            melt_cols = melt_cols[::-1, :]
            this_cols = this_cols[::-1, :]

            check_cols = np.full(melt_cols.shape, np.nan)
            for jj in range(len(ucols)):
                melt_col = melt_cols[:, jj].copy()
                check_col = conv_cols[:, jj].copy()
                this_col = this_cols[:, jj]
                first_valid = np.where(~np.isnan(melt_col))[0]
                if len(first_valid) == 0:
                    continue
                melt_col[: first_valid[0] + 1] = 10
                above_melt = np.where(melt_col >= 20)[0]
                if len(above_melt) == 0:
                    continue
                first_ind = above_melt[0]
                check_col[:first_ind + 1] = 1
                nan_inds = np.where(np.isnan(check_col))[0]
                last_ind = nan_inds[0] - 1 if len(nan_inds) > 0 else len(melt_col) - 1
                last_mixed = np.where(this_col == 1)[0]
                if len(last_mixed) == 0:
                    continue
                last_mixed_idx = last_mixed[-1]
                test_conv = conv_cols[:, jj].copy()
                test_conv[:last_mixed_idx + 1] = 1
                first_nan2 = np.where(np.isnan(test_conv))[0]
                if len(first_nan2) > 0 and last_ind > first_nan2[0]:
                    last_ind = max(first_ind, first_nan2[0])
                check_cols[first_ind: last_ind + 1, jj] = (
                    conv_cols[first_ind: last_ind + 1, jj]
                )

            n_valid_check = np.sum(~np.isnan(check_cols))
            strat_perc = np.sum(check_cols < strat_mixed) / max(n_valid_check, 1)
            med_thick = np.median(np.sum(~np.isnan(check_cols), axis=0))
            if strat_perc > 0.8 and med_thick > 5:
                mask_mixed.ravel()[pix_flat] = 0
                conv.ravel()[pix_flat] = 0

    hor_large = binary_dilation(mask_mixed, structure=_line_h(100))

    # --- Mixed region: enlarge, close, fill, erode ---
    mixed_large1 = binary_dilation(mask_mixed, structure=_disk(enlarge_mixed))
    mixed_large = _sequential_close(mixed_large1, enlarge_mixed * 3)
    if not mixed_large.any() and mixed_large1.any():
        mixed_large = mixed_large1
    mixed_large[np.isnan(conv)] = 0
    mixed_large = binary_fill_holes(mixed_large)
    # border_value=1 here (not scipy's default 0) is required for this
    # erosion to be mathematically extensive relative to its input --
    # see module docstring / border_value fix history.
    mixed_large_e = binary_erosion(mixed_large, structure=_disk(3), border_value=1)
    if not mixed_large_e.any() and mixed_large.any():
        mixed_large_e = mixed_large
    mixed_large = mixed_large_e

    for col_idx in np.where(mixed_large.any(axis=0))[0]:
        col = mixed_large[:, col_idx].copy()
        hor_col = hor_large[:, col_idx]
        lbl_col, n_pieces = label(col)
        if n_pieces > 1:
            for jj in range(1, n_pieces + 1):
                piece = lbl_col == jj
                if not hor_col[piece].any():
                    col[piece] = 0
            mixed_large[:, col_idx] = col

    mixed_large = binary_dilation(mixed_large, structure=_disk(3))

    # --- Convective region: enlarge, close, fill, erode ---
    mask_conv = conv >= mixed_conv
    hor_large2 = binary_dilation(mask_conv, structure=_line_h(100))

    conv_large1 = binary_dilation(mask_conv, structure=_disk(enlarge_conv))
    conv_large = _sequential_close(conv_large1, enlarge_conv * 5)
    if not conv_large.any() and conv_large1.any():
        conv_large = conv_large1
    conv_large[np.isnan(conv)] = 0
    conv_large = binary_fill_holes(conv_large)
    conv_large_e = binary_erosion(conv_large, structure=_disk(3), border_value=1)
    if not conv_large_e.any() and conv_large.any():
        conv_large_e = conv_large
    conv_large = conv_large_e

    for col_idx in np.where(conv_large.any(axis=0))[0]:
        col = conv_large[:, col_idx].copy()
        hor_col = hor_large2[:, col_idx]
        lbl_col, n_pieces = label(col)
        if n_pieces > 1:
            for jj in range(1, n_pieces + 1):
                piece = lbl_col == jj
                if not hor_col[piece].any():
                    col[piece] = 0
            conv_large[:, col_idx] = col

    conv_large = binary_dilation(conv_large, structure=_disk(3))

    result[conv_large] = 3
    result[np.isnan(result) & mixed_large] = 2
    result[np.isnan(result)] = 1
    result[np.isnan(conv)] = np.nan
    return result


# ---------------------------------------------------------------------------
# 1-D minimum-length clump filter (EccoPy-1D)
# ---------------------------------------------------------------------------
#
# There is no MATLAB or C++ "ECCO-1D" reference to port this from -- 1-D
# time-series classification has no upstream ground truth. This is a new
# EccoPy-1D-only addition, designed (not ported) to address one specific,
# named risk: class_basic()'s morphological pipeline already suppresses
# sub-structuring-element speckle via its erode/dilate sequence, but a
# convective run that survives that pipeline can still be physically very
# brief -- e.g. a single erroneous high-convectivity sample sitting right
# at the edge of what enlarge_conv's disk width lets through. The 3-D path
# (eccopy3d/clumping.py) guards against the volumetric equivalent of this
# by rejecting any clump with volume_km3 < min_valid_volume_for_convective
# BEFORE it can be labeled Convective -- see _clump_category()'s first
# check, which demotes undersized clumps to MIXED rather than dropping
# them to Stratiform (an undersized-but-real convective signature is
# "uncertain", not "definitely not convective"). filter_short_convective_
# runs_1d() is the 1-D analogue of that same check: physical LENGTH
# (time or distance, matching whatever unit the caller's coords are in)
# stands in for volume, since a 1-D run has no other size to measure.
#
# Deliberately NOT included here: the 3-D path's two-stage dual-threshold
# SPLITTING logic (ClumpingDualThresh -- see eccopy3d/clumping.py module
# docstring), which exists to re-separate two genuine, distinct storm
# cores that got morphologically bridged into one clump. A 1-D analogue
# (secondary-threshold local-maxima detection + regrow) is possible but
# is a separate, nontrivial design decision -- deferred pending input on
# whether EccoPy-1D actually needs to distinguish "one brief burst" from
# "two adjacent bursts that got closed together" for the intended
# time-series use case, versus just filtering brief/erroneous ones out.


def filter_short_convective_runs_1d(echo_type: np.ndarray,
                                    spacing_base: np.ndarray,
                                    min_length_base: float,
                                    demote_to: int = CATEGORY_MIXED,
                                    target_code: int = 3) -> np.ndarray:
    """
    Demote contiguous convective runs shorter than `min_length_base` to
    `demote_to` (Mixed by default -- matching _clump_category()'s
    treatment of undersized 3-D clumps, not Stratiform).

    Parameters
    ----------
    echo_type : np.ndarray, shape (N,)
        Output of class_basic() (already squeezed back to 1-D by the
        eccopy1d caller): 1=stratiform, 2=mixed, 3=convective, NaN=missing.
    spacing_base : np.ndarray, shape (N,)
        Local point-to-point spacing, in the SAME base unit as
        `min_length_base` (both metres/both seconds/both raw pixel
        counts -- caller's responsibility to match them; see
        eccopy1d.run()'s handling of WindowSpec base units for the
        pattern used to guarantee this).
    min_length_base : float
        Minimum physical (or pixel-count) length a convective run must
        span to remain Convective. A run's length is the sum of
        spacing_base over every point in the run -- see Notes.
    demote_to : int
        Echo code to assign to undersized runs. Default CATEGORY_MIXED.
    target_code : int
        Echo code identifying which runs to check. Default 3
        (Convective) -- the only code this is meaningful for currently,
        but left as a parameter rather than hardcoded.

    Returns
    -------
    result : np.ndarray, shape (N,)
        Copy of echo_type with undersized target_code runs demoted.

    Notes
    -----
    A run's length is computed as the sum of spacing_base over the run's
    own points, which slightly OVER-counts the run's true footprint (the
    last point's spacing value describes the gap to the point AFTER the
    run, not a gap the run itself occupies) -- this is a deliberate,
    conservative choice: it means a run is very slightly more likely to
    survive the filter than a stricter definition would allow, rather
    than less likely, which matters more given False Elimination of a
    real convective signature is a worse failure than the reverse.
    """
    result = np.array(echo_type, dtype=float, copy=True)
    if min_length_base is None or min_length_base <= 0:
        return result

    mask = (result == target_code)
    if not mask.any():
        return result

    labeled, n_runs = label(mask)  # default structure: 1-D adjacency
    spacing_base = np.asarray(spacing_base, dtype=float)

    for ii in range(1, n_runs + 1):
        run = labeled == ii
        run_length = float(np.nansum(spacing_base[run]))
        if run_length < min_length_base:
            result[run] = demote_to

    return result


# ---------------------------------------------------------------------------
# 2-D sub-classification (EccoPy-1D / EccoPy-2D)
# ---------------------------------------------------------------------------

def class_sub_2d(class_in: np.ndarray,
                  height: np.ndarray,
                  topo,
                  melt: np.ndarray,
                  temp: np.ndarray,
                  elev: Optional[np.ndarray] = None,
                  first_row: Optional[int] = None,
                  surf_alt_lim: float = 0.0) -> np.ndarray:
    """
    Sub-classification into echo type codes. Port of f_classSub.m.

    Non-uniform grids: unlike class_basic()
    /class_basic_isotropic(), this function has NO pixel-radius structuring
    elements and never touches a spacing array at all -- every threshold
    here (melt=15, temp=-25 C, surf_alt_lim in metres) is compared
    directly against physical field VALUES at each point independently.
    So class_sub_2d() is already fully non-uniform-grid-safe; the "pixel
    counts, not physical units" caveat documented at the top of this
    module applies only to the basic-classification morphology, not to
    sub-classification.

    *** API CHANGE from the previous class_sub_2d ***: `height` and
    `temp` are NOT alternatives with a fallback between them. The real
    algorithm requires `height` (for AGL / near-surface tests, and to
    force-correct `melt`/`temp` near the surface) together with BOTH
    `melt` and `temp`. `melt` is the primary signal for shallow/low
    classification; `temp` only distinguishes mid from deep/high. All
    three are required together; f_classSub.m has no path that takes a
    subset.

    Parameters
    ----------
    class_in : np.ndarray, shape (Z, X)
        Output of class_basic(): 1=stratiform, 2=mixed, 3=convective.
    height : np.ndarray, shape (Z, X)
        Height field, METRES (MATLAB's `asl`). Note: this differs from
        this package's usual km convention at the public API boundary --
        follow the existing eccopy2d_v.run() pattern of converting
        km->m once at the boundary before calling this function.
    topo : np.ndarray or scalar, broadcastable to (Z, X)
        Terrain height / beam-height correction, METRES.
    melt : np.ndarray, shape (Z, X)
        Melting-layer field (same convention as class_basic's `melt`
        parameter -- NOT optional here, unlike in the previous version
        of this function).
    temp : np.ndarray, shape (Z, X)
        Temperature field, deg C.
    elev, first_row : optional
        Near-aircraft override logic (rarely needed) -- see f_classSub.m.
    surf_alt_lim : float, metres

    Returns
    -------
    result : np.ndarray, shape (Z, X)
        Echo type codes: 14/16/18 (stratiform low/mid/high), 25 (mixed),
        30 (near-aircraft override), 32/34/36/38 (convective
        elevated/shallow/mid/deep).
    """
    class_in = np.asarray(class_in, dtype=float)
    height = np.asarray(height, dtype=float)
    melt_orig = np.asarray(melt, dtype=float)
    temp = np.asarray(temp, dtype=float).copy()  # mutated below; don't touch caller's array

    topo_arr = np.asarray(topo, dtype=float)
    if topo_arr.shape != height.shape:
        topo_arr = np.broadcast_to(topo_arr, height.shape)

    result = np.full(class_in.shape, np.nan)

    conv_mask = (class_in == 3)
    labeled_conv, n_conv = label(conv_mask, structure=_CONN8)

    dist_asl_topo = height - topo_arr

    melt = melt_orig.copy()
    melt[dist_asl_topo < 2000] = 9
    melt[np.isnan(melt_orig)] = np.nan
    temp[(dist_asl_topo < 4000) & (temp < -25)] = -25

    for ii in range(1, n_conv + 1):
        pix_flat = np.where((labeled_conv == ii).ravel())[0]
        rows, cols = np.unravel_index(pix_flat, class_in.shape)

        if elev is not None and first_row is not None:
            plane_mask = rows == first_row
            plane_pix = int(np.sum(plane_mask))
            alt_diff = dist_asl_topo.ravel()[pix_flat]
            alt_plane_pix = alt_diff[plane_mask]
            cols_first = cols[plane_mask]
            elev_plane_pix = elev[cols_first]
            alt_low = int(np.sum((alt_plane_pix < 500) & (elev_plane_pix > 0)))
            plane_pix = plane_pix - alt_low
        else:
            plane_pix = 0

        asl_area = dist_asl_topo.ravel()[pix_flat]
        near_surf_pix = int(np.sum(asl_area < 500 + surf_alt_lim))

        if near_surf_pix == 0:
            if elev is not None and plane_pix > 10 and np.nanmedian(elev[cols]) > 0:
                code = CATEGORY_CONVECTIVE
            else:
                code = CATEGORY_CONVECTIVE_ELEVATED
        else:
            melt_max = np.nanmax(melt.ravel()[pix_flat])
            if melt_max < 15:
                code = CATEGORY_CONVECTIVE if plane_pix > 10 else CATEGORY_CONVECTIVE_SHALLOW
            else:
                min_temp = np.nanmin(temp.ravel()[pix_flat])
                if min_temp >= -25:
                    code = CATEGORY_CONVECTIVE if plane_pix > 10 else CATEGORY_CONVECTIVE_MID
                else:
                    if elev is not None and plane_pix > 10 and np.nanmedian(elev[cols]) > 0:
                        code = CATEGORY_CONVECTIVE
                    else:
                        code = CATEGORY_CONVECTIVE_DEEP
        result.ravel()[pix_flat] = code

    result[(class_in == 1) & (melt < 15)] = CATEGORY_STRATIFORM_LOW
    result[(class_in == 1) & (melt > 15) & (temp >= -25)] = CATEGORY_STRATIFORM_MID
    result[(class_in == 1) & (melt > 15) & (temp < -25)] = CATEGORY_STRATIFORM_HIGH

    result[class_in == 2] = CATEGORY_MIXED

    return result


def assign_echo_type_2d(convectivity: np.ndarray,
                        clumps: list,
                        max_conv_for_strat: float) -> np.ndarray:
    """
    Assign basic (1/2/3) echo type codes for EccoPy-2D-H from a list of
    2-D clumps (see eccopy2d_h/clumping.py:find_clumps_2d()).

    2-D analogue of set_echo_type_3d()'s two-pass structure, simplified:
    there is no height/temp axis for a single horizontal level, so there
    is no Pass-1 shallow/mid/deep/elevated sub-typing -- every clump
    pixel is simply CATEGORY code 3 (Convective), and remaining pixels
    follow the same convectivity-threshold Pass-2 logic as class_basic /
    class_basic_isotropic (which is why this returns plain 1/2/3 codes,
    not the extended 14/16/18/25/32/34/36/38 set that set_echo_type_3d
    can produce -- matching EccoPy-2D-H's documented "no sub-classification
    without depth" contract).

    Parameters
    ----------
    convectivity : np.ndarray, shape (Y, X)
    clumps : list of dicts, each with an 'index' key -- a (iy_arr, ix_arr)
        tuple of integer index arrays (see find_clumps_2d()).
    max_conv_for_strat : float
        Convectivity threshold above which a non-clump point is Mixed
        rather than Stratiform (same role as class_basic's strat_mixed).

    Returns
    -------
    echo_type : np.ndarray, shape (Y, X)
        1=Stratiform, 2=Mixed, 3=Convective, NaN=missing/no data.
    """
    result = np.full(convectivity.shape, np.nan)

    # Pass 1 - clump pixels are Convective, unconditionally (any clump
    # that reached this function already passed find_clumps_2d()'s own
    # area filtering -- see that module for the filtering logic).
    for clump in clumps:
        result[clump['index']] = 3

    # Pass 2 - stratiform/mixed for everything else with valid convectivity
    unassigned = np.isnan(result)
    has_conv = unassigned & ~np.isnan(convectivity)
    mixed_mask = has_conv & (convectivity > max_conv_for_strat)
    result[mixed_mask] = 2
    result[has_conv & ~mixed_mask] = 1

    return result


# ---------------------------------------------------------------------------
# 3-D echo type assignment (EccoPy-3D)
# Port of ConvStratFinder::_setEchoType3D() + StormClump::setEchoType()
# ---------------------------------------------------------------------------

def set_echo_type_3d(convectivity: np.ndarray,
                     clumps: list,
                     height_km: Optional[np.ndarray] = None,
                     temp: Optional[np.ndarray] = None,
                     shallow_threshold_ht: float = 4.5,
                     deep_threshold_ht: float = 9.0,
                     shallow_threshold_temp: float = 0.0,
                     deep_threshold_temp: float = -12.0,
                     terrain_ht_km: Optional[np.ndarray] = None,
                     min_ht_agl_for_mid: float = 2.0,
                     min_ht_agl_for_deep: float = 4.0,
                     max_conv_for_strat: float = 0.4,
                     min_conv_for_conv: float = 0.5,
                     min_vol_km3: float = 20.0,
                     min_vert_extent_km: float = 1.0,
                     min_conv_frac_deep: float = 0.05,
                     min_conv_frac_shallow: float = 0.95,
                     max_shallow_frac_elevated: float = 0.05,
                     max_deep_frac_elevated: float = 0.25,
                     min_strat_frac_strat_below: float = 0.9) -> np.ndarray:
    """
    Assign 3-D echo type codes.

    *** WARNING: this function has NOT been re-validated against this
    session's findings (border_value fix, exact disk masks, sequential
    closing mechanism). It still uses height-threshold logic for
    shallow/mid/deep, which does not apply to the 2-D case
    (class_sub_2d) -- the real algorithm there uses melt/temp instead.
    Whether the 3-D C++ reference (ConvStratFinder) genuinely differs
    from the 2-D MATLAB reference (f_classSub.m) in this respect, or
    whether this function has the same undiscovered bug, has not been
    checked. Do not treat this function as validated. ***

    Exact port of ConvStratFinder::_setEchoType3D() two-pass structure:

    Pass 1: For each clump, call _clump_category() - sets convective pixels.
    Pass 2: Loop all remaining pixels:
              - conv missing or 0 → skip (stays MISSING)
              - conv > max_conv_for_strat → MIXED
              - conv <= max_conv_for_strat → low/mid/high stratiform code,
                based on EITHER height_km OR temp (whichever is supplied)

    If both height_km and temp are None, every non-clump, non-mixed point
    with valid convectivity is simply MIXED-or-missing - pass a basic
    classification mode (CATEGORY_MIXED only, no low/mid/high distinction)
    by leaving both arguments unset; clump pixels are still differentiated
    into shallow/mid/deep/elevated.

    Parameters
    ----------
    convectivity : np.ndarray, shape (nz, ny, nx)
    clumps : list of dicts (see find_clumps_3d / _clump_category)
    height_km : np.ndarray, shape (nz, ny, nx), optional
        Height field, km MSL.
    temp : np.ndarray, shape (nz, ny, nx), optional
        Temperature field, °C.
    shallow_threshold_ht, deep_threshold_ht : float, km
    shallow_threshold_temp, deep_threshold_temp : float, °C
    terrain_ht_km : np.ndarray, shape (ny, nx), optional
    min_ht_agl_for_mid, min_ht_agl_for_deep : float, km
    max_conv_for_strat, min_conv_for_conv, min_vol_km3,
    min_vert_extent_km, min_conv_frac_deep, min_conv_frac_shallow,
    max_shallow_frac_elevated, max_deep_frac_elevated,
    min_strat_frac_strat_below : float

    Returns
    -------
    echo_type : np.ndarray, shape (nz, ny, nx), int16
    """
    nz, ny, nx = convectivity.shape
    echo_type = np.zeros((nz, ny, nx), dtype=np.int16)

    use_height = height_km is not None
    use_temp = (not use_height) and (temp is not None)

    if use_height:
        height_km = np.asarray(height_km, dtype=float)
        shallow_bnd = np.full((ny, nx), shallow_threshold_ht)
        deep_bnd = np.full((ny, nx), deep_threshold_ht)
        if terrain_ht_km is not None:
            terrain_ht_km = np.asarray(terrain_ht_km, dtype=float)
            shallow_bnd = np.maximum(shallow_bnd, terrain_ht_km + min_ht_agl_for_mid)
            deep_bnd = np.maximum(deep_bnd, terrain_ht_km + min_ht_agl_for_deep)
    elif use_temp:
        temp = np.asarray(temp, dtype=float)

    # Pass 1 - assign convective pixels via clumps
    for clump in clumps:
        category = _clump_category(
            clump, min_vol_km3, min_vert_extent_km,
            min_conv_frac_deep, min_conv_frac_shallow,
            max_shallow_frac_elevated, max_deep_frac_elevated,
            min_strat_frac_strat_below,
            convectivity, min_conv_for_conv,
        )
        echo_type[clump['index']] = category

    # Pass 2 - stratiform/mixed for remaining points
    unassigned = (echo_type == CATEGORY_MISSING)
    has_conv = unassigned & ~np.isnan(convectivity) & (convectivity != 0)

    mixed_mask = has_conv & (convectivity > max_conv_for_strat)
    echo_type[mixed_mask] = CATEGORY_MIXED

    strat_mask = has_conv & ~mixed_mask
    if use_height:
        ht_broadcast = height_km
        sh_b = shallow_bnd[np.newaxis, :, :]
        dp_b = deep_bnd[np.newaxis, :, :]
        echo_type[strat_mask & (ht_broadcast <= sh_b)] = CATEGORY_STRATIFORM_LOW
        echo_type[strat_mask & (ht_broadcast >= dp_b)] = CATEGORY_STRATIFORM_HIGH
        echo_type[strat_mask & (ht_broadcast > sh_b) & (ht_broadcast < dp_b)] = CATEGORY_STRATIFORM_MID
    elif use_temp:
        echo_type[strat_mask & (temp >= shallow_threshold_temp)] = CATEGORY_STRATIFORM_LOW
        echo_type[strat_mask & (temp < shallow_threshold_temp)
                  & (temp >= deep_threshold_temp)] = CATEGORY_STRATIFORM_MID
        echo_type[strat_mask & (temp < deep_threshold_temp)] = CATEGORY_STRATIFORM_HIGH
    else:
        echo_type[strat_mask] = CATEGORY_MIXED

    return echo_type


def _clump_category(clump: dict,
                    min_vol_km3: float,
                    min_vert_extent_km: float,
                    min_conv_frac_deep: float,
                    min_conv_frac_shallow: float,
                    max_shallow_frac_elevated: float,
                    max_deep_frac_elevated: float,
                    min_strat_frac_strat_below: float,
                    convectivity: np.ndarray,
                    min_conv_for_conv: float) -> int:
    """
    Determine echo type category for one clump.
    Port of ConvStratFinder::StormClump::setEchoType().

    Decision tree (matching C++ exactly):
      1. vol < min_vol_km3 OR vert_extent < min_vert_extent → MIXED
      2. fracShallow < max_shallow_frac_elevated
         AND stratiformBelow()
           → if fracDeep < max_deep_frac_elevated → ELEVATED
           → else → MIXED
      3. fracShallow > min_conv_frac_shallow → SHALLOW
      4. fracDeep > min_conv_frac_deep → DEEP
      5. else → MID
    """
    if (clump['volume_km3'] < min_vol_km3 or
            clump['vert_extent_km'] < min_vert_extent_km):
        return CATEGORY_MIXED

    n_total = max(clump['n_pts_total'], 1)
    frac_shallow = clump['n_pts_shallow'] / n_total
    frac_deep    = clump['n_pts_deep']    / n_total

    if frac_shallow < max_shallow_frac_elevated:
        if _strat_below(clump['index'], convectivity, min_conv_for_conv,
                        min_strat_frac_strat_below):
            if frac_deep < max_deep_frac_elevated:
                return CATEGORY_CONVECTIVE_ELEVATED
            else:
                return CATEGORY_MIXED

    if frac_shallow > min_conv_frac_shallow:
        return CATEGORY_CONVECTIVE_SHALLOW
    if frac_deep > min_conv_frac_deep:
        return CATEGORY_CONVECTIVE_DEEP
    return CATEGORY_CONVECTIVE_MID


def _strat_below(index: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 convectivity: np.ndarray,
                 min_conv_for_conv: float,
                 min_strat_frac: float) -> bool:
    """
    Check for stratiform echo in the plane IMMEDIATELY below each clump pixel.

    Port of ConvStratFinder::StormClump::stratiformBelow():
      - If any clump pixel is at iz==0, return False immediately.
      - For each (iz, iy, ix) in clump, look at convectivity[iz-1, iy, ix].
      - missing → nMiss++
      - < min_conv_for_conv → nStrat++
      - fractionStrat = nStrat / (nMiss + nStrat) > min_strat_frac → True

    Parameters
    ----------
    index : tuple of (iz_arr, iy_arr, ix_arr) - matches clump['index']
        from find_clumps_3d(), shape (Z, Y, X) convention.
    convectivity : np.ndarray, shape (Z, Y, X)

    `index` is a (Z, Y, X)-ordered tuple from np.where(); the vertical
    step decrements iz_arr, the first axis.
    """
    iz_arr, iy_arr, ix_arr = index
    if len(iz_arr) == 0:
        return False
    if iz_arr.min() == 0:
        return False

    below = convectivity[iz_arr - 1, iy_arr, ix_arr]
    n_miss = int(np.sum(np.isnan(below)))
    n_strat = int(np.sum(below < min_conv_for_conv))  # NaN-safe: NaN < x is False

    n_total = n_miss + n_strat
    if n_total == 0:
        return False
    return (n_strat / n_total) > min_strat_frac