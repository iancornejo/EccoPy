"""
Basic and sub-classification functions.

class_basic()       — port of f_classBasic.m  (EccoPy-1D / EccoPy-2D)
class_sub_2d()      — port of f_classSub.m    (EccoPy-1D / EccoPy-2D)
set_echo_type_3d()  — port of ConvStratFinder::_setEchoType3D() + StormClump

Echo type codes (matching MATLAB f_classSub.m and C++ enum values)
-------------------------------------------------------------------
14  CATEGORY_STRATIFORM_LOW
16  CATEGORY_STRATIFORM_MID
18  CATEGORY_STRATIFORM_HIGH
25  CATEGORY_MIXED
30  CATEGORY_CONVECTIVE (near-aircraft override -- rarely used)
32  CATEGORY_CONVECTIVE_ELEVATED
34  CATEGORY_CONVECTIVE_SHALLOW
36  CATEGORY_CONVECTIVE_MID
38  CATEGORY_CONVECTIVE_DEEP
 0  CATEGORY_MISSING

===============================================================================
VALIDATION HISTORY / KNOWN-GOOD STATUS (read before touching this file)
===============================================================================
This file's morphological logic (class_basic) was extensively debugged
against real ECCO-V MATLAB reference output across three independent
cases (SEA RHI, SPOL RHI). Summary of what's confirmed and what isn't:

CONFIRMED (mathematically proven or bit-exact matched against real
MATLAB intermediate arrays -- not just "looks close"):
  - binary_dilation with border_value=0 exactly reproduces MATLAB's
    imdilate at every radius tested (3, 5, 15, 25), including right up
    to array boundaries. This is NOT in question.
  - The disk masks in _disk() are loaded from MATLAB's own exported
    strel('disk', r).Neighborhood arrays -- NOT a Euclidean-circle
    approximation. A literal circle approximation (x^2+y^2<=r^2) was
    tried first and found to systematically over-reach MATLAB's real
    disk shape at every radius (MATLAB's actual disk has a smaller
    effective bounding radius than the nominal parameter -- e.g.
    strel('disk',25) has a 49x49, not 51x51, bounding box).
  - class_basic's morphological closing steps use _sequential_close(),
    which applies MATLAB's REAL decomposition primitives (exported via
    getsequence()) one at a time, each via "edge-pad the input, dilate,
    erode with border_value=0, crop" -- see _edge_pad_close(). This was
    found by direct comparison against MATLAB's own intermediate arrays,
    not guessed. It is bit-exact for simple axis-aligned line primitives
    and ~97-98% exact-pixel-match for the diagonal/2D primitives in the
    decomposition -- there IS a small remaining residual, not yet fully
    explained, but validated end-to-end results are excellent:
        SEA Test1: 99.7% basic-classification agreement
        SEA Test6: 100.0% (exact)
        SPOL:      99.4%
    (All measured against real MATLAB ECHOTYPE/CONVECTIVITY output,
    collapsed to basic strat/mixed/conv categories.)
  - class_sub_2d below is a corrected port: the ORIGINAL version of this
    function (now removed) used height-threshold logic (4.5/9.0 km AGL)
    that was NEVER checked against the real f_classSub.m and turned out
    to test a different physical quantity entirely. The real algorithm
    uses `melt` (primary signal, threshold 15) and `temp` (threshold
    -25 C, only distinguishes mid vs deep/high) -- see f_classSub.m.
    This was re-verified line-by-line against the actual MATLAB source
    and validated: 100% precision on stratiform sub-classification
    (14/16/18) against real SPOL ground truth, zero cross-contamination.

NOT YET DONE / OPEN ITEMS:
  - class_basic_isotropic() (used by EccoPy-2D-H) has NOT been checked
    for either the border_value issue or the disk-shape issue. It still
    uses scipy's binary_closing() with default border_value and _disk()
    from this file (now the exact-mask version, at least) but the
    closing steps have not been converted to _sequential_close(). Do not
    assume EccoPy-2D-H is validated just because EccoPy-2D-V is.
  - set_echo_type_3d() / _clump_category() / _strat_below() (the 3-D
    path) have NOT been re-checked against this session's findings.
    _strat_below() had its own, separately-fixed axis-order bug
    (documented in its docstring) but the morphological-closing findings
    here have not been ported to any 3-D closing logic if one exists
    elsewhere in the package.
  - The small residual in _sequential_close() for diagonal primitives is
    real but unexplained. Do not assume it is zero for radii other than
    those tested (3, 5, 15, 25).
  - The melt-correction ("rain below melting layer") block in
    class_basic: FIXED via direct comparison against the real
    f_classBasic.m source (lrose-ecco repo). Two things were wrong: (1)
    an earlier threshold change (20/10 -> 15/9) was reverted -- the real
    MATLAB source uses 20/10, matching the ORIGINAL pre-session Python
    constants; (2) a genuine off-by-one bug in `check_col[:first_ind]`
    (should be `[:first_ind + 1]`, matching MATLAB's inclusive
    `checkCol(1:firstInd)=1`) was found and fixed -- a NaN exactly at
    the melt-crossing pixel could silently zero out a column's
    contribution. Tested against the real SPOL case (20220526_084500):
    neither fix changes that case's final output, since the one real
    clump reaching the check has strat_perc=0.66 (genuinely below
    threshold; confirmed via real MATLAB ECHOTYPE that it's 95.7%
    Convective Mid, i.e. correctly not reclassified) and none of its
    columns hit the off-by-one edge case. Both fixes are verified
    directly against MATLAB source, the strongest evidence this project
    uses -- see README "Validation status" for the full writeup.
  - 234 pixels of NaN-pattern mismatch were found between a from-scratch
    Python port of f_reflTexture.m and MATLAB's saved convectivity on
    the SPOL case, despite otherwise-perfect (1e-17 level) value
    agreement. Not investigated further -- flagged for whoever owns
    core/texture.py, since that file was not modified or reviewed here.

===============================================================================
PACKAGING REQUIREMENT -- READ BEFORE DEPLOYING
===============================================================================
_disk() and _sequential_close() depend on data files exported directly
from MATLAB (strel Neighborhood arrays and getsequence() decompositions)
-- they are NOT computed algorithmically in Python. An earlier attempt to
algorithmically reconstruct MATLAB's disk approximation from documented
general principles was tried and FAILED a basic sanity check (produced a
filled square, not a disk) -- do not re-attempt this without a way to
validate against real MATLAB-exported ground truth.

As of this revision, data-file paths are anchored to this module's own
location (Path(__file__).parent / "data" / ...), NOT the caller's
current working directory -- an earlier version used bare relative paths
("disk_strels", "disk_decomp") which broke with a real FileNotFoundError
the first time this was run from a notebook whose CWD didn't happen to
contain those folders. See the comment above _DISK_STREL_DIR /
_DISK_DECOMP_DIR for exactly where these files are expected relative to
this file, and adjust if the package's actual data-file layout differs.

Fixing the path resolution does NOT mean the data files are actually
bundled with the package yet -- that is a separate, still-open step.
The four disk_strel_r{R}.mat files and the two disk_decomp_r{R}_*.mat
sets validated this session are being delivered as a standalone zip
alongside this code change; they still need to be physically placed at
<package_dir>/core/data/disk_strels/ and <package_dir>/core/data/disk_decomp/
(or wherever the paths above are adjusted to point), and wired into the
package's build/install configuration (setup.py package_data /
MANIFEST.in / pyproject.toml, depending on build backend) so a real
`pip install` actually ships them -- a plain file-copy into a dev
checkout is enough to unblock local testing but not enough for
distribution.

Required data files (see export_disk_strels.m / export_disk_decomposition.m
used during this validation session):
  disk_strels/disk_strel_r{R}.mat       for R in {3, enlarge_mixed, enlarge_conv}
  disk_decomp/disk_decomp_r{R}_step{k}.mat  for R in {enlarge_mixed*3, enlarge_conv*5}
  disk_decomp/disk_decomp_r{R}_info.mat

enlarge_mixed and enlarge_conv are user-tunable parameters, both
defaulting to 5 -- confirmed directly against the MATLAB f_classBasic.m
source. (An earlier revision of ClassificationParams incorrectly
defaulted enlarge_conv to 3; that value was never the real MATLAB
default and had only ever been exercised via the enlarge_conv=5 real
validation runs -- fixed.) If a user sets values other than the
validated set {3, 5, 15, 25}, the corresponding .mat files must be
exported and bundled BEFORE those parameters can be used -- _disk()/
_load_decomp() will raise a clear FileNotFoundError rather than
silently falling back to an approximation, which is the failure mode
actually observed when these files were missing entirely.

===============================================================================
enlarge_mixed / enlarge_conv ARE PIXEL COUNTS, NOT PHYSICAL UNITS
===============================================================================
Unlike the texture window (WindowSpec, resolved in km/seconds against a
spacing array -- see core/texture.py's kernel_mode), enlarge_mixed and
enlarge_conv are literal pixel/gate radii, full stop. class_basic() and
class_basic_isotropic() never look at a spacing array at all -- _disk(5)
is a fixed 11x11 pixel mask regardless of what one pixel represents
physically. This traces straight back to f_classBasic.m: nothing in that
source suggests enlarge_mixed/enlarge_conv were ever meant to represent a
fixed physical distance -- they read as gate/pixel counts in the original
MATLAB, and this port preserves that literally, not as an oversight.

Practical consequence: on a NON-UNIFORM grid (e.g. an RHI whose range-gate
spacing changes with elevation angle, or any Z axis whose spacing varies
by row), the SAME enlarge_conv=5 represents a DIFFERENT physical cleanup
radius at different rows -- 5 gates is a different real distance wherever
gate spacing differs. This is not a bug and there is nothing to "fix" the
way the refl_texture_1d row-spacing bug was fixed earlier this session:
there is no known-correct physical-radius reference to port, no MATLAB
source suggesting one was intended, and -- separately -- no way to even
construct the required disk mask at an arbitrary radius in the first
place (see PACKAGING REQUIREMENT above: _disk() only has masks at 4 fixed
integer radii, all MATLAB-exported ground truth, none reconstructible
algorithmically). A genuinely spacing-aware morphological cleanup would
need a different validated disk shape at every distinct local spacing on
the grid, which nothing here can produce.

What IS provided (see resolve_enlarge_radius_px() below): a one-shot,
UNIFORM-GRID-ONLY convenience that converts a desired physical radius
(km) into the nearest AVAILABLE pixel radius, given one representative
spacing value for the whole array (its median, matching the "uniform"
convention already used elsewhere in EccoPy). This is explicitly not a
"varying" counterpart the way refl_texture_1d's kernel_mode has one --
there is no way to offer a spacing-aware version of this, for the reasons
above, so this function doesn't pretend to.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import (
    label, binary_dilation, binary_erosion,
    binary_closing, binary_fill_holes,
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
# NOTE: previously these were bare relative paths ("disk_strels",
# "disk_decomp"), which only resolved correctly if the CURRENT WORKING
# DIRECTORY happened to contain those folders -- i.e. it depended on
# where the *user's script/notebook* was run from, not on where this
# package is installed. That's fragile by construction (works by
# accident in a notebook sitting next to a disk_strels/ folder, breaks
# immediately anywhere else) and was the direct cause of a real
# FileNotFoundError in practice. Anchoring to __file__ makes this work
# regardless of the caller's CWD.
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
    """Exact MATLAB strel('disk', radius).Neighborhood, loaded from
    exported ground truth. NOT a Euclidean-circle approximation -- see
    module docstring for why that was tried and rejected."""
    r = int(radius)
    if r not in _disk_cache:
        try:
            d = loadmat(f"{_DISK_STREL_DIR}/disk_strel_r{r}.mat")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No exported MATLAB disk mask for radius {r}. Run "
                f"export_disk_strels.m with RADII including {r} and bundle "
                f"the resulting .mat file at {_DISK_STREL_DIR}/disk_strel_r{r}.mat."
            ) from e
        _disk_cache[r] = d["nhood"].astype(bool)
    return _disk_cache[r]


def _load_decomp(radius: int):
    """MATLAB's real getsequence(strel('disk', radius)) primitives,
    loaded from exported ground truth."""
    r = int(radius)
    if r not in _decomp_cache:
        try:
            info = loadmat(f"{_DISK_DECOMP_DIR}/disk_decomp_r{r}_info.mat")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No exported MATLAB disk decomposition for radius {r}. Run "
                f"export_disk_decomposition.m with RADII including {r} and "
                f"bundle the resulting .mat files under {_DISK_DECOMP_DIR}/."
            ) from e
        n_steps = int(info["n_steps"][0][0])
        steps = [loadmat(f"{_DISK_DECOMP_DIR}/disk_decomp_r{r}_step{k}.mat")["nhood_k"].astype(bool)
                 for k in range(1, n_steps + 1)]
        _decomp_cache[r] = steps
    return _decomp_cache[r]


def _line_h(length: int) -> np.ndarray:
    return np.ones((1, int(length)), dtype=bool)


# ---------------------------------------------------------------------------
# Physical-radius -> pixel-radius convenience for enlarge_mixed/enlarge_conv
# (see module docstring "PIXEL COUNTS, NOT PHYSICAL UNITS" section).
# ---------------------------------------------------------------------------

def available_enlarge_radii_px() -> list:
    """
    List the pixel radii that have a bundled, MATLAB-exported disk mask
    (see module docstring "PACKAGING REQUIREMENT") -- the only radii
    _disk() can produce without exporting new MATLAB data. Queried live
    from the data directory (not hardcoded) so this can't silently drift
    out of sync with whatever .mat files actually ship with a given
    install.

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
                              representative_spacing_km: float,
                              param_name: str = "enlarge radius") -> int:
    """
    Convert a target PHYSICAL enlarge radius (km) into the nearest pixel
    radius that has a bundled, validated MATLAB disk mask -- a one-shot,
    UNIFORM-GRID-ONLY convenience. See module docstring "PIXEL COUNTS,
    NOT PHYSICAL UNITS" for why this does NOT have, and cannot have, a
    genuinely spacing-aware ("varying", in refl_texture_1d's kernel_mode
    sense) counterpart: class_basic()'s structuring elements are fixed
    MATLAB-exported pixel masks at only 4 discrete radii, not something
    resolvable at an arbitrary point-by-point physical size the way a
    sliding texture window is.

    This function only answers "which bundled pixel radius is closest to
    my target physical radius, given ONE representative spacing value for
    the whole array" -- it does NOT check whether the resulting pixel
    radius, or its enlarge_mixed*3 / enlarge_conv*5 derived closing
    radius, actually has bundled DECOMPOSITION data too (see module
    docstring: enlarge_mixed is only fully usable at 5; enlarge_conv at 3
    or 5). class_basic() / class_basic_isotropic() will still raise their
    own clear FileNotFoundError downstream if you pick a radius whose
    decomposition wasn't exported -- this function can't and doesn't
    pre-empt that, since it only knows about disk_strels, not
    disk_decomp.

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
    param_name : str
        Only used to make the warning message readable (e.g. pass
        "enlarge_mixed" or "enlarge_conv").

    Returns
    -------
    int
        The bundled pixel radius closest to
        target_km / representative_spacing_km.

    Raises
    ------
    ValueError
        If representative_spacing_km is not a positive, finite number.
    FileNotFoundError
        If no disk masks are bundled at all (see available_enlarge_radii_px()).
    """
    if not np.isfinite(representative_spacing_km) or representative_spacing_km <= 0:
        raise ValueError(
            f"representative_spacing_km must be a positive, finite number; "
            f"got {representative_spacing_km!r}."
        )
    available = available_enlarge_radii_px()
    if not available:
        raise FileNotFoundError(
            f"No bundled MATLAB disk masks found under {_DISK_STREL_DIR}/ -- "
            f"cannot resolve any enlarge radius. See module docstring "
            f"'PACKAGING REQUIREMENT'."
        )

    target_px = target_km / representative_spacing_km
    best = min(available, key=lambda r: abs(r - target_px))
    achieved_km = best * representative_spacing_km

    rel_err = abs(achieved_km - target_km) / target_km if target_km != 0 else np.inf
    if rel_err > 0.25:
        warnings.warn(
            f"{param_name}: requested {target_km:.3g} km resolves to the "
            f"nearest AVAILABLE pixel radius {best} px ({achieved_km:.3g} km "
            f"at this grid's representative spacing of "
            f"{representative_spacing_km:.4g} km/px) -- a "
            f"{rel_err:.0%} difference from what was requested, since only "
            f"{available} px are bundled (see module docstring 'PACKAGING "
            f"REQUIREMENT' to export additional MATLAB disk masks for a "
            f"closer match).",
            stacklevel=2,
        )
    return best


def _edge_pad_close(arr: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Single closing step matching MATLAB's real imclose mechanism,
    found by direct comparison against MATLAB ground truth on real SPOL
    data: edge-replicate the INPUT (not the already-dilated array)
    before dilating, then standard dilate+erode with border_value=0,
    then crop back to the original shape.

    Validated bit-exact against MATLAB for simple axis-aligned line
    primitives. Small (~2%) residual remains for diagonal/2D primitives
    -- not yet fully explained, see module docstring. Do not replace
    with scipy.ndimage.binary_closing() (which uses a single global
    border_value and was found to diverge from MATLAB by 3000-9000+
    pixels on real cases -- catastrophically wrong, not a minor
    approximation).
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

    *** UPDATED this session, still NOT validated against any ground
    truth (no MATLAB or C++ "ECCO-H" reference exists to check against,
    unlike class_basic()'s SEA/SPOL cases) -- but it now carries the two
    fixes that WERE independently confirmed against class_basic()'s real
    MATLAB validation to be generically, mathematically necessary for
    ANY closing/erosion of this shape, not just facts about matching one
    specific reference:
      1. border_value=1 on the final erosion steps (mathematically
         required for erosion-after-closing to be extensive -- see
         module docstring "border_value" discussion; this is a fact
         about scipy's erosion semantics, independent of any reference).
      2. _sequential_close() instead of scipy's monolithic binary_closing()
         -- scipy's single global-border_value closing is a real,
         boundary-proximate divergence from true decomposed closing for
         ANY large structuring element, not a MATLAB-specific quirk (see
         _edge_pad_close()'s docstring). This reuses the SAME bundled
         decomposition data files as class_basic() (radius = enlarge*3 /
         enlarge*5), which only cover {15, 25} -- i.e. this still
         requires enlarge_mixed/enlarge_conv at their validated default
         of 5; other values will raise FileNotFoundError exactly as
         class_basic() does, see module docstring "PACKAGING REQUIREMENT".
    What is NOT fixed/known: whether the disk-enlarge-close-fill-erode
    STRUCTURE itself (as opposed to the mechanics of any one closing
    step) is the right algorithm for horizontal composite data at all --
    that's a question about matching some ground truth this package does
    not have access to, not a general morphological fact like the two
    items above.

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
        *** FIXED this session, via direct line-by-line comparison
        against the real f_classBasic.m source (lrose-ecco repo, not
        available earlier in the session). Two things were wrong:
        (1) An earlier attempt to fix this block's apparent real-data
        inertness changed the hardcoded threshold from 20/10 to 15/9,
        reasoning from OTHER parts of this codebase (class_sub_2d,
        VerticalParams). That was WRONG: the real MATLAB source uses
        `meltArea<20` and sentinel 10 -- exactly matching the ORIGINAL
        pre-session Python constants. Reverted back to 20/10.
        (2) A genuine, previously-undiscovered off-by-one translation
        bug: MATLAB's `checkCol(1:firstInd)=1` is INCLUSIVE of firstInd
        (1-indexed); the correct Python translation is
        `check_col[:first_ind + 1] = 1`, but the port wrote
        `check_col[:first_ind] = 1`, excluding first_ind. If the
        convectivity value at the melt-crossing pixel itself is NaN,
        this silently zeros out that column's entire contribution to
        the stratiform-percentage check. Confirmed in isolation (a NaN
        exactly at the crossing point flips a column from contributing
        6 valid points to contributing 0) and fixed by adding the
        missing +1.
        Tested against the real SPOL case (20220526_084500): neither fix
        changes the final output for that case specifically -- the one
        real clump that clears the below_frac gate has strat_perc=0.66
        (genuinely below the 0.8 trigger; confirmed via real MATLAB
        ECHOTYPE ground truth that this clump is 95.7% Convective Mid,
        i.e. real convection, correctly not reclassified), and none of
        its 226 columns happen to have a NaN exactly at the crossing
        pixel, so the off-by-one fix isn't exercised by this particular
        case either. Both fixes are still correct and now verified
        against the real MATLAB source directly (the strongest evidence
        available, per this project's own "ground truth over speculative
        debugging" standard) -- they just don't happen to move the needle
        on this one case's final numbers. ***
    enlarge_mixed, enlarge_conv : int
        Structuring element radii. Must have corresponding exported
        MATLAB data files bundled -- see module docstring.

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
    # *** ACTUALLY FIXED this session, via direct line-by-line comparison
    # against the real f_classBasic.m source (lrose-ecco repo). Two
    # things were wrong:
    #  1. An earlier attempt this session changed the melt threshold from
    #     20/10 to 15/9, reasoning from OTHER parts of this codebase
    #     (class_sub_2d, VerticalParams). That was WRONG -- the real
    #     MATLAB source uses `meltArea<20` and a sentinel of 10, exactly
    #     matching the ORIGINAL (pre-this-session) Python constants.
    #     Reverted back to 20/10 to match ground truth.
    #  2. A genuine, previously-undiscovered off-by-one translation bug:
    #     MATLAB's `checkCol(1:firstInd)=1` is INCLUSIVE of firstInd
    #     (1-indexed). The direct Python translation must be
    #     `check_col[:first_ind + 1] = 1` (0-indexed, inclusive of
    #     first_ind) -- but the port wrote `check_col[:first_ind] = 1`,
    #     EXCLUDING first_ind. Consequence: if the convectivity value at
    #     the melt-crossing pixel itself happens to be NaN, the
    #     subsequent `nan_inds`/`last_ind` search sees that NaN as an
    #     immediate gap (since it was never forced to a valid placeholder
    #     the way MATLAB forces it), producing last_ind < first_ind and
    #     silently zeroing out that column's entire contribution to
    #     check_cols. Confirmed via isolated reproduction: a NaN exactly
    #     at first_ind flips a column from contributing its full 6 valid
    #     points to contributing 0. This directly explains real-data
    #     under-triggering: any column whose crossing-point pixel is NaN
    #     (common near data-coverage edges/gaps) was silently dropped
    #     from strat_perc's denominator AND numerator, biasing strat_perc
    #     in an unpredictable direction depending on which columns happen
    #     to be affected. Fixed by adding the missing +1.
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
                  surf_alt_lim: float = 200.0) -> np.ndarray:
    """
    Sub-classification into echo type codes. Faithful port of the REAL
    f_classSub.m -- this replaces an earlier version of this function
    that used height-threshold logic (4.5/9.0 km AGL) and was never
    actually checked against the real MATLAB source. See module
    docstring for the validation history of that discovery.

    Non-uniform-grid note (checked, not just assumed): unlike class_basic()
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
    three are now required parameters, not optional alternatives. Any
    caller (eccopy2d_v.run(), eccopy1d.run(), eccopy3d.run(), etc.) that
    previously called this with "height OR temp" needs to be updated to
    supply height + melt + temp together -- see downstream TODO note
    below this function and in the calling modules.

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

    # Pass 1 — clump pixels are Convective, unconditionally (any clump
    # that reached this function already passed find_clumps_2d()'s own
    # area filtering -- see that module for the filtering logic).
    for clump in clumps:
        result[clump['index']] = 3

    # Pass 2 — stratiform/mixed for everything else with valid convectivity
    unassigned = np.isnan(result)
    has_conv = unassigned & ~np.isnan(convectivity)
    mixed_mask = has_conv & (convectivity > max_conv_for_strat)
    result[mixed_mask] = 2
    result[has_conv & ~mixed_mask] = 1

    return result


# ---------------------------------------------------------------------------
# 3-D echo type assignment (EccoPy-3D)
# Port of ConvStratFinder::_setEchoType3D() + StormClump::setEchoType()
# *** NOT re-validated this session -- see module docstring "NOT YET DONE" ***
# ---------------------------------------------------------------------------

def set_echo_type_3d(convectivity: np.ndarray,
                     clumps: list,
                     height_km: Optional[np.ndarray] = None,
                     temp: Optional[np.ndarray] = None,
                     shallow_threshold_ht: float = 4.5,
                     deep_threshold_ht: float = 9.0,
                     shallow_threshold_temp: float = 0.0,
                     deep_threshold_temp: float = -12.0,
                     topo_km: Optional[np.ndarray] = None,
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
    shallow/mid/deep, which was found to be WRONG for the 2-D case
    (class_sub_2d) -- the real algorithm there uses melt/temp instead.
    Whether the 3-D C++ reference (ConvStratFinder) genuinely differs
    from the 2-D MATLAB reference (f_classSub.m) in this respect, or
    whether this function has the same undiscovered bug, has not been
    checked. Do not treat this function as validated. ***

    Exact port of ConvStratFinder::_setEchoType3D() two-pass structure:

    Pass 1: For each clump, call _clump_category() — sets convective pixels.
    Pass 2: Loop all remaining pixels:
              - conv missing or 0 → skip (stays MISSING)
              - conv > max_conv_for_strat → MIXED
              - conv <= max_conv_for_strat → low/mid/high stratiform code,
                based on EITHER height_km OR temp (whichever is supplied)

    If both height_km and temp are None, every non-clump, non-mixed point
    with valid convectivity is simply MIXED-or-missing — pass a basic
    classification mode (CATEGORY_MIXED only, no low/mid/high distinction)
    by leaving both arguments unset; clump pixels are still differentiated
    into shallow/mid/deep/elevated.

    Parameters
    ----------
    convectivity : np.ndarray, shape (nz, ny, nx)
    clumps : list of dicts (see find_clumps_3d / _clump_category)
    height_km : np.ndarray, shape (nz, ny, nx), optional
        Height field, km MSL (or AGL — see `topo_km`).
    temp : np.ndarray, shape (nz, ny, nx), optional
        Temperature field, °C.
    shallow_threshold_ht, deep_threshold_ht : float, km
    shallow_threshold_temp, deep_threshold_temp : float, °C
    topo_km : np.ndarray, shape (ny, nx), optional
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
        if topo_km is not None:
            topo_km = np.asarray(topo_km, dtype=float)
            height_km = height_km - topo_km[np.newaxis, :, :]
        shallow_bnd = np.full((ny, nx), shallow_threshold_ht)
        deep_bnd = np.full((ny, nx), deep_threshold_ht)
        if terrain_ht_km is not None:
            terrain_ht_km = np.asarray(terrain_ht_km, dtype=float)
            shallow_bnd = np.maximum(shallow_bnd, terrain_ht_km + min_ht_agl_for_mid)
            deep_bnd = np.maximum(deep_bnd, terrain_ht_km + min_ht_agl_for_deep)
    elif use_temp:
        temp = np.asarray(temp, dtype=float)

    # Pass 1 — assign convective pixels via clumps
    for clump in clumps:
        category = _clump_category(
            clump, min_vol_km3, min_vert_extent_km,
            min_conv_frac_deep, min_conv_frac_shallow,
            max_shallow_frac_elevated, max_deep_frac_elevated,
            min_strat_frac_strat_below,
            convectivity, min_conv_for_conv,
        )
        echo_type[clump['index']] = category

    # Pass 2 — stratiform/mixed for remaining points
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
    index : tuple of (iz_arr, iy_arr, ix_arr) — matches clump['index']
        from find_clumps_3d(), shape (Z, Y, X) convention.
    convectivity : np.ndarray, shape (Z, Y, X)

    Note: an earlier version of this function used variables NAMED
    (ix, iy, iz) that were actually bound from np.where() on a
    (Z, Y, X)-shaped mask — so "iz_arr - 1" was really decrementing the
    LAST axis (X), not the first (Z). Fixed by using the correct
    (Z, Y, X)-ordered index tuple and decrementing iz_arr, vectorized
    via fancy indexing.
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