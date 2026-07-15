"""
2-D clumping for EccoPy-2D-H (horizontal composite data).

*** NEW THIS SESSION -- design, not a port. *** There is no MATLAB or C++
"ECCO-H" reference algorithm to match against (unlike class_basic()'s
SEA/SPOL ground truth, or eccopy3d/clumping.py's LROSE-ECCO ground truth).
This module instead reuses the ALGORITHM STRUCTURE that IS validated --
eccopy3d/clumping.py's two-stage dual-threshold clumping (ClumpingDualThresh,
ported from real lrose-core sources) -- because that algorithm's Stage 2
(the actual splitting logic) is already a genuinely 2-D computation: it
projects each 3-D clump down to a 2-D (Y, X) "composite" (max convectivity
per column) before doing anything else. For single-level EccoPy-2D-H data,
that projection is a no-op (there is only one level per column already),
so reusing the SAME splitting mechanics here is not a guess -- it is the
same 2-D computation the validated 3-D path already performs internally,
just without a Z axis to project away first.

What genuinely differs from eccopy3d/clumping.py, and is NOT validated:

  - AREA, not volume. 3-D clumps are filtered by volume_km3 (spacing^3,
    or more precisely dz*dy*dx per cell). A single horizontal level has
    no dz -- there is no principled way to manufacture one, and the 3-D
    code's each_subclump_min_area_km2 quirk (multiplying by dz at level
    0 of a FULL 3-D grid) is a known artifact of ClumpProps being reused
    for a 2-D-only computation in the C++, not something to replicate
    here on purpose. This module computes real area (km^2) directly from
    each pixel's own dy*dx, and filters against that -- more correct than
    the 3-D quirk, but a genuine behavioural difference, not a match to
    any reference.

    *** CAUTION: this means each_subclump_min_area_km2 (a ClassificationParams
    field shared with eccopy3d) means a DIFFERENT PHYSICAL QUANTITY here
    than it does in eccopy3d/clumping.py -- true area vs. the C++'s
    pseudo-volume quirk. Do not reuse a single ClassificationParams
    instance, tuned/validated against 3-D cases, directly for
    eccopy2d_h.run() and expect matching splitting behaviour at the same
    numeric threshold value -- tune this field separately per module.
    See README "Parameters" for the full warning. ***
  - Primary-envelope connectivity: 4-connectivity (matching _STRUCT_4,
    the same convention eccopy3d/clumping.py's Stage-2 2-D composite
    labeling already uses, and matching LROSE/TITAN interval clumping at
    min_overlap=1). This is DIFFERENT from class_basic()'s 8-connectivity
    (_CONN8, MATLAB bwconncomp convention) used for its melt-correction
    region labeling -- that is a different codebase's convention for a
    different purpose (rain-below-melt column detection, not storm-cell
    identification) and should not be assumed to apply here.
  - No height/temp/vertical-extent geometry at all -- there is nothing
    to compute it from. Clumps carry only their index, area, and point
    count.

ARRAY CONVENTION: (Y, X), matching EccoPy-2D-H's data-agnostic shape
contract (see eccopy2d_h/classify.py).

PERFORMANCE: per-clump work is restricted to each clump's bounding box via
scipy.ndimage.find_objects(), same pattern as eccopy3d/clumping.py.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from scipy.ndimage import label, find_objects

# Reused directly, unmodified, from the validated 3-D path -- both
# operate on plain (Y, X) arrays already, so no adaptation is needed.
from ..eccopy3d.clumping import _grow_regions, _STRUCT_4


def find_clumps_2d(convectivity: np.ndarray,
                   spacing_y: np.ndarray,
                   spacing_x: np.ndarray,
                   min_conv: float,
                   min_area_km2: Optional[float] = None,
                   use_dual_thresholds: bool = True,
                   secondary_threshold: float = 0.65,
                   all_subclumps_min_area_frac: float = 0.33,
                   each_subclump_min_area_frac: float = 0.02,
                   each_subclump_min_area_km2: float = 2.0) -> List[dict]:
    """
    Find 2-D convective clumps in horizontal composite data using the
    same two-stage dual-threshold structure as eccopy3d/clumping.py's
    find_clumps_3d() -- see module docstring for what is/isn't validated.

    Parameters
    ----------
    convectivity : np.ndarray, shape (Y, X)
    spacing_y, spacing_x : np.ndarray, shape (Y, X)
        Local point-to-point spacing, km. Used for per-pixel area
        (spacing_y * spacing_x) in all area filters.
    min_conv : float
        PRIMARY threshold -- builds the Stage-1 envelope clumps.
    min_area_km2 : float, optional
        Minimum clump area, applied BOTH before splitting (Stage 1) and
        after (Stage 3) -- same two-checkpoint pattern as find_clumps_3d's
        min_vol_km3. None (default) disables this floor entirely: clumps
        are still found and (optionally) split, just never dropped for
        being too small -- matching eccopy1d's min_convective_length
        opt-in-filter convention, since there's no validated default to
        offer here.
    use_dual_thresholds : bool
        If False, falls back to single-threshold 4-connected labeling at
        min_conv with only the min_area_km2 filter, no splitting.
    secondary_threshold : float
    all_subclumps_min_area_frac, each_subclump_min_area_frac,
    each_subclump_min_area_km2 : float
        Same roles as find_clumps_3d's identically-named parameters
        (reused ClassificationParams fields) -- each_subclump_min_area_km2
        here means genuine area, unlike the 3-D quirk (see module
        docstring). CAUTION: do not pass in a ClassificationParams
        instance tuned/validated against eccopy3d cases and expect this
        field's numeric value to mean the same thing here -- see module
        docstring "AREA, not volume" and README "Parameters".

    Returns
    -------
    clumps : list of dicts, each with:
        index (an (iy_arr, ix_arr) tuple of integer index arrays -- use
            as echo_type[clump['index']] = ...),
        area_km2, n_pts_total
    """
    convectivity = np.asarray(convectivity, dtype=float)
    spacing_y = np.asarray(spacing_y, dtype=float)
    spacing_x = np.asarray(spacing_x, dtype=float)
    if spacing_y.shape != convectivity.shape or spacing_x.shape != convectivity.shape:
        raise ValueError(
            f"spacing_y/spacing_x shape must match convectivity shape "
            f"{convectivity.shape}; got {spacing_y.shape}, {spacing_x.shape}"
        )
    pixel_area = spacing_y * spacing_x
    area_floor = min_area_km2 if min_area_km2 is not None else 0.0

    def _geom_for_mask(local_mask, bbox, local_area):
        iy_local, ix_local = np.where(local_mask)
        y0, x0 = bbox[0].start, bbox[1].start
        return {
            'index':      (iy_local + y0, ix_local + x0),
            'area_km2':   float(np.sum(local_area[local_mask])),
            'n_pts_total': int(len(iy_local)),
        }

    # ------------------------------------------------------------
    # Fallback: no dual thresholds -- single-stage labeling, area filter.
    # ------------------------------------------------------------
    if not use_dual_thresholds:
        active = (~np.isnan(convectivity)) & (convectivity >= min_conv)
        labeled, n_clumps = label(active, structure=_STRUCT_4)
        if n_clumps == 0:
            return []
        bboxes = find_objects(labeled, max_label=n_clumps)
        clumps = []
        for cid, bbox in enumerate(bboxes, start=1):
            if bbox is None:
                continue
            local_mask = labeled[bbox] == cid
            local_area = pixel_area[bbox]
            clump = _geom_for_mask(local_mask, bbox, local_area)
            if clump['area_km2'] >= area_floor:
                clumps.append(clump)
        return clumps

    # ------------------------------------------------------------
    # Stage 1: primary envelope clumping
    # ------------------------------------------------------------
    active_primary = (~np.isnan(convectivity)) & (convectivity >= min_conv)
    labeled_primary, n_primary = label(active_primary, structure=_STRUCT_4)
    if n_primary == 0:
        return []

    bboxes = find_objects(labeled_primary, max_label=n_primary)

    clumps: List[dict] = []
    for cid, bbox in enumerate(bboxes, start=1):
        if bbox is None:
            continue

        primary_mask = labeled_primary[bbox] == cid
        local_conv = convectivity[bbox]
        local_area = pixel_area[bbox]

        primary_area = float(np.sum(local_area[primary_mask]))
        if primary_area < area_floor:
            # Dropped entirely -- never split, never emitted (matches
            # find_clumps_3d's outer volume check).
            continue

        # ------------------------------------------------------------
        # Stage 2: dual-threshold split within this clump's bbox
        # ------------------------------------------------------------
        sub_masks = _dual_threshold_split_2d(
            primary_mask, local_conv, local_area,
            secondary_threshold=secondary_threshold,
            all_frac=all_subclumps_min_area_frac,
            each_frac=each_subclump_min_area_frac,
            each_km2=each_subclump_min_area_km2,
        )

        # ------------------------------------------------------------
        # Stage 3: final area filter, applied to EACH emitted piece
        # ------------------------------------------------------------
        for sub_mask in sub_masks:
            clump = _geom_for_mask(sub_mask, bbox, local_area)
            if clump['area_km2'] >= area_floor:
                clumps.append(clump)

    return clumps


def _dual_threshold_split_2d(primary_mask: np.ndarray,
                             local_conv: np.ndarray,
                             local_area: np.ndarray,
                             secondary_threshold: float,
                             all_frac: float,
                             each_frac: float,
                             each_km2: float) -> List[np.ndarray]:
    """
    2-D analogue of eccopy3d/clumping.py's _dual_threshold_split(), for
    ONE primary clump already restricted to its local bounding box.
    There is no Z axis to project away here -- the "composite" is simply
    the clump's own convectivity values, everywhere else masked out.

    Returns a list of 2-D boolean masks (same local shape as
    primary_mask), one per accepted sub-clump -- or a single-element list
    containing primary_mask unchanged if splitting wasn't valid.
    """
    composite = np.where(primary_mask, local_conv, np.nan)
    footprint = primary_mask

    n_comp = int(np.sum(footprint))
    if n_comp == 0:
        return [primary_mask]

    active_secondary = footprint & (composite >= secondary_threshold)
    labeled_sec, n_secondary = label(active_secondary, structure=_STRUCT_4)

    if n_secondary == 0:
        return [primary_mask]

    sizes_area = np.array([
        float(np.sum(local_area[labeled_sec == i])) for i in range(1, n_secondary + 1)
    ])
    sizes_px = np.array([
        int(np.sum(labeled_sec == i)) for i in range(1, n_secondary + 1)
    ])
    sum_px = float(np.sum(sizes_px))
    frac_all_parts = sum_px / n_comp

    frac_each = sizes_px.astype(float) / n_comp
    valid = (frac_each >= each_frac) & (sizes_area >= each_km2)
    n_large_enough = int(np.sum(valid))

    if (frac_all_parts < all_frac) or (n_large_enough < 2):
        return [primary_mask]

    valid_grid = np.zeros_like(labeled_sec, dtype=int)
    next_id = 0
    for i in range(1, n_secondary + 1):
        if valid[i - 1]:
            next_id += 1
            valid_grid[labeled_sec == i] = next_id

    # Grow valid sub-clump seeds back out to fill the full envelope
    # footprint -- reused directly from the validated 3-D path.
    grown_grid = _grow_regions(valid_grid, np.where(footprint, local_conv, -np.inf), footprint)

    out_masks = []
    for sub_id in range(1, next_id + 1):
        sub_mask = primary_mask & (grown_grid == sub_id)
        if not np.any(sub_mask):
            continue
        # Unlike the 3-D case, a 2-D sub-region carved from a 4-connected
        # primary clump via 4-connected growth is guaranteed 4-connected
        # itself -- no "largest fragment" step is needed here.
        out_masks.append(sub_mask)

    if not out_masks:
        return [primary_mask]

    return out_masks
