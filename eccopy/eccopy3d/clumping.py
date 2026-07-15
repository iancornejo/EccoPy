"""
3-D clumping for EccoPy-3D.

*** REWRITE (this session) — real two-stage dual-threshold algorithm ***
*** VALIDATED against real LROSE-ECCO truth across 3 independent cases ***
Ported directly from lrose-core sources (NCAR/lrose-core @
b29264bf466d9b702dcea8825aa2486ebf0b8554):
    codebase/libs/euclid/src/clump/ClumpingMgr.cc      (loadClumpVector)
    codebase/libs/euclid/src/clump/ClumpingDualThresh.cc  (compute, _growSubAreas)
    codebase/libs/euclid/src/clump/ClumpProps.cc          (volumeKm3, nPoints2D)
    codebase/libs/radar/src/convstrat/ConvStratFinder.cc  (_performClumping,
                                                            StormClump::computeGeom)

The PREVIOUS version of this module did straight 3-D connected-component
labeling directly at the secondary threshold. Baseline-tested against real
LROSE-ECCO truth output (SPOL Taiwan case, texture_radius=4km): only 6
clumps found, capturing 3,379 of 126,445 truth-convective pixels (2.7%
recall) -- 96.5% of true convective pixels were mis-assigned as Mixed.

The rewritten algorithm below fixed this to 98%+ convective-pixel recall
on that same case, and has since been validated end-to-end (raw DBZ
through texture/convectivity/clumping/classification) against 3
independent real LROSE-ECCO cases -- SPOL, WRF, and SEA, 5 runs total
across 2 texture-radius variants each -- achieving 99.5%-99.8%+ both-echo
pixel agreement on every run. All 4 convective sub-types (Shallow, Mid,
Deep, and Elevated -- the last driven by the `_strat_below()` path below,
only exercised by the SEA 7km case) have been checked against real truth.
See the README's "Validation status" section for the full case-by-case
numbers and the other real bugs (a texture-prefiltering gap in
core/texture.py; a Z-spacing convention bug for non-uniform vertical
grids) found and fixed alongside this rewrite.

ALGORITHM ("dual threshold" clumping, exact port):

  Stage 1 -- PRIMARY ENVELOPE
    3-D connected-component labeling of convectivity >= min_convectivity_
    for_convective (the PRIMARY threshold), 6-connectivity (face-adjacent),
    matching LROSE's interval clumping at min_overlap=1 (the TDRP default
    for min_overlap_for_convective_clumps in every case file seen so far;
    general min_overlap>1 is NOT implemented here -- see note below).
    Clumps below min_valid_volume_for_convective are DROPPED ENTIRELY at
    this stage (never split, never emitted) -- matches loadClumpVector's
    outer volume check gating entry to the dual-threshold split.

  Stage 2 -- PER-CLUMP DUAL-THRESHOLD SPLIT (_dual_threshold_split)
    For each surviving primary clump, restricted to its 3-D bounding box:
      a. Build a 2-D "composite" grid: for each (y,x) column touched by
         the clump, the MAX convectivity among just that clump's OWN
         member voxels in that column (not the full column of the input
         grid) -- ClumpingDualThresh::_fillComposite().
      b. 2-D connected-component label the composite at the SECONDARY
         threshold (0.65 default), 4-connectivity -- candidate sub-clumps.
      c. A sub-clump is VALID if its pixel-count fraction of the envelope
         footprint >= each_subclump_min_area_frac, AND its pixel count *
         dx_km * dy_km * dz_km[0] (see note below) >= each_subclump_min_
         area_km2.
      d. If the summed pixel-count fraction of ALL candidate sub-clumps
         (valid or not) < all_subclumps_min_area_frac, OR fewer than 2
         valid sub-clumps exist -- give up splitting; use the ORIGINAL
         primary clump unchanged (ClumpingDualThresh::compute() returns 1).
      e. Otherwise, GROW each valid sub-clump seed back out to fill the
         full envelope footprint via iterative highest-neighbour-value
         flood fill, 4-connected (_growSubAreas()) -- ties broken by
         value, not by original C++'s randomized visit order (negligible
         difference on real float data).
      f. Re-mask the PRIMARY clump's own 3-D voxels by which grown 2-D
         sub-region each column falls under, then 3-D-label each masked
         sub-volume (6-connectivity). If a sub-region's voxels are not
         3-D-connected, ONLY THE LARGEST fragment is kept -- smaller
         fragments are silently dropped, matching _computeSubClump()'s
         "keep max-points fragment" behaviour.

  Stage 3 -- FINAL VOLUME FILTER
    Each object emitted by stage 2 (whether the original unsplit clump
    or an individual sub-clump) is filtered AGAIN against min_valid_
    volume_for_convective. This is a real second, independent filter in
    the C++ (ConvStratFinder::_performClumping(), after loadClumpVector
    returns) -- sub-clumps that individually fall below the volume floor
    are dropped even though their PARENT clump passed the same test.

QUIRK, faithfully reproduced (found in ClumpProps.cc::_computeProps()):
  each_subclump_min_area_km2 is nominally an AREA (km^2), but the C++
  compares it against a value computed as
      n_pixels_2d * dx_km * dy_km * dz_km_of_LEVEL_ZERO
  -- i.e. an actual VOLUME using the grid's z-spacing at index 0 of the
  FULL grid, regardless of the sub-clump's real vertical position. This
  looks like a latent unit-convention artifact of ClumpProps being reused
  for a 2-D-only computation, but since the C++ output is our ground
  truth, we reproduce it exactly rather than "fixing" it.

CONNECTIVITY CAVEAT: TITAN's underlying interval-based clumping supports
min_overlap_for_convective_clumps > 1 (require >1 column of horizontal
overlap between adjacent rows/planes to connect). All three validation
cases (SPOL, WRF, SEA) use the TDRP default of 1, which is exactly
equivalent to standard 6-connectivity (3-D) / 4-connectivity (2-D)
labeling -- so scipy.ndimage.label is a safe stand-in here. If a future
case sets min_overlap_for_convective_clumps > 1, this module will need a
true interval-overlap implementation; it will silently give 6-/4-
connectivity behaviour instead, which is a real (currently unvalidated)
gap.

ARRAY CONVENTION: (Z, Y, X), matching EccoPy-3D's data-agnostic shape
contract (see eccopy3d.classify.run()).

TOPO/AGL SUPPORT: `topo_km` here performs a literal height_km - topo_km
AGL subtraction. NOTE: this mechanism does NOT appear to exist in the
real 3-D C++ path (ConvStratFinder only has `terrainHt`, which RAISES the
shallow/deep threshold BOUNDARIES -- see set_echo_type_3d's `terrain_ht_km`
-- it never subtracts anything from the height field itself). `topo_km`
was carried over from the 2-D MATLAB f_classSub.m port and is UNVALIDATED
for the 3-D path; none of the 3 real test cases populate it, so it hasn't
mattered yet. Flagging for awareness, not fixing now.

PERFORMANCE: per-clump work is restricted to each clump's tight bounding
box via scipy.ndimage.find_objects(), as before -- cost scales with total
clump volume, not n_clumps x grid_size.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import label, find_objects, generate_binary_structure

_STRUCT_6 = np.zeros((3, 3, 3), dtype=int)
_STRUCT_6[1, 1, :] = 1
_STRUCT_6[1, :, 1] = 1
_STRUCT_6[:, 1, 1] = 1

_STRUCT_4 = generate_binary_structure(2, 1)  # 2-D cross (4-connectivity)


def find_clumps_3d(convectivity: np.ndarray,
                   spacing: np.ndarray,
                   min_conv: float,
                   min_vol_km3: float,
                   height_km: Optional[np.ndarray] = None,
                   temp: Optional[np.ndarray] = None,
                   topo_km: Optional[np.ndarray] = None,
                   shallow_threshold_ht: float = 4.5,
                   deep_threshold_ht: float = 9.0,
                   shallow_threshold_temp: float = 0.0,
                   deep_threshold_temp: float = -12.0,
                   use_dual_thresholds: bool = True,
                   secondary_threshold: float = 0.65,
                   all_subclumps_min_area_frac: float = 0.33,
                   each_subclump_min_area_frac: float = 0.02,
                   each_subclump_min_area_km2: float = 2.0,
                   dx_km: Optional[float] = None,
                   dy_km: Optional[float] = None,
                   dz0_km: Optional[float] = None) -> List[dict]:
    """
    Find 3-D convective clumps using the real two-stage dual-threshold
    algorithm (see module docstring for the full port description).

    Parameters
    ----------
    convectivity : np.ndarray, shape (Z, Y, X)
    spacing : np.ndarray, shape (Z, Y, X)
        Local point-to-point spacing, km. Used to compute each grid
        cell's volume (spacing^3) for volume filters and geometry.
    min_conv : float
        PRIMARY threshold -- now genuinely used to build the outer
        envelope clumps (stage 1), unlike the previous version.
    min_vol_km3 : float
        Volume filter, applied BOTH before splitting (stage 1) and after
        (stage 3) -- see module docstring.
    height_km, temp, topo_km : see previous version; unchanged semantics.
    shallow_threshold_ht, deep_threshold_ht : float, km
    shallow_threshold_temp, deep_threshold_temp : float, °C
    use_dual_thresholds : bool
        If False, falls back to single-threshold clumping at min_conv
        with no splitting -- matches loadClumpVector's non-dual-threshold
        branch.
    secondary_threshold : float
    all_subclumps_min_area_frac, each_subclump_min_area_frac,
    each_subclump_min_area_km2 : float
        See ClassificationParams / TDRP all_subclumps_min_area_fraction,
        each_subclump_min_area_fraction, each_subclump_min_area_km2.
    dx_km, dy_km : float, optional
        Representative (scalar) horizontal grid spacing, km. Required if
        use_dual_thresholds=True. For non-uniform horizontal grids, pass
        a representative value (e.g. median spacing) -- matches the C++,
        which also uses a single dx/dy "at centroid" per clump, not a
        genuinely spatially-varying one.
    dz0_km : float, optional
        Z-spacing at vertical level 0 of the FULL grid. Required if
        use_dual_thresholds=True. Used ONLY to reproduce the
        each_subclump_min_area_km2 quirk described in the module
        docstring -- deliberately NOT the sub-clump's own local dz.

    Returns
    -------
    clumps : list of dicts, each with:
        index (a (iz_arr, iy_arr, ix_arr) tuple of integer index arrays —
            use as echo_type[clump['index']] = ...),
        volume_km3, vert_extent_km,
        n_pts_total, n_pts_shallow, n_pts_mid, n_pts_deep
    """
    convectivity = np.asarray(convectivity, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != convectivity.shape:
        raise ValueError(
            f"spacing shape {spacing.shape} must match convectivity shape "
            f"{convectivity.shape}"
        )

    use_height = height_km is not None
    use_temp = (not use_height) and (temp is not None)
    if use_height:
        height_km = np.asarray(height_km, dtype=float)
        if topo_km is not None:
            topo_km = np.asarray(topo_km, dtype=float)
            height_km = height_km - topo_km[np.newaxis, :, :]
    elif use_temp:
        temp = np.asarray(temp, dtype=float)

    def _geom_for_mask(local_mask, bbox, local_spacing, local_height, local_temp):
        return _compute_geom(
            local_mask, bbox, local_spacing, local_height, local_temp,
            shallow_threshold_ht, deep_threshold_ht,
            shallow_threshold_temp, deep_threshold_temp,
        )

    # ------------------------------------------------------------
    # Fallback: no dual thresholds -- single-stage labeling at min_conv,
    # straight volume filter. Matches loadClumpVector's non-dual branch.
    # ------------------------------------------------------------
    if not use_dual_thresholds:
        active = (~np.isnan(convectivity)) & (convectivity >= min_conv)
        labeled, n_clumps = label(active, structure=_STRUCT_6)
        if n_clumps == 0:
            return []
        bboxes = find_objects(labeled, max_label=n_clumps)
        clumps = []
        for cid, bbox in enumerate(bboxes, start=1):
            if bbox is None:
                continue
            local_mask = labeled[bbox] == cid
            local_spacing = spacing[bbox]
            local_height = height_km[bbox] if use_height else None
            local_temp = temp[bbox] if use_temp else None
            clump = _geom_for_mask(local_mask, bbox, local_spacing, local_height, local_temp)
            if clump['volume_km3'] >= min_vol_km3:
                clumps.append(clump)
        return clumps

    # ------------------------------------------------------------
    # Stage 1: primary envelope clumping
    # ------------------------------------------------------------
    if dx_km is None or dy_km is None or dz0_km is None:
        raise ValueError(
            "dx_km, dy_km, and dz0_km are required when use_dual_thresholds=True"
        )

    active_primary = (~np.isnan(convectivity)) & (convectivity >= min_conv)
    labeled_primary, n_primary = label(active_primary, structure=_STRUCT_6)
    if n_primary == 0:
        return []

    bboxes = find_objects(labeled_primary, max_label=n_primary)

    clumps: List[dict] = []
    for cid, bbox in enumerate(bboxes, start=1):
        if bbox is None:
            continue

        primary_mask = labeled_primary[bbox] == cid
        local_conv = convectivity[bbox]
        local_spacing = spacing[bbox]
        local_height = height_km[bbox] if use_height else None
        local_temp = temp[bbox] if use_temp else None

        primary_vol = float(np.sum(local_spacing[primary_mask] ** 3))
        if primary_vol < min_vol_km3:
            # Dropped entirely -- never split, never emitted (matches the
            # outer volume check gating entry to _dualT->compute()).
            continue

        # ------------------------------------------------------------
        # Stage 2: dual-threshold split within this clump's bbox
        # ------------------------------------------------------------
        sub_masks = _dual_threshold_split(
            primary_mask, local_conv,
            secondary_threshold=secondary_threshold,
            all_frac=all_subclumps_min_area_frac,
            each_frac=each_subclump_min_area_frac,
            each_km2=each_subclump_min_area_km2,
            dx_km=dx_km, dy_km=dy_km, dz0_km=dz0_km,
        )

        # ------------------------------------------------------------
        # Stage 3: final volume filter, applied to EACH emitted piece
        # ------------------------------------------------------------
        for sub_mask in sub_masks:
            clump = _geom_for_mask(sub_mask, bbox, local_spacing, local_height, local_temp)
            if clump['volume_km3'] >= min_vol_km3:
                clumps.append(clump)

    return clumps


def _dual_threshold_split(primary_mask: np.ndarray,
                          local_conv: np.ndarray,
                          secondary_threshold: float,
                          all_frac: float,
                          each_frac: float,
                          each_km2: float,
                          dx_km: float,
                          dy_km: float,
                          dz0_km: float) -> List[np.ndarray]:
    """
    Port of ClumpingDualThresh::compute() for ONE primary clump, already
    restricted to its local bounding box.

    Returns a list of 3-D boolean masks (same local shape as
    primary_mask), one per accepted sub-clump -- or a single-element list
    containing primary_mask unchanged if splitting wasn't valid.
    """
    # (a) composite: max convectivity among the CLUMP'S OWN member voxels
    # per (y, x) column (not the whole column of the input grid).
    masked_conv = np.where(primary_mask, local_conv, -np.inf)
    composite = np.max(masked_conv, axis=0)
    footprint = np.any(primary_mask, axis=0)
    composite = np.where(footprint, composite, np.nan)

    n_comp = int(np.sum(footprint))  # envelope footprint size (pixels)
    if n_comp == 0:
        return [primary_mask]

    # (b) 2-D label the composite at the secondary threshold
    active_secondary = footprint & (composite >= secondary_threshold)
    labeled_sec, n_secondary = label(active_secondary, structure=_STRUCT_4)

    if n_secondary == 0:
        return [primary_mask]

    # (c) validity of each candidate sub-clump
    sizes = np.array([np.sum(labeled_sec == i) for i in range(1, n_secondary + 1)])
    sum_size = float(np.sum(sizes))
    frac_all_parts = sum_size / n_comp

    pseudo_vol_km3 = sizes.astype(float) * dx_km * dy_km * dz0_km  # QUIRK, see module docstring
    frac_each = sizes.astype(float) / n_comp
    valid = (frac_each >= each_frac) & (pseudo_vol_km3 >= each_km2)
    n_large_enough = int(np.sum(valid))

    # (d) give up splitting -> use original clump unchanged
    if (frac_all_parts < all_frac) or (n_large_enough < 2):
        return [primary_mask]

    # renumber valid sub-clumps 1..n_large_enough, in original label order
    valid_grid = np.zeros_like(labeled_sec, dtype=int)
    next_id = 0
    for i in range(1, n_secondary + 1):
        if valid[i - 1]:
            next_id += 1
            valid_grid[labeled_sec == i] = next_id

    # (e) grow valid sub-clumps back out to fill the full envelope
    grown_grid = _grow_regions(valid_grid, composite, footprint)

    # (f) re-mask the primary clump's own 3-D voxels by grown sub-region,
    # 3-D label each, keep only the largest fragment if it splits
    out_masks = []
    for sub_id in range(1, next_id + 1):
        column_mask = (grown_grid == sub_id)  # (Y, X)
        sub_mask_3d = primary_mask & column_mask[np.newaxis, :, :]
        if not np.any(sub_mask_3d):
            continue
        labeled_3d, n3d = label(sub_mask_3d, structure=_STRUCT_6)
        if n3d <= 1:
            out_masks.append(sub_mask_3d)
        else:
            # keep only the largest connected fragment
            counts = np.bincount(labeled_3d.ravel())
            counts[0] = 0  # ignore background
            best = int(np.argmax(counts))
            out_masks.append(labeled_3d == best)

    if not out_masks:
        # shouldn't normally happen, but fall back safely
        return [primary_mask]

    return out_masks


def _grow_regions(seed_grid: np.ndarray,
                  value_grid: np.ndarray,
                  footprint: np.ndarray) -> np.ndarray:
    """
    Port of ClumpingDualThresh::_growSubAreas(): iteratively grow the
    labeled seed regions in `seed_grid` (0 = unassigned) out to fill the
    full `footprint`, 4-connected. Each still-unassigned footprint cell
    adopts the label of whichever already-assigned neighbour has the
    HIGHEST value_grid value, repeating until nothing changes.

    Vectorized: each iteration considers all 4 neighbour directions at
    once (rather than the C++'s randomized single-cell visit order).
    Ties (identical composite values among competing neighbours) are
    broken by direction-scan order -- a negligible difference on real
    floating-point convectivity data.
    """
    grown = seed_grid.copy()
    # value grid used for comparisons; unassigned/out-of-footprint cells
    # act as always-losing neighbours (-inf) so they never propagate.
    vals = np.where(footprint, value_grid, -np.inf)

    unassigned = footprint & (grown == 0)
    if not np.any(unassigned):
        return grown

    while True:
        best_val = np.full(grown.shape, -np.inf)
        best_id = np.zeros(grown.shape, dtype=int)

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb_id = np.roll(np.roll(grown, dy, axis=0), dx, axis=1)
            nb_val = np.roll(np.roll(vals, dy, axis=0), dx, axis=1)
            # zero-out wrapped-around edges (roll wraps; we don't want that)
            if dy == -1:
                nb_id[-1, :] = 0; nb_val[-1, :] = -np.inf
            elif dy == 1:
                nb_id[0, :] = 0; nb_val[0, :] = -np.inf
            if dx == -1:
                nb_id[:, -1] = 0; nb_val[:, -1] = -np.inf
            elif dx == 1:
                nb_id[:, 0] = 0; nb_val[:, 0] = -np.inf

            better = (nb_id > 0) & (nb_val > best_val)
            best_val = np.where(better, nb_val, best_val)
            best_id = np.where(better, nb_id, best_id)

        can_grow = unassigned & (best_id > 0)
        if not np.any(can_grow):
            break  # converged (or isolated unreachable cells -- shouldn't
                    # happen since footprint is 4-connected by construction
                    # of the original clump's projection, but be safe)

        grown = np.where(can_grow, best_id, grown)
        unassigned = footprint & (grown == 0)
        if not np.any(unassigned):
            break

    return grown


def _compute_geom(local_mask: np.ndarray,
                  bbox: Tuple[slice, slice, slice],
                  local_spacing: np.ndarray,
                  local_height_km: Optional[np.ndarray],
                  local_temp: Optional[np.ndarray],
                  shallow_threshold_ht: float,
                  deep_threshold_ht: float,
                  shallow_threshold_temp: float,
                  deep_threshold_temp: float) -> dict:
    """
    Compute volume, vertical extent, and shallow/mid/deep point counts
    for ONE clump, given arrays already cropped to that clump's bounding
    box — so every operation here touches only the clump's small local
    volume, not the full grid.
    """
    iz_local, iy_local, ix_local = np.where(local_mask)
    n_total = len(iz_local)

    z0 = bbox[0].start
    y0 = bbox[1].start
    x0 = bbox[2].start
    iz_arr = iz_local + z0
    iy_arr = iy_local + y0
    ix_arr = ix_local + x0

    cell_vol = local_spacing[local_mask] ** 3
    vol = float(np.sum(cell_vol))

    if local_height_km is not None:
        h = local_height_km[local_mask]
        vert_extent = float(np.nanmax(h) - np.nanmin(h))
        # NOTE: <= for shallow (not <) to match ConvStratFinder::
        # StormClump::computeGeom() exactly: "zKm <= shallowBoundaryKm".
        n_shallow = int(np.sum(h <= shallow_threshold_ht))
        n_deep = int(np.sum(h >= deep_threshold_ht))
        n_mid = n_total - n_shallow - n_deep
    elif local_temp is not None:
        t = local_temp[local_mask]
        vert_extent = float((iz_local.max() - iz_local.min() + 1)
                            * np.nanmedian(local_spacing[local_mask]))
        n_shallow = int(np.sum(t >= shallow_threshold_temp))
        n_deep = int(np.sum(t < deep_threshold_temp))
        n_mid = n_total - n_shallow - n_deep
    else:
        vert_extent = float((iz_local.max() - iz_local.min() + 1)
                            * np.nanmedian(local_spacing[local_mask])) if n_total else 0.0
        n_shallow = n_deep = 0
        n_mid = n_total

    return {
        'index':          (iz_arr, iy_arr, ix_arr),
        'volume_km3':     vol,
        'vert_extent_km': vert_extent,
        'n_pts_total':    n_total,
        'n_pts_shallow':  n_shallow,
        'n_pts_mid':      n_mid,
        'n_pts_deep':     n_deep,
    }


def col_max_convectivity(conv_3d: np.ndarray) -> np.ndarray:
    """Column-maximum convectivity (Z,Y,X) → (Y,X)."""
    return np.nanmax(conv_3d, axis=0)
