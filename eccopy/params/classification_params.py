"""Classification parameters — defaults match ConvStratFinder constructor,
except enlarge_mixed/enlarge_conv, which come from a different codebase
entirely (MATLAB ECCO-V's f_classBasic.m) -- confirmed against that source.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ClassificationParams:
    """
    Parameters controlling basic and sub-classification.
    Most defaults are taken directly from ConvStratFinder constructor in
    lrose-core. enlarge_mixed/enlarge_conv are the one exception: they do
    not appear anywhere in ConvStratFinder.cc or paramdef.ConvStrat (checked
    directly against source) -- they come from the separate MATLAB ECCO-V
    codebase (f_classBasic.m) instead, and their defaults below are taken
    from that source.

    Basic (EccoPy-1D / EccoPy-2D)
    --------------------------
    strat_mixed : float        Convectivity threshold strat/mixed.      Default: 0.4
    mixed_conv  : float        Convectivity threshold mixed/conv.       Default: 0.5
    enlarge_mixed : int        Dilation radius for mixed (pixels).      Default: 5
    enlarge_conv  : int        Dilation radius for convective (pixels). Default: 5
                                (matches MATLAB f_classBasic.m; NOTE an
                                earlier revision of this package incorrectly
                                defaulted this to 3, which was never the
                                real MATLAB default and was only ever
                                validated at 5 -- see core/classification.py
                                module docstring for the validated data-file
                                radius set)

                                *** These are LITERAL PIXEL/GATE COUNTS,
                                not physical distances -- unlike the
                                texture WindowSpec, class_basic() never
                                looks at a spacing array. On a non-uniform
                                grid (e.g. an RHI with elevation-angle-
                                dependent range-gate spacing), the SAME
                                enlarge_conv represents a DIFFERENT
                                physical cleanup radius at different rows.
                                This is not a bug -- see core/classification.py
                                module docstring "PIXEL COUNTS, NOT
                                PHYSICAL UNITS" for why there's no
                                physically-correct alternative to offer,
                                and core.classification.resolve_enlarge_
                                radius_px() for a uniform-grid-only
                                km-to-pixel convenience if you want to
                                target a physical radius on a grid that's
                                at least approximately uniform. ***

    Shared
    ------
    surf_alt_lim : float       Min AGL for valid echo [m].             Default: 200
    min_convectivity_for_convective : float                            Default: 0.5
    max_convectivity_for_stratiform : float                            Default: 0.4

    3-D volume/extent filters (from ConvStratFinder constructor)
    -------------------------------------------------------------
    min_valid_volume_for_convective : float   km³  Default: 20.0
    min_vert_extent_for_convective  : float   km   Default: 1.0

    3-D terrain AGL adjustments (from ConvStratParams)
    ---------------------------------------------------
    min_ht_km_agl_for_mid  : float   km   Default: 2.0
    min_ht_km_agl_for_deep : float   km   Default: 4.0

    3-D clumping (dual-threshold)
    -----------------------------
    use_dual_thresholds             : bool   Default: True
    secondary_convectivity          : float  Default: 0.65
    all_subclumps_min_area_frac     : float  Default: 0.33
    each_subclump_min_area_frac     : float  Default: 0.02
    each_subclump_min_area_km2      : float  Default: 2.0
    min_overlap_for_convective_clumps : int  Default: 1

    3-D sub-type thresholds (from ConvStratFinder constructor)
    ----------------------------------------------------------
    min_conv_fraction_for_deep              : float  Default: 0.05
    min_conv_fraction_for_shallow           : float  Default: 0.95
    max_shallow_conv_fraction_for_elevated  : float  Default: 0.05
    max_deep_conv_fraction_for_elevated     : float  Default: 0.25
    min_strat_fraction_for_strat_below      : float  Default: 0.9
    """

    # Basic (2D morphological) -- enlarge_mixed/enlarge_conv from MATLAB
    # f_classBasic.m, NOT ConvStratFinder (see module/class docstring).
    # PIXEL/GATE COUNTS, not physical distances -- see class docstring
    # and core/classification.py's "PIXEL COUNTS, NOT PHYSICAL UNITS".
    strat_mixed:  float = 0.4
    mixed_conv:   float = 0.5
    enlarge_mixed: int  = 5
    enlarge_conv:  int  = 5    # FIXED: was 3, confirmed against MATLAB
                                # source to be wrong; real MATLAB default
                                # is 5, matching enlarge_mixed
    surf_alt_lim:  float = 200.0

    # Point-wise convectivity thresholds
    min_convectivity_for_convective: float = 0.5
    max_convectivity_for_stratiform: float = 0.4

    # Volume / extent filters
    min_valid_volume_for_convective:  float = 20.0   # was 4.0 — FIXED
    min_vert_extent_for_convective:   float = 1.0    # was 1.5 — FIXED

    # Terrain AGL
    min_ht_km_agl_for_mid:  float = 2.0    # new
    min_ht_km_agl_for_deep: float = 4.0    # new

    # Dual-threshold clumping
    use_dual_thresholds:              bool  = True
    secondary_convectivity:           float = 0.65
    all_subclumps_min_area_frac:      float = 0.33
    each_subclump_min_area_frac:      float = 0.02
    each_subclump_min_area_km2:       float = 2.0
    min_overlap_for_convective_clumps: int  = 1      # new

    # Sub-type classification
    min_conv_fraction_for_deep:             float = 0.05   # was 0.1  — FIXED
    min_conv_fraction_for_shallow:          float = 0.95   # was 0.1  — FIXED
    max_shallow_conv_fraction_for_elevated: float = 0.05   # was 0.1  — FIXED
    max_deep_conv_fraction_for_elevated:    float = 0.25   # was 0.1  — FIXED
    min_strat_fraction_for_strat_below:     float = 0.9    # was 0.5  — FIXED
