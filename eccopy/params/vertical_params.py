"""Vertical level parameters - defaults match ConvStratFinder constructor."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class VerticalParams:
    """
    Vertical level / temperature threshold parameters.

    *** SCOPE: this is for the 3-D path (set_echo_type_3d /
    ConvStratFinder) only. EccoPy-2D-V's class_sub_2d() does NOT use
    these fields -- the real 2-D algorithm (f_classSub.m) hardcodes a
    melt threshold of 15 and a temperature threshold of -25 deg C
    internally, not as configurable parameters. Passing VerticalParams
    into class_sub_2d was a real bug found and fixed in a prior session
    -- see core/classification.py's module docstring "VALIDATION
    HISTORY" section. Do not reintroduce that coupling. ***

    vert_levels_type : 'by_temp' | 'by_height'
        Default: 'by_height'  (matches C++ VERT_LEVELS_BY_HT default)

    shallow_threshold_temp : float   °C   Default:   0.0
    deep_threshold_temp    : float   °C   Default: -12.0  (was -25 - FIXED)

    shallow_threshold_ht   : float   km   Default:  4.5   (was 4.0 - FIXED)
    deep_threshold_ht      : float   km   Default:  9.0

    min_valid_height : float   km   Default:  0.0
    max_valid_height : float   km   Default: 25.0
        Vertical band the analysis is restricted to. Levels whose height
        falls outside are treated as missing before texture is computed,
        matching ConvStratFinder. The defaults span any realistic radar or
        model grid, so they are inert unless deliberately narrowed.

    The reflectivity floor lives on TextureParams.min_valid_dbz, not
    here.

    Values marked FIXED below are taken from the C++ ConvStratFinder
    defaults rather than being tunable in the reference.
    """

    vert_levels_type: Literal["by_temp", "by_height"] = "by_height"  # FIXED

    shallow_threshold_temp: float = 0.0
    deep_threshold_temp:    float = -12.0   # was -25.0 - FIXED

    shallow_threshold_ht:   float = 4.5    # was 4.0 - FIXED
    deep_threshold_ht:      float = 9.0

    min_valid_height: float = 0.0
    max_valid_height: float = 25.0
