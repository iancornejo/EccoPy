"""Vertical level parameters — defaults match ConvStratFinder constructor."""

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
    deep_threshold_temp    : float   °C   Default: -12.0  (was -25 — FIXED)

    shallow_threshold_ht   : float   km   Default:  4.5   (was 4.0 — FIXED)
    deep_threshold_ht      : float   km   Default:  9.0

    min_valid_height : float   km   Default:  0.0
    max_valid_height : float   km   Default: 25.0
    min_valid_dbz    : float   dBZ  Default:  0.0

    NOTE on the "-25 — FIXED" / "4.0 — FIXED" comments above: these
    describe changes made against the 3-D C++ ConvStratFinder reference
    and have NOT been independently re-verified against real reference
    output this session (unlike the 2-D MATLAB findings referenced
    above, which were checked against real ECCO-V output). Treat the
    3-D path as unvalidated regardless of what these comments say.
    """

    vert_levels_type: Literal["by_temp", "by_height"] = "by_height"  # FIXED

    shallow_threshold_temp: float = 0.0
    deep_threshold_temp:    float = -12.0   # was -25.0 — FIXED

    shallow_threshold_ht:   float = 4.5    # was 4.0 — FIXED
    deep_threshold_ht:      float = 9.0

    min_valid_height: float = 0.0
    max_valid_height: float = 25.0
    min_valid_dbz:    float = 0.0
