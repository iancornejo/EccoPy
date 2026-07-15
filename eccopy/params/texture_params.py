"""Texture calculation parameters — defaults match ConvStratFinder constructor."""

from __future__ import annotations
from dataclasses import dataclass, field
from .window import WindowSpec


@dataclass
class TextureParams:
    """
    Parameters controlling texture calculation.

    1-D window (EccoPy-1D / EccoPy-2D)
    --------------------------------
    window_1d : WindowSpec      Half-width of sliding window (time or range).
    upper_lim_dbz : float       Texture value → convectivity=1 (linear). Default: 29.
    dbz_base : float            Subtracted before texture.                 Default: 0.

    2-D radial (EccoPy-3D) — all defaults from ConvStratFinder constructor
    -----------------------------------------------------------------------
    texture_radius       : WindowSpec  Circular neighbourhood radius. Default: 7 km.
    min_frac_texture     : float       Min coverage for texture.       Default: 0.25.
    min_frac_fit         : float       Min coverage for planar fit.    Default: 0.67.
    texture_limit_low    : float       Texture → conv=NaN below this.  Default: 0.
    texture_limit_high   : float       Texture → conv=1 above this.    Default: 30.
    use_dbz_col_max      : bool        Use col-max DBZ for texture
                                       (same texture copied to all levels).
                                       Default: False.
    min_valid_dbz        : float       DBZ below this → missing.       Default: 0.
    dbz_for_echo_tops    : float       DBZ threshold for echo tops.    Default: 18.
    """

    # 1-D
    window_1d:     WindowSpec = field(default_factory=lambda: WindowSpec(19))
    upper_lim_dbz: float = 29.0
    dbz_base:      float = 0.0

    # 3-D
    texture_radius:    WindowSpec = field(default_factory=lambda: WindowSpec((7.0, "km")))
    min_frac_texture:  float = 0.25
    min_frac_fit:      float = 0.67
    texture_limit_low: float = 0.0
    texture_limit_high: float = 30.0
    use_dbz_col_max:   bool  = False   # new
    min_valid_dbz:     float = 0.0
    dbz_for_echo_tops: float = 18.0    # new
