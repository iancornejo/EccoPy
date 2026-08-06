"""Texture calculation parameters — defaults match ConvStratFinder constructor."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
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
    min_valid_dbz        : float|None  DBZ below this → missing, applied
                                       before texture is computed.
                                       Default: None = use each module's
                                       reference default (see below).
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
    min_valid_dbz:     Optional[float] = None
    dbz_for_echo_tops: float = 18.0    # new

    def resolve_min_valid_dbz(self, module_default: float) -> float:
        """
        Resolve `min_valid_dbz` for a calling module.

        `None` means "use this module's reference default", which differs
        because the parameter has a different provenance in each family:

          eccopy2d_h / eccopy3d : 0.0
              ConvStratFinder::computeEchoType() prefilters the volume at
              this threshold before computing anything downstream.
          eccopy1d / eccopy2d_v : -inf (no gating)
              f_reflTexture.m has no equivalent prefilter. EccoPy-2D-V
              agrees with real MATLAB ECCO-V output at 99.4-100% without
              one; gating at 0 dBZ moves that output by roughly 13%.

        Any value the caller sets explicitly is honoured by every module.
        """
        if self.min_valid_dbz is None:
            return module_default
        return float(self.min_valid_dbz)
