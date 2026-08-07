"""Texture calculation parameters - defaults match ConvStratFinder constructor."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextureParams:
    """
    Parameters controlling texture calculation.

    The texture window itself is a `run()` argument, not a field here:
    every module takes `window=WindowSpec(...)`.

    Defaults follow the ConvStratFinder constructor.
    ------------------------------------------------
    dbz_base             : float       Subtracted before texture.      Default: 0.
    min_frac_texture     : float       Min coverage for texture.       Default: 0.25.
    min_frac_fit         : float       Min coverage for planar fit.    Default: 0.67.
    texture_limit_low    : float       Texture below this -> convectivity
                                       is missing. Convectivity is scaled
                                       across [low, high].            Default: 0.
    texture_limit_high   : float       Texture -> conv=1 above this.   Default: 30.
    use_dbz_col_max      : bool        Compute texture once from the
                                       column-maximum reflectivity and
                                       copy it to every level
                                       (eccopy3d only).               Default: False.
    min_valid_dbz        : float|None  DBZ below this → missing, applied
                                       before texture is computed.
                                       Default: None = use each module's
                                       reference default (see below).
    dbz_for_echo_tops    : float       Reflectivity threshold defining
                                       echo top; populates Result3D.
                                       echo_top_km.                   Default: 18.
    """

    dbz_base:          float = 0.0
    min_frac_texture:  float = 0.25
    min_frac_fit:      float = 0.67
    texture_limit_low: float = 0.0
    texture_limit_high: float = 30.0
    use_dbz_col_max:   bool  = False
    min_valid_dbz:     Optional[float] = None
    dbz_for_echo_tops: float = 18.0

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
