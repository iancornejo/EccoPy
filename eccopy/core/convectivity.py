"""
Texture to convectivity transfer function.

Port of ConvStratFinder::_computeConvectivity(). Convectivity is missing
rather than zero where texture falls below texture_limit_low. The
coverage condition the C++ applies alongside it - missing where
fractionActive < minValidFractionForTexture - is applied upstream by
refl_texture_2d(), which returns NaN texture at those points.
"""

from __future__ import annotations
import numpy as np


def texture_to_convectivity_linear(texture: np.ndarray,
                                   upper_lim: float,
                                   lower_lim: float = 0.0) -> np.ndarray:
    """
    Linear scaling of texture onto convectivity.

        texture <  lower_lim   -> NaN (missing)
        texture >  upper_lim   -> 1.0
        otherwise              -> (texture - lower_lim) / (upper_lim - lower_lim)

    Port of the convectivity loop in ConvStratFinder::_computeConvectivity().
    At the default lower_lim of 0 this reduces to texture / upper_lim
    clipped to [0, 1].

    NaN where texture is NaN.
    """
    span = upper_lim - lower_lim
    if span <= 0:
        raise ValueError(
            f"texture_limit_high ({upper_lim}) must exceed texture_limit_low "
            f"({lower_lim})"
        )
    conv = (texture - lower_lim) / span
    conv = np.clip(conv, 0.0, 1.0)
    # Below lower_lim is missing, not zero - matching the C++, which leaves
    # convectivity at its missing value rather than flooring it.
    if lower_lim > 0.0:
        conv = np.where(texture < lower_lim, np.nan, conv)
    return conv
