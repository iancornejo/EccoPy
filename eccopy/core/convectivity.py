"""
Texture → convectivity transfer functions.

Both functions exactly replicate ConvStratFinder::_computeConvectivity().
Key behavioural notes (from C++ source):
  - convectivity is NaN (missing) when texture < texture_limit_low, NOT 0
  - convectivity is NaN when fractionActive < minValidFractionForTexture
  - The 2-D linear case (EccoPy-1D/2D) uses a simpler direct scaling from MATLAB
"""

from __future__ import annotations
import numpy as np


def texture_to_convectivity_linear(texture: np.ndarray,
                                   upper_lim: float) -> np.ndarray:
    """
    Linear scaling used by EccoPy-1D and EccoPy-2D (EccoPy-V).

        convectivity = texture / upper_lim,  clipped to [0, 1]

    NaN where texture is NaN.
    """
    conv = texture / upper_lim
    return np.clip(conv, 0.0, 1.0)


def texture_to_convectivity_piecewise(texture: np.ndarray,
                                      fraction_active: np.ndarray,
                                      min_frac_texture: float,
                                      limit_low: float,
                                      limit_high: float) -> np.ndarray:
    """
    Piecewise linear transfer function used by EccoPy-3D.

    Exact port of ConvStratFinder::_computeConvectivity():

        if fractionActive < minValidFractionForTexture → NaN (missing)
        elif texture < limit_low                       → NaN (missing)
        elif texture > limit_high                      → 1.0
        else                                           → (texture - low) / (high - low)

    Note: texture below limit_low gives NaN, NOT zero. This matches the C++.

    Parameters
    ----------
    texture : np.ndarray (any shape)
    fraction_active : np.ndarray, same shape as first two dims of texture
        Coverage fraction from col-max DBZ. Shape (nx, ny) broadcast over nz.
    min_frac_texture : float
        Minimum coverage fraction; points below → NaN convectivity.
    limit_low, limit_high : float
    """
    span = limit_high - limit_low
    if span <= 0:
        raise ValueError(f"limit_high ({limit_high}) must be > limit_low ({limit_low}).")

    texture = np.asarray(texture, dtype=float)
    fraction_active = np.asarray(fraction_active, dtype=float)

    # Broadcast fraction_active over z if needed
    if texture.ndim == 3 and fraction_active.ndim == 2:
        fa = fraction_active[:, :, np.newaxis]  # (nx, ny, 1)
    else:
        fa = fraction_active

    conv = np.full_like(texture, np.nan)

    active = fa >= min_frac_texture
    above_low = texture >= limit_low
    above_high = texture >= limit_high

    # Piecewise linear where active and texture in range
    in_range = active & above_low & ~above_high
    conv[in_range] = (texture[in_range] - limit_low) / span

    # Cap at 1.0 where active and above upper limit
    conv[active & above_high] = 1.0

    # Where active but below lower limit: stays NaN (missing), per C++
    # Where not active: stays NaN

    return conv
