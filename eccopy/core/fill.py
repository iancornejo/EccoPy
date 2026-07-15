"""Nearest-neighbour fill for eroded edge pixels."""

from __future__ import annotations
import numpy as np
from scipy.ndimage import label
from sklearn.neighbors import KDTree


def fill_regions_closest_pixel(VEL_mask: np.ndarray,
                                VEL_small: np.ndarray,
                                texture: np.ndarray,
                                VEL: np.ndarray) -> np.ndarray:
    """
    Fill eroded-edge pixels within each connected cloud region using the
    nearest valid (non-NaN) texture value in that same region.

    Port of MATLAB f_fillRegionsClosestPixel.m.

    Parameters
    ----------
    VEL_mask : np.ndarray of uint8
        Binary mask of originally valid pixels (1 = valid).
    VEL_small : np.ndarray of uint8
        Eroded mask (1 = survived erosion).
    texture : np.ndarray
        Texture array (NaN at eroded edges).
    VEL : np.ndarray
        Original velocity/data array — used only for shape/indexing.

    Returns
    -------
    texture : np.ndarray
        Texture with eroded pixels filled.
    """
    texture = texture.copy()
    labeled, n_objects = label(VEL_mask)

    for ii in range(1, n_objects + 1):
        cloud_flat = np.where((labeled == ii).ravel())[0]

        valid_flat = cloud_flat[~np.isnan(texture.ravel()[cloud_flat])]
        if len(valid_flat) == 0:
            continue

        to_fill_flat = cloud_flat[
            (VEL_mask.ravel()[cloud_flat] == 1) &
            (VEL_small.ravel()[cloud_flat] == 0)
        ]
        if len(to_fill_flat) == 0:
            continue

        rows_valid, cols_valid = np.unravel_index(valid_flat, VEL.shape)
        rows_fill, cols_fill = np.unravel_index(to_fill_flat, VEL.shape)

        tree = KDTree(np.column_stack([rows_valid, cols_valid]))
        _, idx = tree.query(np.column_stack([rows_fill, cols_fill]))
        idx = idx.ravel()

        texture[rows_fill, cols_fill] = texture[rows_valid[idx], cols_valid[idx]]

    return texture
