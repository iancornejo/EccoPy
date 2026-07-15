"""Frozen pre-Numba reference implementation, for cross-validation only.

*** UPDATED: ddof=0 -> ddof=1 ***. This reference was originally frozen
before the "sample variance (N-1), matching MATLAB's std(..., 'omitnan')
default" fix landed in the production core (see core/texture.py's
_sliding_texture_core docstring/comments for the validation history --
found against real MATLAB output, moved downstream classification
agreement from 98.2% to 99.4%). That fix only touched production; this
frozen copy was never updated to match, so it was silently comparing the
CORRECT, MATLAB-validated Numba core against the OLD, disproven ddof=0
formula -- the opposite of a conversion bug, but functionally identical
as a test failure. Updated here so this file goes back to doing its job:
catching real Numba-conversion discrepancies against the CURRENT
validated formula, not flagging a difference that production was
correct to introduce.
"""
import numpy as np

def sliding_texture_core_reference(data_padded, radius_field, pad, nRows, nCols):
    texture = np.full((nRows, nCols), np.nan)

    for ii in range(nCols):
        r = int(radius_field[ii])
        if r < 1:
            continue
        lo = pad + ii - r
        hi = pad + ii + r + 1
        block = data_padded[:, lo:hi]
        W = block.shape[1]
        if W < 2:
            continue

        x1 = np.arange(1, W + 1, dtype=float)
        X = np.tile(x1, (nRows, 1))

        sumX = np.sum(X, axis=1)
        sumY = np.nansum(block, axis=1)
        sumXY = np.nansum(block * X, axis=1)
        sumX2 = np.sum(X ** 2, axis=1)

        denom = W * sumX2 - sumX ** 2
        safe = denom != 0
        a = np.where(safe, (sumY * sumX2 - sumX * sumXY) / np.where(safe, denom, 1), 0)
        b = np.where(safe, (W * sumXY - sumX * sumY) / np.where(safe, denom, 1), 0)

        newY = a[:, None] + b[:, None] * X
        mean_block = np.nanmean(block, axis=1, keepdims=True)
        corrected = block - newY + mean_block
        corrected[corrected < 1] = 1

        # Sample variance (N-1), NaNs dropped per row -- matches
        # production's _sliding_texture_core exactly (see module
        # docstring). np.nanstd's ddof=1 already returns NaN for a
        # row with < 2 non-NaN values, and 0.0 is not produced for
        # exactly 1 valid value the way the scalar-loop production code
        # special-cases it -- see NUMERICAL NOTE in test_numba_cross_
        # validation.py for why this can't perfectly agree pointwise
        # in that specific edge case; the atol=0.01 tolerance covers it.
        # Sample variance (N-1), NaNs dropped per row -- matches
        # production's _sliding_texture_core exactly (see module
        # docstring). np.nanstd's ddof=1 divides by (n_valid - 1), which
        # is undefined (NaN, with a RuntimeWarning) for a row with
        # EXACTLY 1 non-NaN value -- but production explicitly special-
        # cases that to var=0.0 ("matches MATLAB's std() of a single
        # sample == 0", see core/texture.py). Without this override,
        # rows with exactly 1 valid point in the window would silently
        # come out NaN here but 0.0 in production -- a real, position-
        # dependent NaN-vs-value mismatch, not just a rounding
        # difference, so it isn't covered by the atol=0.01 numerical
        # tolerance the way near-zero-variance rounding noise is (see
        # NUMERICAL NOTE in test_numba_cross_validation.py).
        n_valid_row = np.sum(~np.isnan(corrected), axis=1)
        row_texture = np.sqrt(np.nanstd(corrected ** 2, axis=1, ddof=1))
        row_texture = np.where(n_valid_row == 1, 0.0, row_texture)
        texture[:, ii] = row_texture

    return texture
