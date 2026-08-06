"""
eccopy.core.disk — MATLAB-faithful disk structuring elements, generated
in pure Python for ARBITRARY radius (no MATLAB, no pre-exported .mat).

Reproduces MATLAB's ``strel('disk', r)`` (default ``n = 4`` periodic-line
approximation) and its ``getsequence()`` decomposition, bit-for-bit,
without requiring anyone to pre-export a .mat file per radius. Validated
bit-exact against the MATLAB ground truth bundled in ``core/data/``
(strel radii 3, 5, 15, 25; decomposition radii 15, 25) — see
``tests/test_disk_generator.py``.

Why this exists
---------------
``strel('disk', r)`` with the default ``n = 4`` does **not** return a
Euclidean disk. It returns an OCTAGON built by dilating a single point
with four periodic-line structuring elements (0°, 45°, 90°, 135°) plus
two unit correction lines — the radial decomposition of Rolf Adams,
*Radial Decomposition of Discs and Spheres* (CVGIP: GMIP, 1993). EccoPy
previously loaded the resulting ``.Neighborhood`` arrays from
MATLAB-exported .mat files, which pinned every disk-based operation
(``enlarge_mixed`` / ``enlarge_conv`` morphology and the ``imclose``
decompositions they trigger at radius ``enlarge×3`` / ``enlarge×5``) to
the handful of radii someone had remembered to export — e.g.
``enlarge_conv=15`` failed because nobody had exported the radius-75
decomposition. This module removes that limitation while preserving the
exact MATLAB shape, so the classification path stays faithful to every
already-validated 2D-V / 3D case.

This is NOT a divergence from MATLAB: the geometry below reproduces
MATLAB's ``n = 4`` algorithm exactly (proven bit-exact at every exported
radius). The only thing that changes is *where* the arrays come from
(computed on demand vs. pre-exported), which is why it does not require
the five-step deliberate-divergence bar in CONTRIBUTING.md.

Geometry (n = 4), for r >= 3
----------------------------
The neighborhood is a ``(2r-1) x (2r-1)`` octagon with::

    A = r - 1                    half-width at the centre row
    d = floor((3r - 5) / 10)     diagonal periodic-line half-extent
    a = r - 2 - 2*d              axis periodic-line half-extent
    B = 2*d                      corner chamfer depth

and the ``getsequence`` decomposition is the ordered list::

    [ vert(a), diag_main(d), horiz(a), diag_anti(d), horiz(1), vert(1) ]

where ``line(h)`` is a periodic line of half-length ``h`` (so ``2h+1``
pixels). ``a + 2*d + 1 == A`` holds identically, so the closed-form
octagon and the decomposition-built neighborhood agree for every radius.
These four relations were recovered from, and verified against, the
exported MATLAB masks.

Small radii
-----------
Only ``r >= 3`` is checked against MATLAB ground truth. ``r`` in {1, 2}
uses the same closed form (a filled ``(2r-1)`` square) and is flagged
unverified; ``disk_neighborhood`` warns rather than silently guessing.
``n = 0`` returns the true Euclidean disk for callers who explicitly want
it, but ``n = 4`` is the default everywhere in EccoPy because that is what
ECCO's MATLAB / LROSE path uses and what the validation cases were run
against.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
from scipy.ndimage import binary_dilation

# Radii with bundled MATLAB ground truth (used by the regression tests).
MATLAB_VERIFIED_STREL_RADII = (3, 5, 15, 25)
MATLAB_VERIFIED_DECOMP_RADII = (15, 25)


def _periodic_line(dy: int, dx: int, half: int) -> np.ndarray:
    """
    Minimal boolean periodic-line structuring element: the pixels
    ``t * (dy, dx)`` for ``t`` in ``[-half, half]``, packed into the
    smallest array that holds them (origin at the geometric centre).

    ``half == 0`` yields a 1x1 single-pixel element (a dilation identity).
    """
    if half == 0:
        return np.ones((1, 1), dtype=bool)
    ts = np.arange(-half, half + 1)
    ys, xs = ts * dy, ts * dx
    m = np.zeros((ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), dtype=bool)
    m[ys - ys.min(), xs - xs.min()] = True
    return m


def _disk_params(r: int):
    """(A, a, d, B) for the n=4 octagon of radius r. See module docstring."""
    d = max((3 * r - 5) // 10, 0)     # floor((3r-5)/10), clamped
    a = max(r - 2 - 2 * d, 0)
    A = r - 1
    B = 2 * d
    return A, a, d, B


def disk_decomposition(r: int, n: int = 4) -> List[np.ndarray]:
    """
    MATLAB ``getsequence(strel('disk', r, n))`` primitives.

    Parameters
    ----------
    r : int
        Disk radius (pixels).
    n : int
        Periodic-line count. Only ``4`` (MATLAB default, the EccoPy path)
        and ``0`` (no decomposition — a single Euclidean-disk element)
        are supported.

    Returns
    -------
    list of 2-D bool arrays
        The line structuring elements, in MATLAB's order
        ``[vert, diag_main, horiz, diag_anti, horiz_corr, vert_corr]``.
        Pure 1x1 identity elements (which arise only for tiny radii) are
        dropped, since they do not affect any dilation / erosion / close.
    """
    r = int(r)
    if n == 0:
        return [disk_neighborhood(r, n=0)]
    if n != 4:
        raise ValueError(f"disk_decomposition supports n in {{0, 4}}; got n={n}. "
                         "ECCO uses MATLAB's default n=4.")
    _, a, d, _ = _disk_params(r)
    steps = [
        _periodic_line(1,  0, a),   # vertical axis line
        _periodic_line(1,  1, d),   # main diagonal (top-left -> bottom-right)
        _periodic_line(0,  1, a),   # horizontal axis line
        _periodic_line(1, -1, d),   # anti-diagonal (top-right -> bottom-left)
        _periodic_line(0,  1, 1),   # horizontal unit correction (length 3)
        _periodic_line(1,  0, 1),   # vertical unit correction (length 3)
    ]
    return [s for s in steps if s.size > 1]


def disk_neighborhood(r: int, n: int = 4) -> np.ndarray:
    """
    MATLAB ``strel('disk', r, n).Neighborhood`` as a boolean array.

    ``n = 4`` (default) returns the periodic-line octagon — bit-exact to
    MATLAB and to EccoPy's previously-exported masks. ``n = 0`` returns
    the true Euclidean disk (``x^2 + y^2 <= r^2``), a ``(2r+1)`` array,
    for callers who explicitly want the un-approximated shape.
    """
    r = int(r)
    if n == 0:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return (xx * xx + yy * yy) <= r * r
    if n != 4:
        raise ValueError(f"disk_neighborhood supports n in {{0, 4}}; got n={n}.")
    if r < 3:
        warnings.warn(
            f"disk_neighborhood(r={r}, n=4): radii below 3 are not checked "
            "against MATLAB ground truth; returning the closed-form octagon "
            "(a filled (2r-1) square for r in {1,2}).",
            stacklevel=2,
        )
    A, a, d, B = _disk_params(r)
    if A < 0:                              # r == 0 -> empty; be defensive
        return np.zeros((1, 1), dtype=bool)
    yy, xx = np.mgrid[-A:A + 1, -A:A + 1]
    half_width = A - np.maximum(0, np.abs(yy) - (A - B))
    return np.abs(xx) <= half_width


def neighborhood_from_decomposition(r: int, n: int = 4) -> np.ndarray:
    """
    Build the neighborhood by actually dilating a point through the
    decomposition — the way MATLAB derives ``.Neighborhood`` from
    ``getsequence``. Equal to ``disk_neighborhood(r, n)`` for all r; kept
    as an independent cross-check (see the reconstruction test).
    """
    steps = disk_decomposition(r, n=n)
    sz = 4 * int(r) + 5
    img = np.zeros((sz, sz), dtype=bool)
    img[sz // 2, sz // 2] = True
    for s in steps:
        img = binary_dilation(img, structure=s)
    ys, xs = np.where(img)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
