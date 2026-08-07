"""
Ground-truth tests for eccopy.core.disk.

The pure-Python disk generator must reproduce MATLAB's strel('disk', r, 4)
and its getsequence() decomposition bit-for-bit. These tests pin that
against the exported MATLAB .mat masks bundled under core/data/ - turning
those fixtures into regression guards - and check internal consistency
(the closed-form octagon equals the decomposition-built neighborhood)
across a wide radius range where no MATLAB export exists.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from eccopy.core.disk import (
    disk_neighborhood,
    disk_decomposition,
    neighborhood_from_decomposition,
    MATLAB_VERIFIED_STREL_RADII,
    MATLAB_VERIFIED_DECOMP_RADII,
)

_DATA = Path(__file__).resolve().parent.parent / "core" / "data"
_STRELS = _DATA / "disk_strels"
_DECOMP = _DATA / "disk_decomp"


@pytest.mark.parametrize("r", MATLAB_VERIFIED_STREL_RADII)
def test_neighborhood_bit_exact_vs_matlab(r):
    ref = loadmat(_STRELS / f"disk_strel_r{r}.mat")["nhood"].astype(bool)
    gen = disk_neighborhood(r)
    assert gen.shape == ref.shape
    assert np.array_equal(gen, ref)


@pytest.mark.parametrize("r", MATLAB_VERIFIED_DECOMP_RADII)
def test_decomposition_bit_exact_vs_matlab(r):
    info = loadmat(_DECOMP / f"disk_decomp_r{r}_info.mat")
    n_steps = int(info["n_steps"][0][0])
    ref = [
        loadmat(_DECOMP / f"disk_decomp_r{r}_step{k}.mat")["nhood_k"].astype(bool)
        for k in range(1, n_steps + 1)
    ]
    gen = disk_decomposition(r)
    assert len(gen) == len(ref)
    for k, (g, rf) in enumerate(zip(gen, ref), 1):
        assert g.shape == rf.shape, f"step {k} shape {g.shape} != {rf.shape}"
        assert np.array_equal(g, rf), f"step {k} contents differ"


@pytest.mark.parametrize("r", list(range(3, 41)))
def test_decomposition_reconstructs_neighborhood(r):
    # MATLAB derives .Neighborhood by dilating a point through the
    # sequence; the closed-form octagon must equal that reconstruction.
    assert np.array_equal(neighborhood_from_decomposition(r), disk_neighborhood(r))


@pytest.mark.parametrize("r", list(range(3, 41)))
def test_octagon_shape_invariants(r):
    n = disk_neighborhood(r)
    assert n.shape == (2 * r - 1, 2 * r - 1)          # MATLAB n=4 disk size
    assert np.array_equal(n, n[::-1])                 # top-bottom symmetric
    assert np.array_equal(n, n[:, ::-1])              # left-right symmetric
    assert np.array_equal(n, n.T)                     # transpose symmetric
    widths = n.sum(1)
    half = widths[: len(widths) // 2 + 1]
    assert np.all(np.diff(half) >= 0)                 # monotone taper to centre


def test_euclidean_option_differs_from_octagon():
    # n=0 is the true disk (2r+1), n=4 is the octagon (2r-1) - different by design.
    eu = disk_neighborhood(15, n=0)
    oc = disk_neighborhood(15, n=4)
    assert eu.shape == (31, 31)
    assert oc.shape == (29, 29)


def test_small_radius_warns():
    with pytest.warns(UserWarning):
        disk_neighborhood(2, n=4)
