"""
min_valid_dbz resolution across the four modules.

The parameter ports ConvStratFinder::computeEchoType()'s prefilter, which
has no analog in f_reflTexture.m. Its default is therefore None, resolved
per module: 0.0 for the C++-derived 2-D-H / 3-D paths, no gating for the
MATLAB-derived 1-D / 2-D-V paths. An explicitly-set value is honoured by
all four.
"""
import numpy as np
import pytest

from eccopy import eccopy1d, eccopy2d_h, eccopy2d_v, eccopy3d
from eccopy.core.texture import refl_texture_1d, refl_texture_1d_with_fit
from eccopy.params import WindowSpec, TextureParams


def _same(a, b):
    """Array comparison that treats NaN as equal (2-D-H/2-D-V use NaN)."""
    if a.dtype.kind == "f":
        return np.array_equal(np.nan_to_num(a, nan=-9e9),
                              np.nan_to_num(b, nan=-9e9))
    return np.array_equal(a, b)


# ---------------------------------------------------------------- resolution

def test_default_is_none():
    assert TextureParams().min_valid_dbz is None


def test_none_resolves_to_the_calling_module_default():
    tp = TextureParams()
    assert tp.resolve_min_valid_dbz(0.0) == 0.0
    assert tp.resolve_min_valid_dbz(-np.inf) == -np.inf


def test_explicit_value_overrides_every_module_default():
    tp = TextureParams(min_valid_dbz=7.5)
    assert tp.resolve_min_valid_dbz(0.0) == 7.5
    assert tp.resolve_min_valid_dbz(-np.inf) == 7.5


@pytest.mark.parametrize("value", [0.0, -30.0, 12.0])
def test_explicit_zero_is_not_treated_as_unset(value):
    """0.0 is falsy; it must not be confused with None."""
    assert TextureParams(min_valid_dbz=value).resolve_min_valid_dbz(-np.inf) == value


# ------------------------------------------------------------------ 1-D core

def _ramp(n=60, seed=3):
    rng = np.random.default_rng(seed)
    dbz = np.linspace(-20, 40, n) + rng.normal(0, 2, n)
    return dbz


def test_refl_texture_1d_default_does_not_gate():
    dbz = _ramp()
    ungated = refl_texture_1d(dbz, WindowSpec(5))
    explicit = refl_texture_1d(dbz, WindowSpec(5), min_valid_dbz=-np.inf)
    assert _same(ungated, explicit)


def test_refl_texture_1d_gates_when_set():
    dbz = _ramp()
    ungated = refl_texture_1d(dbz, WindowSpec(5))
    gated = refl_texture_1d(dbz, WindowSpec(5), min_valid_dbz=0.0)
    assert not _same(ungated, gated)


def test_refl_texture_1d_with_fit_agrees_with_plain_when_gated():
    """The debug entry point must gate identically to the production one."""
    dbz = _ramp()
    plain = refl_texture_1d(dbz, WindowSpec(5), min_valid_dbz=5.0)
    tex, _, _ = refl_texture_1d_with_fit(dbz, WindowSpec(5), min_valid_dbz=5.0)
    assert _same(plain, tex)


# --------------------------------------------------------------- end-to-end

def _profile(n=200, seed=5):
    rng = np.random.default_rng(seed)
    dbz = np.full(n, -15.0) + rng.normal(0, 3, n)
    dbz[80:120] += 45 * np.exp(-0.5 * ((np.arange(-20, 20)) / 6) ** 2)
    return dbz, np.arange(n, dtype=float)


def _volume(nz=8, ny=24, nx=24, seed=6):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.5, 8.0, nz)
    y = np.linspace(-12, 12, ny)
    x = np.linspace(-12, 12, nx)
    dbz = 5 - 0.9 * z[:, None, None] + rng.normal(0, 2, (nz, ny, nx))
    for iz in range(nz):
        dbz[iz, 9:15, 9:15] += 35 * np.exp(-((iz - 2) ** 2) / 8)
    return dbz, z, y, x


def test_eccopy1d_default_matches_no_gating():
    dbz, coords = _profile()
    default = eccopy1d.run(dbz, coords=coords, window=WindowSpec(9))
    explicit = eccopy1d.run(dbz, coords=coords, window=WindowSpec(9),
                            texture_params=TextureParams(min_valid_dbz=-np.inf))
    assert _same(default.echo_type, explicit.echo_type)


def test_eccopy1d_honours_an_explicit_value():
    dbz, coords = _profile()
    default = eccopy1d.run(dbz, coords=coords, window=WindowSpec(9))
    gated = eccopy1d.run(dbz, coords=coords, window=WindowSpec(9),
                         texture_params=TextureParams(min_valid_dbz=0.0))
    assert not _same(default.echo_type, gated.echo_type)


def test_eccopy2d_v_default_matches_no_gating():
    dbz, z, _, x = _volume()
    dbz2 = dbz[:, 12, :]
    height = np.broadcast_to(z[:, None], dbz2.shape).copy()
    temp = 20 - 6.5 * height
    melt = np.where(temp <= 0, 20.0, 10.0)
    kw = dict(coords_z=z, coords_x=x, height=height, temp=temp, melt=melt,
              window=WindowSpec(5))
    default = eccopy2d_v.run(dbz2, **kw)
    explicit = eccopy2d_v.run(dbz2, texture_params=TextureParams(min_valid_dbz=-np.inf), **kw)
    assert _same(default.echo_type, explicit.echo_type)


def test_eccopy3d_default_matches_explicit_zero():
    """The 3-D default is 0.0, matching ConvStratFinder -- not no-gating."""
    dbz, z, y, x = _volume()
    kw = dict(coords_z=z, coords_y=y, coords_x=x, window=WindowSpec((5, "km")))
    default = eccopy3d.run(dbz, **kw)
    explicit = eccopy3d.run(dbz, texture_params=TextureParams(min_valid_dbz=0.0), **kw)
    assert _same(default.echo_type, explicit.echo_type)


def test_eccopy2d_h_default_matches_explicit_zero():
    dbz, _, y, x = _volume()
    field = np.nanmax(dbz, axis=0)
    kw = dict(coords_y=y, coords_x=x, window=WindowSpec((5, "km")))
    default = eccopy2d_h.run(field, **kw)
    explicit = eccopy2d_h.run(field, texture_params=TextureParams(min_valid_dbz=0.0), **kw)
    assert _same(default.echo_type, explicit.echo_type)
