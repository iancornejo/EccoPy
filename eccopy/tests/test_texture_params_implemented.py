"""
Coverage for the TextureParams fields wired up in v1.0.

`texture_limit_low`, `use_dbz_col_max` and `dbz_for_echo_tops` were
declared and documented but read by no code. Each is now consumed, and
each is inert at its default so existing results are unchanged.
"""
import dataclasses

import numpy as np
import pytest

from eccopy import eccopy3d
from eccopy.core.convectivity import texture_to_convectivity_linear
from eccopy.params import TextureParams, ClassificationParams, WindowSpec

CLASSES = [14, 16, 18, 25, 30, 32, 34, 36, 38]


def _volume(nz=10, ny=26, nx=26, seed=13):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.5, 10.0, nz)
    y = np.linspace(-13, 13, ny)
    x = np.linspace(-13, 13, nx)
    dbz = 14 - 0.9 * z[:, None, None] + rng.normal(0, 1.5, (nz, ny, nx))
    for iz in range(nz):
        dbz[iz, 9:17, 9:17] += 34 * np.exp(-((iz - 3) ** 2) / 9)
    height = np.broadcast_to(z[:, None, None], dbz.shape).copy()
    return dbz, z, y, x, height


def _run(dbz, z, y, x, **kw):
    return eccopy3d.run(dbz, coords_z=z, coords_y=y, coords_x=x,
                        window=WindowSpec((5, "km")), **kw)


# --------------------------------------------------------- removed fields

@pytest.mark.parametrize("name", ["window_1d", "texture_radius", "upper_lim_dbz"])
def test_redundant_window_fields_are_gone(name):
    """Superseded by the `window` argument every run() already takes."""
    assert name not in {f.name for f in dataclasses.fields(TextureParams)}
    with pytest.raises(TypeError):
        TextureParams(**{name: 5})


@pytest.mark.parametrize("name", ["strat_mixed", "mixed_conv"])
def test_duplicate_threshold_aliases_are_gone(name):
    """These shadowed max/min_convectivity_* with identical defaults."""
    assert name not in {f.name for f in dataclasses.fields(ClassificationParams)}
    with pytest.raises(TypeError):
        ClassificationParams(**{name: 0.3})


# ------------------------------------------------------ texture_limit_low

def test_convectivity_default_low_limit_is_plain_ratio():
    texture = np.array([0.0, 15.0, 30.0, 45.0, np.nan])
    got = texture_to_convectivity_linear(texture, upper_lim=30.0)
    np.testing.assert_allclose(got, [0.0, 0.5, 1.0, 1.0, np.nan])


def test_convectivity_low_limit_rescales_and_masks():
    """Below lower_lim is missing, not zero - matching ConvStratFinder."""
    texture = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    got = texture_to_convectivity_linear(texture, upper_lim=30.0, lower_lim=5.0)
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], [0.0, 0.2, 0.6, 1.0])


def test_convectivity_rejects_inverted_limits():
    with pytest.raises(ValueError, match="must exceed"):
        texture_to_convectivity_linear(np.array([1.0]), upper_lim=10.0, lower_lim=20.0)


def test_texture_limit_low_reaches_the_pipeline():
    dbz, z, y, x, _ = _volume()
    base = _run(dbz, z, y, x).echo_type
    raised = _run(dbz, z, y, x,
                  texture_params=TextureParams(texture_limit_low=8.0)).echo_type
    mask = np.isin(base, CLASSES) | np.isin(raised, CLASSES)
    assert not np.array_equal(base[mask], raised[mask])


# ------------------------------------------------------- use_dbz_col_max

def test_use_dbz_col_max_default_false_is_per_level():
    dbz, z, y, x, _ = _volume()
    default = _run(dbz, z, y, x).texture
    explicit = _run(dbz, z, y, x,
                    texture_params=TextureParams(use_dbz_col_max=False)).texture
    np.testing.assert_array_equal(np.nan_to_num(default, nan=-9e9),
                                  np.nan_to_num(explicit, nan=-9e9))


def test_use_dbz_col_max_gives_one_texture_per_column():
    """Every echo-bearing level in a column shares the column-max texture."""
    dbz, z, y, x, _ = _volume()
    r = _run(dbz, z, y, x, texture_params=TextureParams(use_dbz_col_max=True))

    finite = np.isfinite(r.texture)
    columns = finite.sum(axis=0) > 1
    assert columns.any(), "test volume has no multi-level columns"

    hi = np.where(finite, r.texture, -np.inf).max(axis=0)
    lo = np.where(finite, r.texture, np.inf).min(axis=0)
    assert (hi[columns] - lo[columns]).max() == pytest.approx(0.0, abs=1e-9)


def test_use_dbz_col_max_differs_from_per_level():
    dbz, z, y, x, _ = _volume()
    per_level = _run(dbz, z, y, x).texture
    col_max = _run(dbz, z, y, x,
                   texture_params=TextureParams(use_dbz_col_max=True)).texture
    both = np.isfinite(per_level) & np.isfinite(col_max)
    assert both.any()
    assert not np.allclose(per_level[both], col_max[both])


# ----------------------------------------------------- dbz_for_echo_tops

def test_echo_top_km_is_none_without_height():
    dbz, z, y, x, _ = _volume()
    assert _run(dbz, z, y, x).echo_top_km is None


def test_echo_top_km_is_the_highest_level_reaching_the_threshold():
    dbz, z, y, x, height = _volume()
    tp = TextureParams(dbz_for_echo_tops=18.0)
    r = _run(dbz, z, y, x, height=height, texture_params=tp)

    assert r.echo_top_km.shape == dbz.shape[1:]
    reaches = np.isfinite(dbz) & (dbz >= 18.0)
    expected = np.where(reaches.any(axis=0),
                        np.where(reaches, height, -np.inf).max(axis=0),
                        np.nan)
    np.testing.assert_allclose(r.echo_top_km, expected, equal_nan=True)


def test_raising_dbz_for_echo_tops_lowers_the_tops():
    dbz, z, y, x, height = _volume()
    low = _run(dbz, z, y, x, height=height,
               texture_params=TextureParams(dbz_for_echo_tops=5.0)).echo_top_km
    high = _run(dbz, z, y, x, height=height,
                texture_params=TextureParams(dbz_for_echo_tops=35.0)).echo_top_km
    both = np.isfinite(low) & np.isfinite(high)
    assert both.any()
    assert np.all(high[both] <= low[both])
