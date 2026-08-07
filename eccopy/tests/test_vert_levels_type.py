"""
VerticalParams.vert_levels_type selects the field used to assign vertical
levels in the 3-D path.

find_clumps_3d() and set_echo_type_3d() both prefer `height` whenever it is
supplied, so honouring "by_temp" means withholding height from them. The
parameter is declared but was read by no code, making it a silent
no-op.
"""
import warnings

import numpy as np
import pytest

from eccopy import eccopy3d
from eccopy.params import WindowSpec, VerticalParams

CLASSIFIED = [14, 16, 18, 25, 30, 32, 34, 36, 38]


def _volume(nz=12, ny=28, nx=28, seed=11):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.5, 11.0, nz)
    y = np.linspace(-14, 14, ny)
    x = np.linspace(-14, 14, nx)
    dbz = 12 - 0.9 * z[:, None, None] + rng.normal(0, 1.5, (nz, ny, nx))
    for iz in range(nz):
        dbz[iz, 10:18, 10:18] += 34 * np.exp(-((iz - 3) ** 2) / 10)
    height = np.broadcast_to(z[:, None, None], dbz.shape).copy()
    temp = 22.0 - 6.5 * height
    return dbz, z, y, x, height, temp


def _run(dbz, z, y, x, **kw):
    return eccopy3d.run(dbz, coords_z=z, coords_y=y, coords_x=x,
                        window=WindowSpec((5, "km")), **kw)


def _differs(a, b):
    m = np.isin(a, CLASSIFIED) | np.isin(b, CLASSIFIED)
    return not np.array_equal(a[m], b[m])


def test_default_is_by_height():
    assert VerticalParams().vert_levels_type == "by_height"


def test_by_height_default_emits_no_warning():
    dbz, z, y, x, height, temp = _volume()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(dbz, z, y, x, height=height, temp=temp)
    assert not [w for w in caught if "vert_levels_type" in str(w.message)]


def test_by_temp_changes_the_result_when_height_is_also_given():
    """The bug: this used to be a no-op because height always won."""
    dbz, z, y, x, height, temp = _volume()
    by_height = _run(dbz, z, y, x, height=height, temp=temp).echo_type
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_temp = _run(dbz, z, y, x, height=height, temp=temp,
                       vert_params=VerticalParams(vert_levels_type="by_temp")).echo_type
    assert _differs(by_height, by_temp)


def test_by_temp_ignores_height_entirely():
    """Passing height under by_temp must equal not passing it at all."""
    dbz, z, y, x, height, temp = _volume()
    vp = VerticalParams(vert_levels_type="by_temp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_height = _run(dbz, z, y, x, height=height, temp=temp, vert_params=vp).echo_type
    without_height = _run(dbz, z, y, x, temp=temp, vert_params=vp).echo_type
    assert np.array_equal(with_height, without_height)


def test_by_temp_warns_that_height_is_unused():
    dbz, z, y, x, height, temp = _volume()
    with pytest.warns(UserWarning, match="is NOT used"):
        _run(dbz, z, y, x, height=height, temp=temp,
             vert_params=VerticalParams(vert_levels_type="by_temp"))


def test_by_temp_without_temp_raises():
    dbz, z, y, x, height, _ = _volume()
    with pytest.raises(ValueError, match="requires a temp field"):
        _run(dbz, z, y, x, height=height,
             vert_params=VerticalParams(vert_levels_type="by_temp"))


def test_by_height_without_height_falls_back_to_temp_silently():
    """Passing temp alone is an unambiguous request, not a mistake."""
    dbz, z, y, x, _, temp = _volume()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallback = _run(dbz, z, y, x, temp=temp).echo_type
    assert not [w for w in caught if "vert_levels_type" in str(w.message)]

    explicit = _run(dbz, z, y, x, temp=temp,
                    vert_params=VerticalParams(vert_levels_type="by_temp")).echo_type
    assert np.array_equal(fallback, explicit)


def test_no_vertical_field_at_all_is_silent():
    """Neither height nor temp: nothing to warn about."""
    dbz, z, y, x, _, _ = _volume()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(dbz, z, y, x)
    assert not [w for w in caught if "vert_levels_type" in str(w.message)]


# ---------------------------------------------------------------------
# min_valid_height / max_valid_height
# ---------------------------------------------------------------------

def test_min_valid_dbz_is_not_a_vertical_param():
    """
    The reflectivity floor lives on TextureParams. VerticalParams used to
    carry a same-named field that no code read, which made it possible to
    set the wrong one and silently get nothing.
    """
    import dataclasses
    names = {f.name for f in dataclasses.fields(VerticalParams)}
    assert "min_valid_dbz" not in names
    with pytest.raises(TypeError):
        VerticalParams(min_valid_dbz=5.0)


def test_default_height_band_is_inert():
    """0-25 km spans any realistic grid, so defaults must change nothing."""
    dbz, z, y, x, height, temp = _volume()
    wide = _run(dbz, z, y, x, height=height, temp=temp,
                vert_params=VerticalParams(min_valid_height=-1e9,
                                           max_valid_height=1e9)).echo_type
    default = _run(dbz, z, y, x, height=height, temp=temp).echo_type
    assert np.array_equal(default, wide)


def test_max_valid_height_excludes_levels_above_it():
    dbz, z, y, x, height, temp = _volume()
    capped = _run(dbz, z, y, x, height=height, temp=temp,
                  vert_params=VerticalParams(max_valid_height=5.0)).echo_type
    classified = np.isin(capped, CLASSIFIED)
    assert classified.any()
    assert height[classified].max() <= 5.0


def test_min_valid_height_excludes_levels_below_it():
    dbz, z, y, x, height, temp = _volume()
    floored = _run(dbz, z, y, x, height=height, temp=temp,
                   vert_params=VerticalParams(min_valid_height=3.0)).echo_type
    classified = np.isin(floored, CLASSIFIED)
    assert classified.any()
    assert height[classified].min() >= 3.0


def test_height_band_falls_back_to_the_z_coordinate():
    """With no height field, the band is applied against coords_z."""
    dbz, z, y, x, _, temp = _volume()
    capped = _run(dbz, z, y, x, temp=temp,
                  vert_params=VerticalParams(max_valid_height=5.0)).echo_type
    zz = np.broadcast_to(z[:, None, None], dbz.shape)
    classified = np.isin(capped, CLASSIFIED)
    assert classified.any()
    assert zz[classified].max() <= 5.0


def test_band_is_applied_before_texture():
    """
    Excluded levels must not leak into the statistic via the kernel. A band
    equal to a manual pre-mask of the input must give the same answer.
    """
    dbz, z, y, x, height, temp = _volume()
    banded = _run(dbz, z, y, x, height=height, temp=temp,
                  vert_params=VerticalParams(max_valid_height=6.0)).echo_type
    premasked = _run(np.where(height > 6.0, np.nan, dbz), z, y, x,
                     height=height, temp=temp).echo_type
    assert np.array_equal(banded, premasked)
