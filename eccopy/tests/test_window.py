"""Tests for eccopy.params.window.WindowSpec."""

import numpy as np
import pytest

from eccopy.params.window import WindowSpec


def test_pixel_radius_bare_int():
    ws = WindowSpec(7)
    assert ws.is_pixel
    spacing = np.full(10, 1.0)
    radii = ws.pixel_radius_field(spacing)
    assert np.all(radii == 7)


def test_km_window_uniform_spacing():
    ws = WindowSpec((5, "km"))
    assert not ws.is_pixel
    spacing_m = np.full(10, 1000.0)  # 1 km spacing, in metres
    radii = ws.pixel_radius_field(spacing_m)
    assert np.all(radii == 5)


def test_km_window_variable_spacing():
    ws = WindowSpec((10, "km"))
    spacing_m = np.array([1000.0, 2000.0, 500.0])  # 1km, 2km, 0.5km
    radii = ws.pixel_radius_field(spacing_m)
    np.testing.assert_array_equal(radii, [10, 5, 20])


def test_minute_window_resolves_against_seconds():
    ws = WindowSpec((3, "minute"))
    spacing_s = np.full(5, 30.0)  # 30-second spacing
    radii = ws.pixel_radius_field(spacing_s)
    # 3 minutes = 180s; 180/30 = 6
    assert np.all(radii == 6)


def test_unit_aliases_equivalent():
    ws_km = WindowSpec((5, "km"))
    ws_kilometers = WindowSpec((5, "kilometers"))
    spacing_m = np.full(5, 500.0)
    np.testing.assert_array_equal(
        ws_km.pixel_radius_field(spacing_m),
        ws_kilometers.pixel_radius_field(spacing_m),
    )


def test_unrecognised_unit_raises():
    with pytest.raises(ValueError):
        WindowSpec((5, "furlongs"))


def test_zero_or_negative_spacing_yields_zero_radius():
    ws = WindowSpec((5, "km"))
    spacing_m = np.array([0.0, -100.0, 1000.0])
    radii = ws.pixel_radius_field(spacing_m)
    assert radii[0] == 0
    assert radii[1] == 0
    assert radii[2] == 5


def test_pixel_radius_single_index():
    ws = WindowSpec((5, "km"))
    spacing_m = np.array([1000.0, 500.0, 2000.0])
    assert ws.pixel_radius(spacing_m, 0) == 5
    assert ws.pixel_radius(spacing_m, 1) == 10
    assert ws.pixel_radius(spacing_m, 2) == 2


def test_repr_pixel():
    ws = WindowSpec(7)
    assert "7" in repr(ws)


def test_repr_physical():
    ws = WindowSpec((5, "km"))
    assert "5" in repr(ws)
    assert "km" in repr(ws)
