"""
Tests for eccopy.core.colormaps -- item 3 of the v0.1 checklist:
colormaps for convectivity, basic classification, and sub-classification,
with the latter two sharing coincident colors/norms/labels for their
common categories.
"""

import numpy as np
import pytest

from eccopy.core import colormaps as cm


def test_basic_and_sub_cmaps_have_expected_sizes():
    assert cm.basic_echo_type_cmap().N == 3
    assert cm.echo_type_cmap().N == 9


def test_basic_and_sub_norms_bracket_their_code_ranges():
    assert cm.basic_echo_type_norm().vmin == 0.5
    assert cm.basic_echo_type_norm().vmax == 3.5
    assert cm.echo_type_norm().vmin == 0.5
    assert cm.echo_type_norm().vmax == 9.5


def test_basic_and_sub_colormaps_are_coincident_for_shared_categories():
    """
    The whole-category basic codes (1=Stratiform, 2=Mixed, 3=Convective)
    must render in the SAME color as their sub-classified "plain"
    counterpart (Strat Mid=16, Mixed=25, Conv=30) -- so a reader can
    switch between a basic-only panel and a sub-classified panel for a
    different case and still read blue=stratiform, green=mixed, red=
    convective consistently.
    """
    basic_colors = cm.basic_echo_type_cmap().colors
    sub_colors = cm.echo_type_cmap().colors

    basic_strat = basic_colors[cm.BASIC_ECHO_TYPE_TO_IDX[1] - 1]
    sub_strat_mid = sub_colors[cm.ECHO_TYPE_TO_IDX[16] - 1]
    np.testing.assert_allclose(basic_strat, sub_strat_mid)

    basic_mixed = basic_colors[cm.BASIC_ECHO_TYPE_TO_IDX[2] - 1]
    sub_mixed = sub_colors[cm.ECHO_TYPE_TO_IDX[25] - 1]
    np.testing.assert_allclose(basic_mixed, sub_mixed)

    basic_conv = basic_colors[cm.BASIC_ECHO_TYPE_TO_IDX[3] - 1]
    sub_conv = sub_colors[cm.ECHO_TYPE_TO_IDX[30] - 1]
    np.testing.assert_allclose(basic_conv, sub_conv)


def test_basic_and_sub_labels_agree_on_shared_categories():
    assert cm.BASIC_ECHO_TYPE_LABELS[cm.BASIC_ECHO_TYPE_TO_IDX[1] - 1] == "Stratiform"
    assert cm.ECHO_TYPE_LABELS[cm.ECHO_TYPE_TO_IDX[16] - 1] == "Strat Mid"
    assert cm.BASIC_ECHO_TYPE_LABELS[cm.BASIC_ECHO_TYPE_TO_IDX[2] - 1] == "Mixed"
    assert cm.ECHO_TYPE_LABELS[cm.ECHO_TYPE_TO_IDX[25] - 1] == "Mixed"


def test_remap_echo_type_autodetects_basic_vs_sub():
    basic = np.array([1, 2, 3, np.nan])
    sub = np.array([14, 25, 38, np.nan])
    remapped_basic = cm.remap_echo_type(basic)
    remapped_sub = cm.remap_echo_type(sub)
    assert np.array_equal(remapped_basic[:3], [1, 2, 3])
    assert np.array_equal(remapped_sub[:3], [1, 4, 9])


def test_convectivity_cmap_and_norm():
    cmap = cm.convectivity_cmap()
    norm = cm.convectivity_norm()
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0
    # Endpoints should map to the documented stratiform-blue / convective-red
    # anchors, and be visually distinct from each other and the midpoint.
    low = np.array(cmap(norm(0.0)))
    mid = np.array(cmap(norm(0.5)))
    high = np.array(cmap(norm(1.0)))
    assert not np.allclose(low, high)
    assert not np.allclose(low, mid)
    assert not np.allclose(mid, high)
    # Low end should be blue-dominant, high end red-dominant (RGBA order).
    assert low[2] > low[0]    # more blue than red at convectivity=0
    assert high[0] > high[2]  # more red than blue at convectivity=1


def test_convectivity_cmap_shares_endpoint_hues_with_classification_cmaps():
    """Loose visual-consistency check: convectivity=0 should be closer in
    hue to the Stratiform color than to the Convective color, and vice
    versa for convectivity=1."""
    conv_cmap = cm.convectivity_cmap()
    conv_norm = cm.convectivity_norm()
    strat_color = np.array(cm.basic_echo_type_cmap().colors[cm.BASIC_ECHO_TYPE_TO_IDX[1] - 1])
    conv_color = np.array(cm.basic_echo_type_cmap().colors[cm.BASIC_ECHO_TYPE_TO_IDX[3] - 1])

    low = np.array(conv_cmap(conv_norm(0.0)))[:3]
    high = np.array(conv_cmap(conv_norm(1.0)))[:3]

    assert np.linalg.norm(low - strat_color) < np.linalg.norm(low - conv_color)
    assert np.linalg.norm(high - conv_color) < np.linalg.norm(high - strat_color)


# ---------------------------------------------------------------------------
# Texture-window ring overlay (draw_window_ring)
# ---------------------------------------------------------------------------


def _new_axis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    return fig, ax, Circle


def test_draw_window_ring_adds_circle_of_correct_radius():
    fig, ax, Circle = _new_axis()
    coords_x = np.arange(0, 100, 1.0)
    coords_y = np.arange(0, 80, 1.0)
    ring = cm.draw_window_ring(ax, coords_x, coords_y, radius_km=7.0, label=False)
    assert ring is not None
    circles = [p for p in ax.patches if isinstance(p, Circle)]
    assert len(circles) == 1
    assert circles[0].radius == 7.0
    # centred on the domain
    cx, cy = circles[0].center
    assert cx == pytest.approx(np.mean([coords_x[0], coords_x[-1]]))
    assert cy == pytest.approx(np.mean([coords_y[0], coords_y[-1]]))


def test_draw_window_ring_skips_non_physical_radius():
    """None / non-finite / non-positive radii draw nothing (a bare-pixel
    window on a unit-agnostic grid degrades quietly rather than drawing a
    misleading ring)."""
    fig, ax, Circle = _new_axis()
    coords_x = np.arange(0, 100, 1.0)
    coords_y = np.arange(0, 80, 1.0)
    for bad in (None, 0.0, -3.0, float("nan"), float("inf")):
        assert cm.draw_window_ring(ax, coords_x, coords_y, radius_km=bad) is None
    assert not [p for p in ax.patches if isinstance(p, Circle)]


def test_draw_window_ring_honours_explicit_center():
    fig, ax, Circle = _new_axis()
    coords_x = np.arange(0, 100, 1.0)
    coords_y = np.arange(0, 80, 1.0)
    cm.draw_window_ring(ax, coords_x, coords_y, radius_km=5.0,
                        center=(10.0, 20.0), label=False)
    circ = [p for p in ax.patches if isinstance(p, Circle)][0]
    assert circ.center == (10.0, 20.0)
