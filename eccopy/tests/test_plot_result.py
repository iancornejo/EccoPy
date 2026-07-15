"""
Regression tests for the plot_result() figure lifecycle, across all four
modules.

Guards a bug where plot_result() forced the Agg backend and then closed
the figure unconditionally. In a notebook / interactive session that meant
the cell ran without error but displayed NOTHING -- the returned figure was
already closed and the global backend had been hijacked. The contract these
tests lock in:

  * an interactive-style call (no outfile, show=False) returns a figure
    that is still OPEN (registered with pyplot), so the inline backend
    renders it;
  * a file-render call (outfile=...) closes the figure, so repeated calls
    in a loop don't leak figures;
  * plot_result never forces a global backend switch.
"""

import numpy as np
import pytest

import matplotlib
import matplotlib.pyplot as plt


def _close_all():
    plt.close("all")


def _grid(ny=40, nx=50):
    cy = np.arange(ny) * 1.0
    cx = np.arange(nx) * 1.0
    YY, XX = np.meshgrid(cy, cx, indexing="ij")
    rng = np.random.default_rng(0)
    dbz = 20 + 25 * np.exp(-(((XX - 25) ** 2 + (YY - 20) ** 2) / 120)) \
        + rng.normal(0, 2, (ny, nx))
    return cy, cx, dbz


def test_2dh_plot_result_returns_open_figure_without_outfile():
    from eccopy import eccopy2d_h
    from eccopy.eccopy2d_h.plot import plot_result
    cy, cx, dbz = _grid()
    r = eccopy2d_h.run(dbz, cy, cx)
    _close_all()
    fig = plot_result(r, dbz, cy, cx)
    assert fig.number in plt.get_fignums()   # still open -> inline renders it
    _close_all()


def test_2dh_plot_result_closes_figure_when_writing_file(tmp_path):
    from eccopy import eccopy2d_h
    from eccopy.eccopy2d_h.plot import plot_result
    cy, cx, dbz = _grid()
    r = eccopy2d_h.run(dbz, cy, cx)
    _close_all()
    out = tmp_path / "h.png"
    fig = plot_result(r, dbz, cy, cx, outfile=str(out))
    assert out.exists()
    assert fig.number not in plt.get_fignums()   # closed -> no leak in a loop
    _close_all()


def test_3d_plot_result_returns_open_figure_without_outfile():
    from eccopy import eccopy3d
    from eccopy.eccopy3d.plot import plot_result
    cy, cx, dbz2d = _grid()
    nz = 8
    cz = np.arange(nz) * 1.0
    dbz = np.stack([dbz2d * np.exp(-k / 6) for k in range(nz)])
    r = eccopy3d.run(dbz, cz, cy, cx)
    _close_all()
    fig = plot_result(r, dbz, cy, cx, cz)
    assert fig.number in plt.get_fignums()
    _close_all()


def test_1d_and_2dv_plot_result_return_open_figures():
    from eccopy import eccopy1d, eccopy2d_v
    from eccopy.eccopy1d.plot import plot_result as plot_1d
    from eccopy.eccopy2d_v.plot import plot_result as plot_2dv

    # 1D
    coords = np.arange(60) * 1.0
    rng = np.random.default_rng(1)
    prof = 20 + 25 * np.exp(-((coords - 30) ** 2) / 40) + rng.normal(0, 1.5, 60)
    r1 = eccopy1d.run(prof, coords)
    _close_all()
    fig1 = plot_1d(r1, prof, coords)
    assert fig1.number in plt.get_fignums()
    _close_all()

    # 2D-V
    nz, nx = 30, 50
    cz = np.arange(nz) * 0.5
    cx = np.arange(nx) * 1.0
    ZZ, XX = np.meshgrid(cz, cx, indexing="ij")
    dbz = 20 + 25 * np.exp(-(((XX - 25) ** 2) / 100 + ((ZZ - 5) ** 2) / 20)) \
        + rng.normal(0, 1.5, (nz, nx))
    r2 = eccopy2d_v.run(dbz, cz, cx)
    _close_all()
    fig2 = plot_2dv(r2, dbz, cz, cx)
    assert fig2.number in plt.get_fignums()
    _close_all()


def test_plot_result_does_not_force_a_global_backend_switch():
    """plot_result must not hijack the caller's backend (the old code called
    matplotlib.use('Agg'), which broke every subsequent inline plot in the
    same session)."""
    from eccopy import eccopy2d_h
    from eccopy.eccopy2d_h.plot import plot_result
    cy, cx, dbz = _grid()
    r = eccopy2d_h.run(dbz, cy, cx)
    backend_before = matplotlib.get_backend()
    plot_result(r, dbz, cy, cx)
    assert matplotlib.get_backend() == backend_before
    _close_all()
