#!/usr/bin/env python3
"""
EccoPy-1D worked example: time series and transects.

Runs the full texture -> convectivity -> classification chain on the
bundled disdrometer record. This is the script form of
notebooks/eccopy1d_workflow.ipynb.

Run from anywhere in a checkout:

    python examples/example_1d.py
    python examples/example_1d.py --outdir figs       # also save a figure

Needs the sample data under notebooks/data/, which ships with the
repository but not with the installed wheel. Reading it needs netCDF4:

    pip install -e ".[dev,plot]" netCDF4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eccopy import eccopy1d, stats, time_to_distance_km
from eccopy.params import ClassificationParams, TextureParams, WindowSpec

DATA = Path(__file__).resolve().parent.parent / "notebooks" / "data"


def load():
    """
    Read the drop-size distribution and derive reflectivity from it.

    The file stores no reflectivity field. EccoPy takes plain arrays from
    any source, so anything you can turn into a 1-D sequence of dBZ is
    valid input:

        Z = sum_i N(D_i) * D_i**6 * dD_i        dBZ = 10 log10(Z)
    """
    import netCDF4 as nc

    ds = nc.Dataset(DATA / "disdrometer_dsd_20080627.nc")
    fill = lambda name: np.ma.filled(ds.variables[name][:].astype(float), np.nan)

    diameter = fill("particle_size")            # mm
    bin_width = fill("particle_size_bin_width")  # mm
    nd = fill("Nd")                             # mm^-1 m^-3

    z_linear = np.nansum(nd * diameter ** 6 * bin_width, axis=1)
    dbz = np.where(z_linear > 0,
                   10.0 * np.log10(np.where(z_linear > 0, z_linear, 1.0)),
                   np.nan)

    minutes = fill("time")
    t_seconds = (minutes - minutes[0]) * 60.0
    return dbz, t_seconds


def main(outdir: Path | None) -> None:
    dbz, t_seconds = load()
    cadence_s = float(np.median(np.diff(t_seconds)))
    print(f"input {dbz.shape}: {np.isfinite(dbz).sum()} of {dbz.size} samples "
          f"carry precipitation, {cadence_s:.0f} s cadence")
    print(f"reflectivity {np.nanmin(dbz):.1f} to {np.nanmax(dbz):.1f} dBZ")

    # A fixed-point instrument gives you time. With a wind speed you could
    # instead build a distance coordinate through Taylor's hypothesis and
    # size the window in km, directly comparable to the 2-D/3-D modules:
    distance_km = time_to_distance_km(t_seconds, wind_speed_ms=12.0)
    print(f"record spans {t_seconds[-1] / 3600:.1f} h, "
          f"or {distance_km[-1]:.0f} km advected at 12 m/s")

    # ------------------------------------------------------------------
    # Parameters. Every value below is the package default.
    # ------------------------------------------------------------------
    # WindowSpec is unit-typed and must agree with `coords`. coords are in
    # seconds here, so the window is in minutes. WindowSpec(15) would mean
    # 15 samples instead, regardless of cadence.
    window = WindowSpec((15, "min"))

    texture_params = TextureParams(
        dbz_base=0.0,                  # subtracted before the statistic
        texture_limit_low=0.0,         # texture below this -> convectivity
                                       # missing rather than zero
        texture_limit_high=30.0,       # the normaliser
        min_valid_dbz=None,            # None -> no gating on this path
    )

    class_params = ClassificationParams(
        max_convectivity_for_stratiform=0.4,   # below -> stratiform
        min_convectivity_for_convective=0.5,   # at/above -> convective
        enlarge_mixed=5,               # closing radius, mixed runs
        enlarge_conv=5,                # closing radius, convective runs
    )

    # ------------------------------------------------------------------
    result = eccopy1d.run(
        dbz,
        coords=t_seconds,
        coord_mode="position",         # t_seconds is cumulative position
        window=window,
        texture_params=texture_params,
        class_params=class_params,
        min_convective_length=None,    # demoted to Mixed below this; see
                                       # the sweep at the end
        kernel_mode="uniform",
        return_intermediates=True,
    )

    print("\nintermediates")
    for name in ("texture", "convectivity", "fitted_dbz",
                 "detrended_dbz", "echo_type"):
        arr = getattr(result, name)
        print(f"  {name:14s} {str(arr.shape):8s} {np.isfinite(arr).mean():6.1%} finite")

    # ------------------------------------------------------------------
    print("\nclassification")
    fractions = stats.echo_type_fractions(result.echo_type)
    print(f"  classified samples {fractions['n_valid']}")
    for name in ("stratiform", "mixed", "convective"):
        print(f"    {name:12s} {fractions[name]:6.1%}")

    runs = convective_runs(result.echo_type)
    cadence_min = cadence_s / 60.0
    if len(runs):
        print(f"  {len(runs)} convective runs; longest "
              f"{runs.max() * cadence_min:.0f} min, total "
              f"{runs.sum() * cadence_min / 60:.1f} h")

    # min_convective_length demotes brief bursts to Mixed. It takes the
    # same unit as `coords`, so a time series wants a time WindowSpec.
    print("\nmin_convective_length")
    for minutes_required in (None, 5, 15, 30):
        mcl = None if minutes_required is None else WindowSpec((minutes_required, "min"))
        r = eccopy1d.run(dbz, coords=t_seconds, coord_mode="position",
                         window=window, texture_params=texture_params,
                         class_params=class_params, min_convective_length=mcl)
        label = "none" if minutes_required is None else f"{minutes_required} min"
        print(f"  {label:>8s}: {len(convective_runs(r.echo_type)):3d} runs, "
              f"convective {stats.convective_percentage(r.echo_type):5.1f}%")

    # These fractions are of RAINING MINUTES at one point, not of any area,
    # and 1-D texture is temporal rather than spatial variability. Do not
    # compare them directly against 2-D or 3-D fractions.

    if outdir is not None:
        plot(outdir, t_seconds, dbz, result)


def convective_runs(echo: np.ndarray) -> np.ndarray:
    """Lengths, in samples, of each contiguous convective run."""
    conv = (echo == 3).astype(int)
    edges = np.diff(np.concatenate([[0], conv, [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def plot(outdir, t_seconds, dbz, result):
    """Reflectivity with the classification as a colour strip beneath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eccopy.core.colormaps import (BASIC_ECHO_TYPE_LABELS,
                                       basic_echo_type_cmap,
                                       basic_echo_type_norm, remap_echo_type)

    outdir.mkdir(parents=True, exist_ok=True)
    t_hours = t_seconds / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 3.6), sharex=True,
                             height_ratios=[3, 1])
    axes[0].plot(t_hours, dbz, lw=0.8, color="0.25")
    axes[0].set_ylabel("dBZ")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("EccoPy-1D - MRR/2DVD disdrometer, 2008-06-27", fontsize=10)

    # imshow rather than pcolormesh: a single-row strip gives pcolormesh no
    # way to infer cell height, so it draws an empty axis.
    strip = remap_echo_type(result.echo_type)[None, :]
    m = axes[1].imshow(strip, aspect="auto", origin="lower",
                       extent=[t_hours[0], t_hours[-1], 0, 1],
                       cmap=basic_echo_type_cmap(), norm=basic_echo_type_norm(),
                       interpolation="nearest")
    axes[1].set_yticks([])
    axes[1].set_xlabel("hours from start of record")
    cb = fig.colorbar(m, ax=axes, ticks=range(1, 4), pad=0.01, aspect=12)
    cb.ax.set_yticklabels(BASIC_ECHO_TYPE_LABELS)
    axes[0].set_xlim(0, t_hours[-1])

    path = outdir / "example_1d.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write a figure here (needs matplotlib)")
    main(ap.parse_args().outdir)
