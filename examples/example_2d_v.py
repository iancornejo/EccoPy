#!/usr/bin/env python3
"""
EccoPy-2D-V worked example: vertical cross-sections.

Runs the full texture -> convectivity -> classification chain on the
bundled S-Pol RHI, showing every parameter and every intermediate. This is
the script form of notebooks/eccopy2d_v_workflow.ipynb; the notebook has
more explanation, this has less scrolling.

Run from anywhere in a checkout:

    python examples/example_2d_v.py
    python examples/example_2d_v.py --outdir figs      # also save a figure

Needs the sample data under notebooks/data/, which ships with the
repository but not with the installed wheel. Reading it needs netCDF4:

    pip install -e ".[dev,plot]" netCDF4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eccopy import eccopy2d_v, stats
from eccopy.core.temperature import broadcast_temp_field, melt_layer_from_temp
from eccopy.params import ClassificationParams, TextureParams, WindowSpec

DATA = Path(__file__).resolve().parent.parent / "notebooks" / "data"


def load():
    """Read the RHI and its co-located sounding into plain arrays."""
    import netCDF4 as nc

    ds = nc.Dataset(DATA / "spol_rhi_20220526.nc")
    fill = lambda name: np.ma.filled(ds.variables[name][:].astype(float), np.nan)

    z_km = fill("Z") / 1000.0          # (Z,)  height above radar
    r_km = fill("R") / 1000.0          # (X,)  ground range
    dbz = fill("DBZ_F")                # (Z, X)

    # The sounding ships as a profile; interpolate it onto the radar's
    # vertical grid, then broadcast across range.
    temp_profile = np.interp(z_km, fill("sounding_altitude") / 1000.0,
                             fill("sounding_temperature"))
    return z_km, r_km, dbz, temp_profile


def main(outdir: Path | None) -> None:
    z_km, r_km, dbz, temp_profile = load()
    print(f"input {dbz.shape}: {np.isfinite(dbz).mean():.1%} valid, "
          f"{np.nanmin(dbz):.1f} to {np.nanmax(dbz):.1f} dBZ")

    # ------------------------------------------------------------------
    # Auxiliary fields. Sub-classification needs all three together; with
    # any of them missing, run() returns basic 1/2/3 codes instead.
    # ------------------------------------------------------------------
    height = np.broadcast_to(z_km[:, None], dbz.shape).copy()   # km
    temp = broadcast_temp_field(temp_profile, dbz.shape)        # degC
    melt = melt_layer_from_temp(temp)                           # binary 10/20

    print(f"0 degC level: {np.interp(0.0, -temp_profile, z_km):.2f} km")

    # ------------------------------------------------------------------
    # Parameters. Every value below is the package default, spelled out so
    # you can see what there is to change.
    # ------------------------------------------------------------------
    window = WindowSpec((7, "km"))     # texture half-width along X.
                                       # Larger -> smoother texture.

    texture_params = TextureParams(
        dbz_base=0.0,                  # subtracted before the statistic
        texture_limit_low=0.0,         # texture below this -> convectivity
                                       # missing rather than zero
        texture_limit_high=30.0,       # the normaliser: convectivity =
                                       # texture / this, clipped to 1
        min_valid_dbz=None,            # None -> no gating on this path,
                                       # matching f_reflTexture.m
    )

    class_params = ClassificationParams(
        max_convectivity_for_stratiform=0.4,   # below -> stratiform
        min_convectivity_for_convective=0.5,   # at/above -> convective
                                               # between the two -> mixed
        enlarge_mixed=5,               # closing radius, mixed regions (px)
        enlarge_conv=5,                # closing radius, convective regions
        surf_alt_lim=0.0,              # metres; near-surface convective test
    )

    # ------------------------------------------------------------------
    result = eccopy2d_v.run(
        dbz,
        coords_z=z_km,
        coords_x=r_km,
        height=height,
        temp=temp,
        melt=melt,
        window=window,
        texture_params=texture_params,
        class_params=class_params,
        kernel_mode="uniform",         # "varying" resolves per point;
                                       # identical on an even grid
        remove_surface_echo=False,     # mask below surf_alt_lim pre-texture
        return_intermediates=True,     # adds fitted_dbz / detrended_dbz /
                                       # echo_basic
    )

    print("\nintermediates")
    for name in ("texture", "convectivity", "fitted_dbz",
                 "detrended_dbz", "echo_basic", "echo_type"):
        arr = getattr(result, name)
        print(f"  {name:14s} {str(arr.shape):10s} {np.isfinite(arr).mean():6.1%} finite")

    print(f"\ntexture: median {np.nanmedian(result.texture):.1f}, "
          f"max {np.nanmax(result.texture):.1f}")

    # ------------------------------------------------------------------
    print("\nclassification")
    fractions = stats.echo_type_fractions(result.echo_type)
    print(f"  classified points {fractions['n_valid']:,}")
    for name in ("stratiform", "mixed", "convective"):
        print(f"    {name:12s} {fractions[name]:6.1%}")

    codes, counts = np.unique(result.echo_type[np.isfinite(result.echo_type)],
                              return_counts=True)
    print("  by code: " + "  ".join(f"{int(c)}:{n / counts.sum():.1%}"
                                    for c, n in zip(codes, counts)))

    top = stats.convective_top_height(result.echo_type, height, axis=0)
    if np.isfinite(top).any():
        print(f"  convective top: mean {np.nanmean(top):.1f} km, "
              f"max {np.nanmax(top):.1f} km")

    if outdir is not None:
        plot(outdir, z_km, r_km, dbz, result, class_params)


def plot(outdir, z_km, r_km, dbz, result, class_params):
    """Reflectivity, convectivity and echo type, using EccoPy's colormaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eccopy.core.colormaps import (ECHO_TYPE_LABELS, convectivity_cmap,
                                       convectivity_norm, echo_type_cmap,
                                       echo_type_norm, remap_echo_type)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, sharey=True)

    m = axes[0].pcolormesh(r_km, z_km, dbz, cmap="turbo", vmin=-10, vmax=60,
                           shading="nearest")
    fig.colorbar(m, ax=axes[0], pad=0.01, label="dBZ")
    axes[0].set_title("Reflectivity", fontsize=10)

    m = axes[1].pcolormesh(
        r_km, z_km, result.convectivity, shading="nearest",
        norm=convectivity_norm(),
        cmap=convectivity_cmap(
            strat_mixed=class_params.max_convectivity_for_stratiform,
            mixed_conv=class_params.min_convectivity_for_convective))
    fig.colorbar(m, ax=axes[1], pad=0.01, label="convectivity")
    axes[1].set_title("Convectivity", fontsize=10)

    m = axes[2].pcolormesh(r_km, z_km, remap_echo_type(result.echo_type),
                           cmap=echo_type_cmap(), norm=echo_type_norm(),
                           shading="nearest")
    cb = fig.colorbar(m, ax=axes[2], ticks=range(1, 10), pad=0.01)
    cb.ax.set_yticklabels(ECHO_TYPE_LABELS)
    axes[2].set_title("Echo type", fontsize=10)

    for ax in axes:
        ax.set_ylabel("height (km)")
    axes[-1].set_xlabel("range (km)")
    plt.tight_layout()

    path = outdir / "example_2d_v.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write a figure here (needs matplotlib)")
    main(ap.parse_args().outdir)
