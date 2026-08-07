#!/usr/bin/env python3
"""
EccoPy-3D worked example: full volumes.

Runs the full texture -> convectivity -> clumping -> classification chain
on the bundled WRF volume. This is the script form of
notebooks/eccopy3d_workflow.ipynb.

Run from anywhere in a checkout:

    python examples/example_3d.py
    python examples/example_3d.py --outdir figs       # also save a figure

Needs the sample data under notebooks/data/, which ships with the
repository but not with the installed wheel. Reading it needs netCDF4:

    pip install -e ".[dev,plot]" netCDF4
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from eccopy import eccopy3d, stats
from eccopy.params import (ClassificationParams, TextureParams, VerticalParams,
                           WindowSpec)

DATA = Path(__file__).resolve().parent.parent / "notebooks" / "data"
CLASSES = [14, 16, 18, 25, 30, 32, 34, 36, 38]


def load():
    """Read the volume. `height` ships as a 1-D profile; broadcast it."""
    import netCDF4 as nc

    ds = nc.Dataset(DATA / "wrf_volume_20220526.nc")
    fill = lambda name: np.ma.filled(ds.variables[name][:].astype(float), np.nan)

    z_km = fill("z")          # (Z,)  nominal height
    y_km = fill("y")          # (Y,)
    x_km = fill("x")          # (X,)
    dbz = fill("dbz")         # (Z, Y, X)
    temp = fill("Temp")       # (Z, Y, X)

    # Geopotential height varies horizontally by only 12-20 m, so it ships
    # as a profile. It is NOT interchangeable with the nominal z
    # coordinate: the two drift apart by up to 0.5 km aloft.
    height = np.broadcast_to(fill("height")[:, None, None], dbz.shape).copy()
    return z_km, y_km, x_km, dbz, temp, height


def main(outdir: Path | None) -> None:
    z_km, y_km, x_km, dbz, temp, height = load()
    print(f"input {dbz.shape}: {np.isfinite(dbz).mean():.1%} valid, "
          f"{np.nanmin(dbz):.1f} to {np.nanmax(dbz):.1f} dBZ")

    # ------------------------------------------------------------------
    # Parameters. Every value below is the package default.
    # ------------------------------------------------------------------
    window = WindowSpec((7, "km"))     # radius of the 2-D radial kernel,
                                       # applied within each level

    texture_params = TextureParams(
        dbz_base=0.0,
        texture_limit_low=0.0,         # sub-low texture -> missing
        texture_limit_high=30.0,       # the normaliser
        min_valid_dbz=None,            # None -> 0.0 here (ConvStratFinder)
        min_frac_texture=0.25,
        min_frac_fit=0.67,
        use_dbz_col_max=False,         # True computes texture once from
                                       # column-max dBZ, copied to all levels
        dbz_for_echo_tops=18.0,        # threshold defining echo_top_km
    )

    class_params = ClassificationParams(
        # thresholds
        max_convectivity_for_stratiform=0.4,
        min_convectivity_for_convective=0.5,
        # clump identification and splitting
        use_dual_thresholds=True,
        secondary_convectivity=0.65,
        all_subclumps_min_area_frac=0.33,
        each_subclump_min_area_frac=0.02,
        each_subclump_min_area_km2=2.0,   # a pseudo-VOLUME in 3-D
        min_valid_volume_for_convective=20.0,
        min_vert_extent_for_convective=1.0,
        # clump sub-typing
        min_conv_fraction_for_deep=0.05,
        min_conv_fraction_for_shallow=0.95,
        max_shallow_conv_fraction_for_elevated=0.05,
        max_deep_conv_fraction_for_elevated=0.25,
        min_strat_fraction_for_strat_below=0.9,
        # only active when terrain_ht is supplied
        min_ht_km_agl_for_mid=2.0,
        min_ht_km_agl_for_deep=4.0,
    )

    vert_params = VerticalParams(
        vert_levels_type="by_height",  # or "by_temp" for the temperature pair
        shallow_threshold_ht=4.5,      # km
        deep_threshold_ht=9.0,         # km
        shallow_threshold_temp=0.0,    # degC, used when by_temp
        deep_threshold_temp=-12.0,     # degC
        min_valid_height=0.0,          # analysis band; inert at these
        max_valid_height=25.0,         # defaults on any realistic grid
    )

    # ------------------------------------------------------------------
    result = eccopy3d.run(
        dbz,
        coords_z=z_km,
        coords_y=y_km,
        coords_x=x_km,
        height=height,
        temp=temp,
        terrain_ht=None,               # raises the shallow/deep boundaries
        window=window,
        texture_params=texture_params,
        class_params=class_params,
        vert_params=vert_params,
        kernel_mode="uniform",
        n_threads=1,
    )

    print(f"\nclumps found   {result.n_clumps}")
    print(f"texture radius {result.texture_radius:.1f} km")
    print(f"echo_type dtype {result.echo_type.dtype} "
          f"(int16, 0 = no echo - np.isfinite() is the WRONG mask test here)")

    classified = np.isin(result.echo_type, CLASSES)
    print(f"classified voxels {classified.sum():,} of {result.echo_type.size:,}")
    if result.echo_top_km is not None:
        print(f"echo top (>= {texture_params.dbz_for_echo_tops:.0f} dBZ): "
              f"mean {np.nanmean(result.echo_top_km):.1f} km, "
              f"max {np.nanmax(result.echo_top_km):.1f} km")

    # ------------------------------------------------------------------
    print("\nclassification")
    fractions = stats.echo_type_fractions(result.echo_type)
    for name in ("stratiform", "mixed", "convective"):
        print(f"    {name:12s} {fractions[name]:6.1%}")
    codes, counts = np.unique(result.echo_type[classified], return_counts=True)
    print("  by code: " + "  ".join(f"{int(c)}:{n / counts.sum():.1%}"
                                    for c, n in zip(codes, counts)))

    top = stats.convective_top_height(result.echo_type, height, axis=0)
    print(f"  convective top: mean {np.nanmean(top):.1f} km, "
          f"max {np.nanmax(top):.1f} km")

    # Sub-typing keys off the height/temperature boundaries. Switching to
    # temperature warns that the supplied height is not used for that.
    print("\nvert_levels_type")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        by_temp = eccopy3d.run(
            dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km, height=height,
            temp=temp, window=window, texture_params=texture_params,
            class_params=class_params,
            vert_params=replace(vert_params, vert_levels_type="by_temp"))
        for w in caught:
            print(f"  warning: {str(w.message).splitlines()[0]}")
    for label, echo in (("by_height", result.echo_type),
                        ("by_temp", by_temp.echo_type)):
        present = sorted(int(c) for c in np.unique(echo[np.isin(echo, CLASSES)]))
        print(f"  {label:9s} codes {present}")

    if outdir is not None:
        plot(outdir, z_km, y_km, x_km, dbz, result)


def plot(outdir, z_km, y_km, x_km, dbz, result):
    """Column-max reflectivity and a vertical slice of the classification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eccopy.core.colormaps import (ECHO_TYPE_LABELS, echo_type_cmap,
                                       echo_type_norm, remap_echo_type)

    outdir.mkdir(parents=True, exist_ok=True)
    col_max = np.nanmax(dbz, axis=0)
    iy = int(np.unravel_index(np.nanargmax(col_max), col_max.shape)[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), width_ratios=[1, 1.15])

    m = axes[0].pcolormesh(x_km, y_km, col_max, cmap="turbo", vmin=-10, vmax=65,
                           shading="nearest")
    axes[0].axhline(y_km[iy], color="k", lw=1.0, ls="--")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    axes[0].set_aspect("equal")
    axes[0].set_title("Column-max reflectivity", fontsize=10)
    fig.colorbar(m, ax=axes[0], pad=0.01, label="dBZ")

    # int16 with a 0 sentinel: convert to float/NaN before remapping.
    slice_et = result.echo_type[:, iy, :].astype(float)
    slice_et[slice_et == 0] = np.nan
    m = axes[1].pcolormesh(x_km, z_km, remap_echo_type(slice_et),
                           cmap=echo_type_cmap(), norm=echo_type_norm(),
                           shading="nearest")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("height (km)")
    axes[1].set_title(f"Echo type at y = {y_km[iy]:.0f} km", fontsize=10)
    cb = fig.colorbar(m, ax=axes[1], ticks=range(1, 10), pad=0.01)
    cb.ax.set_yticklabels(ECHO_TYPE_LABELS)

    plt.tight_layout()
    path = outdir / "example_3d.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write a figure here (needs matplotlib)")
    main(ap.parse_args().outdir)
