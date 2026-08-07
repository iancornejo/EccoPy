#!/usr/bin/env python3
"""
EccoPy-2D-H worked example: horizontal composites.

Runs the full texture -> convectivity -> classification chain on the
bundled MRMS composite. This is the script form of
notebooks/eccopy2d_h_workflow.ipynb.

Run from anywhere in a checkout:

    python examples/example_2d_h.py
    python examples/example_2d_h.py --outdir figs     # also save a figure

Needs the sample data under notebooks/data/, which ships with the
repository but not with the installed wheel. Reading it needs netCDF4:

    pip install -e ".[dev,plot]" netCDF4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eccopy import eccopy2d_h, stats
from eccopy.core.coords import latlon_to_xy_spacing
from eccopy.params import ClassificationParams, TextureParams, WindowSpec

DATA = Path(__file__).resolve().parent.parent / "notebooks" / "data"


def load():
    """Read the composite and convert its lat/lon mesh to physical spacing."""
    import netCDF4 as nc

    ds = nc.Dataset(DATA / "mrms_composite_20211215.nc")
    fill = lambda name: np.ma.filled(ds.variables[name][:].astype(float), np.nan)

    lat = fill("latitude")
    lon = fill("longitude")
    dbz = fill("MergedReflectivityQCComposite")

    # EccoPy works in physical distance, not degrees. latlon_to_xy_spacing
    # takes 2-D latitude and longitude FIELDS, so build a meshgrid first.
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dy_km, dx_km = latlon_to_xy_spacing(LAT, LON)
    return lat, lon, dbz, dy_km, dx_km


def main(outdir: Path | None) -> None:
    lat, lon, dbz, dy_km, dx_km = load()
    print(f"input {dbz.shape}: {np.isfinite(dbz).mean():.1%} carries echo, "
          f"max {np.nanmax(dbz):.1f} dBZ")
    print(f"domain {lat.min():.1f}-{lat.max():.1f} N, "
          f"{lon.min():.1f}-{lon.max():.1f} E")
    print(f"dx varies {dx_km.min():.3f}-{dx_km.max():.3f} km "
          f"({100 * (dx_km.max() / dx_km.min() - 1):.1f}% across the domain); "
          f"dy constant at {dy_km.mean():.3f} km")

    # ------------------------------------------------------------------
    # Parameters. Every value below is the package default.
    # ------------------------------------------------------------------
    window = WindowSpec((7, "km"))     # radius of the 2-D radial kernel

    texture_params = TextureParams(
        dbz_base=0.0,                  # subtracted before the statistic
        texture_limit_low=0.0,         # sub-low texture -> missing
        texture_limit_high=30.0,       # the normaliser
        min_valid_dbz=None,            # None -> 0.0 here (ConvStratFinder)
        min_frac_texture=0.25,         # kernel coverage to compute texture
        min_frac_fit=0.67,             # kernel coverage to fit the plane
    )

    class_params = ClassificationParams(
        max_convectivity_for_stratiform=0.4,   # below -> stratiform
        min_convectivity_for_convective=0.5,   # at/above -> convective
        use_dual_thresholds=True,      # split merged cells
        secondary_convectivity=0.65,   # inner threshold used for the split
        all_subclumps_min_area_frac=0.33,
        each_subclump_min_area_frac=0.02,
        each_subclump_min_area_km2=2.0,  # a true AREA here; a pseudo-volume
                                         # in eccopy3d - do not share one
                                         # ClassificationParams between them
    )

    # ------------------------------------------------------------------
    result = eccopy2d_h.run(
        dbz,
        coords_y=dy_km,
        coords_x=dx_km,
        coord_mode="spacing",          # these are cell sizes, not positions
        window=window,
        texture_params=texture_params,
        class_params=class_params,
        kernel_mode="uniform",
        min_convective_area=None,      # km^2 floor on finished clumps
    )

    print(f"\nclumps found   {result.n_clumps}")
    print(f"texture radius {result.texture_radius:.1f} km")
    print(f"texture: median {np.nanmedian(result.texture):.1f}, "
          f"max {np.nanmax(result.texture):.1f}")

    print("\nclassification")
    fractions = stats.echo_type_fractions(result.echo_type)
    print(f"  classified pixels {fractions['n_valid']:,}")
    for name in ("stratiform", "mixed", "convective"):
        print(f"    {name:12s} {fractions[name]:6.1%}")

    # min_convective_area barely moves the map while changing the cell
    # count several-fold: judge clump parameters by n_clumps, not coverage.
    print("\nmin_convective_area (km^2)")
    for area in (None, 4.0, 25.0, 100.0):
        r = eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km,
                           coord_mode="spacing", window=window,
                           texture_params=texture_params,
                           class_params=class_params, min_convective_area=area)
        print(f"  {str(area):>6s}: n_clumps {r.n_clumps:4d}, "
              f"convective {stats.convective_percentage(r.echo_type):5.1f}%")

    # kernel_mode matters here because the grid is genuinely non-uniform.
    # "uniform" reproduces the reference; "varying" is arguably closer to
    # the physical intent. Neither is ground truth for the other.
    varying = eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km,
                             coord_mode="spacing", window=window,
                             texture_params=texture_params,
                             class_params=class_params, kernel_mode="varying")
    both = np.isin(result.echo_type, [1, 2, 3]) | np.isin(varying.echo_type, [1, 2, 3])
    agree = (result.echo_type[both] == varying.echo_type[both]).mean()
    print(f"\nkernel_mode uniform vs varying: {100 * (1 - agree):.2f}% of "
          f"pixels differ")

    if outdir is not None:
        plot(outdir, lat, lon, dbz, result, class_params)


def plot(outdir, lat, lon, dbz, result, class_params):
    """Reflectivity, convectivity and echo type, using EccoPy's colormaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eccopy.core.colormaps import (BASIC_ECHO_TYPE_LABELS,
                                       basic_echo_type_cmap,
                                       basic_echo_type_norm, convectivity_cmap,
                                       convectivity_norm, draw_window_ring,
                                       remap_echo_type)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))

    m = axes[0].pcolormesh(lon, lat, dbz, cmap="turbo", vmin=-10, vmax=60,
                           shading="nearest")
    fig.colorbar(m, ax=axes[0], pad=0.01, label="dBZ")
    axes[0].set_title("Reflectivity", fontsize=10)
    # The panel is in degrees, so the ring must be told that: a km circle
    # is an ellipse in degree space.
    draw_window_ring(axes[0], lon, lat, result.texture_radius,
                     coord_units="degrees")

    m = axes[1].pcolormesh(
        lon, lat, result.convectivity, shading="nearest",
        norm=convectivity_norm(),
        cmap=convectivity_cmap(
            strat_mixed=class_params.max_convectivity_for_stratiform,
            mixed_conv=class_params.min_convectivity_for_convective))
    fig.colorbar(m, ax=axes[1], pad=0.01, label="convectivity")
    axes[1].set_title("Convectivity", fontsize=10)

    m = axes[2].pcolormesh(lon, lat, remap_echo_type(result.echo_type),
                           cmap=basic_echo_type_cmap(),
                           norm=basic_echo_type_norm(), shading="nearest")
    cb = fig.colorbar(m, ax=axes[2], ticks=range(1, 4), pad=0.01)
    cb.ax.set_yticklabels(BASIC_ECHO_TYPE_LABELS)
    axes[2].set_title("Echo type", fontsize=10)

    for ax in axes:
        ax.set_ylabel("latitude")
    axes[-1].set_xlabel("longitude")
    plt.tight_layout()

    path = outdir / "example_2d_h.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write a figure here (needs matplotlib)")
    main(ap.parse_args().outdir)
