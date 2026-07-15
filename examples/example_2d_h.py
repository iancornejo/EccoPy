"""
EccoPy-2D-H example workflow.

Use case: a single horizontal level or composite (e.g. column-max
reflectivity composite, a CAPPI, a single PPI sweep regridded to
Cartesian), shape (Y, X).

Only basic classification (Stratiform / Mixed / Convective) is possible
here -- there's no vertical axis to determine cloud depth from, so
eccopy2d_h.run() has no height/temp parameters at all.

This example also shows the `kernel_mode` option, relevant if your grid
spacing is non-uniform (e.g. a lat/lon grid).

Replace the "YOUR DATA" section with real arrays.
"""
# Run with:  python3 examples/example_2d_h.py
# (from the eccopy_pkg/ directory, or after `pip install -e .`)

import numpy as np
from eccopy import eccopy2d_h
from eccopy.params import WindowSpec


# ---------------------------------------------------------------------
# YOUR DATA — replace this block with real arrays
# ---------------------------------------------------------------------
# dbz: reflectivity, shape (Y, X), dBZ, NaN where missing
# coords_y: N-S coordinate, shape (Y,) -- km
# coords_x: E-W coordinate, shape (X,) -- km
np.random.seed(2)
ny, nx = 60, 80
coords_y = np.linspace(-30, 30, ny)
coords_x = np.linspace(-40, 40, nx)

dbz = 15 + np.random.normal(0, 0.5, (ny, nx))
yy, xx = np.meshgrid(coords_y, coords_x, indexing="ij")
dbz += 30 * np.exp(-((yy - 5) ** 2 + (xx - 10) ** 2) / 30)   # one storm cell
# ---------------------------------------------------------------------

window = WindowSpec((7, "km"))

result = eccopy2d_h.run(
    dbz, coords_y=coords_y, coords_x=coords_x,
    window=window,
    kernel_mode="uniform",   # use "varying" for wide-latitude-range lat/lon grids
)

print("codes:", np.unique(result.echo_type[~np.isnan(result.echo_type)]))
# -> only {1, 2, 3}: Stratiform / Mixed / Convective

# result.echo_type        -> (Y, X) int array
# result.convectivity     -> (Y, X) float, 0-1
# result.texture           -> (Y, X) float, dB
# result.fraction_active  -> (Y, X) float, 0-1: kernel coverage fraction
#                            (useful for spotting low-confidence regions
#                            near the edge of radar coverage)

iy = int(np.argmin(np.abs(coords_y - 5)))
ix = int(np.argmin(np.abs(coords_x - 10)))
print(f"At the storm cell ({coords_y[iy]:.0f}, {coords_x[ix]:.0f} km): "
      f"echo_type={result.echo_type[iy, ix]:.0f}, "
      f"convectivity={result.convectivity[iy, ix]:.3f}")

# ---------------------------------------------------------------------
# WORKING WITH LAT/LON INSTEAD OF A NATIVE X/Y GRID
# ---------------------------------------------------------------------
# If your data is on a lat/lon grid rather than a projected x/y grid,
# convert it to local km spacing first using the haversine helper, then
# pass coord_mode="spacing" (since these are now point-to-point deltas,
# not coordinate positions):
#
#   from eccopy import latlon_to_xy_spacing
#   dy_km, dx_km = latlon_to_xy_spacing(lat, lon)   # lat, lon: (Y, X) arrays
#   result = eccopy2d_h.run(
#       dbz, coords_y=dy_km, coords_x=dx_km, coord_mode="spacing",
#       window=WindowSpec((7, "km")),
#       kernel_mode="varying",   # recommended for lat/lon grids spanning
#                                 # a wide latitude range -- see README
#   )
