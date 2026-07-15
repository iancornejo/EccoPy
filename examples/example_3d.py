"""
EccoPy-3D example workflow.

Use case: a full 3-D Cartesian reflectivity volume, shape (Z, Y, X)
(e.g. a gridded radar volume, a model output volume).

This example shows:
  A. Basic only (no height/temp)
  B. With a height field (full sub-classification)
  C. Tuning the clumping volume threshold
  D. Working from a lat/lon grid instead of native x/y

Replace the "YOUR DATA" section with real arrays.
"""
# Run with:  python3 examples/example_3d.py
# (from the eccopy_pkg/ directory, or after `pip install -e .`)

import numpy as np
from eccopy import eccopy3d
from eccopy.params import WindowSpec, ClassificationParams, VerticalParams


# ---------------------------------------------------------------------
# YOUR DATA — replace this block with real arrays
# ---------------------------------------------------------------------
# dbz: reflectivity, shape (Z, Y, X), dBZ, NaN where missing
# coords_z: vertical coordinate, shape (Z,) -- km
# coords_y: N-S coordinate, shape (Y,) -- km
# coords_x: E-W coordinate, shape (X,) -- km
# height (optional): shape (Z, Y, X) -- km
# temp (optional): shape (Z, Y, X) -- deg C
np.random.seed(3)
nz, ny, nx = 12, 40, 50
coords_z = np.linspace(0.5, 11.5, nz)
coords_y = np.linspace(-20, 20, ny)
coords_x = np.linspace(-25, 25, nx)

dbz = 15 - 0.8 * coords_z[:, None, None] + np.random.normal(0, 0.7, (nz, ny, nx))
# A large, deep storm cell
for iz in range(nz):
    weight = np.exp(-((iz - 4) ** 2) / 14)
    dbz[iz, 15:25, 20:32] += 32 * weight
# Two small, sharp, shallow cells -- intense enough to register as their
# own clumps, but small enough in volume to be filtered out by a
# strict min_valid_volume_for_convective (see part C below)
for iz in range(2):
    dbz[iz, 5:7, 5:7] += 35
    dbz[iz, 32:34, 38:40] += 35

height = np.broadcast_to(coords_z[:, None, None], dbz.shape).copy()
# ---------------------------------------------------------------------

window = WindowSpec((5, "km"))

print("=" * 60)
print("A. BASIC MODE (no height, no temp)")
print("=" * 60)
result_basic = eccopy3d.run(
    dbz, coords_z=coords_z, coords_y=coords_y, coords_x=coords_x,
    window=window,
)
print(f"n_clumps: {result_basic.n_clumps}")
print(f"codes: {np.unique(result_basic.echo_type[result_basic.echo_type > 0])}")
# Note: in basic mode, convective sub-type (here likely 36=ConvMid)
# is a FALLBACK, not a measurement -- it cannot distinguish
# shallow/mid/deep/elevated without height or temp. Only the
# clump/no-clump distinction (and mixed vs convective) is meaningful.

print()
print("=" * 60)
print("B. WITH HEIGHT (full sub-classification)")
print("=" * 60)
vert_params = VerticalParams(shallow_threshold_ht=4.5, deep_threshold_ht=9.0)
result_height = eccopy3d.run(
    dbz, coords_z=coords_z, coords_y=coords_y, coords_x=coords_x,
    height=height, window=window, vert_params=vert_params,
)
codes = np.unique(result_height.echo_type[result_height.echo_type > 0])
label = {14: "StratLow", 16: "StratMid", 18: "StratHigh", 25: "Mixed",
         32: "ConvElevated", 34: "ConvShallow", 36: "ConvMid", 38: "ConvDeep"}
print(f"n_clumps: {result_height.n_clumps}")
print(f"codes: {codes} -> {[label[int(c)] for c in codes]}")

print()
print("=" * 60)
print("C. TUNING THE CLUMPING VOLUME THRESHOLD")
print("=" * 60)
# min_valid_volume_for_convective: minimum 3-D volume (km^3) a clump
# must have to be classified as convective rather than falling back to
# Mixed. Default is 20.0; the validated SPOL Taiwan reference case used
# 30.0 -- there is no universal "correct" value, it depends on your
# radar/model resolution and what you want to call "convective".
cp_loose = ClassificationParams(min_valid_volume_for_convective=1.0)
cp_default = ClassificationParams()  # default: 20.0 km^3
cp_strict = ClassificationParams(min_valid_volume_for_convective=100.0)

result_loose = eccopy3d.run(
    dbz, coords_z=coords_z, coords_y=coords_y, coords_x=coords_x,
    window=window, class_params=cp_loose,
)
result_default = eccopy3d.run(
    dbz, coords_z=coords_z, coords_y=coords_y, coords_x=coords_x,
    window=window, class_params=cp_default,
)
result_strict = eccopy3d.run(
    dbz, coords_z=coords_z, coords_y=coords_y, coords_x=coords_x,
    window=window, class_params=cp_strict,
)
print(f"loose   (1 km^3 min):   n_clumps={result_loose.n_clumps}")
print(f"default (20 km^3 min):  n_clumps={result_default.n_clumps}")
print(f"strict  (100 km^3 min): n_clumps={result_strict.n_clumps}")
print("(the two small, sharp cells get filtered out as the threshold rises,")
print(" leaving only the large storm; this is exactly what the volume filter")
print(" is for -- ignoring small, intense, but probably noise-driven blips)")

# result.echo_type        -> (Z, Y, X) int array
# result.convectivity     -> (Z, Y, X) float, 0-1
# result.texture           -> (Z, Y, X) float, dB
# result.fraction_active  -> (Y, X) float, 0-1 (per-level kernel coverage)
# result.n_clumps          -> int, number of convective clumps found

print()
print("=" * 60)
print("D. WORKING FROM A LAT/LON GRID (commented template)")
print("=" * 60)
print("""
If your data is on a lat/lon grid rather than native x/y:

    from eccopy import latlon_to_xy_spacing

    # lat, lon: shape (Y, X) -- a single representative horizontal slice
    # is enough; spacing doesn't usually vary with height.
    dy_km, dx_km = latlon_to_xy_spacing(lat, lon)

    result = eccopy3d.run(
        dbz, coords_z=z_km,           # vertical axis is usually still a
                                        # plain 1-D km array
        coords_y=dy_km, coords_x=dx_km,
        coord_mode="spacing",          # these are now deltas, not positions
        height=height,
        window=WindowSpec((7, "km")),
        kernel_mode="varying",         # recommended for grids spanning a
                                         # wide latitude range -- see README
    )
""")
