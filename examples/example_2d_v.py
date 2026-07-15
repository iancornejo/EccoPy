"""
EccoPy-2D-V example workflow.

Use case: a vertical cross-section of reflectivity (RHI, model x-z slice,
range-height indicator, etc.), shape (Z, X).

This example shows all THREE modes:
  A. Basic only (no height/temp)          -> codes {1, 2, 3}
  B. With a height field                  -> codes {14,16,18,25,32,34,36,38}
  C. With a temperature field              -> same sub-classified codes

Replace the "YOUR DATA" section with real arrays.
"""
# Run with:  python3 examples/example_2d_v.py
# (from the eccopy_pkg/ directory, or after `pip install -e .`)

import numpy as np
from eccopy import eccopy2d_v
from eccopy.params import WindowSpec, VerticalParams


# ---------------------------------------------------------------------
# YOUR DATA — replace this block with real arrays
# ---------------------------------------------------------------------
# dbz: reflectivity, shape (Z, X), dBZ, NaN where missing
# coords_z: vertical coordinate, shape (Z,) -- km
# coords_x: horizontal coordinate, shape (X,) -- km
# height (optional): height field, shape (Z, X) -- km. Enables full
#   sub-classification. Use AGL (height above ground) if you have it;
#   MSL is fine over flat terrain.
# temp (optional): temperature field, shape (Z, X) -- deg C. Used only
#   if `height` is not supplied.
np.random.seed(1)
nz, nx = 20, 150
coords_z = np.linspace(0.5, 12, nz)     # km
coords_x = np.linspace(0, 100, nx)      # km

dbz = 18 - 0.7 * coords_z[:, None] + np.random.normal(0, 0.5, (nz, nx))
for iz in range(nz):
    dbz[iz, 70:80] += 25 * np.exp(-((iz - 4) ** 2) / 15)   # convective tower

# Simple height field: same altitude profile broadcast across X.
# In real data this usually comes straight from your grid's z-coordinate.
height = np.broadcast_to(coords_z[:, None], dbz.shape).copy()

# Simple temperature field from a constant lapse rate, for demonstration.
# In real data this would come from a model sounding or reanalysis.
temp = 20.0 - 6.5 * height
# ---------------------------------------------------------------------

window = WindowSpec((5, "km"))

print("=" * 60)
print("A. BASIC MODE (no height, no temp)")
print("=" * 60)
result_basic = eccopy2d_v.run(
    dbz, coords_z=coords_z, coords_x=coords_x, window=window,
)
print("codes:", np.unique(result_basic.echo_type[~np.isnan(result_basic.echo_type)]))
# -> only {1, 2, 3}: Stratiform / Mixed / Convective

print()
print("=" * 60)
print("B. WITH HEIGHT (full sub-classification)")
print("=" * 60)
vert_params = VerticalParams(
    shallow_threshold_ht=4.5,   # km -- boundary for shallow convection
    deep_threshold_ht=9.0,      # km -- boundary for deep convection
)
result_height = eccopy2d_v.run(
    dbz, coords_z=coords_z, coords_x=coords_x,
    height=height, window=window, vert_params=vert_params,
)
codes = np.unique(result_height.echo_type[~np.isnan(result_height.echo_type)])
label = {14: "StratLow", 16: "StratMid", 18: "StratHigh", 25: "Mixed",
         32: "ConvElevated", 34: "ConvShallow", 36: "ConvMid", 38: "ConvDeep"}
print("codes:", codes, "->", [label[int(c)] for c in codes])

print()
print("=" * 60)
print("C. WITH TEMPERATURE (used since height is not given here)")
print("=" * 60)
result_temp = eccopy2d_v.run(
    dbz, coords_z=coords_z, coords_x=coords_x,
    temp=temp, window=window,
    vert_params=VerticalParams(shallow_threshold_temp=0.0, deep_threshold_temp=-12.0),
)
codes_t = np.unique(result_temp.echo_type[~np.isnan(result_temp.echo_type)])
print("codes:", codes_t, "->", [label[int(c)] for c in codes_t])

# Note: if BOTH height and temp are supplied, height takes priority.

# All three results carry the same fields:
# result.echo_type     -> (Z, X) int array of codes
# result.convectivity  -> (Z, X) float, 0-1
# result.texture        -> (Z, X) float, dB
