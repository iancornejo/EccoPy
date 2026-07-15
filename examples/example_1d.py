"""
EccoPy-1D example workflow.

Use case: a single time series or 1-D distance profile of reflectivity
(e.g. one ray, one transect, one vertically-pointing radar time series).

Replace the "YOUR DATA" section with your real arrays — everything else
should work unchanged.
"""
# Run with:  python3 examples/example_1d.py
# (from the eccopy_pkg/ directory, or after `pip install -e .`)

import numpy as np
from eccopy import eccopy1d
from eccopy.params import WindowSpec, TextureParams, ClassificationParams


# ---------------------------------------------------------------------
# YOUR DATA — replace this block with real arrays
# ---------------------------------------------------------------------
# dbz: reflectivity, shape (N,), dBZ, NaN where missing
# coords: position OR spacing along the same axis, shape (N,)
#   - if these are actual locations (e.g. distance in km from a fixed
#     point, or elapsed time in seconds), coord_mode="position" below
#   - if these are already point-to-point deltas, coord_mode="spacing"
np.random.seed(0)
n = 300
coords = np.linspace(0, 150, n)                       # km, monotonic positions
dbz = 18 + 2 * np.sin(coords / 20) + np.random.normal(0, 0.5, n)
dbz[140:156] += 28 * np.exp(-0.5 * ((np.arange(-8, 8)) / 3) ** 2)  # embedded storm
dbz[50:55] = np.nan                                    # a data gap, for realism
# ---------------------------------------------------------------------

# Step 1: choose your texture window.
# WindowSpec((5, 'km')) -> 5 km radius, resolved per-point against `coords`.
# Use WindowSpec(7) instead if you want a fixed 7-PIXEL radius regardless
# of physical spacing.
window = WindowSpec((5, "km"))

# Step 2 (optional): override default thresholds. Skip this if the
# defaults (texture_limit_high=30, strat/mixed/conv at 0.4/0.5) are fine.
texture_params = TextureParams(texture_limit_high=30.0)
class_params = ClassificationParams(
    max_convectivity_for_stratiform=0.4,
    min_convectivity_for_convective=0.5,
)

# Step 3: run the pipeline.
result = eccopy1d.run(
    dbz,
    coords=coords,
    window=window,
    coord_mode="position",          # "position" | "spacing" | "auto"
    texture_params=texture_params,
    class_params=class_params,
)

# Step 4: use the result.
# result.echo_type    -> (N,) int array: 1=Stratiform, 2=Mixed, 3=Convective
# result.convectivity -> (N,) float array, 0-1
# result.texture       -> (N,) float array, dB

print("echo_type unique codes:", np.unique(result.echo_type[~np.isnan(result.echo_type)]))
n_strat = np.sum(result.echo_type == 1)
n_mixed = np.sum(result.echo_type == 2)
n_conv = np.sum(result.echo_type == 3)
print(f"Stratiform: {n_strat}  Mixed: {n_mixed}  Convective: {n_conv}")

# Quick sanity check: convectivity should be high inside the storm, low outside
print(f"Convectivity at storm center (idx 148): {result.convectivity[148]:.3f}")
print(f"Convectivity in background (idx 10):    {result.convectivity[10]:.3f}")
