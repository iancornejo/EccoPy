# EccoPy

Data-agnostic Python implementation of the **ECCO** (Echo Classification for
Convection and Others) and **ConvStratFinder** radar algorithms.

EccoPy operates purely on numpy arrays — it never reads or writes files, and
makes no assumptions about coordinate systems, file formats, or radar type.
You give it reflectivity (and optionally height/temperature) arrays plus
coordinate or spacing arrays; it gives you back classification arrays of the
same shape.

Algorithm references: 
    - [Romatschke and Dixon (2022), JTECH](https://doi.org/10.1175/JTECH-D-22-0019.1) / [Dixon and Romatschke (2022), JTECH](https://doi.org/10.1175/JTECH-D-22-0018.1)
    
Original MATLAB/C++ source: 
    - [NCAR/lrose-ecco](https://github.com/NCAR/lrose-ecco) / [NCAR/lrose-core](https://github.com/ncar/lrose-core)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development/validation
workflow and [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Installation

```bash
pip install eccopy
# with plotting support:
pip install "eccopy[plot]"
```

---

## The four modules

| Module        | Input shape         | Use case                                  | Sub-classification |
|---------------|----------------------|--------------------------------------------|---------------------|
| `eccopy1d`    | `(T,)` or `(N,)`     | Single time series / 1-D profile           | No — basic only     |
| `eccopy2d_v`  | `(Z, X)`             | Vertical cross-section (RHI, model x-z)    | Yes, with height or temp |
| `eccopy2d_h`  | `(Y, X)`             | Single horizontal level / composite        | No — basic only     |
| `eccopy3d`    | `(Z, Y, X)`          | Full 3-D Cartesian volume                  | Yes, with height or temp |

**Basic** classification (always available): every pixel gets one of
`1` (Stratiform), `2` (Mixed), `3` (Convective).

**Sub-classified** (eccopy2d_v / eccopy3d, when `height` or `temp` is given):
`14`/`16`/`18` (Stratiform Low/Mid/High), `25` (Mixed),
`32`/`34`/`36`/`38` (Convective Elevated/Shallow/Mid/Deep).

---

## Workflow

All four modules follow the same pipeline:

```
reflectivity texture  ->  convectivity  ->  classification
```

`eccopy3d` adds a 3-D clumping step (validated against LROSE
ConvStratFinder output) between convectivity and final classification.

---

## Runnable examples

`examples/` contains a complete, runnable script for each module
(`example_1d.py`, `example_2d_v.py`, `example_2d_h.py`, `example_3d.py`),
each with a "YOUR DATA" section marking exactly what to swap in for your
own arrays:

```bash
pip install -e .
python3 examples/example_3d.py
```

`notebooks/workflow_examples/` contains a self-contained (synthetic
data, no external files) Jupyter notebook per module covering every
parameter option in detail, `return_intermediates`/`eccopy.stats`
usage, and plotting with EccoPy's colormaps -- start here if you're new
to the package.

`notebooks/` (one level up) also contains minimal Jupyter workflow
templates for validating against real MATLAB ECCO-V or LROSE-ECCO
output (`eccov_workflow.ipynb`, `eccopy3d_workflow.ipynb`): load a case
→ clean fill values → run EccoPy → load the reference output.
Comparison/metrics/plotting are left to you. See "Validation status"
below for known gaps and fill-value gotchas found while building these.

---

## Quick start

### EccoPy-1D — time series / 1-D profile

```python
import numpy as np
from eccopy import eccopy1d
from eccopy.params import WindowSpec

dbz     = np.array([...])    # shape (T,), dBZ, NaN where missing
coords  = np.array([...])    # shape (T,), e.g. time in seconds or distance in km

result = eccopy1d.run(dbz, coords=coords, window=WindowSpec((5, "km")))

result.echo_type       # (T,) -- values in {1, 2, 3}
result.convectivity    # (T,) -- float, 0-1
result.texture          # (T,) -- float
```

### EccoPy-2D-V — vertical cross-section

```python
from eccopy import eccopy2d_v
from eccopy.params import WindowSpec

dbz      = ...                # (Z, X)
height   = ...                # (Z, X), km -- optional, enables sub-classification
temp     = ...                # (Z, X), C -- optional, used if height not given

result = eccopy2d_v.run(
    dbz, coords_z=z_km, coords_x=x_km,
    height=height,                       # or temp=temp, or neither
    window=WindowSpec((7, "km")),
)
result.echo_type   # (Z, X) -- basic {1,2,3} or sub-classified codes
```

If neither `height` nor `temp` is supplied, `eccopy2d_v` falls back to basic
stratiform/mixed/convective classification automatically.

### EccoPy-2D-H — single horizontal level / composite

```python
from eccopy import eccopy2d_h
from eccopy.params import WindowSpec

dbz_composite = ...   # (Y, X) -- e.g. a column-max composite

result = eccopy2d_h.run(
    dbz_composite, coords_y=y_km, coords_x=x_km,
    window=WindowSpec((7, "km")),
)
result.echo_type   # (Y, X) -- values in {1, 2, 3} (no sub-classification possible)
```

### EccoPy-3D — full 3-D volume

```python
from eccopy import eccopy3d
from eccopy.params import WindowSpec

dbz    = ...   # (Z, Y, X)
height = ...   # (Z, Y, X), km -- optional

result = eccopy3d.run(
    dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km,
    height=height,                       # or temp=..., or neither
    window=WindowSpec((7, "km")),
)
result.echo_type    # (Z, Y, X)
result.n_clumps      # number of convective clumps found
```

---

## Coordinate / spacing arrays

`coords_z`, `coords_y`, `coords_x` (and `eccopy1d`'s `coords`) can be supplied
in two ways, controlled by `coord_mode`:

- **`coord_mode="position"`** -- actual coordinate values at each point
  (e.g. `[-50, -49, -48, ..., 50]` km). Local spacing is derived via
  `np.diff`.
- **`coord_mode="spacing"`** -- already point-to-point spacing
  (e.g. `[1, 1, 1, ..., 1]` km if the grid is uniform with 1 km spacing).
- **`coord_mode="auto"`** (default) -- detects monotonic arrays as
  "position", everything else as "spacing".

Arrays can be 1-D (one value/spacing per index along that axis, broadcast
across the others) or full-rank (a genuinely varying field, e.g. a 2-D
`(Z, X)` spacing array for a range-height grid where spacing changes with
range and height).

### Working with latitude/longitude

EccoPy doesn't know about map projections, but provides a haversine helper
to convert a lat/lon grid into local x/y spacing:

```python
from eccopy import latlon_to_xy_spacing

dy_km, dx_km = latlon_to_xy_spacing(lat, lon)   # lat, lon: (Y, X) arrays
result = eccopy2d_h.run(dbz, coords_y=dy_km, coords_x=dx_km,
                        coord_mode="spacing", window=WindowSpec((7, "km")))
```

For idealized model data with native x/y grids in km (no lat/lon at all),
just pass those coordinate arrays directly -- no conversion needed.

```python
from eccopy import haversine_distance
d_km = haversine_distance(lat1, lon1, lat2, lon2)   # great-circle distance
```

---

## WindowSpec — physical-unit texture windows

```python
from eccopy.params import WindowSpec

WindowSpec(7)              # bare 7-pixel radius
WindowSpec((5, "km"))      # 5 km radius -- resolved against the spacing array
WindowSpec((3, "minute"))  # 3-minute radius -- for eccopy1d time series
```

When a physical-unit `WindowSpec` is used, the number of grid points it
covers is computed **per point** from the local spacing -- so the same
`WindowSpec((5, "km"))` will use more grid points where spacing is finer and
fewer where spacing is coarser, rather than assuming one fixed pixel count
for the whole array.

---

## Non-uniform grids (e.g. lat/lon): `kernel_mode`

`eccopy2d_h` and `eccopy3d` use a circular kernel to compute texture at each
point. On a lat/lon grid, `dx` (east-west spacing in km) shrinks with
`cos(latitude)` while `dy` stays roughly constant — so a uniform lat/lon grid
is *not* a uniform km grid. Both functions expose a `kernel_mode` argument
to control how this is handled:

```python
result = eccopy2d_h.run(
    dbz, coords_y=dy_km, coords_x=dx_km, coord_mode="spacing",
    window=WindowSpec((7, "km")),
    kernel_mode="uniform",   # or "varying"
)
```

- **`"uniform"` (default)** — builds one kernel from the median `dy`/`dx`
  across the whole grid, reused everywhere. This is what the validated
  LROSE ConvStratFinder algorithm itself does for lat/lon grids — it
  computes a single representative `dx`/`dy` at the domain's mean latitude
  rather than adjusting per point. Fast (one kernel built once). Appropriate
  for radar-sized domains (a few hundred km) or grids where spacing doesn't
  vary much. Becomes increasingly approximate as the domain's latitude
  range grows.

- **`"varying"`** — rebuilds the kernel from the *local* spacing at every
  grid point, so the kernel's physical footprint stays correct everywhere
  regardless of domain size. This is **not** what LROSE does, and has not
  been validated against LROSE output — it's a genuine algorithmic
  extension for domains where the single-kernel approximation breaks down
  (continental or global lat/lon grids). Noticeably slower, since no kernel
  is reused between points.

If `kernel_mode="varying"` and the texture radius is smaller than the local
grid spacing anywhere in the domain, EccoPy raises a `UserWarning` — at
those points the kernel collapses to a single cell and texture is always 0
there, which usually means the radius needs to be larger or the grid needs
to be coarser for that domain.

---

## Parameters

Default thresholds match the validated `Ecco.spol` / ConvStratFinder
parameter set:

```python
from eccopy.params import TextureParams, ClassificationParams, VerticalParams

texture_params = TextureParams(
    texture_limit_low=0.0,
    texture_limit_high=30.0,
)
class_params = ClassificationParams(
    min_convectivity_for_convective=0.5,
    max_convectivity_for_stratiform=0.4,
    min_valid_volume_for_convective=30.0,   # eccopy3d clumping
    secondary_convectivity=0.65,             # eccopy3d dual-threshold clumping
)
vert_params = VerticalParams(
    shallow_threshold_ht=4.5,       # km
    deep_threshold_ht=9.0,          # km
    shallow_threshold_temp=0.0,     # C
    deep_threshold_temp=-12.0,      # C
)

result = eccopy3d.run(
    dbz, coords_z=z_km, coords_y=y_km, coords_x=x_km, height=height,
    texture_params=texture_params,
    class_params=class_params,
    vert_params=vert_params,
)
```

> **Warning — `each_subclump_min_area_km2` means different physical
> quantities in `eccopy3d` vs. `eccopy2d_h`.** Both reuse the same
> `ClassificationParams` field, but `eccopy3d` compares it against a
> C++-quirk pseudo-*volume* (`n_pixels_2d * dx_km * dy_km * dz_km` at
> level 0 of the full grid — faithfully reproduced from LROSE's
> `ClumpProps.cc`, see `eccopy3d/clumping.py`), while `eccopy2d_h`
> compares it against genuine *area* (`spacing_y * spacing_x`, summed
> per pixel — the physically correct choice for a single level with no
> `dz`). If you reuse a `ClassificationParams` object tuned/validated
> against your 3-D cases directly for `eccopy2d_h`, the same numeric
> threshold now gates a different quantity and sub-clump splitting
> behavior will not transfer the way you'd expect. Tune this parameter
> separately for each module rather than sharing one `ClassificationParams`
> instance across both.

---

## Performance

The per-pixel hot loops (2-D radial texture, 1-D sliding-window texture,
and `isotherm_height`) are JIT-compiled with [Numba](https://numba.pydata.org/),
giving roughly a 35-300x speedup over the pure-Python implementation they
started as — e.g. the 30-level, 500x500 SPOL Taiwan validation case (see
below) runs in ~7 seconds instead of an estimated 34 minutes. The first
call in a given Python process pays a one-time JIT compilation cost
(a few seconds, cached to disk afterwards); every call after that runs
at full compiled speed.

Every Numba-converted function was cross-validated against a frozen copy
of the pre-conversion pure-Python implementation across tens of thousands
of randomized test points (`eccopy/tests/test_numba_cross_validation.py`)
before being accepted — this caught a real translation bug (`sqrt(var)`
instead of the correct `sqrt(sqrt(var))`) that none of the higher-level
entry-point tests noticed, since "looks like a plausible echo
classification" doesn't catch "the underlying number is wrong."

---

## Validation status

### EccoPy-3D — validated across 3 independent real LROSE-ECCO cases

EccoPy-3D has been validated end-to-end (raw DBZ → texture → convectivity →
clumping → classification, the full pipeline, not just a partial/decoded
comparison) against real LROSE ConvStratFinder truth output, across **5
runs spanning 3 independent radar/model cases and 2 texture-radius
variants each**:

| Case | Grid (Z,Y,X) | texture_radius | Overall agreement | Both-echo agreement |
|------|--------------|-----------------|--------------------|----------------------|
| SPOL Taiwan (RHI, 2022-05-26) | 30×500×500 | 4 km | 99.94% | 99.81% |
| SPOL Taiwan | 30×500×500 | 7 km | 99.87% | 99.61% |
| SEA (2025-01-11) | 40×801×801 | 4 km | 99.98% | 99.83% |
| SEA | 40×801×801 | 7 km | 99.93% | 99.50% |
| WRF ensemble (2022-05-26) | 46×750×750 | 4 km | 99.80% | 99.49% |

Every echo-type sub-classification code — Stratiform Low/Mid/High,
Mixed, and **all four** convective sub-types (Shallow, Mid, Deep, and
Elevated, the last of which only appears in the SEA 7km case and was
otherwise completely unexercised) — has been checked against real truth
and matches closely. The Missing/echo boundary matches truth exactly or
near-exactly in every case.

This is the result of two real bugs found and fixed by comparing against
this truth data, both significant:

1. **Clumping was structurally wrong.** The original port did straight
   3-D connected-component labeling directly at the secondary
   (0.65) convectivity threshold. Real LROSE-ECCO output revealed this
   captured only 2.7% of true convective pixels (96.5% of truth-convective
   pixels were mis-assigned as Mixed) on the first real test case. The
   actual algorithm is a two-stage "envelope → 2-D composite → secondary-
   threshold subclump-split → grow → 3-D remask" process, ported exactly
   from lrose-core's `ClumpingDualThresh.cc`/`ClumpingMgr.cc`/
   `ClumpProps.cc` sources. See `eccopy/eccopy3d/clumping.py`'s module
   docstring for the full algorithm description, including a faithfully
   reproduced unit-convention quirk in the C++'s `each_subclump_min_area_km2`
   check. Fixed this from 2.7% to 98%+ convective-pixel recall.
2. **`min_valid_dbz` only gated the coverage-fraction count, not the
   actual texture computation.** The real algorithm nulls every dBZ value
   below `min_valid_dbz` to missing *before* computing column-max,
   coverage fraction, AND texture — ours kept sub-threshold values as
   valid texture-contributing neighbors, just excluded from the coverage
   count. This inflated texture/convectivity at the edges of valid-data
   regions. Fixed in `refl_texture_2d()` (`eccopy/core/texture.py`);
   closed a 191,000-pixel (2.5%) Missing-boundary mismatch down to zero
   in the case that exposed it.

Also fixed along the way:
  - A `<=` vs `<` boundary-convention mismatch in the shallow-height
    stratiform sub-classification check (`_compute_geom()` in
    `clumping.py`), found by line-by-line comparison against
    `StormClump::computeGeom()`.
  - `resolve_spacing()`'s simple forward-difference Z-spacing convention
    diverges from `PjgGridGeom::dzKm()`'s centered-difference convention
    for **non-uniform** vertical grids (identical for uniform grids like
    SPOL/SEA, but WRF's grid coarsens from 0.25 km near-surface to 1 km
    aloft). Fixed with a scoped, 3-D-only centered-difference calculation
    in `eccopy3d/classify.py` — does not touch `resolve_spacing()` itself,
    so the validated 2-D-V texture-kernel path is unaffected.
  - The original WRF test case's input file turned out to have a
    genuinely wrong/mismatched DBZ field (dominated by a `-40.0` fill
    sentinel covering 37% of the volume, absent from the corrected file,
    with a mean absolute difference of ~5.2 dBZ where both were finite).
    This was diagnosed by noticing texture agreement degraded smoothly
    with altitude (correlation 0.998 near-surface down to 0.84 aloft)
    despite the texture *algorithm* cross-validating bit-exact against a
    from-scratch reference reimplementation — i.e. correct code, wrong
    input data. Confirmed by re-running against a corrected DBZ field.

**Known unvalidated / lower-confidence areas:**
  - `min_overlap_for_convective_clumps` > 1 (all 3 cases use the TDRP
    default of 1, which is exactly equivalent to standard 6-/4-
    connectivity labeling — see `clumping.py`'s module docstring). A
    true interval-overlap implementation would be needed for cases using
    a higher value.
  - `topo_km`'s literal `height_km - topo_km` AGL-subtraction mechanism
    (in both `find_clumps_3d()` and `set_echo_type_3d()`) does not appear
    to exist in the real 3-D C++ path at all — `ConvStratFinder` only has
    `terrainHt`, which *raises the shallow/deep threshold boundaries*
    (the already-validated `terrain_ht_km` parameter), and never
    subtracts anything from the height field itself. `topo_km` was
    carried over from the 2-D MATLAB `f_classSub.m` port; none of the 3
    real test cases populate it, so this has not been exercised or
    validated for the 3-D path.
  - The flood-fill sub-clump growth step (`_grow_regions()` in
    `clumping.py`) is a vectorized simultaneous-direction approximation
    of the C++'s randomized single-cell visit order; ties are broken by
    direction-scan order instead. Negligible on real floating-point data
    in all cases checked, but not bit-identical by construction.

### EccoPy-2D-V — validated against real MATLAB ECCO-V output

EccoPy-2D-V has been validated end-to-end against real MATLAB ECCO-V
output across 3 cases (SEA Test1, SEA Test6, SPOL RHI with temperature
data), achieving **99.7% (SEA Test1), 100.0% (SEA Test6), and 99.4%
(SPOL)** agreement, after a series of real bugs were found and fixed via
ground-truth comparison against MATLAB-exported intermediate arrays at
each algorithmic stage:
  - **scipy `border_value` bug**: `binary_erosion`/`binary_closing`'s
    default `border_value=0` violates the mathematical extensivity of
    closing; fixed with `border_value=1` for standalone erosion.
  - **Disk structuring-element mismatch**: a naive Euclidean
    approximation over-reaches MATLAB's `strel('disk', r)`; fixed by
    loading the exact MATLAB-exported neighborhood arrays.
  - **Morphological closing decomposition**: scipy's monolithic
    dilate-then-erode diverges from MATLAB's per-primitive sequential
    decomposition; fixed via `_sequential_close()` — bit-exact for
    axis-aligned primitives, ~97-98% for diagonal (see "Known remaining
    gap" below).
  - **`class_sub_2d()` logic**: was testing height-threshold AGL
    thresholds against the wrong physical quantity; the real MATLAB
    algorithm uses melt-field values (threshold 15) as the primary
    signal, temperature (threshold -25°C) to distinguish mid from
    deep/high, with near-surface AGL corrections (2000m/4000m) applied
    to melt and temp before use.
  - `eccopy2d_v.run()`'s `height` parameter was documented as km but
    passed straight through to `class_sub_2d` (which expects metres),
    silently shrinking every height value by 1000x. Fixed;
    `height`/`topo` are now correctly converted km→m at the
    `eccopy2d_v.run()` boundary.
  - `topo` (terrain height / earth-curvature beam-height correction) was
    not exposed at all in `eccopy2d_v.run()` or `eccopy3d.run()`. Added
    as a proper parameter to both, plus `terrain_ht` for `eccopy3d.run()`'s
    separate threshold-raising terrain mechanism.
  - `find_clumps_3d()` (shared 3-D-adjacent code) had an OOM crash on
    WRF-scale grids from full-grid boolean masking per clump; fixed with
    `scipy.ndimage.find_objects()` bounding-box restriction.
  - `_strat_below()` was decrementing the X axis instead of Z.
  - Data-file paths in `classification.py` were bare relative paths
    (real `FileNotFoundError` when run from notebooks); fixed to be
    `__file__`-anchored.
  - `ClassificationParams.enlarge_conv` defaulted to 3, which was never
    the real MATLAB default and had only ever been exercised via
    `enlarge_conv=5` in the real validation runs above; confirmed
    against the MATLAB `f_classBasic.m` source and fixed to 5, matching
    `enlarge_mixed`.

**API/packaging clarifications made along the way:** `eccopy2d_v.run()`
requires `height`, `melt`, and `temp` together for sub-classification;
`VerticalParams` is 3-D-path-only (its threshold fields were removed
from the 2-D call site); `surf_alt_lim` is correctly sourced from
`ClassificationParams`; MATLAB-exported `disk_strel`/`disk_decomp` `.mat`
files are packaged with the library.

**Known open items:**
  - `class_basic_isotropic()` has not been separately re-validated
    against real MATLAB output after the structural fixes above (this
    session's EccoPy-3D work exercised adjacent `classification.py`
    logic, but not this function specifically).
  - A small, unexplained residual disagreement remains for diagonal
    morphological-decomposition primitives (~97-98% vs. bit-exact for
    axis-aligned ones).

**Fixed via direct comparison against the real f_classBasic.m source
(lrose-ecco repo):**
  - The rain-below-melting-layer correction block in `class_basic()`
    (see `melt` parameter docstring) had two real issues, found by
    reading the actual MATLAB source line-by-line rather than inferring
    from input/output data alone:
    1. An earlier attempt this session to fix the block's apparent
       real-data inertness changed its hardcoded threshold from 20/10 to
       15/9, reasoning from `class_sub_2d`'s documented convention. That
       was **wrong** -- the real MATLAB source uses `meltArea<20` and a
       sentinel of 10, exactly matching the *original*, pre-session
       Python constants. Reverted back to 20/10.
    2. A genuine, previously-undiscovered **off-by-one translation bug**:
       MATLAB's `checkCol(1:firstInd)=1` is inclusive of `firstInd`
       (1-indexed); the direct Python translation needs
       `check_col[:first_ind + 1] = 1`, but the port wrote
       `check_col[:first_ind] = 1`, excluding `first_ind`. If the
       convectivity value exactly at the melt-crossing pixel is NaN,
       this silently zeroes out that column's entire contribution to the
       stratiform-percentage check. Confirmed in isolation and fixed.

    Tested against the real SPOL case (`ECCO_V_CASE_3_SPOL_NEW.zip`,
    20220526_084500): neither fix changes that case's final output --
    the one real clump that clears the `below_frac` gate has
    `strat_perc=0.66` (genuinely below the 0.8 trigger; cross-checked
    against real MATLAB `ECHOTYPE` ground truth showing that clump is
    95.7% Convective Mid, i.e. real embedded convection, correctly not
    reclassified), and none of its 226 columns happen to hit the
    off-by-one edge case. Both fixes are nonetheless verified directly
    against the real MATLAB source -- the strongest evidence this
    project's own validation standard calls for -- rather than inferred
    from behavior on a single case that doesn't happen to exercise every
    path. A case with a genuine rain-below-melt signature (or MATLAB's
    own intermediate `classBasic` output) would be needed to get a
    numeric before/after agreement delta, but the fixes themselves don't
    depend on that to be correct.

**Also found, NOT a package bug:** Real LROSE/MATLAB MDV-derived data
commonly uses a literal `-9999.0` fill value that is NOT NaN. EccoPy has
no way to know this convention on its own, so **clean `-9999`-style fill
values to NaN yourself before calling any EccoPy function** — see
`notebooks/` for the exact one-line fix. Skipping this on the SPOL case
dropped apparent agreement from 99.4% to 18.2%, even though nothing in
EccoPy itself was wrong.

EccoPy-1D has not yet been validated against real output (no 1-D ECCO-V
test case was available during development; the 2-D-V validation above
exercises the same underlying `refl_texture_1d`/`class_basic` code path
EccoPy-1D uses).

EccoPy-2D-H has not yet been validated against real output (it shares
`refl_texture_2d()` with EccoPy-3D, so the texture/convectivity stage is
covered by the EccoPy-3D validation above, but the basic-classification
logic downstream of that has not been separately checked).

---

## Inspecting intermediate calculations

Two ways to inspect what happens between reflectivity and the final
texture value, at two different scales:

### Whole-array: `return_intermediates=True`

Every module's `run()` accepts `return_intermediates=True` to attach the
fitted-trend and detrended fields (the values computed just before
texture, for every point at once) directly to the returned `Result`
object, at no cost when left at its default `False`:

```python
result = eccopy1d.run(dbz, coords=coords, window=WindowSpec((5, "km")),
                      return_intermediates=True)
result.fitted_dbz      # (N,) -- local linear-fit value at each point
result.detrended_dbz   # (N,) -- fit removed + re-centred, clipped >= 1

# eccopy2d_v additionally exposes:
result.echo_basic      # basic strat/mixed/conv classification BEFORE
                        # sub-classification is applied

# eccopy3d additionally requires an explicit levels=[...] list (the
# per-point debug loop it uses is slow -- see below):
result = eccopy3d.run(..., return_intermediates=True, levels=[3, 4, 5])
result.fitted_dbz            # (len(levels), Y, X)
result.intermediate_levels   # the Z-indices fitted_dbz corresponds to
```

`eccopy1d`/`eccopy2d_v` compute these via the same fast Numba core as
production texture (negligible extra cost). `eccopy2d_h`/`eccopy3d` use
a plain Python per-point loop instead (`core.debug.refl_texture_2d_field_debug`)
— noticeably slower than the rest of `run()`, intended for debugging a
specific level/region rather than routine use on full volumes.

### Single-point: debug functions

For understanding or sanity-checking *why one specific point* got the
value it did, two lower-level functions expose every intermediate
quantity for a single point, reproducing production's output exactly
(including edge cases like array borders and NaN gaps):

```python
from eccopy.core.debug import refl_texture_1d_debug, refl_texture_2d_debug

# 1-D (eccopy1d / eccopy2d_v)
dbg = refl_texture_1d_debug(dbz, window, index=15, spacing=spacing_m)
dbg.window_dbz     # raw values in the window
dbg.fitted_line     # the linear trend that gets removed
dbg.detrended        # residual after detrending
dbg.texture           # final value -- matches refl_texture_1d(...)[15] exactly

# 2-D (eccopy2d_h / eccopy3d)
dbg = refl_texture_2d_debug(dbz_level, radius_km, dy, dx, iy=12, ix=8)
dbg.kernel_dbz       # raw values in the circular kernel
dbg.plane_a, dbg.plane_b   # fitted plane coefficients
dbg.detrended         # residual after the plane is removed
dbg.fraction_active   # kernel coverage fraction
dbg.texture           # final value -- matches refl_texture_2d(...) at [iy,ix] exactly
```

Both are exhaustively tested against the production functions (`eccopy/tests/test_debug.py`)
across thousands of random configurations, including NaN gaps, array
borders, and non-uniform spacing. `return_intermediates=True` is itself
tested against these single-point functions for consistency.

---

## Statistics: `eccopy.stats`

Generic, data-agnostic statistics computed from any module's `echo_type`
output (and, for height statistics, a co-located `height` array):

```python
from eccopy import stats

stats.convective_percentage(result.echo_type)          # float, 0-100
stats.stratiform_percentage(result.echo_type)
stats.n_clumps(result.echo_type)                        # connected convective regions
stats.clump_sizes(result.echo_type, spacing=spacing)     # physical sizes if spacing given
stats.convective_depth(result.echo_type, height)         # per-column, 2D-V/3D only
stats.convective_top_height(result.echo_type, height)
stats.summarize(result.echo_type, height=height)         # bundle of the above
```

See `eccopy/stats/basic_stats.py` for the full function list and
docstrings, including the important note on `stats.n_clumps()` not
numerically matching `Result3D.n_clumps` (they're computed at different
pipeline stages — see that function's docstring).

---

## Plotting

Each module has an optional `plot.py` with a `plot_result()` function
(requires `pip install "eccopy[plot]"`):

```python
from eccopy.eccopy3d.plot import plot_result
plot_result(result, dbz, coords_y=y_km, coords_x=x_km, coords_z=z_km,
           outfile="result.png")
```

For custom figures, `eccopy.core.colormaps` provides matching
colormaps/norms/labels for basic classification, sub-classification, and
convectivity:

```python
from eccopy.core.colormaps import (
    basic_echo_type_cmap, basic_echo_type_norm, BASIC_ECHO_TYPE_LABELS,  # 3-code
    echo_type_cmap, echo_type_norm, ECHO_TYPE_LABELS,                    # 9-code sub-classified
    convectivity_cmap, convectivity_norm,                                # 0-1, hard breaks at thresholds
)
```

The basic and sub-classified colormaps assign the **same color** to
their shared categories (Stratiform &harr; Strat Mid, Mixed &harr;
Mixed, Convective &harr; Conv), so panels stay visually consistent
whether a given case ends up basic-only or sub-classified.

`convectivity_cmap()` ramps continuously *within* each class but has
**hard breaks** at the two classification thresholds — strat/mixed
(`ClassificationParams.strat_mixed`, default 0.4) and mixed/conv
(`ClassificationParams.mixed_conv`, default 0.5). Blue / teal / red on a
convectivity panel therefore correspond directly to Stratiform / Mixed /
Convective, while shading within a band shows how marginal or how
firmly-held that call is. The breaks are placed to match the
classifier's inclusive-upper comparison (`convectivity >= strat_mixed`
is Mixed), so a pixel's color and its `echo_type` always agree.

If you classify with **non-default thresholds**, pass them through or
the breaks will land in the wrong place:

```python
cmap = convectivity_cmap(params.strat_mixed, params.mixed_conv)
```

See `notebooks/workflow_examples/` for a full walkthrough of every
module's parameters and plotting, using synthetic data end to end.

---

## Tests

```bash
pip install "eccopy[dev]"
pytest eccopy/tests/
```
