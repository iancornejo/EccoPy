# Changelog

All notable changes to EccoPy are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Sample datasets** under `notebooks/data/` (4.5 MB total), one per
  module, with `make_sample_data.py` documenting provenance and subset
  selection and a `README.md` carrying attribution. Committed to the
  repository but excluded from the built wheel and sdist.
- **`eccopy.__version__`**, single-sourced from `eccopy/__init__.py`.
  `pyproject.toml` now declares `dynamic = ["version"]` and reads the
  attribute, so the version is defined in exactly one place.
- **`eccopy.core.disk`**: pure-Python generation of MATLAB
  `strel('disk', r)` neighborhoods and their `getsequence()`
  decompositions (the default `n=4` periodic-line octagon, Adams 1993),
  for **any** radius. Validated bit-exact against the MATLAB-exported
  ground truth (`tests/test_disk_generator.py`, 84 cases). An `n=0`
  Euclidean-disk option is exposed for callers who explicitly want it.

### Removed
- **`topo` / `topo_km` from the 3-D path** (`eccopy3d.run()`,
  `find_clumps_3d()`, `set_echo_type_3d()`). It performed a literal
  `height_km - topo_km` AGL subtraction with no analog in
  `ConvStratFinder`, which handles terrain solely via `terrainHt` by
  *raising the shallow/deep threshold boundaries* — already implemented
  as `terrain_ht_km`. The 3-D path therefore carried two terrain
  mechanisms where the reference has one, the extra being an
  unvalidated carry-over from the 2-D `f_classSub.m` port and unexercised
  by any validation case. **Breaking:** `eccopy3d.run(..., topo=...)` now
  raises `TypeError`; use `terrain_ht=` instead. `topo` is unchanged in
  `eccopy2d_v`, where it is faithful to `f_classSub.m`.
- **`eccopy.core.fill` / `fill_regions_closest_pixel()`** — a port of
  `f_fillRegionsClosestPixel.m`, which belongs to ECCO-V's *velocity*
  texture path. It had no callers inside EccoPy and no test coverage.
- **`vel_cmap()` / `_VEL_COLORS`** (`eccopy.core.colormaps`) — HCR
  velocity colormap (port of `velCols.m`), unused by any EccoPy
  classification or plotting path.
- **`scikit-learn` runtime dependency**, which existed solely to supply
  `KDTree` to `fill_regions_closest_pixel()`.
- Duplicate `strel('disk', r)` fixtures under `examples/disk_strels/`
  (byte-identical to the copies under `eccopy/core/data/disk_strels/`,
  plus a `disk.zip` holding a third copy). The MATLAB export scripts are
  retained as the provenance record and now default to writing into
  `eccopy/core/data/disk_strels/`.

### Fixed
- `pyproject.toml` `Repository` URL and the `CHANGELOG` release link
  pointed at upstream/placeholder repositories rather than EccoPy's own.
- `CONTRIBUTING.md` quoted a stale test count.
- `.gitignore` did not cover `build/`, `dist/`, `*.egg-info/`,
  `.pytest_cache/` or coverage output, so a local `python -m build` or
  `pytest --cov` left artefacts staged for commit.
- `build` added to the `dev` extra: `test_packaging.py` `importorskip`s
  it, so a documented dev install previously skipped the wheel-build
  check and reported a lower test count than a full run.

### Changed
- **`min_overlap_for_convective_clumps` now raises `NotImplementedError`
  for any value other than 1.** The parameter was declared and documented
  but never read by any code, so a non-default value silently did nothing.
  EccoPy's `scipy.ndimage.label` clumping reproduces TITAN/LROSE interval
  clumping only at `min_overlap=1`, where the two are exactly equivalent.
  The name is retained for TDRP correspondence; validation lives in a new
  `ClassificationParams.__post_init__`.
- **`enlarge_mixed`/`enlarge_conv` are no longer limited to pre-exported
  radii.** `_disk()` / `_load_decomp()` generate the strel and closing
  decomposition on demand via `eccopy.core.disk` (using a bundled `.mat`
  as a byte-identical override when present), so previously-unusable
  values like `enlarge_conv=15` now work instead of raising
  `FileNotFoundError`. Results for the previously-validated radii `{3, 5}`
  are byte-identical.
- **`resolve_enlarge_radius_px()`** now resolves to the exact rounded
  pixel radius (no snapping to a bundled set, no mismatch warning), since
  any radius is generatable.
- The bundled `disk_strel`/`disk_decomp` `.mat` files are demoted from a
  runtime dependency to optional bit-exact regression fixtures.

## [0.1.0] — v0.1 pre-release

First public pre-release. Four data-agnostic, array-in/array-out
classification modules (`eccopy1d`, `eccopy2d_v`, `eccopy2d_h`,
`eccopy3d`), plus supporting statistics, plotting, and debugging
utilities.

### Fixed
- **`plot_result()` rendered nothing in notebooks / interactive sessions.**
  All four modules' `plot_result()` forced `matplotlib.use("Agg")` and
  then closed the figure unconditionally, so a cell would execute without
  error but display no figure (only code that plotted inline outside
  `plot_result` showed up). `plot_result()` no longer switches the global
  backend, and now closes the figure only when writing to `outfile`;
  interactive calls (no `outfile`, `show=False`) return an open figure
  that renders inline. Regression-tested in `test_plot_result.py`. The
  workflow-example notebooks have been re-executed, so their stored
  figures now include the previously-missing texture/convectivity/echo-type
  panels.

### Added
- **Intermediate/debug field outputs**: `run(..., return_intermediates=True)`
  on all four modules, exposing `fitted_dbz`/`detrended_dbz` (the
  per-point values computed just before texture) and, for `eccopy2d_v`,
  `echo_basic` (classification before sub-classification). Fast (same
  Numba core) for `eccopy1d`/`eccopy2d_v`; a slower explicit
  plain-Python path for `eccopy2d_h`/`eccopy3d` (the latter requires an
  explicit `levels=[...]` list).
- **`eccopy.stats` subpackage**: generic, data-agnostic statistics —
  `convective_percentage`, `stratiform_percentage`, `mixed_percentage`,
  `n_clumps`, `clump_sizes`, `convective_depth`,
  `convective_top_height`, `convective_base_height`, `summarize()` — all
  operating on any module's `echo_type` output.
- **`convectivity_cmap()`/`convectivity_norm()`**: a colormap for the
  0-1 convectivity field that ramps continuously within each class
  (dark→light blue, dark→light teal, light→dark red) but breaks hard at
  the strat/mixed and mixed/conv thresholds, so a convectivity panel can
  be read for class directly. Breaks default to 0.4/0.5 and are
  overridable via `convectivity_cmap(strat_mixed, mixed_conv)` for
  non-default `ClassificationParams`; they are positioned to match the
  classifier's inclusive-upper comparison, so color and `echo_type`
  never disagree at a threshold. The existing basic/sub-classification
  colormaps remain coincident for their shared categories; that
  invariant is regression-tested.
- **Texture-window footprint overlay**: `core.colormaps.draw_window_ring()`,
  plus a `show_window=True` (default) option on the `eccopy2d_h` and
  `eccopy3d` `plot_result()` functions, drawing the texture
  neighbourhood as a dashed circle at each plan-view panel's centre so
  users can see the window size relative to their features. Both
  `Result2DH` and `Result3D` now carry a `texture_radius` field (km, the
  physical radius actually used) that the overlay reads; it is None only
  for a bare-pixel window on a unit-agnostic grid, in which case the ring
  is silently skipped.
- **`eccopy1d.plot.plot_result()`**: brings EccoPy-1D to parity with
  the other three modules' plotting helpers.
- **Workflow example notebooks** (`notebooks/workflow_examples/`): one
  self-contained, synthetic-data notebook per module covering every
  parameter option and plotting with EccoPy's colormaps.
- Packaging: `LICENSE`, `.gitignore`, `MANIFEST.in`, GitHub Actions CI
  (`test` across Python 3.10-3.12 / Linux/macOS/Windows, plus a
  `packaging` job), `CONTRIBUTING.md`, this changelog, and a
  `conda-recipe/meta.yaml`.

### Fixed
- **Packaging bug**: the `.mat` disk-strel/disk-decomp reference data
  files (required at runtime by `class_basic()`/`class_basic_isotropic()`
  whenever `enlarge_conv`/`enlarge_mixed` load a strel) were silently
  missing from built wheels/sdists — no `package-data` configuration
  existed. Anyone installing EccoPy via `pip install eccopy` rather than
  running from the source tree would have hit a runtime `FileNotFoundError`.
  Fixed via `[tool.setuptools.package-data]` in `pyproject.toml`; a
  regression test (`test_packaging.py`) now builds a real wheel/sdist
  and confirms the files are present.
- **Hard matplotlib dependency**: `import eccopy` transitively required
  `matplotlib` even though it's declared as an optional `[plot]` extra,
  because `eccopy/core/__init__.py` imported colormap functions eagerly.
  Fixed by resolving those names lazily via `__getattr__` (PEP 562);
  `import eccopy` and all classification work no longer require
  matplotlib at all.

### Known limitations (see README "Validation status" for full detail)
- `eccopy1d` / `eccopy2d_h`: not yet validated against real reference
  output (only share code paths with validated modules).
- `eccopy3d`: `min_overlap_for_convective_clumps > 1` is unimplemented
  and rejected at construction time.
- `stats.n_clumps()` labels connectivity on the *final* `echo_type`
  array and will not numerically match `Result3D.n_clumps` (computed
  earlier, on convectivity, by the dual-threshold clumping algorithm).

[0.1.0]: https://github.com/iancornejo/EccoPy/releases/tag/v0.1.0
