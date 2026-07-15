# Changelog

All notable changes to EccoPy are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- `eccopy3d`: `min_overlap_for_convective_clumps > 1` and the
  `topo_km` AGL-subtraction path are unvalidated / unexercised by any
  real test case.
- `stats.n_clumps()` labels connectivity on the *final* `echo_type`
  array and will not numerically match `Result3D.n_clumps` (computed
  earlier, on convectivity, by the dual-threshold clumping algorithm).

[0.1.0]: https://github.com/NCAR/eccopy/releases/tag/v0.1.0
