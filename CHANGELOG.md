# Changelog

All notable changes to EccoPy are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **`eccopy/tests/test_notebooks.py`** — structural checks on the committed
  notebooks: no error outputs, well-formed markdown tables, no cell that
  calls `print()`/`plt.show()` without output, and one notebook per module.
  These catch rendering faults that are otherwise invisible until someone
  opens the file on GitHub. Skipped when `notebooks/` is absent.
- **`notebooks/eccopy1d_workflow.ipynb`** — full walkthrough of the 1-D
  pipeline on the bundled disdrometer record: deriving reflectivity from a
  drop-size distribution, unit-typed time windows, wind-advected distance
  coordinates via `time_to_distance_km()`, `min_convective_length`, and a
  sweep helper. Completes one notebook per module; the synthetic-data
  notebooks under `notebooks/workflow_examples/` are removed, and
  `notebooks/README.md` indexes the four that replace them.
- **`eccopy.time_to_distance_km`** is now exported from `eccopy` and
  `eccopy.core`, alongside `haversine_distance`, `latlon_to_xy_spacing`
  and `resolve_spacing`. It was public, tested, and recommended by
  `eccopy1d`'s own docstring, but reachable only as
  `eccopy.core.coords.time_to_distance_km`.
- **`notebooks/eccopy2d_h_workflow.ipynb`** — full walkthrough of the 2-D-H
  pipeline on the bundled MRMS composite: lat/lon to physical spacing via
  `latlon_to_xy_spacing()`, the `kernel_mode` comparison (this is the one
  bundled dataset with genuinely non-uniform spacing), 2-D clumping, and a
  sweep helper reporting clump count alongside echo-type change.
- **`notebooks/eccopy3d_workflow.ipynb`** — full walkthrough of the 3-D
  pipeline on the bundled WRF volume: clump-based sub-typing, the int16
  sentinel convention, every parameter the 3-D path consumes with a
  measured sensitivity tier, and a sweep helper that reports clump count
  alongside echo-type change.
- **`notebooks/eccopy2d_v_workflow.ipynb`** — full walkthrough of the
  2-D-V pipeline on the bundled S-Pol RHI: every intermediate array, every
  tunable parameter with its direction, mechanism and a measured
  sensitivity tier, statistics, and a reusable parameter-sweep helper.
  Replaces the synthetic-data notebook under `workflow_examples/`.
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

- **Redundant and unimplemented parameter fields**, found in a
  pre-release audit of every file:
  - `TextureParams.window_1d`, `TextureParams.texture_radius` and
    `TextureParams.upper_lim_dbz` - the first two superseded by the
    `window` argument every `run()` takes, the third having no
    ConvStratFinder counterpart and being superseded by
    `texture_limit_high`.
  - `ClassificationParams.strat_mixed` and `.mixed_conv`, which shadowed
    `max_convectivity_for_stratiform` and
    `min_convectivity_for_convective` with identical defaults while being
    read by nothing.
  - `texture_to_convectivity_piecewise()`, uncalled and made fully
    redundant by `texture_to_convectivity_linear()`'s new `lower_lim`.
    The coverage condition it additionally applied is already applied
    upstream by `refl_texture_2d()`.
  - `col_max_convectivity()` in `eccopy3d.clumping`, uncalled and
    unexported.
  - The unused `param_name` argument of `resolve_enlarge_radius_px()`,
    plus five unused imports.

### Removed
- **`VerticalParams.min_valid_dbz`.** It was read by no code and shadowed
  `TextureParams.min_valid_dbz`, which is the live parameter the 3-D path
  actually consumes -- so setting it looked correct and did nothing. The
  reflectivity floor now has exactly one home.
- **The four `plot.py` modules and their `plot_result()` functions**
  (`eccopy1d`, `eccopy2d_h`, `eccopy2d_v`, `eccopy3d`), plus
  `tests/test_plot_result.py`. Every EccoPy output has the same shape as
  the reflectivity input, so any code that can plot the input can plot the
  result; the four modules were 400 lines of near-duplicate layout opinion
  (38-53% pairwise line similarity) that had to be fixed four times over
  whenever anything changed. `eccopy.core.colormaps` is **retained** --
  the echo-type code/colour/label mapping and the threshold-aligned
  convectivity colormap are the parts with real value. The workflow
  notebooks now serve as the plotting reference.
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
- **`examples/example_2d_v.py` was broken**, passing a `vert_params`
  argument removed from `eccopy2d_v.run()`, and nothing exercised it.
  `tests/test_examples.py` now checks every keyword each example passes to
  a `run()` against that module's real signature, and runs each script
  end to end under the `slow` marker.
- **`conda-recipe/meta.yaml` still required `scikit-learn`**, removed as a
  dependency in this release, and pointed its `home`/`doc_url`/`dev_url` at
  a non-existent repository. The recipe-maintainers placeholder is also
  filled in.
- **A malformed row in the 3-D notebook's sub-typing table.** One row was
  missing its Mechanism cell, and GitHub-flavoured markdown rejects an
  entire table when any row's column count differs from the header -- so
  the whole block rendered as a run-on paragraph on GitHub while the two
  tables above it in the same cell rendered normally.
- **The 1-D notebook's echo-type strip rendered as an empty axis.** A
  single-row `pcolormesh` gives matplotlib no way to infer cell height;
  replaced with `imshow` and an explicit extent.
- **`draw_window_ring()` drew a wildly oversized footprint on
  latitude/longitude panels.** It placed a circle of radius `radius_km`
  directly in axis units, so a 7 km window on a degree-based axis was drawn
  as a 7 *degree* circle -- roughly a hundredfold overstatement, and
  plausible enough to pass a glance. A `coord_units` argument now selects
  between `"km"` (unchanged, still a circle) and `"degrees"`, which draws
  the physically correct ellipse sized from the ring centre's latitude
  using the same earth radius as `core.coords`. An unrecognised value
  raises rather than guessing.
- **`eccopy.stats` miscounted every `eccopy3d` result.** The validity test
  was `~np.isnan(echo_type)`, but `eccopy3d` returns `int16` with
  `CATEGORY_MISSING` (0) as its no-echo sentinel rather than NaN, so every
  sentinel voxel counted as valid. `n_valid` came back as the full array
  size and all three category fractions were divided by that inflated
  denominator -- on the bundled WRF volume, convective coverage was
  reported as 21.5% instead of 33.8%, and the three fractions summed to
  0.64 rather than 1.0. Affected `echo_type_fractions()`,
  `convective_percentage()`, `stratiform_percentage()`,
  `mixed_percentage()`, `codes_for_category()`'s basic/sub auto-detection,
  and `summarize()`. Float/NaN results from the other three modules were
  never affected and their numbers are unchanged.
- **`VerticalParams.vert_levels_type` was a silent no-op.** It was
  declared, typed and documented, but read by no code: the height-vs-
  temperature choice was made implicitly by which array the caller
  supplied, so `vert_levels_type="by_temp"` did nothing whenever a
  `height` field was also passed. It is now authoritative --
  `"by_temp"` assigns vertical levels from temperature and emits a
  `UserWarning` stating that the supplied `height` is not used for that
  purpose, and raises `ValueError` if no `temp` field is given. Passing
  `temp` alone under the default `"by_height"` still falls through to
  temperature silently, since that is an unambiguous request rather than
  a mistake. Default-parameter output is byte-identical to the previous
  release.
- `pyproject.toml` `Repository` URL and the `CHANGELOG` release link
  pointed at upstream/placeholder repositories rather than EccoPy's own.
- `CONTRIBUTING.md` quoted a stale test count.
- `.gitignore` did not cover `build/`, `dist/`, `*.egg-info/`,
  `.pytest_cache/` or coverage output, so a local `python -m build` or
  `pytest --cov` left artefacts staged for commit.
- `build` added to the `dev` extra: `test_packaging.py` `importorskip`s
  it, so a documented dev install previously skipped the wheel-build
  check and reported a lower test count than a full run.

- **`examples/` rewritten** to mirror the workflow notebooks: each script
  now runs on the real bundled data, spells out every parameter that
  module consumes with its default, prints the intermediates and
  statistics, and optionally writes a figure with `--outdir`. Adds
  `examples/README.md`.

### Changed
- **`texture_limit_low`, `use_dbz_col_max` and `dbz_for_echo_tops` are now
  implemented.** All three were declared and documented but read by no
  code. `texture_limit_low` scales convectivity across
  `[low, high]` and makes sub-low texture missing rather than zero, per
  `ConvStratFinder::_computeConvectivity()`. `use_dbz_col_max` computes
  texture once from the column-maximum reflectivity and copies it to every
  level. `dbz_for_echo_tops` populates a new `Result3D.echo_top_km` field -
  an echo top derived from reflectivity, distinct from
  `stats.echo_top_height()`, which works from the classification. Each is
  inert at its default, and output with default parameters is
  byte-identical to the previous release in all four modules.
- **README rewritten** for v1.0: 693 lines to 351. Adds full citations for
  both algorithm papers, a Mermaid diagram of the texture ->
  convectivity -> classification chain, a per-module parameter matrix with
  defaults, and an acknowledgements section. Drops the validation-status
  section and the per-module quick starts, which the workflow notebooks
  now cover.
- **Comment and docstring rewrite, part 3 of 3** (`params/`, `stats/`,
  and the test suite), completing the pass. Comments across the package
  now describe behaviour and name their upstream counterpart rather than
  recording development history. Regression tests keep their description
  of the behaviour they pin, since that is the point of the test, but
  drop the temporal framing.
- **Comment and docstring rewrite, part 2 of 3** (`eccopy/core/`).
  `classification.py` drops from 1244 to 1027 lines and `texture.py` from
  1426 to 1397, with no functional change; the validation-history and
  open-item narrative in their module docstrings is replaced by
  descriptions of the conventions the code actually relies on -
  `border_value=1` on erosion, `_sequential_close()` over
  `binary_closing()`, and MATLAB's octagonal disk over a Euclidean one.
- **Comment and docstring rewrite, part 1 of 3** (the four module
  packages, plus a repository-wide typographic pass). Comments now
  describe what the code does and name the upstream function it
  corresponds to, rather than recording the history of how it came to be
  written. Each module header lists its upstream analogues once, with the
  source commit pinned (`lrose-ecco @ ad85c56`, `lrose-core @ b29264bf`),
  so individual references stay short and do not go stale. Em- and
  en-dashes are normalised to ASCII hyphens throughout (176 occurrences).
  No functional change; all tests pass unmodified.
- **`VerticalParams.min_valid_height` / `max_valid_height` are now
  implemented.** Both were declared but read by no code. Levels whose
  height falls outside the band are treated as missing *before* texture is
  computed, so excluded data cannot leak back in through the texture
  kernel -- matching ConvStratFinder, and verified equal to pre-masking the
  input by hand. The band is applied against the `height` field when one is
  supplied, otherwise `coords_z`. The defaults (0 to 25 km) span any
  realistic radar or model grid, so default-parameter output is
  byte-identical to the previous release.
- **`ClassificationParams.surf_alt_lim` now defaults to `0.0` m** (was
  `200.0`), so no near-surface adjustment is applied unless a caller asks
  for one. `class_sub_2d()`'s own default changes to match. The parameter
  sets the near-surface convective test at `echo base < 500 m +
  surf_alt_lim`, so convection based between 500 m and 700 m AGL now
  classifies as elevated (32) rather than surface-based; outside that band
  nothing changes, and the bundled S-Pol case is unaffected.
- **`min_valid_dbz` is now honoured by `eccopy1d` and `eccopy2d_v`.** It
  was declared in `TextureParams` and consumed by `eccopy2d_h`/`eccopy3d`,
  but `refl_texture_1d()` had no such parameter, so setting it silently
  did nothing on the 1-D and 2-D-V paths. Both 1-D texture entry points
  now accept and apply it.

  Its default changes from `0.0` to `None`, meaning "use the calling
  module's reference default", resolved by
  `TextureParams.resolve_min_valid_dbz()`:

  | Module | `None` resolves to | Provenance |
  |---|---|---|
  | `eccopy2d_h`, `eccopy3d` | `0.0` | `ConvStratFinder::computeEchoType()` prefilter |
  | `eccopy1d`, `eccopy2d_v` | no gating | `f_reflTexture.m` has no prefilter |

  An explicitly-set value is honoured by all four modules. Output with
  default parameters is byte-identical to the previous release in every
  module. Gating the 2-D-V path at 0 dBZ moves its output by roughly 13%,
  which is why the default must not do so: that module agrees with real
  MATLAB ECCO-V output at 99.4-100% without gating.
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
