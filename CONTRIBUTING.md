# Contributing to EccoPy

## Development setup

```bash
git clone <this-repo>
cd eccopy
pip install -e ".[dev,plot]"
pytest eccopy/tests/
```

All 195 tests should pass before and after any change. If you add a
feature, add tests for it in `eccopy/tests/` — see the existing files
for the project's style (plain `pytest` functions, synthetic data
generated in the test file itself, no fixtures files).

## The ground-truth-first standard

This is the single most important convention in this codebase, and the
reason EccoPy exists as a faithful *port* rather than a reimplementation:

**Every fix to the classification algorithm itself (texture, detrending,
convectivity, morphology, clumping, sub-classification) must be proven
against real reference output** — the original MATLAB ECCO-V source
(`f_classBasic.m`, `f_classSub.m`, etc.) or the C++
ConvStratFinder/StormClump source, or real intermediate arrays/output
from running that reference code. Pattern-matching a fix that "looks
right", or one based on assumptions about what the algorithm *should* do
without checking the source, is not acceptable for anything on the
classification critical path — see `eccopy/core/classification.py`'s
module docstring for the kind of subtle, easy-to-miss bugs (off-by-one
indexing, wrong default border values, decomposition-vs-single-op
morphology) that this standard has caught in the past.

**Evidence hierarchy, strongest first:**
1. Direct line-by-line comparison against the real source file.
2. Array-level comparison against real reference *output* (e.g. a real
   LROSE-ECCO case's `ECCO_OUT_*.npz`, or real MATLAB intermediate
   arrays).
3. Anything else (physical reasoning, "this seems more correct") is a
   *hypothesis* to go check against (1) or (2), not a fix to merge on
   its own.

If you don't have access to the original MATLAB/C++ source or a real
reference case and want to propose a change to the classification path,
please open an issue describing what you've observed rather than a PR —
someone with source/reference access can help verify it properly.

**This standard does NOT apply** to code that doesn't affect
classification numerics: the `eccopy.stats` module, plotting/colormaps,
packaging, documentation, and the debug/intermediate-output machinery
(`return_intermediates`, `core.debug`) are all validated against the
*production code they're derived from* (e.g. "does this debug function's
texture output exactly match the production texture function's output"),
not against an external MATLAB/C++ reference — normal software-engineering
testing practices apply there.

## Honest validation accounting

If you add support for a new configuration (e.g. a new parameter
combination, a new module code path), please update the "Validation
status" section of `README.md` to say plainly whether it's been checked
against real reference output, and at what agreement level — don't leave
new code paths silently implied to be validated when they haven't been.
This project's credibility rests on that section being accurate, not
optimistic.

## Code style

- No enforced formatter/linter currently — match the surrounding code's
  style (docstrings on every public function, type hints on parameters,
  dataclasses for `Result*` objects).
- Numba-accelerated cores (`@njit` functions) should stay small,
  self-contained, and have a plain-Python or already-validated debug
  equivalent that can be cross-checked against them (see
  `eccopy/tests/test_numba_cross_validation.py` and
  `test_debug.py` for the pattern).
- Prefer adding a new function over modifying a validated one in place
  when the new behavior is opt-in/debug-only — see
  `core/texture.py`'s `_sliding_texture_core_with_fit` next to
  `_sliding_texture_core` for the convention: the hot, validated path
  is never touched by an unrelated feature addition.

## Running the full test suite, including the slower packaging check

```bash
pytest eccopy/tests/ -q                 # fast, ~10-15s, run this normally
pytest eccopy/tests/ -q -m slow         # includes building a real wheel; slower
```
