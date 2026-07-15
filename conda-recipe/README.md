# conda-recipe

A `conda-build`-style recipe for EccoPy. **Not build-tested in this
environment** (no `conda`/`mamba` available here, and this sandbox's
network is restricted to a small allowlist that doesn't include conda
channels) — please build-test it yourself before submitting to
conda-forge:

```bash
conda install -n base conda-build
conda build conda-recipe/
```

This builds from the local source tree (`source: path: ..`) by default,
which is convenient for testing changes before a release. Once EccoPy
has a tagged release / PyPI upload, switch `source:` to the commented-out
`url:`/`sha256:` block instead — conda-forge's own tooling (`grayskull`,
or the `staged-recipes` PR template) can regenerate a lot of this
automatically from the PyPI sdist if you'd rather start from that.

## Before submitting to conda-forge

1. Fill in the real `sha256` once you have a released sdist.
2. Replace `your-github-username-here` under `extra.recipe-maintainers`.
3. Confirm `home`/`doc_url`/`dev_url` point at the real repository once
   it exists (currently placeholder `NCAR/eccopy` URLs — see the note
   in the top-level README/CHANGELOG about updating `pyproject.toml`'s
   `Repository` URL to match).
4. Run the `test:` section locally first — it re-runs the full pytest
   suite from the installed package and specifically checks that the
   `.mat` reference data files installed correctly (a real bug found
   and fixed for v0.1.0 — see CHANGELOG.md).
5. Follow conda-forge's [staged-recipes](https://github.com/conda-forge/staged-recipes)
   process for the initial submission.
