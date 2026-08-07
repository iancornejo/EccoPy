# Example scripts

One runnable script per module, doing the same thing as the notebooks in
`notebooks/` on the same bundled data, for anyone who would rather not use
Jupyter.

| Script | Module | Data |
|---|---|---|
| `example_1d.py` | `eccopy1d` | MRR/2DVD disdrometer time series |
| `example_2d_v.py` | `eccopy2d_v` | S-Pol gridded RHI + sounding |
| `example_2d_h.py` | `eccopy2d_h` | MRMS composite reflectivity |
| `example_3d.py` | `eccopy3d` | WRF simulated reflectivity volume |

Each one loads real data, spells out every parameter that module consumes
with its default, runs the full chain, and prints the intermediates and
statistics. Read them top to bottom; they are meant to be copied and
edited.

```bash
python examples/example_2d_v.py                # print results
python examples/example_2d_v.py --outdir figs  # also save a figure
```

They resolve their own paths, so they run from anywhere in a checkout.

## Requirements

The sample data lives in `notebooks/data/`. It ships with the repository
but **not** with the installed wheel, so these run from a checkout only.
Reading NetCDF needs `netCDF4`, and the figures need `matplotlib`:

```bash
pip install -e ".[dev,plot]" netCDF4
```

Without `--outdir`, nothing is plotted and `matplotlib` is not imported.

## Also here

`export_disk_strels.m` and `run_export_disk_strels.sh` export MATLAB's
`strel('disk', r)` neighbourhoods and their `getsequence()`
decompositions. `eccopy.core.disk` now generates these in pure Python at
any radius, so they are not needed to run EccoPy - they are kept as the
provenance record for the regression fixtures under
`eccopy/core/data/`.
