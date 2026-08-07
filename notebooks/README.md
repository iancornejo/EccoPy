# Workflow notebooks

One notebook per EccoPy module, each running end-to-end on real data
bundled under `data/`.

| Notebook | Module | Data |
|---|---|---|
| `eccopy1d_workflow.ipynb` | `eccopy1d` | MRR/2DVD disdrometer time series |
| `eccopy2d_v_workflow.ipynb` | `eccopy2d_v` | S-Pol gridded RHI + sounding |
| `eccopy2d_h_workflow.ipynb` | `eccopy2d_h` | MRMS composite reflectivity |
| `eccopy3d_workflow.ipynb` | `eccopy3d` | WRF simulated reflectivity volume |

Each one walks the full chain — texture, convectivity, classification —
shows the intermediate arrays, documents every parameter that module
consumes, runs through the applicable statistics, and ends with a reusable
helper for sweeping parameters on your own data.

They are committed with their outputs so they render on GitHub. To run
them yourself:

```bash
pip install -e ".[dev,plot]"
cd notebooks
jupyter lab                      # or:
jupyter nbconvert --to notebook --execute --inplace eccopy2d_v_workflow.ipynb
```

Paths are relative to this directory, so run them from here.

Sample data provenance and attribution are in `data/README.md`. The
datasets are committed to the repository but excluded from the built wheel
and sdist, so `pip install eccopy` does not download them.

## A note on the parameter guidance

Each notebook labels parameters with a sensitivity tier — high, moderate or
low. Those tiers were measured on that notebook's specific case and are
meant as a guide to where attention is worth spending, not as constants
that transfer to other data. The sweep helper at the end of each notebook
is there so you can measure them on yours.
