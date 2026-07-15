# EccoPy workflow example notebooks

Four self-contained notebooks (synthetic data, no external files needed)
walking through each EccoPy module's options and parameters in detail,
plus plotting with EccoPy's shared colormaps:

- `eccopy1d_workflow.ipynb` — 1-D time/distance profiles
- `eccopy2d_v_workflow.ipynb` — 2-D vertical cross-sections (RHI-style)
- `eccopy2d_h_workflow.ipynb` — 2-D horizontal composites / single levels
- `eccopy3d_workflow.ipynb` — full 3-D volumes

Each notebook covers: generating/loading data, `WindowSpec` /
`TextureParams` / `ClassificationParams` (and `VerticalParams` for
EccoPy-3D), debugging with `return_intermediates=True`, statistics via
`eccopy.stats`, and plotting with `eccopy.core.colormaps`.

Run top-to-bottom with `jupyter notebook` / `jupyter lab`, or
non-interactively with:

```bash
jupyter nbconvert --to notebook --execute --inplace eccopy1d_workflow.ipynb
```

These are separate from the case-study notebooks one level up
(`../eccopy3d_workflow.ipynb`, `../eccov_workflow.ipynb`, etc.), which
walk through validating EccoPy against real LROSE-ECCO/MATLAB reference
output and require external case-file data.

`_build_scripts/` contains the (non-shipped) authoring scripts used to
generate these notebooks with `nbformat` — useful as a reference if you
want to regenerate or extend them programmatically, not needed to just
run the notebooks.
