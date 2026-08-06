# Sample data

Four small datasets, one per EccoPy module, used by the workflow
notebooks in `notebooks/`. Each is a subset of a larger original,
prepared by `make_sample_data.py` in this directory — that script is the
provenance record and documents exactly which subset was taken from
which source.

These files are committed to the repository but **excluded from the
built wheel and sdist**, so `pip install eccopy` does not download them.

| File | Module | Contents | Size |
|---|---|---|---|
| `disdrometer_dsd_20080627.nc` | `eccopy1d` | 24 h × 1-min drop-size distribution (1440 × 110) | 0.11 MB |
| `mrms_composite_20211215.nc` | `eccopy2d_h` | 600 × 900 composite reflectivity crop, 39–45 °N / 100–91 °W | 0.24 MB |
| `spol_rhi_20220526.nc` | `eccopy2d_v` | 41 × 601 gridded RHI plus co-located sounding profile | 0.10 MB |
| `wrf_volume_20220526.nc` | `eccopy3d` | 45 × 200 × 200 model volume with 3-D temperature | 4.07 MB |

## Attribution

**`disdrometer_dsd_20080627.nc`** — Combined MRR and 2DVD drop-size
distribution from TiMREX/SoWMEX (2008). Courtesy of Ian Cornejo and
Wei-Yu Chang.

**`mrms_composite_20211215.nc`** — NOAA Multi-Radar/Multi-Sensor
`MergedReflectivityQCComposite`, 2021-12-15 22:38 UTC (serial derecho).
NOAA/National Severe Storms Laboratory; public domain.

**`spol_rhi_20220526.nc`** — NSF NCAR S-Pol gridded RHI from
PRECIP-2022, 2022-05-26 17:56 UTC, azimuth 264°. Gridded product
courtesy of Michael M. Bell, Colorado State University; the source
product states CC-BY-4.0. The bundled temperature profile is from
NCAR/EOL sounding 46757 (PRECIP-2022), launched 2022-05-26 17:14 UTC at
121.0142 °E, 24.8279 °N — 42 minutes before the scan, at the same site.

The source file's global attributes name the instrument as SEA-POL. That
is a processing-template artifact carried over from other work by the
same group; this scan is S-Pol, consistent with the co-located sounding
and the scan location in Taiwan.

**`wrf_volume_20220526.nc`** — WRF simulation output, 2022-05-26
08:30 UTC, converted via MDV. Courtesy of Kao-Shen Chung, National
Central University.

These are teaching samples, not scientific data products. Cite the
original sources rather than this repository if you use them in
published work.

## Notes on the extracts

**Reflectivity is not stored in the 1-D file.** It holds the raw drop-size
distribution, and the notebook derives reflectivity from it as
`Z = Σ N(D)·D⁶·ΔD`. This keeps the file small and demonstrates that
EccoPy consumes plain arrays from any source.

**Missing-data sentinels are pre-cleaned.** MRMS uses −999 (outside radar
coverage) and −99 (no echo); the S-Pol product uses −9999. None of these are
declared as `_FillValue` in the originals, so `netCDF4`'s automatic
masking does not catch them and they read back as enormous negative
reflectivities. All are converted to `NaN` here.

**The 3-D file is packed as int16** with a 0.01 dBZ quantization, matching
how MDV stored it. Measured end-to-end through `eccopy3d` against the
full-precision field, this reproduces 99.98 % of echo-type codes with a
convectivity MAE of 2.2 × 10⁻⁵ — an order of magnitude below EccoPy-3D's
own agreement with the C++ reference, so the quantization cannot be
mistaken for a port error.

**Height in the 3-D file is a 1-D profile.** The source `geo_hgt` field
varies horizontally by only 12–20 m, negligible against 1 km grid
spacing. It is *not* interchangeable with the nominal `z` coordinate,
though: geopotential height drifts from it by up to 0.5 km aloft, so the
notebook passes `height` explicitly rather than broadcasting `z`.

## Regenerating

`make_sample_data.py` expects the original files, which are not in this
repository. Contact the attributed sources for access.

```bash
python make_sample_data.py
```
