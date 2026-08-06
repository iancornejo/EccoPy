"""
Build EccoPy's shipped sample datasets from their original sources.

Each output is a self-contained, compressed NetCDF4 file holding only the
fields its workflow notebook uses. This script is the provenance record:
it documents which source each extract came from, which subset was taken,
and why. It is not run at install or test time -- the built files are
committed under notebooks/data/.

Original sources are NOT redistributed; see notebooks/data/README.md for
attribution and licensing.

Reflectivity in the 3-D file is packed as int16 with a 0.01 dBZ
quantization. Measured end-to-end through eccopy3d against the
full-precision field, that reproduces 99.98% of echo-type codes with a
convectivity MAE of 2.2e-05 -- an order of magnitude below EccoPy-3D's
own agreement with the C++ reference, so sample-data quantization can
never be mistaken for a port error.
"""
from __future__ import annotations

import os

import netCDF4 as nc
import numpy as np

SRC = "data/Test_Data"
RHI_SRC = "data/gridded_rhi_20220526_1756_225_0.nc"
OUT = "out"

COMP = dict(zlib=True, complevel=5, shuffle=True)
DBZ_QUANT = 0.01          # dBZ, see module docstring
FILL_I2 = np.int16(-32768)


def _finalize(ds, title, summary, source, attribution, extra=None):
    ds.title = title
    ds.summary = summary
    ds.source = source
    ds.attribution = attribution
    ds.Conventions = "CF-1.8"
    ds.note = ("Subset prepared for the EccoPy workflow notebooks. Not a "
               "scientific data product; see notebooks/data/README.md.")
    for k, v in (extra or {}).items():
        setattr(ds, k, v)


def _pack_i2(ds, name, arr, dims, scale, units, long_name, comment=None):
    """Store `arr` as int16 with scale/offset, mapping non-finite to _FillValue."""
    finite = np.isfinite(arr)
    offset = float(np.floor(arr[finite].min() / scale) * scale) + 32767 * scale
    packed = np.where(finite, np.round((arr - offset) / scale), FILL_I2).astype("i2")

    v = ds.createVariable(name, "i2", dims, fill_value=FILL_I2, **COMP)
    # The values below are ALREADY packed. netCDF4 applies scale_factor /
    # add_offset on write by default, which would pack them a second time.
    v.set_auto_maskandscale(False)
    v.scale_factor = np.float32(scale)
    v.add_offset = np.float32(offset)
    v.units = units
    v.long_name = long_name
    if comment:
        v.comment = comment
    v[:] = packed


def _f4(ds, name, arr, dims, units, long_name, comment=None):
    v = ds.createVariable(name, "f4", dims, fill_value=np.float32(np.nan), **COMP)
    v.units = units
    v.long_name = long_name
    if comment:
        v.comment = comment
    v[:] = arr


# ---------------------------------------------------------------------
# 1-D: disdrometer drop-size distribution
# ---------------------------------------------------------------------
def build_1d(start_idx=91000, n=1440):
    """
    24 h of 1-min DSD spanning convective cells embedded in stratiform
    rain (2008-06-27; 816 of 1440 minutes have precipitation).

    Reflectivity is deliberately NOT stored: the notebook derives it from
    the DSD as Z = sum(N(D) D^6 dD), which demonstrates that EccoPy takes
    plain arrays from any source.
    """
    f = nc.Dataset(f"{SRC}/MRR_2DVD_2008.nc")
    sl = slice(start_idx, start_idx + n)
    tvar = f.variables["time"]

    out = nc.Dataset(f"{OUT}/disdrometer_dsd_20080627.nc", "w", format="NETCDF4")
    out.createDimension("time", n)
    out.createDimension("particle_size", len(f.dimensions["particle_size"]))

    v = out.createVariable("time", "f8", ("time",), **COMP)
    v[:] = tvar[sl]
    v.units = tvar.units
    v.calendar = getattr(tvar, "calendar", "standard")
    v.long_name = "time"

    _f4(out, "particle_size", f.variables["particle_size"][:], ("particle_size",),
        "mm", "drop diameter (bin centre)")
    _f4(out, "particle_size_bin_width", f.variables["particle_size_bin_width"][:],
        ("particle_size",), "mm", "drop diameter bin width")
    _f4(out, "Nd", np.ma.filled(f.variables["Nd"][sl].astype("f4"), np.nan),
        ("time", "particle_size"), "mm^-1 m^-3",
        "number concentration (combined MRR + 2DVD DSD)",
        "Reflectivity for EccoPy-1D is derived from this field: "
        "Z = sum(Nd * D**6 * dD); dBZ = 10*log10(Z).")

    _finalize(out,
              "MRR/2DVD drop-size distribution, 2008-06-27",
              "24 h of 1-min DSD spanning convective cells embedded in "
              "stratiform rain; sample input for the EccoPy-1D notebook.",
              "MRR + 2DVD combined drop-size distribution, TiMREX/SoWMEX (2008)",
              "TiMREX/SoWMEX; Ian Cornejo and Wei-Yu Chang")
    out.close()
    f.close()


# ---------------------------------------------------------------------
# 2-D-H: MRMS composite reflectivity
# ---------------------------------------------------------------------
def build_2dh(i0=1000, j0=3000, ny=600, nx=900):
    """
    Regional crop of the 2021-12-15 serial derecho over the central
    Plains (39-45 N, 100-91 W). Full CONUS is 3500x7000 (~98 MB as
    float32); this crop keeps the convective line and the surrounding
    stratiform shield at native 0.01 deg resolution.

    MRMS uses two negative sentinels that are NOT declared as _FillValue,
    so netCDF4's auto-masking does not catch them:
        -999  outside radar coverage
         -99  within coverage, no echo
    Both become NaN here.
    """
    f = nc.Dataset(f"{SRC}/MRMS_MergedReflectivityQCComposite_00.50_20211215-223839.nc")
    lat = f.variables["latitude"][i0:i0 + ny]
    lon = f.variables["longitude"][j0:j0 + nx]
    lon = np.where(lon > 180, lon - 360, lon)

    d = np.array(f.variables["MergedReflectivityQCComposite"][0,
                                                              i0:i0 + ny,
                                                              j0:j0 + nx], "f4")
    d = np.where(d <= -99, np.nan, d)

    out = nc.Dataset(f"{OUT}/mrms_composite_20211215.nc", "w", format="NETCDF4")
    out.createDimension("latitude", ny)
    out.createDimension("longitude", nx)

    v = out.createVariable("latitude", "f8", ("latitude",), **COMP)
    v[:] = lat
    v.units = "degrees_north"
    v.long_name = "latitude"

    v = out.createVariable("longitude", "f8", ("longitude",), **COMP)
    v[:] = lon
    v.units = "degrees_east"
    v.long_name = "longitude"

    _f4(out, "MergedReflectivityQCComposite", d, ("latitude", "longitude"),
        "dBZ", "merged QC'd composite reflectivity",
        "MRMS -999 (no coverage) and -99 (no echo) sentinels converted to NaN.")

    _finalize(out,
              "MRMS composite reflectivity, 2021-12-15 22:38 UTC",
              "Regional crop over the 2021-12-15 serial derecho; sample "
              "input for the EccoPy-2D-H notebook.",
              "NOAA Multi-Radar/Multi-Sensor MergedReflectivityQCComposite",
              "NOAA National Severe Storms Laboratory (public domain)",
              {"valid_time": "2021-12-15T22:38:39Z"})
    out.close()
    f.close()


# ---------------------------------------------------------------------
# 2-D-V: S-Pol gridded RHI + co-located sounding
# ---------------------------------------------------------------------
def _read_eol(path):
    """EOL Sounding Format v1.1: 17 whitespace-separated columns, -999 missing."""
    rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) != 17:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    a = np.array(rows)
    temp, alt = a[:, 5], a[:, 13]          # Temp (degC), GeoPoAlt (m)
    good = (temp > -900) & (alt > -900)
    temp, alt = temp[good], alt[good]
    order = np.argsort(alt)
    return alt[order], temp[order]


def build_2dv(max_alt_m=12000, thin_m=10.0):
    """
    One gridded RHI (Z x R) plus the co-located sounding, launched 42 min
    earlier at the same site. Distilling the sounding to a height/
    temperature profile keeps the whole case in a single file; the
    notebook broadcasts it with broadcast_temp_field() and derives the
    melt field with melt_layer_from_temp().

    Fields use -9999 as a sentinel with no declared _FillValue, so
    netCDF4's auto-masking does not catch it. Converted to NaN here.

    NOTE: the source file's global attributes name the instrument as
    SEA-POL. That is a processing-template artifact; this scan is from
    the NSF NCAR S-Pol deployment for PRECIP-2022 in Taiwan, consistent
    with the co-located EOL sounding and the scan location.
    """
    f = nc.Dataset(RHI_SRC)
    dbz = np.array(f.variables["DBZ_F"][0], "f4")
    dbz = np.where(dbz < -900, np.nan, dbz)

    alt, temp = _read_eol(f"{SRC}/46757_2022052618.L2.eol")
    keep = alt <= max_alt_m
    alt, temp = alt[keep], temp[keep]
    # The 1 s sonde is ~5 m in the vertical, far denser than the 250 m
    # radar grid needs; thin to ~10 m.
    idx = np.unique(np.searchsorted(alt, np.arange(alt[0], alt[-1], thin_m)))
    idx = idx[idx < len(alt)]
    alt, temp = alt[idx], temp[idx]

    out = nc.Dataset(f"{OUT}/spol_rhi_20220526.nc", "w", format="NETCDF4")
    out.createDimension("Z", len(f.dimensions["Z"]))
    out.createDimension("R", len(f.dimensions["R"]))
    out.createDimension("sounding_level", len(alt))

    _f4(out, "Z", f.variables["Z"][:], ("Z",), "m", "height above radar")
    _f4(out, "R", f.variables["R"][:], ("R",), "m", "ground range from radar")
    _f4(out, "DBZ_F", dbz, ("Z", "R"), "dBZ", "filtered reflectivity",
        "Source -9999 sentinel converted to NaN.")
    _f4(out, "sounding_altitude", alt, ("sounding_level",), "m",
        "geopotential altitude of sounding level")
    _f4(out, "sounding_temperature", temp, ("sounding_level",), "degC",
        "sounding temperature",
        "EOL sounding 46757, launched 2022-05-26 17:14:13 UTC at "
        "121.0142E 24.8279N -- 42 min before this RHI, same site. Thinned "
        f"to ~{thin_m:g} m and truncated at {max_alt_m} m.")

    _finalize(out,
              "S-Pol gridded RHI with co-located sounding, 2022-05-26",
              "Vertical cross-section (Z x R) plus the sounding temperature "
              "profile needed for sub-classification; sample input for the "
              "EccoPy-2D-V notebook.",
              "NSF NCAR S-Pol gridded RHI; NCAR/EOL sounding (PRECIP-2022)",
              "Radar: NSF NCAR S-Pol, PRECIP-2022. Gridded product courtesy "
              "of Michael M. Bell, Colorado State University. Sounding: "
              "NCAR/EOL, PRECIP-2022, site 46757.",
              {"valid_time": "2022-05-26T17:56:56Z",
               "azimuth_deg": float(f.variables["azimuth"][0]),
               "license": "CC-BY-4.0 (as stated by the source gridded product)"})
    out.close()
    f.close()


# ---------------------------------------------------------------------
# 3-D: WRF volume
# ---------------------------------------------------------------------
def build_3d(i0=425, j0=475, n=200, ztop_km=15.0):
    """
    200 x 200 km subdomain of a WRF forecast volume covering a deep
    convective complex (99% echo coverage, 67 dBZ peak). The full grid is
    46 x 750 x 750 (~310 MB as float32).

    Uses `dbz`, NOT `dbz_3d`. The z=0 level is entirely missing in the
    source and is dropped.

    `height` ships as a 1-D profile: the source `geo_hgt` varies
    horizontally by only 12-20 m, negligible against 1 km grid spacing.
    It is not interchangeable with the nominal `z0` coordinate, though --
    geopotential height drifts from it by up to 0.5 km aloft.
    """
    f = nc.Dataset(f"{SRC}/20220526_083000.mdv.cf.nc")
    f.set_auto_maskandscale(True)

    z = f.variables["z0"][:]
    kz = (z <= ztop_km) & (z > 0)
    ysl, xsl = slice(i0, i0 + n), slice(j0, j0 + n)

    def grab(name):
        return np.ma.filled(f.variables[name][0, kz, ysl, xsl].astype("f4"), np.nan)

    dbz = grab("dbz")
    temp = grab("Temp")
    height_1d = np.nanmean(grab("geo_hgt") / 1000.0, axis=(1, 2))

    out = nc.Dataset(f"{OUT}/wrf_volume_20220526.nc", "w", format="NETCDF4")
    out.createDimension("z", int(kz.sum()))
    out.createDimension("y", n)
    out.createDimension("x", n)

    _f4(out, "z", z[kz], ("z",), "km", "nominal height above ground")
    _f4(out, "y", f.variables["y0"][ysl], ("y",), "km",
        "distance north of grid origin")
    _f4(out, "x", f.variables["x0"][xsl], ("x",), "km",
        "distance east of grid origin")

    _pack_i2(out, "dbz", dbz, ("z", "y", "x"), DBZ_QUANT, "dBZ",
             "simulated reflectivity",
             "Source `dbz` field (not `dbz_3d`). Quantized to "
             f"{DBZ_QUANT} dBZ; see make_sample_data.py.")
    _pack_i2(out, "Temp", temp, ("z", "y", "x"), 0.01, "degC",
             "air temperature")
    _f4(out, "height", height_1d, ("z",), "km", "geopotential height profile",
        "Horizontal-mean of the source `geo_hgt` field (gpm -> km); "
        "horizontal variation is 12-20 m.")

    _finalize(out,
              "WRF simulated reflectivity volume, 2022-05-26 08:30 UTC",
              "200 x 200 km subdomain of a WRF forecast volume over deep "
              "convection; sample input for the EccoPy-3D notebook.",
              "WRF output converted via MDV (20220526_083000.mdv.cf.nc)",
              "Kao-Shen Chung, National Central University",
              {"valid_time": "2022-05-26T08:30:00Z"})
    out.close()
    f.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    build_1d()
    build_2dh()
    build_2dv()
    build_3d()
    total = 0
    for fn in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, fn))
        total += size
        print(f"  {fn:34s} {size / 1e6:6.2f} MB")
    print(f"  {'TOTAL':34s} {total / 1e6:6.2f} MB")
