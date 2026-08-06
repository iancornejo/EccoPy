#!/bin/bash
# Exports exact MATLAB strel('disk', r) masks. OPTIONAL: eccopy.core.disk
# now generates these shapes in pure Python for any radius (bit-exact to
# these exports). Use this only to (re)generate the ground-truth
# regression fixtures under eccopy/core/data/, not to enable a radius.
#
# Usage:
#   ./run_export_disk_strels.sh
#   ./run_export_disk_strels.sh "3,5,15,25" ../eccopy/core/data/disk_strels
#
# Args (optional): [radii_csv] [outdir]
# Defaults: radii = 3,5,15,25 (enlarge_mixed=5, enlarge_conv=5 case);
#           outdir = ../eccopy/core/data/disk_strels (where the fixtures live)

set -e

export RADII="${1:-3,5,15,25}"
export OUTDIR="${2:-../eccopy/core/data/disk_strels}"

echo "Exporting strel('disk', r) masks for radii: $RADII"
echo "Output directory: $OUTDIR"

matlab -nodisplay -nosplash -r "run('export_disk_strels.m'); exit"

echo "Done. .mat files written to $OUTDIR"
