#!/bin/bash
# Exports exact MATLAB strel('disk', r) masks for the radii EccoPy uses,
# so the Python port can load bit-exact structuring elements instead of
# the approximate Euclidean-circle _disk(r).
#
# Usage:
#   ./run_export_disk_strels.sh
#   ./run_export_disk_strels.sh "3,5,15,25" ./disk_strels
#
# Args (optional): [radii_csv] [outdir]
# Defaults: radii = 3,5,15,25 (enlarge_mixed=5, enlarge_conv=5 case);
#           outdir = ./disk_strels

set -e

export RADII="${1:-3,5,15,25}"
export OUTDIR="${2:-./disk_strels}"

echo "Exporting strel('disk', r) masks for radii: $RADII"
echo "Output directory: $OUTDIR"

matlab -nodisplay -nosplash -r "run('export_disk_strels.m'); exit"

echo "Done. .mat files written to $OUTDIR"
