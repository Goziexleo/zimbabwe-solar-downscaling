#!/usr/bin/env bash
# Re-download ERA5 over the corrected bounding box, staged.
#
# The raw ERA5 on disk was requested for [-15, 25, -22, 33], half a degree short
# of the domain Section 3.2.1 states and of the country itself, which reaches
# 22.42 S. The download scripts now carry [-15, 25, -22.5, 33.5]; both skip files
# that already exist, so they write to a staging directory here rather than
# silently skipping all 160 existing files.
#
# Everything else already covers the wider box: CMIP6 is global, SRTM runs to
# 22.75 S, the land-cover subset to 22.5 S, and WorldPop, WDPA and GADM stop at
# the national boundary by construction, which is correct.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
STAGE="$PWD/data/raw/era5_wide"
export ERA5_RAW_DIR="$STAGE/predictors"
export ERA5_VAL_RAW_DIR="$STAGE/predictors/validation"
mkdir -p "$ERA5_VAL_RAW_DIR"

echo "staging to $STAGE"
echo "=== training predictors, 1985-2010 ($(date +%H:%M:%S)) ==="
$PY download_era5_predictors2.py

echo "=== validation predictors, 2011-2024 ($(date +%H:%M:%S)) ==="
$PY download_era5_validation_predictors.py

echo
echo "training files:   $(ls "$ERA5_RAW_DIR"/*.nc 2>/dev/null | wc -l | tr -d ' ')"
echo "validation files: $(ls "$ERA5_VAL_RAW_DIR"/*.nc 2>/dev/null | wc -l | tr -d ' ')"
echo "=== DOWNLOAD COMPLETE ($(date +%H:%M:%S)) ==="
