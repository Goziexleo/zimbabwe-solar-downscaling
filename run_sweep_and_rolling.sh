#!/usr/bin/env bash
# The two long reruns, in sequence, on the current target and configurations.
#
# Both exceed a two-hour window, so this is meant to be run in a terminal the
# user can watch rather than backgrounded. Sequential, not parallel: 8 GB will
# not hold two of these at once, and parallelising them is what killed an
# earlier rolling-origin run.
#
#   optimise_unet.py    14 variants, fit on 1985-2004 and selected on 2005-2010,
#                       the withheld record touched once per variant. The
#                       September file it replaces predates both the clear-sky
#                       rebuild and the dropout adoption.
#   compute_rolling_origin.py
#                       four expanding windows. Now reads the deployed pixel-wise
#                       configurations instead of hardcoding the superseded ones,
#                       and stamps what it used into a config column.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
mkdir -p logs
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1  U-Net variant sweep, all 14"
$PY -u optimise_unet.py --all 2>&1 | tee logs/unet_sweep_rerun.log

stage "2  rolling origin, deployed configurations"
ROLLING_N_JOBS=2 ROLLING_TIMEOUT=14400 $PY -u compute_rolling_origin.py 2>&1 | tee logs/rolling_origin_rerun.log

stage "BOTH COMPLETE"
echo "Tell Claude; the chapters are not regenerated here on purpose, because the"
echo "sweep's findings feed prose in Section 4.5 that may need rewording."
