#!/usr/bin/env bash
# Resume the clear-sky rerun from stage 10. Stages 1-9 completed; rolling origin
# died on its largest fold with a joblib worker timeout.
#
# Cause: the machine has 8 GB and the default pool is four workers, each holding
# its own copy of the fold's predictor slice. The last fold trains on 384 months
# across 5,751 cells, so four copies exhaust memory, a worker is killed, and the
# remaining tasks time out. Two workers and a longer ceiling trade wall time for
# the headroom to finish.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
export ROLLING_N_JOBS=2 ROLLING_TIMEOUT=14400

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "10 rolling origin, two workers"
$PY compute_rolling_origin.py

stage "11 figures"
$PY make_figures.py
$PY make_suitability_maps.py
$PY make_study_area_map.py

stage "12 chapters, document, verification"
./finalise_dissertation.sh

stage "13 sync deliverables"
$PY brief/build_brief_formats.py --pdf --sync >/dev/null
echo "synced to the project folder"

printf '\n========== RESUME COMPLETE  (%s) ==========\n' "$(date +%H:%M:%S)"
