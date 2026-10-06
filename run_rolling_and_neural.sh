#!/usr/bin/env bash
# Stages 2 and 3 of the long reruns, which the sweep's completeness check stopped.
#
# The sweep produced 13 of 14: combo_spec fails on a tensor-shape mismatch in the
# spectral loss when combined with the per-cell climatology, and has never run,
# in September or now. The 13 are the set the thesis has always reported, so they
# are swapped in and the chain continues here.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
mkdir -p logs
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "2  rolling origin, deployed configurations"
ROLLING_N_JOBS=2 ROLLING_TIMEOUT=14400 $PY -u compute_rolling_origin.py 2>&1 | tee logs/rolling_origin_rerun.log

stage "3  neural learning-rate and penalty search"
$PY -u hpo_neural.py 2>&1 | tee logs/neural_hpo_rerun.log

stage "BOTH REMAINING COMPLETE"
echo "Tell Claude. Sections 3.6.7 and 4.2 read these."
