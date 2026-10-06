#!/usr/bin/env bash
# The three long reruns, sequentially, each written somewhere it cannot corrupt
# what the chapters read until it is complete.
#
# Lessons this encodes:
#   - the sweep writes to a STAGING path and is swapped in only when all 14
#     variants are present. An interrupted run that wrote to the live file left
#     Section 4.5 comparing variants from two model generations.
#   - nohup, so closing the terminal tab does not kill it. Tab closure is what
#     killed the first attempt after five variants.
#   - sequential, because all three want the MPS device and 8 GB will not hold
#     two of them.
#   - the chapters are NOT regenerated here. The sweep feeds prose in Section
#     4.5 that may need rewriting rather than renumbering.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
mkdir -p logs data/_staging
STAGE="$PWD/data/_staging/unet_optimisation.csv"
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1  U-Net sweep, 14 variants, to a staging file"
rm -f "$STAGE"
UNET_OPT_OUT="$STAGE" $PY -u optimise_unet.py --all 2>&1 | tee logs/unet_sweep_rerun.log
n=$($PY -c "import pandas,sys; print(len(pandas.read_csv(sys.argv[1])))" "$STAGE")
if [ "$n" -eq 14 ]; then
  cp "$STAGE" data/processed/evaluation/unet_optimisation.csv
  echo "all 14 variants present; swapped into the live file"
else
  echo "only $n of 14 variants; live file left alone, staging kept at $STAGE"
  exit 1
fi

stage "2  rolling origin, deployed configurations"
ROLLING_N_JOBS=2 ROLLING_TIMEOUT=14400 $PY -u compute_rolling_origin.py 2>&1 | tee logs/rolling_origin_rerun.log

stage "3  neural learning-rate and penalty search"
$PY -u hpo_neural.py 2>&1 | tee logs/neural_hpo_rerun.log

stage "ALL THREE COMPLETE"
echo "Tell Claude. Sections 3.6.7, 4.2 and 4.5 all read these and need checking."
