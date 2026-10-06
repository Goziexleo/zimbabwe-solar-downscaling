#!/usr/bin/env bash
# The neural learning-rate and penalty search, queued behind the two long reruns.
#
# All three want the MPS device and 8 GB is not enough for two at once, so this
# waits for optimise_unet.py and compute_rolling_origin.py to leave the process
# table before starting. It polls rather than using a sentinel file so that a
# crash in either upstream job still releases it.
#
# The September file this replaces predates the clear-sky rebuild. Its
# is_deployed column was also wrong: hpo_neural.py hardcoded CNN lr 0.0005 with
# lambda_gp 0.01 and U-Net lr 0.0002 while the training scripts deployed 0.001
# for all three, so the flag marked rows the search had not selected. The script
# now reads the deployed values from the training scripts and asserts they are
# in the grid.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
mkdir -p logs

printf 'waiting for the sweep and rolling origin to finish (%s)\n' "$(date +%H:%M:%S)"
while pgrep -f "optimise_unet.py|compute_rolling_origin.py" >/dev/null; do sleep 60; done
printf 'upstream clear, starting the neural search (%s)\n' "$(date +%H:%M:%S)"

$PY -u hpo_neural.py 2>&1 | tee logs/neural_hpo_rerun.log
printf '\nNEURAL SEARCH COMPLETE (%s)\n' "$(date +%H:%M:%S)"
echo "Tell Claude: Section 3.6.7 quotes these inner-split figures and needs updating."
