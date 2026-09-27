#!/bin/zsh
# Regenerate everything that depends on the CNN and U-Net, after the honest
# hyperparameter re-search changed both configurations (Section 3.6.7).
#
# Written with explicit failure handling because the first version of this
# cascade printed COMPLETE after all fourteen steps had failed: `nohup sh` did
# not inherit conda, and piping each step to `tail` meant the status checked was
# always tail's. Here the environment is activated explicitly, an import guard
# fails fast if it did not take, and every step's own exit status is tested.

set -u
cd "$(dirname "$0")" || exit 1

source /opt/anaconda3/etc/profile.d/conda.sh || { echo "FATAL: no conda"; exit 1; }
conda activate climate_stack || { echo "FATAL: cannot activate climate_stack"; exit 1; }

# torch and the OMP-linked libraries collide in one process on this machine, so
# the project already sets this when it spawns training (compute_rolling_origin,
# optimise_unet). Without it the import guard below fails on a healthy
# environment, which is what happened on the first run of this cascade.
export KMP_DUPLICATE_LIB_OK=TRUE

python -c "import torch, xarray, pandas, docx, pptx" || {
  echo "FATAL: environment is missing packages - wrong interpreter?"; exit 1; }

LOG_DIR="logs/neural_cascade"
mkdir -p "$LOG_DIR"
FAILED=0

step () {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  printf '%-52s ' "$name"
  "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "ok"
  else
    echo "FAILED (exit $rc) -> $log"
    FAILED=$((FAILED + 1))
    tail -5 "$log" | sed 's/^/      /'
  fi
  return 0
}

echo "=== retrain both neural models on the honest hyperparameters ==="
step train_cnn            python train_cnn_downscaler.py
step train_unet           python train_unet_downscaler.py

echo
echo "=== evaluation ==="
step evaluate_cnn         python evaluate_cnn.py
step evaluate_unet        python evaluate_unet.py
step validation_fields    python generate_validation_spatial_fields.py
step table_3_3            python compute_table33.py
step bootstrap_ci         python compute_bootstrap_ci.py
step bca_centred_rmse     python compute_bca_centred_rmse.py
step power_spectra        python compute_power_spectra.py
step rolling_origin       python compute_rolling_origin.py
step information_content  python compute_information_content.py

echo
echo "=== projections and aggregation ==="
step proj_cnn             python generate_future_projections_cnn.py
step proj_unet            python generate_future_projections.py
step mme_cnn              env PROJECTIONS_DIR="$PWD/data/processed/projections_cnn" \
                              MME_OUTPUT_DIR="$PWD/data/processed/mme_aggregations_cnn" \
                              python compute_mme_aggregations.py
step mme_unet             env PROJECTIONS_DIR="$PWD/data/processed/projections" \
                              MME_OUTPUT_DIR="$PWD/data/processed/mme_aggregations" \
                              python compute_mme_aggregations.py
step scenario_screen      python compute_scenario_discrimination.py
step uncertainty          python compute_uncertainty_decomposition.py
step suitability          python compute_suitability.py

echo
echo "=== figures and documents ==="
step figures              python make_figures.py
step suitability_maps     python make_suitability_maps.py
step chapter4             python build_chapter4.py
step chapter5             python build_chapter5.py

echo
if [ $FAILED -eq 0 ]; then
  echo "CASCADE COMPLETE - every step exited 0"
else
  echo "CASCADE FINISHED WITH $FAILED FAILED STEP(S) - see $LOG_DIR"
fi
exit $FAILED
