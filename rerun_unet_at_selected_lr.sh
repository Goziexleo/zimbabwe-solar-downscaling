#!/usr/bin/env bash
# Refit the U-Net at the learning rate its own search selects, then regenerate
# everything downstream of it.
#
# Section 3.6.7's rerun search preferred 2e-4 to the deployed 1e-3. The U-Net was
# the only model left at a setting its own search did not select, and Section 4.4
# declines to deploy it precisely because its margin over XGBoost is not
# established, so a better-tuned U-Net could change that decision under the
# thesis's own rule. This removes the risk instead of disclosing it.
#
# The chapters are not regenerated here. If the margin becomes established the
# deployment decision changes, and that is not a change to make inside a script.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
mkdir -p logs
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1  retrain the U-Net at 2e-4"
$PY train_unet_downscaler.py

stage "2  validation fields and headline metrics"
$PY evaluate_unet.py
$PY generate_validation_spatial_fields.py
$PY compute_table33.py
$PY compute_spatial_verification.py

stage "3  intervals, spectra, information content, baselines"
$PY compute_bootstrap_ci.py
$PY compute_bca_centred_rmse.py
$PY compute_power_spectra.py
$PY compute_information_content.py
$PY compute_baselines.py

stage "4  U-Net projections and ensemble"
$PY generate_future_projections.py
PROJECTIONS_DIR="$PWD/data/processed/projections" \
MME_OUTPUT_DIR="$PWD/data/processed/mme_aggregations" $PY compute_mme_aggregations.py

stage "5  screen, uncertainty budget, operational baseline"
$PY compute_scenario_discrimination.py
$PY compute_uncertainty_decomposition.py
$PY compute_projection_baselines.py

stage "6  derived products and figures"
$PY compute_dropout_variant_comparison.py
$PY make_figures.py

stage "UNET REFIT COMPLETE"
$PY - <<'PYEOF2'
import pandas as pd
t = pd.read_csv("data/processed/evaluation/table_3_3.csv").set_index("model")
b = pd.read_csv("data/processed/evaluation/bca_intervals.csv")
r = b[(b.model_a == "XGBoost") & (b.model_b == "U-Net") & (b.metric == "RMSE")]
if r.empty:
    r = b[(b.model_a == "U-Net") & (b.model_b == "XGBoost") & (b.metric == "RMSE")]
r = r.iloc[0]
print("U-Net %.4f  XGBoost %.4f  over Zimbabwe" % (t.loc["U-Net","RMSE_zw"], t.loc["XGBoost","RMSE_zw"]))
print("RMSE difference %.4f  BCa [%.4f, %.4f]  spans zero: %s"
      % (r.plug_in, r.bca_lo, r.bca_hi, bool(r.bca_spans_zero)))
print("DEPLOYMENT DECISION UNCHANGED" if bool(r.bca_spans_zero)
      else "*** THE MARGIN IS NOW ESTABLISHED - the deployment rule points at the U-Net ***")
PYEOF2
