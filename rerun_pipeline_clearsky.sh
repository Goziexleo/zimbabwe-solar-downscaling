#!/usr/bin/env bash
# Full rerun after putting the clear-sky denominator on a 24-hour basis.
#
# The clear-sky climatology is the denominator of the training target, so every
# trained model, projection and suitability layer downstream of it is invalid
# until refitted. compute_finegrid_clearsky_ghi.py has already been re-run; this
# script does everything that depends on it, in dependency order.
#
# The per-cell models MUST be persisted or the projection stage has nothing to
# apply. That is what PERSIST_RF_MODELS and PERSIST_XGB_MODELS are for.
#
# Order note: the long rolling-origin evaluation runs late, so that if it is
# interrupted the rest of the results are already rebuilt. The chapters are last,
# because they read every CSV above them.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
export PERSIST_RF_MODELS=1 PERSIST_XGB_MODELS=1
R="$PWD"

stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1  rebuild the CSI target and ML-ready datasets"
$PY build_ml_features_and_targets.py
$PY build_ml_validation_dataset.py

stage "2  retrain all four architectures"
$PY train_pixelwise_rf.py
$PY train_pixelwise_xgb.py
$PY train_cnn_downscaler.py
$PY train_unet_downscaler.py

stage "3  validation fields and headline metrics"
$PY evaluate_cnn.py
$PY evaluate_unet.py
$PY generate_validation_spatial_fields.py
$PY compute_table33.py
$PY compute_spatial_verification.py
$PY compute_feature_importance.py

stage "4  intervals, spectra, information content"
$PY compute_bootstrap_ci.py
$PY compute_bca_centred_rmse.py
$PY compute_power_spectra.py
$PY compute_information_content.py

stage "5  baselines and the satellite comparison"
$PY compute_baselines.py
$PY validate_against_sarah.py

stage "6  projections for every architecture"
$PY generate_future_projections_rf.py
$PY generate_future_projections_xgb.py
$PY generate_future_projections_cnn.py
$PY generate_future_projections.py

stage "7  multi-model ensembles"
PROJECTIONS_DIR="$R/data/processed/projections_rf"  MME_OUTPUT_DIR="$R/data/processed/mme_aggregations_rf"  $PY compute_mme_aggregations.py
PROJECTIONS_DIR="$R/data/processed/projections_xgb" MME_OUTPUT_DIR="$R/data/processed/mme_aggregations_xgb" $PY compute_mme_aggregations.py
PROJECTIONS_DIR="$R/data/processed/projections_cnn" MME_OUTPUT_DIR="$R/data/processed/mme_aggregations_cnn" $PY compute_mme_aggregations.py
PROJECTIONS_DIR="$R/data/processed/projections"     MME_OUTPUT_DIR="$R/data/processed/mme_aggregations"     $PY compute_mme_aggregations.py

stage "8  screen, uncertainty budget, operational baseline"
$PY compute_scenario_discrimination.py
$PY compute_uncertainty_decomposition.py
$PY compute_projection_baselines.py

stage "9  suitability"
$PY build_suitability_layers.py
$PY compute_suitability.py
$PY compute_robust_set_geography.py
$PY compute_robustness_monte_carlo.py

stage "10 rolling origin (the long one)"
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

printf '\n========== PIPELINE COMPLETE  (%s) ==========\n' "$(date +%H:%M:%S)"
