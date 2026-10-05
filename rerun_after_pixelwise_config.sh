#!/usr/bin/env bash
# Regenerate everything downstream of the pixel-wise configurations.
#
# train_pixelwise_rf.py and train_pixelwise_xgb.py now carry the
# cross-validated configurations rather than the a priori defaults, so both
# models and every artefact computed from them are stale. The CSI target, the
# CNN and the U-Net are unchanged and are not retrained here; their validation
# fields are rebuilt anyway because generate_validation_spatial_fields.py writes
# all four into one file.
#
# The per-cell models MUST be persisted or the projection stage has nothing to
# apply. Rolling origin is deliberately left out: it is the long job and it is
# run separately so an interruption does not cost the rest of the rebuild.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD"
export PERSIST_RF_MODELS=1 PERSIST_XGB_MODELS=1
R="$PWD"
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1  retrain the two pixel-wise models"
$PY train_pixelwise_xgb.py
$PY train_pixelwise_rf.py

stage "2  validation fields and headline metrics"
$PY generate_validation_spatial_fields.py
$PY compute_table33.py
$PY compute_spatial_verification.py
$PY compute_feature_importance.py

stage "3  intervals, spectra, information content"
$PY compute_bootstrap_ci.py
$PY compute_bca_centred_rmse.py
$PY compute_power_spectra.py
$PY compute_information_content.py

stage "4  baselines and the satellite comparison"
$PY compute_baselines.py
$PY validate_against_sarah.py

stage "5  projections for the two pixel-wise models"
$PY generate_future_projections_rf.py
$PY generate_future_projections_xgb.py

stage "6  their ensembles"
PROJECTIONS_DIR="$R/data/processed/projections_rf"  MME_OUTPUT_DIR="$R/data/processed/mme_aggregations_rf"  $PY compute_mme_aggregations.py
PROJECTIONS_DIR="$R/data/processed/projections_xgb" MME_OUTPUT_DIR="$R/data/processed/mme_aggregations_xgb" $PY compute_mme_aggregations.py

stage "7  screen, uncertainty budget, operational baseline"
$PY compute_scenario_discrimination.py
$PY compute_uncertainty_decomposition.py
$PY compute_projection_baselines.py

stage "8  suitability, which reads the XGBoost ensemble"
$PY build_suitability_layers.py
$PY compute_suitability.py
$PY compute_robust_set_geography.py
$PY compute_robustness_monte_carlo.py

stage "9  the configuration comparison, now a priori against deployed"
$PY compute_pixelwise_cv_config_check.py

stage "10 figures"
$PY make_figures.py
$PY make_suitability_maps.py

stage "11 refresh the brief's tables and sync"
$PY brief/sync_brief_tables.py
$PY brief/build_brief_formats.py --pdf --sync >/dev/null

printf '\n========== REBUILD COMPLETE  (%s) ==========\n' "$(date +%H:%M:%S)"
