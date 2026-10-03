#!/usr/bin/env bash
# Regenerate everything downstream of the U-Net checkpoint.
#
# The U-Net was retrained after the clear-sky rerun had already produced the
# evaluation artefacts, so the checkpoint on disk is newer than every number
# computed from it. Reporting the old figures against the new model would be
# describing a model that no longer exists.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
R="$PWD"
stage() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M:%S)"; }

stage "1 validation fields and headline metrics"
$PY evaluate_unet.py
$PY generate_validation_spatial_fields.py
$PY compute_table33.py
$PY compute_spatial_verification.py
$PY compute_feature_importance.py

stage "2 intervals, spectra, information content, baselines"
$PY compute_bootstrap_ci.py
$PY compute_bca_centred_rmse.py
$PY compute_power_spectra.py
$PY compute_information_content.py
$PY compute_baselines.py
$PY validate_against_sarah.py

stage "3 U-Net projections and ensemble"
$PY generate_future_projections.py
PROJECTIONS_DIR="$R/data/processed/projections" \
MME_OUTPUT_DIR="$R/data/processed/mme_aggregations" $PY compute_mme_aggregations.py

stage "4 screen, uncertainty budget, operational baseline"
$PY compute_scenario_discrimination.py
$PY compute_uncertainty_decomposition.py
$PY compute_projection_baselines.py

stage "5 figures"
$PY make_figures.py
$PY make_suitability_maps.py

stage "6 chapters, document, verification"
./finalise_dissertation.sh

stage "7 sync"
$PY brief/build_brief_formats.py --pdf --sync >/dev/null
echo "synced"
printf '\n========== COMPLETE (%s) ==========\n' "$(date +%H:%M:%S)"
