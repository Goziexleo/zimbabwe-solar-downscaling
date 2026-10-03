#!/usr/bin/env bash
# Re-measure the dropout-free U-Net on the CURRENT target.
#
# The comparison in Sections 4.5 and 5.6 was made before the clear-sky rebuild,
# so it set a model trained on the old target against a deployed model trained on
# the new one. That is not a comparison.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE UNET_DROPOUT=0.0
export UNET_MODEL_PATH="$PWD/data/processed/models/unet/unet_csi_downscaler_drop0.pth"
export UNET_PROJECTIONS_OUTPUT_DIR="$PWD/data/processed/projections_unet_drop0"
echo "=== train (dropout 0, current target) ==="
$PY train_unet_downscaler.py
echo "=== validation fields ==="
VALIDATION_FIELDS_OUTPUT="$PWD/data/processed/evaluation/validation_fields_unet_drop0.nc" \
  $PY generate_validation_spatial_fields.py
echo "=== project and aggregate ==="
$PY generate_future_projections.py
PROJECTIONS_DIR="$UNET_PROJECTIONS_OUTPUT_DIR" \
MME_OUTPUT_DIR="$PWD/data/processed/mme_aggregations_unet_drop0" $PY compute_mme_aggregations.py
echo "=== DONE ==="
