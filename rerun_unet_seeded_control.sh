#!/usr/bin/env bash
# Train the DEPLOYED U-Net configuration under the seed, as a control.
#
# The deployed model was trained on Oct 2, before commit 44d4363 added a random
# seed to this script on Oct 3. The dropout-free variant was trained today, with
# the seed. So the 8.62 against 11.66 comparison in Section 4.5 confounds the
# dropout change with a seeded-against-unseeded draw, and this script removes
# that confound by training dropout 0.3 under the same seed as the variant.
#
# Nothing the thesis currently reports is overwritten: the deployed model,
# its validation fields and its projections are left exactly where they are.
set -e
cd "$(dirname "$0")"
PY=/opt/anaconda3/envs/climate_stack/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
export UNET_DROPOUT=0.3 UNET_SEED=42
export UNET_MODEL_PATH="$PWD/data/processed/models/unet/unet_csi_downscaler_seed42.pth"
echo "=== train (dropout 0.3, seed 42, current target) ==="
$PY train_unet_downscaler.py
echo "=== validation fields ==="
VALIDATION_FIELDS_OUTPUT="$PWD/data/processed/evaluation/validation_fields_unet_seed42.nc" \
  $PY generate_validation_spatial_fields.py
echo "=== DONE ==="
