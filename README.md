# ML downscaling of GCM output for solar suitability over Zimbabwe

MSc project code. Statistical downscaling of CMIP6 output to 0.1° over Zimbabwe,
benchmarking four architectures and projecting to 2100 under two SSP scenarios.

`PROJECT_STATUS.md` is the authoritative record of results, findings and
limitations. This file covers only how to run things.

## Environment

```bash
conda env create -f environment.yml -n climate_stack
conda activate climate_stack
```

The repo-local `env/` directory is **not** the ML environment and must not be
used — it has no torch, sklearn or xgboost.

### The OpenMP workaround

Every run must be prefixed:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3 <script>.py
```

Without it the process aborts with `OMP: Error #15: Initializing libomp.dylib,
but found libomp.dylib already initialized.`

This is a **workaround, not a fix**. Two copies of the OpenMP runtime are linked
into the process — typically one from PyTorch and one from the conda-installed
LLVM openmp that scikit-learn and XGBoost link against. Setting the variable
tells the runtime to continue anyway. The Intel documentation warns this "may
cause crashes or silently produce incorrect results".

Nothing in this project has been traced to it, and results have been stable
across repeated retrains, but the risk is real and the proper fix is to remove
the duplicate installation so a single libomp is linked. Recorded here rather
than left as folklore in shell history.

## Running the pipeline

Order matters; each step consumes the previous step's outputs.

```bash
# 1. Static inputs
python generate_topographic_features.py            # SRTM 90 m -> 0.1 deg
python compute_finegrid_clearsky_ghi.py            # clear-sky ceiling

# 2. Predictors  (FORCE_REMERGE=1 to recompute an existing file)
FORCE_REMERGE=1 python merge_predictor_stack.py
EDCM_INCLUDE_HISTORICAL=1 python apply_edcm_bias_correction.py
# enforce physical bounds on the corrected fields (EDCM overshoots at the tails)
python apply_qc_bounds.py --apply

# 3. ML-ready datasets
python build_ml_features_and_targets.py
python build_ml_validation_dataset.py

# 4. Train  (defaults now reproduce the deployed configuration)
PERSIST_RF_MODELS=1 python train_pixelwise_rf.py
python train_pixelwise_xgb.py
python train_cnn_downscaler.py
python train_unet_downscaler.py

# 5. Evaluate
python generate_validation_spatial_fields.py
python compute_table33.py
python compute_spatial_verification.py
python compute_information_content.py
python compute_power_spectra.py
python compute_feature_importance.py
python test_perfect_prognosis.py

# 6. Project
python generate_future_projections.py
python generate_future_projections_rf.py
python compute_mme_aggregations.py
PROJECTIONS_DIR="$PWD/data/processed/projections_rf" \
MME_OUTPUT_DIR="$PWD/data/processed/mme_aggregations_rf" \
  python compute_mme_aggregations.py
python compute_uncertainty_decomposition.py

# 7. Figures
python make_figures.py
```

## Tests

```bash
pytest tests/ -v
```

The suite encodes the three silent bugs that actually occurred (temporal
misalignment, fabricated constant regions, dead predictor channels) plus grid
and physical-bound invariants. Tests skip cleanly when the datasets are absent.

## Notable environment variables

| Variable | Effect |
|---|---|
| `KMP_DUPLICATE_LIB_OK=TRUE` | Required on every run, see above |
| `MODEL_PREDICTORS` | Override the model input set; used for the `rsds` ablation |
| `FORCE_REMERGE=1` | Recompute predictors even if already merged |
| `EDCM_INCLUDE_HISTORICAL=1` | Also bias-correct the historical period, for the transfer test |
| `PERSIST_RF_MODELS=1` | Write the 5,751 per-cell forests (~4 GB measured) |
| `NAN_FRACTION_LIMIT` | Fraction of NaN above which `safe_nan_to_num` raises (default 0.01) |

## What is not in git

`data/` (16 GB), `figures/`, `logs/` and `env/` are excluded. Everything under
`data/processed/` is regenerable from the pipeline above given the raw
downloads.
