# Instruction: topography correction and full pipeline re-run

**Project:** Machine Learning-Based Downscaling of GCM Outputs for Solar Energy Suitability Mapping over Zimbabwe
**Project root:** `/Users/gozie/Documents/MCSM PROJECT AGY/`
**Canonical methodology reference:** Chapter 3 (`CR_Madukwe_Chapter3_Final.md`). Where this instruction and Chapter 3 disagree, stop and ask. Do not silently deviate.

---

## 0. Standing constraints

- Do not introduce a three-way train/validate/test split. The split is strictly two-period: training 1985 to 2010, validation 2011 to 2024.
- Do not change the canonical grid definitions in `ml_dataset_common.py`. Fine grid is 71 x 81 cells, lat -22.0 to -15.0, lon 25.0 to 33.0, at 0.1 degrees.
- Do not add `np.nan_to_num` calls beyond those already present, and do not remove the existing ones without flagging it first.
- Do not refactor beyond what each step asks for. Targeted edits only.
- Report every metric you produce. Do not summarise away a regression.

---

## 1. Background: what was wrong and what is already fixed

`generate_topographic_features.py` has already been rewritten and re-run. **Do not re-derive this work.** It is described here so you understand what changed downstream.

Three defects existed in the previous version:

1. **Sky-view factor was 100% NaN across the entire domain.** `approximate_svf` divided by `np.max(dem) - np.min(dem)`, and because the DEM was opened with `masked=True`, nodata cells became NaN, so both `np.max` and `np.min` returned NaN. Every output cell was NaN. Every downstream script applies `np.nan_to_num(..., nan=0.0)`, so SVF entered all four models as an all-zero constant channel. All four models were therefore trained on 11 effective features, not 12 (CNN: 8, not 9).

2. **The SVF formula was not the specified algorithm.** It was a 5x5 local elevation-contrast proxy, not the "radial sampling algorithm with 36 azimuth directions and 100 m maximum search radius" of Section 3.5.2. It is now a 36-azimuth horizon search with the Dozier and Frew (1990) sky-view integral.

3. **Aggregation to 0.1 degrees used `.interp(method="linear")`,** which point-samples the 90 m field at target cell centres and discards roughly 14,000 of the ~14,400 cells inside each target cell. Section 3.5.2 specifies a cell mean. It is now a true areal mean by block aggregation.

### Measured effect of the correction

| Field | New vs old correlation | Note |
|---|---|---|
| elevation | 0.9932 | mean 890.79 to 884.72 m, negligible |
| slope | **0.7255** | max 43.96 to 18.51 deg; the old field was point-sampled noise |
| svf | not computable | old field was entirely NaN |

NaN fraction fell from 0.0565 to 0.0109. The residual 1.09% is the lon 33.85 column, which lies outside the DEM's eastern bound (33.75) and is outside the canonical fine grid anyway.

**The slope change is the reason retraining is necessary.** The SVF restoration on its own is unlikely to move metrics much, because `corr(svf, slope) = -0.9663` and the aggregated SVF spans only 0.963 to 1.000. Expect a modest effect at best from SVF, and a real effect from slope.

### New variable

The file now also carries `slope_frac_gt15`, the fraction of 90 m cells within each 0.1 degree cell exceeding 15 degrees. This exists for Section 3.9.2 and is not a model predictor. **Do not add it to any feature list.** Rationale: only 10 cells of 7,452 have a *mean* slope above 15 degrees, so applying the Section 3.9.2 exclusion to mean slope is very nearly a no-op. By sub-grid fraction, 83 cells exceed 30% steep area and 627 exceed 10%.

---

## 2. Step 0: confirm the environment

Before running anything, resolve which interpreter runs the training.

`env/lib/python3.14/site-packages` contains only `cdsapi`, `shapely`, `xarray`, `numba`, `netCDF4`, `pandas`, `dask` and similar. It has **no** `torch`, `sklearn`, `xgboost`, `rasterio`, `rioxarray`, or `scipy`, so the ML scripts cannot be running from it.

Locate the interpreter that has `torch`, `sklearn`, `xgboost`, `rasterio`, `rioxarray` and `scipy` installed. Report its path. Use it for every step below and state which one you used. If no such environment exists, stop and report rather than installing packages into `env/`.

Verify the corrected topography file before proceeding:

```
python -c "
import xarray as xr, numpy as np
d = xr.open_dataset('data/processed/topography/zimbabwe_topographic_features_0.1deg.nc')
assert 'slope_frac_gt15' in d.data_vars, 'stale topography file, re-run generate_topographic_features.py'
for v in ['elevation','slope','svf']:
    a = d[v].values
    print(v, 'nan %.4f' % np.isnan(a).mean(), 'mean %.4f' % np.nanmean(a))
    assert np.isnan(a).mean() < 0.02, v + ' has too many NaN'
print('OK')
"
```

Expected: `elevation nan 0.0109 mean 884.7242`, `slope nan 0.0109 mean 2.4034`, `svf nan 0.0109 mean 0.9980`.

If the assertion on `slope_frac_gt15` fails, run `python generate_topographic_features.py` first (about 40 seconds at the default 100 m search radius) and re-verify.

---

## 3. Step 1: rebuild the ML datasets

```
python build_ml_features_and_targets.py
python build_ml_validation_dataset.py
```

Both read `data/processed/topography/zimbabwe_topographic_features_0.1deg.nc` and interpolate it onto the coarse ERA5 grid, so both must be re-run.

**Verification, mandatory before proceeding.** For each output dataset, confirm that `svf` is now a varying field rather than a constant:

```
python -c "
import xarray as xr, numpy as np
for p in ['data/processed/ml_ready/<train_file>.nc','data/processed/ml_ready/<val_file>.nc']:
    d = xr.open_dataset(p)
    for v in ['elevation','slope','svf']:
        a = d[v].values
        print(p.split('/')[-1], v, 'nan %.4f std %.6f' % (np.isnan(a).mean(), np.nanstd(a)))
"
```

`svf` must have a non-zero standard deviation. If it is zero or the field is all-NaN, stop and report. Substitute the actual filenames from `data/processed/ml_ready/`.

---

## 4. Step 2: retrain all four models

Run in this order and **capture full stdout to a log file for each**:

```
python train_pixelwise_rf.py    2>&1 | tee logs/retrain_rf.log
python train_pixelwise_xgb.py   2>&1 | tee logs/retrain_xgb.log
python train_cnn_downscaler.py  2>&1 | tee logs/retrain_cnn.log
python train_unet_downscaler.py 2>&1 | tee logs/retrain_unet.log
```

Create `logs/` if it does not exist.

**Use the currently deployed configurations. Do not re-run hyperparameter optimisation.** The Section 3.6.7 HPO round has already concluded and the decisions were:

- Random Forest: reverted to Chapter 3 defaults, `n_estimators=500`, `max_features=sqrt`, `min_samples_leaf=5`
- XGBoost: reverted to Chapter 3 defaults, `max_depth=6`, `eta=0.05`, `subsample=0.8`, `min_child_weight=3`
- CNN: kept its tuned configuration, `lr=0.0005`, `lambda_gp=0.01`
- U-Net: adopted `lr=0.0002`

Confirm each script already holds these values before running. If any script does not, report the discrepancy rather than editing silently.

---

## 5. Step 3: re-evaluate and produce the revised Table 3.3

```
python evaluate_unet.py 2>&1 | tee logs/eval_unet.log
python evaluate_cnn.py  2>&1 | tee logs/eval_cnn.log
```

RF and XGBoost metrics are printed by their training scripts.

Produce a comparison table with these columns for all four models: RMSE (W/m2), MAE (W/m2), Pearson R, MBE (W/m2), Skill Score, R2. Place the previous values alongside the new ones.

Previous Table 3.3, validation period 2011 to 2024, post-HPO:

| Model | RMSE | MAE | Pearson R | MBE | Skill Score | R2 |
|---|---|---|---|---|---|---|
| Random Forest | 18.72 | 13.74 | 0.877 | +0.074 | 0.422 | 0.758 |
| XGBoost | 23.15 | 16.59 | 0.818 | +0.790 | 0.285 | 0.630 |
| CNN | 26.21 | 19.38 | 0.814 | -2.03 | 0.190 | 0.525 |
| U-Net | 20.96 | 16.21 | 0.857 | -6.01 | 0.353 | — |

Table 3.3 targets are R > 0.90 and |MBE| < 5 W/m2. No model previously cleared the R threshold.

**Do not present a metric change as an improvement unless it is one.** If the corrected topography makes a model worse, say so plainly. That is a legitimate outcome and it is more useful than a flattering one.

---

## 6. Step 4: regenerate projections and downstream products

The deployed model families are U-Net and Random Forest. Both projection sets must be regenerated because both consume topography.

```
python generate_future_projections.py    2>&1 | tee logs/proj_unet.log
python generate_future_projections_rf.py 2>&1 | tee logs/proj_rf.log
```

Then the multi-model-ensemble aggregations. `compute_mme_aggregations.py` reads `PROJECTIONS_DIR` and writes to `MME_OUTPUT_DIR`, both overridable by environment variable, so run it twice:

```
python compute_mme_aggregations.py

PROJECTIONS_DIR="$PWD/data/processed/projections_rf" \
MME_OUTPUT_DIR="$PWD/data/processed/mme_aggregations_rf" \
python compute_mme_aggregations.py
```

Then the verification and uncertainty products:

```
python generate_validation_spatial_fields.py
python compute_spatial_verification.py
python compute_uncertainty_decomposition.py
```

Expected outputs in `data/processed/evaluation/`: `validation_spatial_fields.nc`, `taylor_diagram_stats.csv`, `spatial_verification_maps.nc`, `uncertainty_decomposition_unet.nc`, `uncertainty_decomposition_rf.nc`, `uncertainty_decomposition_summary.csv`.

Report the revised Taylor statistics against the previous values, computed on the time-mean climatological GHI pattern over 2011 to 2024 at all 71 x 81 fine-grid cells, reference spatial standard deviation 7.98 W/m2:

| Model | Spatial correlation | Std ratio | Centred RMSE | Domain-mean bias |
|---|---|---|---|---|
| Baseline (bilinear) | 0.9998 | 0.9994 | 0.175 | -0.44 |
| Random Forest | 0.9977 | 1.0083 | 0.546 | +0.07 |
| XGBoost | 0.9953 | 1.0769 | 1.012 | +0.79 |
| CNN | 0.8982 | 0.9103 | 3.508 | -2.03 |
| U-Net | 0.8732 | 0.8101 | 3.921 | -6.01 |

The CNN and U-Net spatial correlations are the values most likely to respond to the corrected topography, since the shared-weight architectures rely on the spatial covariates in a way the pixel-wise models do not.

---

## 7. Step 5: sanity checks

Before reporting completion, confirm all of the following and state each result explicitly:

1. No output NetCDF contains an all-NaN or all-zero variable.
2. Latitude ordering is ascending and consistent across the coarse grid, the fine grid, and topography. A vertically flipped model has been a real failure mode on this project.
3. No fabricated hard-zero band exists at the domain edge in any CSI or GHI field. Check the outer ring of the 71 x 81 grid specifically. A previous version of the fine grid extended beyond the coarse grid's coverage and produced exactly this artefact.
4. Projection GHI fields contain no NaN. NaN in the topographic input has previously poisoned every BatchNorm channel and produced all-NaN output.
5. The `svf` channel varies in every training and projection input.
6. Row counts and time coverage match: 312 monthly training samples and 168 monthly validation samples, 5,751 fine-grid cells.

---

## 8. What is explicitly out of scope for this run

- **Section 3.9 multi-criteria suitability analysis.** The auxiliary data (ESA CCI land cover, OpenStreetMap roads and power lines, WDPA protected areas, WorldPop) has not been acquired. Acquisition scripts are being prepared separately. Do not begin this.
- **Hyperparameter re-optimisation.** Section 3.6.7 is closed.
- **The daily-resolution experiment.** It remains a documented limitation. The finding stands: `rsds` is a reused copy of the `ssrd` field the clear-sky index target is derived from, so a high daily R largely reflects the model reproducing its own input.
- **Chapter 3 text amendments.** Sections 3.5.2, 3.9.2 and 3.3.4 need revision, but that is a writing task, not a pipeline task.

---

## 9. Deliverable

A single report containing:

1. The interpreter path used.
2. Revised Table 3.3, with previous values alongside, and an honest statement of whether each model improved, regressed, or held steady.
3. Revised Taylor diagram statistics, with previous values alongside.
4. Revised uncertainty decomposition summary. Downscaling uncertainty previously dominated at 98.3% to 99.7% of variance across all periods and both model families; state whether that still holds.
5. The results of all six sanity checks in Section 7.
6. Any script that failed, any discrepancy between a script's configuration and the values listed in Section 4, and anything you had to change to make the run complete.
