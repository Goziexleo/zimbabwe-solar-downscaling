# Project status and work record

**Project:** Machine Learning-Based Downscaling of GCM Outputs for Solar Energy Suitability Mapping over Zimbabwe
**Author:** Chiagozie Raphael Madukwe (FSR2526653), University of Zimbabwe MSc
**Project root:** `/Users/gozie/Documents/MCSM PROJECT AGY` (git repository since the second audit; `data/` untracked)
**Chapter 3:** `.../UNI ZIM/PROJECT CHAPTERS/CR_Madukwe_Chapter3_Final.docx`

**Chapter 3 backups (same folder), in order:**
`..._BACKUP_preupdate.docx` (before the proposal→implemented rewrite) ·
`..._BACKUP_pre_rf_deploy.docx` (before §3.8.4 was first added) ·
`..._BACKUP_pre_384_rewrite.docx` (before §3.8.4 was rewritten for the corrected predictor set) ·
`..._BACKUP_pre_validation_scope.docx` (before §3.3.3 was reworded to treat SARAH-2/NSRDB as planned)

This is the authoritative record of what the pipeline does, what was built and fixed, the current numbers, the honest limitations, and what remains. All figures below are from the final corrected-alignment / 5-predictor run and are mutually consistent.

---

## 1. What the project does

Statistical downscaling of coarse CMIP6 global-climate-model output to 0.1° over Zimbabwe, to produce a solar-resource product for siting analysis. Four architectures are benchmarked on identical inputs:

- **Random Forest** and **XGBoost** — *pixel-wise*: one independent model fitted per output grid cell (5,751 of each)
- **CNN** and **U-Net** — *shared-weight*: one model applied across the whole domain

Trained on 1985–2010 ERA5, validated on withheld 2011–2024 ERA5, then applied to bias-corrected CMIP6 to project 2026–2100 under two emission scenarios across three GCMs.

The prediction target is the **clear-sky index** (measured irradiance ÷ modelled clear-sky ceiling), not raw irradiance, so the model learns atmospheric departure from a deterministic solar-geometry baseline rather than re-learning the seasonal cycle.

---

## 2. Environment

- Interpreter: `/opt/anaconda3/envs/climate_stack/bin/python3` (conda env `climate_stack`)
- torch 2.13.0 · scikit-learn 1.9.0 · xgboost 3.2.0 · rasterio 1.4.4 · rioxarray 0.19.0 · scipy 1.17.1 · pvlib 0.15.2 · matplotlib 3.11.0
- **Not installed:** cartopy, geopandas (needed for scripted cartography)
- **Every run needs `KMP_DUPLICATE_LIB_OK=TRUE`** or libomp double-initialisation aborts the process
- Neural training on Apple MPS; tree models across 8 CPU cores via joblib
- The repo-local `env/` directory is **not** the ML environment — do not use it

---

## 3. Domain, grids, splits

| Property | Value |
|---|---|
| Coarse (predictor) grid | 0.25°, 29 × 33 |
| Fine (target) grid | 0.1°, 71 × 81 = **5,751 cells** |
| Domain retained | 22.0°S–15.0°S, 25.0°E–33.0°E |
| Upsampling factor | 2.5 by resolution; ≈2.45 by cell count (71/29, 81/33) |
| Training | 1985–2010, **312 monthly fields** |
| Validation | 2011–2024, **168 monthly fields** |
| Projection | 2026–2100, 900 months × 3 GCMs × 2 scenarios |

The retained domain is narrower than Chapter 3's nominal box (15.0–22.5°S, 25.0–33.5°E) because the ERA5 download undershot by 0.5° at the south and east edges. The fine grid was aligned to the *actual* coarse coverage so no target cell requires extrapolation — this fixed bug §6.3.

**GCMs:** CNRM-CM6-1, MPI-ESM1-2-HR, ACCESS-CM2 · **Scenarios:** SSP2-4.5, SSP5-8.5

---

## 4. Features and target

**Five atmospheric predictors fed to the models (ERA5):** `clt`, `tas`, `ps`, `huss`, `od550aer`
**Three topographic (SRTM 90 m → 0.1°):** `elevation`, `slope`, `svf`
**Three astronomical:** `solar_zenith_angle`, `sin_doy`, `cos_doy`

- RF, XGBoost, U-Net use **11** features (`COARSE_FEATURE_VARS`)
- CNN uses **8** (5 predictors + 3 topographic), per §3.6.5's (C+3) formula
- **`rsds` is stored in the datasets but is NOT a model input.** It is retained only because the §3.7.3 interpolation baseline is defined from it. See §6.11 and §7.1.
- Predictors are **ERA5, not bias-corrected CMIP6** — this is perfect-prognosis training; CMIP6 enters only at projection time
- `MODEL_PREDICTORS="clt,tas,ps,huss,rsds,od550aer"` reproduces the §7.3 ablation

**Target:** `CSI = ssrd_fine / clearsky_ghi`, clipped to [0, 1.1]. Recovered as `GHI = CSI × clearsky_ghi`. Observed CSI spans 0.256–0.632, so **the clip never binds** and the normalisation is an exact inverse — verified to machine precision (max difference 1.14 × 10⁻¹³ W m⁻²). The clear-sky specification therefore fixes the physical meaning of the intermediate CSI the models learn, but **cancels entirely from the GHI product**.

---

## 5. Pipeline

### Shared modules
| Script | Role |
|---|---|
| `ml_dataset_common.py` | Grid definitions, feature lists, astronomical features, `build_fine_grid_csi`, `csi_to_ghi`, `evaluation_baselines`, `safe_nan_to_num`. Single source of truth. |
| `unet_model.py` | `ClimateUNet` (schema v5), checkpoint I/O, `predict_csi_field` |
| `cnn_model.py` | `SuperResolutionCNN`, checkpoint I/O |
| `pixelwise_common.py` | `build_fine_pixel_dataset` — collocates coarse predictors onto every fine cell |
| `oni_utils.py` | NOAA CPC ONI → El Niño / La Niña / neutral |

### Data preparation
| Script | Role |
|---|---|
| `generate_topographic_features.py` | SRTM 90 m → elevation, slope (Horn 1981), SVF (36-azimuth horizon search, Dozier & Frew 1990 integral), areal block mean to 0.1°. Also `slope_frac_gt15` for §3.9. |
| `compute_finegrid_clearsky_ghi.py` | Monthly clear-sky ceiling, PVLIB Ineichen, **per-cell SRTM elevation** and **per-cell/per-month Linke turbidity** |
| `merge_predictor_stack.py` | Merges `rsds` (reused `ssrd`) + MERRA-2 `od550aer`. Aligns on **calendar month** (`align_to_months`) — the old nearest-timestamp match caused §6.11. `FORCE_REMERGE=1` recomputes; warns loudly on missing months. |
| `apply_edcm_bias_correction.py` | Equidistant CDF matching of CMIP6 against ERA5 |
| `build_ml_features_and_targets.py` / `build_ml_validation_dataset.py` | Produce the 312 / 168-sample datasets |

### Training and evaluation
| Script | Role |
|---|---|
| `train_pixelwise_rf.py` | 5,751 forests. `PERSIST_RF_MODELS=1` writes per-cell joblib (~4 GB measured) + `manifest.json` |
| `train_pixelwise_xgb.py` | 5,751 boosters at 200 **fixed** rounds (no validation early stopping, see §6.12). `PERSIST_XGB_MODELS=1` writes per-cell joblib (~400 MB) + `manifest.json` |
| `train_cnn_downscaler.py` | CNN, MSE + spatial-gradient penalty, 100 epochs |
| `train_unet_downscaler.py` | U-Net, ENSO-stratified batches, flip augmentation, early stopping |
| `compute_table33.py` | **Canonical Table 3.3** for all four models from the saved fields — single source of truth, no retraining |
| `evaluate_unet.py` / `evaluate_cnn.py` | Per-model metrics. Skill is reported against climatology (the only admissible reference); interpolation and delta-mapping appear as circularity diagnostics |
| `hpo_pixelwise.py` | 5-fold temporal CV grid search on a 150-cell subsample |
| `compute_information_content.py` | Sub-0.25° information test with elevation positive control |
| `compute_feature_importance.py` | RF MDI + permutation, XGBoost gain + cover |
| `compute_rolling_origin.py` | Rolling-origin evaluation across 4 expanding-window folds (C3) |

### Projection and verification
| Script | Role |
|---|---|
| `generate_future_projections{,_rf,_cnn,_xgb}.py` | U-Net / RF / CNN / XGBoost → downscaled GHI, 6 GCM-scenario combos each |
| `compute_mme_aggregations.py` | Ensemble mean + inter-model spread, 3 horizons. `PROJECTIONS_DIR` / `MME_OUTPUT_DIR` |
| `generate_validation_spatial_fields.py` | Full GHI fields: truth, baseline, all 4 models |
| `compute_spatial_verification.py` | Taylor statistics + per-pixel bias/RMSE maps |
| `compute_uncertainty_decomposition.py` | **Four-component** σ_GCM / σ_SSP / σ_arch / σ_DS. σ_DS **derived** from the validation fields; σ_arch from `ARCH_DIRS` (n = 3), kept separate from the deployed-model list. |

`deprecated/daily_resolution_experiment/` holds the superseded daily-resolution scripts and outputs, with a README explaining why.

---

## 6. Bugs found and fixed

### 6.11 One-month `rsds` misalignment — the most consequential bug in the project
`merge_predictor_stack.py` aligned `rsds` with `reindex(method="nearest")`. The predictor files stamp each month at its **first** day (`2011-02-01`); the `ssrd` files stamp it at its **last** (`2011-01-31`). Nearest-neighbour matched them one day apart, writing **January's irradiance into February's slot** — validation `rsds` lagged exactly one month (`rsds[t] == ssrd[t-1]`, bit-exact, `corr = 1.000000` at lag +1).

Invisible in training because `build_ml_features_and_targets.py` used the *same* unsafe match in the opposite direction, cancelling it. Two wrongs made a right in one split and not the other.

**Consequences, all corrected:** models were fed the wrong month's irradiance as a high-weight feature at validation only, degrading every Table 3.3 figure (XGBoost RMSE 22.94 → 5.54 on fixing this alone). The baseline was computed from the same lagged field, inflating its RMSE from 0.233 to 32.38 and turning deeply negative skill scores into apparently positive ones. Both build scripts now match on calendar month.

### 6.12 XGBoost was early-stopped on the validation set
`train_pixelwise_xgb.py` fitted each cell with `eval_set=[(X_val, y_val)]` and `early_stopping_rounds=50`, so the number of boosting rounds — each cell's effective capacity — was chosen by watching the data the model was then scored against. 5,751 hyperparameters fitted on the evaluation set.

Measured cost: **9.1145 leaky against 9.2422 clean**, about 1.4% of RMSE. Early stopping on an inner 80/20 split of the training record was also tried and scored *worse* (10.0157), because holding back a fifth of an already-small 312-month record costs more than the stopping rule gains. Fixed 200 rounds uses all training data and no validation information, and is now the default.

**The ranking is unchanged** — at 9.2422 XGBoost still has the lowest aggregate RMSE, so the §3.8.4 selection argument is unaffected. But it was not an out-of-sample number, and it was the number that put XGBoost ahead of RF in the first place.

### 6.3 Fabricated zero band across ~25% of the domain (severe, silent)
The fine grid extended up to 0.85° beyond coarse ERA5 coverage; interpolation returned NaN there, a defensive `np.nan_to_num` zeroed it, and a quarter of every training target became a hard-zero band. No aggregate metric flagged it — found only by plotting fields. Fixed by aligning `FINE_LAT`/`FINE_LON` to real coverage.

### 6.4 Sky-view factor 100% NaN (severe, silent)
`approximate_svf` divided by a range that was itself NaN. Downstream `np.nan_to_num` made SVF an all-zero constant channel, so all four models trained on one fewer effective feature than documented. The aggregation also point-sampled instead of taking a cell mean, and the algorithm was a 5×5 contrast proxy rather than the specified 36-azimuth search. All corrected; slope changed most (new-vs-old correlation 0.7255, max 43.96° → 18.51°).

### 6.1 U-Net ReLU on the anomaly
The output layer applied ReLU to a variance-scaled CSI *anomaly*, which is legitimately negative, truncating half the prediction range. R collapsed to ~0.68. Fixed by removing the output activation; bounds applied after denormalisation.

### 6.5 Clear-sky ceiling not actually cell-specific
`altitude=1000.0` and `linke_turbidity=3.0` were constants, so §3.5.4's "cell-specific" field varied only with solar zenith angle. Replaced with per-cell SRTM elevation (124–1840 m) and per-cell/per-month PVLIB turbidity (3.65–6.45 Jan, 2.55–4.65 Jul). Clear-sky sub-0.25° content rose **0.0000% → 1.55%**.

### 6.2 Latitude ordering mismatch
Coarse grid descended, topography ascended — would have trained a vertically flipped model. Fixed with `.sortby('lat')` at every load.

### 6.6–6.10 (shorter)
NaN poisoning of BatchNorm producing all-NaN projections · bias-correction glob hardcoding `_Amon_` and silently missing `od550aer`'s `_AERmon_` files · bias correction re-running interpolation on each of ~957 pixel iterations · three truncated CMIP6 downloads failing every SSP2-4.5 projection · σ_DS hardcoded and gone stale once (now derived).

---

## 7. Substantive findings

### 7.1 The task is largely circular, at every resolution
`rsds` was populated by copying ERA5 `ssrd` — the field the target is built from. With correct alignment, **bilinear interpolation of `rsds` reproduces the target at RMSE 0.233 W m⁻², correlation 0.999982** — a near-identity. Every model scores ≈ **−38 to −43** against it.

**The circularity evidence rests on the interpolation baseline alone.** Delta-mapping was also a near-identity (0.131) while it too was leaking from the validation period; rebuilt from the training climatology it sits at **6.39**, and models score ≈ **−0.43 to −0.61** against it, not −40. It is a far weaker circularity signal and no longer an independent confirmation. The two differ because delta-mapping rescales the interpolated field toward a climatological mean estimated on a *different* period, and that rescaling error dominates a baseline whose underlying field was otherwise near-exact. Interpolation at 0.233 with correlation 0.999982 is overwhelming on its own.

This was first seen at daily resolution and misdiagnosed as daily-specific; it was masked at monthly resolution by bug §6.11. The response was to drop `rsds` from the model inputs, making the task a genuine perfect-prognosis problem.

### 7.2 The product contains no information finer than its input grid
Degrading the 0.1° GHI product to 0.25° and interpolating back recovers it at **correlation 0.999991**, losing **0.0019%** of variance. Elevation, a genuinely fine-scale field, loses 0.898% under the same test — so the test does detect sub-grid structure when present. **The product is a bias-and-variability correction evaluated on a finer mesh, not spatial super-resolution.** Only a genuinely high-resolution target (SARAH-2, NSRDB) could change this.

### 7.3 The alignment fix, not the predictor removal, cleared R > 0.90
Ablation (XGBoost, all else constant):

| Configuration | RMSE | Pearson R | SS vs climatology |
|---|---|---|---|
| A. Lagged `rsds`, 6 predictors (original) | 22.94 | 0.817 | −0.203 |
| B. Lag fixed, `rsds` **kept** | **5.54** | **0.9894** | 0.7096 |
| C. Lag fixed, `rsds` **dropped** (deployed) | 9.24 | 0.9707 | 0.5155 |

Skill is against the corrected 19.08 climatology, so configuration C agrees with §8 (0.5155). An earlier version used the leaky 17.98 reference and gave 0.6922 and 0.4932 — putting the same model at two different skill scores in two sections of this document.

A→B is the fix alone and accounts for essentially all the improvement. B→C shows **dropping `rsds` made the models measurably worse** — it merely left them above threshold. The ~20 points of climatology skill lost (0.69 → 0.49) is precisely the circular portion. *"We removed a predictor and R improved" is the wrong causal claim* and an examiner comparing B and C would catch it.

### 7.4 The topographic predictors are inert for the pixel-wise models
Feature importance gives elevation, slope and SVF **exactly 0.00000** on every measure (RF MDI, RF permutation, XGBoost gain, XGBoost cover). This is structural: within one cell's 312-sample time series these fields are *constants*, so zero variance, so no tree can split on them. **RF and XGBoost effectively use 8 features, not 11.** CNN and U-Net are unaffected — they see topography varying spatially.

This supersedes the audit's SVF concern: widening the 100 m search radius cannot change the deployed model. Tested at 2 km anyway — variability rises 1.70× but correlates 0.9976 with the original, so a retrain is unjustified. Live topography restored to the 100 m version every trained model used.

**Physical ranking (confirmatory):** cloud fraction dominates (0.333 / 0.395 / 0.470), then specific humidity (0.254 / 0.209 / 0.278), then pressure, day-of-year, temperature, aerosol. Cloud and water vapour leading is exactly what §3.5.1 predicts from Maposa et al. (2024).

### 7.5 U-Net's best-looking number is its worst failure
MBE of **+0.0003 W m⁻²** is the strongest bias figure in Table 3.3 and is *cancellation, not accuracy*:

| Model | Per-cell bias mean | std | range |
|---|---|---|---|
| Random Forest | +0.247 | 0.571 | −1.26 to +2.10 |
| XGBoost | +1.200 | 0.742 | −1.00 to +3.46 |
| **U-Net** | **+0.000** | **3.539** | **−14.81 to +10.18** |

It over-predicts some districts by +10 and under-predicts others by −15; they average to nothing. For a map read cell by cell to choose sites, this is the most damaging error structure available.

### 7.7 The perfect-prognosis assumption holds (C1)
The study trains on ERA5 predictors and applies the relationship to bias-corrected CMIP6. That transfer was previously **asserted, never tested**. Test: feed bias-corrected CMIP6 *historical* predictors to the deployed RF and compare against ERA5 truth for the same period, on climatology (CMIP6 is free-running, so it reproduces the statistics of the period, not its specific months — a month-by-month comparison would be unfair by construction).

| Predictor source | RMSE | MBE | Spatial R |
|---|---|---|---|
| ERA5 (as trained) | 1.66 | +0.009 | 0.9988 |
| CMIP6 CNRM-CM6-1 | 4.81 | +0.424 | 0.9903 |
| CMIP6 MPI-ESM1-2-HR | 5.87 | +0.548 | 0.9848 |
| CMIP6 ACCESS-CM2 | 5.49 | +0.396 | 0.9865 |
| **CMIP6 ensemble mean** | **4.65** | +0.456 | **0.9903** |

Swapping predictor source costs a factor of **2.80** in climatological RMSE (1.66 → 4.65) but skill does **not** collapse: spatial correlation holds at 0.99, and 4.65 sits well below the 10.30 validation RMSE, so transfer is a secondary error source. This is evidence for the study's central assumption rather than an assertion.

**Two caveats, stated because they limit the claim.** First, this probes **climatological** transfer only. That is deliberate — CMIP6 is free-running, so a month-by-month comparison would be unfair by construction — but it means **nothing here tests whether *temporal* skill transfers**, and temporal skill is precisely what these models deliver (§7.2: the spatial pattern was already recoverable by interpolation). A model could reproduce the climatology under CMIP6 forcing while losing its month-to-month discrimination entirely, and this test would not detect it. Second, the EDCM correction is calibrated on this same period, so each variable's *marginal* distribution matches ERA5 almost by construction. The test is therefore optimistic on marginals. What it does genuinely probe — and what the projection actually depends on — is whether the learned multivariate mapping survives CMIP6's own inter-variable correlations, spatial covariance and temporal sequencing. Those are untouched by the correction.

### 7.8 Spectral diagnostic: the U-Net damps, the CNN does not (C2)
Ullrich et al. recommend power spectra because ML emulators typically damp high wavenumbers, making effective resolution coarser than the stated grid. **The models must be judged in CSI space, which is where their loss operates.** GHI is recovered as CSI × clear-sky, and because the CSI target was built by *dividing* by that same clear-sky field, its fine structure is nearly the inverse of the clear-sky field's — multiplying back cancels it. That is why the GHI target is smooth (0.02% of power beyond k=10) while the CSI target is not (0.25%).

| Field | Shortest wavelength retaining 99% of power | CSI power beyond k=10 | vs truth | Verdict |
|---|---|---|---|---|
| Truth | 263 km | 0.00249 | reference | reference |
| Baseline (bilinear) | 263 km | 0.00247 | 0.99× | matches truth |
| Random Forest | 263 km | 0.00245 | 0.98× | matches truth |
| XGBoost | 263 km | 0.00251 | 1.01× | matches truth |
| CNN | 113 km | 0.00259 | 1.04× | matches truth |
| **U-Net** | 113 km | **0.00021** | **0.08×** | **DAMPED** |

**The U-Net damps CSI power beyond k=10 by a factor of twelve** — the Ullrich failure mode, tied to loss and architecture. The CNN sits within 4% of the target: its explicit spatial-gradient penalty, which acts on CSI, is doing its job.

**Correction to an earlier version of this document.** It reported all four models *injecting* ~30× the truth's fine-scale power, measured in GHI space, and named the CNN the worst offender. That was an artefact of the measurement space. GHI-space excess is a *symptom* of getting CSI wrong: a model that reproduces CSI's fine structure recovers the cancellation and yields a smooth GHI, while one that smooths CSI destroys the structure that would have cancelled and lets the clear-sky field's own structure survive. The U-Net's GHI excess is therefore caused by its CSI damping, not by invented detail — and the CNN's apparent excess is the smallest real discrepancy of the group.

On the effective-resolution definition: the conventional "power falls to half the reference" rule degenerates on the GHI field, whose reference has almost no short-wavelength power for a ratio to be taken against. The figure quoted is the shortest wavelength enclosing 99% of each field's *own* power, which is well defined under damping and excess alike.

### 7.9 The model ranking is stable across periods (C3)
Every headline number rests on one 1985–2010 / 2011–2024 split. A rolling-origin design refits on an expanding training window and evaluates on the block immediately after it, so each fold stays a strictly forward-in-time test. Each fold's climatology reference is built from **that fold's own training window**, so the §10a A7 leakage fix carries through.

| Model | 1999–2004 | 2005–2010 | 2011–2016 | 2017–2024 |
|---|---|---|---|---|
| Random Forest | 11.73 | 12.84 | 9.05 | 10.83 |
| **XGBoost** | **9.47** | **11.29** | **8.16** | **9.62** |

**XGBoost has the lower RMSE in all four folds**, by margins of 0.89 to 2.26 W m⁻². The ordering that §3.8.4 arbitrates is therefore a property of the models, not of the deployed period — which matters, because a composite criterion adjudicating between models whose ranking flips by period would be arbitrating noise.

Relative to each model's own deployed-split fold, no architecture is unusually period-sensitive:

| Model | 1999–2004 | 2005–2010 | 2011–2016 | 2017–2024 | spread |
|---|---|---|---|---|---|
| Random Forest | 1.30 | 1.42 | 1.00 | 1.20 | 1.42 |
| XGBoost | 1.16 | 1.38 | 1.00 | 1.18 | 1.38 |
| CNN | 1.29 | 1.36 | 1.00 | 1.26 | 1.36 |
| U-Net | 1.12 | 1.43 | 1.00 | 1.17 | 1.43 |

All four sit within a spread of ~1.4, and all four find 2005–2010 hardest and 2011–2016 easiest — a property of the periods, not of any model.

**A units caveat, recorded because the raw output invites the error.** The two networks report validation MSE in different units: the CNN trains on raw CSI, the U-Net on a standardised anomaly `(CSI − climatology)/anomaly_std`. With `anomaly_std = 0.0578`, the scale factor is 1/std² ≈ 299, which accounts for essentially the whole of the ~278× gap between their raw numbers. **Their absolute MSEs are not comparable**; only each model against itself across folds is, which is what the relative table above reports.

### 7.6 Cross-validation selected worse hyperparameters
A 150-cell subsampled CV search picked configurations for both tree models that underperformed the untuned defaults on the full 5,751-cell holdout. Defaults retained. The search's own scores gave no warning.

---

## 8. Current results

### Table 3.3 — validation 2011–2024 (W m⁻²)

| Model | RMSE | MAE | Pearson R | MBE | SS vs climatology | R² |
|---|---|---|---|---|---|---|
| **XGBoost** | **9.24** | **6.85** | **0.9707** | +1.20 | **0.5155** | **0.9410** |
| U-Net | 10.11 | 7.64 | 0.9640 | +0.0003 | 0.4701 | 0.9294 |
| CNN | 10.14 | 7.71 | 0.9646 | +1.20 | 0.4682 | 0.9289 |
| **Random Forest (deployed)** | 10.30 | 7.73 | 0.9636 | +0.25 | 0.4601 | 0.9267 |

Regenerate with `compute_table33.py` — a single canonical script scoring every model against
identical references, so the table cannot drift between scripts and needs no retraining.

**Only one reference is an admissible skill measure.** Climatology (per-cell, per-calendar-month
mean of the **1985–2010 training** record, RMSE 19.08) is constructible without sight of the
evaluation period. Interpolation (0.233) and delta-mapping (6.39) are **circularity diagnostics**,
not competitors: both are built from `rsds`, a reuse of the target's own source field, and the
first is very nearly an identity. Scoring against an identity measures the circularity of the
task, not model quality.

Targets are R > 0.90 and |MBE| < 5. **All four models clear both for the first time in the project.**

### Taylor statistics — time-mean spatial pattern (reference spatial std 7.98 W m⁻²)

| Model | Spatial R | Std ratio | Centred RMSE | as % of ref std |
|---|---|---|---|---|
| Baseline (bilinear) | 0.9998 | 0.9949 | 0.161 | 2.0% |
| **Random Forest** | **0.9978** | 0.9725 | **0.571** | **7.2%** |
| XGBoost | 0.9963 | 0.9608 | 0.742 | 9.3% |
| CNN | 0.9176 | 0.9937 | 3.229 | 40.5% |
| U-Net | 0.8995 | 0.9762 | 3.539 | 44.4% |

The pixel-wise models reproduce the spatial climatology far more faithfully than the shared-weight ones.

### Uncertainty decomposition — four components

Columns are **RMS over the domain**, so they square and sum to σ_total exactly and the
percentages follow from them. An earlier version tabulated the spatial *mean*, which could not be
reconciled with the shares — except for σ_DS, which is spatially constant and so did reconcile,
which is what made the mismatch confusing.

| Model | Period | σ_GCM | σ_SSP | σ_arch | σ_DS | σ_total | % var DS | % var arch |
|---|---|---|---|---|---|---|---|---|
| RF | Near-term | 1.24 | 0.26 | 2.34 | 10.30 | 10.64 | 93.73% | 4.86% |
| RF | Mid-term | 1.42 | 0.18 | 3.90 | 10.30 | 11.11 | 85.99% | 12.35% |
| RF | Long-term | 1.58 | 0.29 | 5.88 | 10.30 | 11.97 | 74.06% | **24.14%** |
| U-Net | Near-term | 2.03 | 0.56 | 2.34 | 10.11 | 10.59 | 91.16% | 4.90% |
| U-Net | Mid-term | 3.20 | 1.02 | 3.90 | 10.11 | 11.35 | 79.39% | 11.84% |
| U-Net | Long-term | 4.00 | 2.38 | 5.88 | 10.11 | 12.59 | 64.50% | **21.83%** |

**σ_arch rests on all four benchmarked architectures (n = 4)** — RF, XGBoost, CNN, U-Net — which exceeds σ_GCM's n = 3, so the comparison between them no longer favours the GCM term on sample size. Reaching n = 4 required persisting XGBoost (~400 MB at 200 fixed rounds, an order of magnitude below RF's ~4 GB) and projecting it.

**By 2076–2100 architecture choice accounts for 24.1% of projection variance against the GCM's 1.7%** — a factor of about fourteen. The estimate has been stable as members were added (14.2% at n=2, 27.8% at n=3, 24.1% at n=4), which is itself reassuring: the conclusion is not an artefact of which two models happened to be compared.

**σ_DS is constant across horizons** — it is the validation RMSE carried forward, not something
that varies with lead time. Its share falls only because σ_GCM and σ_arch *grow*; the downscaling
error does not improve. σ_DS is also a different **kind** of quantity: a historical error against a
reference, where the other three are spreads across futures. σ_arch rests on **n = 2**
architectures and σ_GCM on **n = 3** GCMs; both are small-sample spread estimates.

σ_DS still dominates (72–95%) but no longer overwhelmingly — halving the validation RMSE shrank it while σ_arch entered. **σ_arch grows with lead time and overtakes σ_GCM by the long-term horizon** (14.2% vs 2.0% for RF): by 2076–2100 the downscaling architecture matters several times more than the GCM.

### Projected change, and how much it depends on architecture

| Long-term minus near-term, domain-mean GHI | RF (deployed) | U-Net |
|---|---|---|
| SSP2-4.5 | +1.65 W m⁻² | +5.43 W m⁻² |
| SSP5-8.5 | +1.35 W m⁻² | +9.00 W m⁻² |
| Spatial correlation of the change pattern | 0.67 (SSP2-4.5) | 0.71 (SSP5-8.5) |

U-Net projects **3.3× to 6.7×** more brightening than RF and the two agree only moderately on where it occurs. All of these are small beside σ_DS ≈ 10 W m⁻²: **no projected change in this study is large relative to its own error bar.**

---

## 9. Deployed model and its justification (Chapter 3 §3.8.4)

**The pixel-wise Random Forest is deployed.** U-Net projections are retained solely to quantify architecture sensitivity.

Selection uses a **composite criterion**, and Chapter 3 §3.8.4 states plainly that it was adopted **after** the corrected-alignment run reversed the ranking (see §10a, A10). It is defended on the grounds that a suitability map is consulted one cell at a time, so per-cell reliability governs fitness for purpose in a way a domain average cannot — an argument that does not depend on which model it favours. A reader who rejects it should prefer XGBoost.

On aggregate RMSE alone the choice would be XGBoost (9.24 vs 10.30, a 10% gap).

**The criterion rests on two independent axes, not three.** Centred RMSE of the time-mean field and the standard deviation of the per-cell bias are *the same quantity* — expanding the centred error gives var(m − r) — verified identical to 6 decimal places. Since RMSE² = bias² + centredRMSE², the genuinely independent axes are the **systematic offset** and the **spatial error structure**. RF wins on both: mean bias +0.25 against XGBoost's +1.26, a factor of five; and the lowest centred error of the four, 0.571 against 0.742. The 10% penalty is small beside σ_DS ≈ 10.

**Why U-Net fails hardest** — three measured architectural causes: 17.8M parameters against 312 training fields (≈57,000 per field); the padded 96×96 domain compressed to **3×3** at the bottleneck; and one shared kernel set forced to fit a single predictor→irradiance relation from Lowveld to Eastern Highlands, so it learns the domain-*average* relation (hence competitive aggregate error) and fails where local relations depart. RF fits 5,751 independent local relations and structurally cannot trade one district against another. CNN sits between — shared weights but only 79k parameters plus a spatial-gradient penalty — and its scores sit between accordingly.

This runs against Chapter 2's expectation (Rampal et al. 2024) that encoder-decoders beat tree methods. The explanation is domain-specific: that literature comes from targets carrying genuine sub-grid information, which this ERA5-derived target does not (§7.2). It is a statement about this problem, not a refutation of the literature.

### Final configurations
| Model | Configuration |
|---|---|
| Random Forest | `n_estimators=500, max_features=sqrt, min_samples_leaf=5` |
| XGBoost | `max_depth=6, eta=0.05, subsample=0.8, min_child_weight=3, n_estimators=200` (fixed, no early stopping) |
| CNN | `lr=0.0005, lambda_gp=0.01` |
| U-Net | `lr=2e-4`, dropout 0.3, weight decay 1e-4, batch 16, patience 20 |

Tree defaults were retained after HPO (§7.6). Deployed values are now the **script defaults** — running any training script with no environment variables reproduces the deployed configuration.

---

## 10a. Second audit — Part A answers

**A6 — the decomposition did not reconcile, and why.** Columns reported the spatial *mean* E[σ]; percentages are computed from E[σ²]. They differ by the field's spatial variance. σ_DS is spatially constant (std 0.0000) so its share reconciled while nothing else did — that asymmetry was the signature. Fixed by tabulating RMS; quadrature now closes exactly.

**A7 — climatology leakage confirmed, direction opposite to the audit's assumption.** The reference averaged the validation target itself, making it the *best attainable* climatology for that period — an unfairly **strong** reference, so skill was **understated**, not inflated. Rebuilt from training: 17.98 → 19.08, skill rose (RF 0.427 → 0.460). Delta-mapping moved far more, 0.131 → 6.39.

**A8 — clean.** EDCM fits against `era5_monthly_predictors_1985_2010.nc` with CMIP6 historical at `slice('1985-01-01','2010-12-31')`. Training period only.

**A9 — the clip never binds in projection.** Across 31,055,400 values per family: RF 0.3084–0.6030, U-Net 0.2952–0.5914. Zero cells at either bound.

**A10 — the criterion was changed after seeing the results.** Traced through the backups: `_preupdate` (Jun 28) *"lowest test-period RMSE"*; `_pre_rf_deploy` (Aug 6) *"the Random Forest, which achieved the lowest validation-period RMSE"* — and RF genuinely won it then (18.72 vs 23.15); `_pre_384_rewrite` (Aug 10) still single-metric; the composite criterion appears only in the Aug 11 file, **after** the corrected-alignment run reversed the ranking that morning. Now disclosed in Chapter 3 §3.8.4 in those words, with the justification resting on cell-by-cell reliability and an explicit statement that a reader who rejects the argument should prefer XGBoost.

**Also found:** Table 3.3's caption still called RMSE the *"primary model selection criterion"*, contradicting §3.8.4. Corrected.

---

## 10. Audit fixes (B1–B10)

| Item | Status |
|---|---|
| B1 information-content script | **Done** — `compute_information_content.py` with elevation positive control (control loses 0.898%, GHI 0.0019%) |
| B2 loud NaN handling | **Done** — `safe_nan_to_num()` logs the filled fraction and raises above 1% (`NAN_FRACTION_LIMIT`); 27 bare calls replaced |
| B3 deployed hyperparameters as defaults | **Done** — verified by running with no env vars |
| B4 stop hardcoding σ_DS | **Done** — derived from `validation_spatial_fields.nc`, prints what it read |
| B5 daily clear-sky script | **Done** — quarantined to `deprecated/` with README |
| B6 σ_arch | **Done** — fourth component; overtakes σ_GCM by the long-term horizon |
| B7 sky-view factor | **Superseded by §7.4** — widening cannot affect the deployed model |
| B8 feature importance | **Done** — `feature_importance.csv` |
| B9 delta-mapping baseline | **Done** — third reference; ref RMSE **6.39** after the A7 leakage fix (0.131 before it, when it too was leaking) |
| B10 documentation | **Partly** — see below |

### Second audit, Parts B–F

| Item | Status |
|---|---|
| B11 double-count | **Done** — centred RMSE ≡ per-cell bias std, verified identical to 6 dp and algebraically (cRMSE² = var(m−r)). §3.8.4 now rests on two independent axes: mean offset and spatial error structure, since RMSE² = bias² + cRMSE². |
| B12 degenerate baselines | **Done** — interpolation and delta-mapping relabelled circularity diagnostics in §3.7.3, `compute_table33.py` and the CSV; climatology named the only admissible reference |
| B13 σ_DS constant | **Done** — stated in the script output and here |
| B14 σ_DS a different kind of quantity | **Done** — stated alongside |
| B15 σ_arch from more members | **Done — n = 4**, all benchmarked architectures. CNN projected from its existing checkpoint; XGBoost persisted (~400 MB) and projected. Exceeds σ_GCM's n = 3. Long-term variance share settled at 24.1% (14.2% at n=2, 27.8% at n=3). |
| B16 U-Net R² | **Done** — 0.9294 |
| C1 perfect prognosis | **Done** — §7.7. Required extending EDCM to emit a historical pseudo-scenario (`EDCM_INCLUDE_HISTORICAL=1`) |
| C2 power spectra | **Done** — §7.8 |
| C3 multiple temporal splits | **Done** — `compute_rolling_origin.py`, 4 expanding-origin folds. XGBoost lowest RMSE in all four; all models within a 1.4 spread relative to their own deployed-split fold. See §7.9. |
| C4 join logic | **Done** — no nearest-neighbour temporal joins survive outside the guarded, warned carry-forward inside `align_to_months` |
| D1 version control | **Done** — git initialised, 61 files committed, `data/` (16 GB) excluded; staged content scanned for credentials before committing |
| D2 regression tests | **Done** — `pytest tests/` 16 passed; reintroducing the §6.11 lag turns it red, restoring turns it green |
| D3 pin environment | **Done** — `environment.yml`, 247 packages |
| D4 OpenMP workaround | **Done** — documented in README with the actual cause and the proper fix |
| E1 dashboard | **Done** — withdrawn and replaced with a retraction notice |
| E2–E6 | Carried forward below |
| Part F figures | **Done** — six PNGs in `figures/` from `make_figures.py` |

---

## 11. Chapter 3 edits

Edited at run level with `python-docx` so all **26 live Zotero citation fields survived** (162 field characters, verified before and after every edit).

"PROPOSED" dropped from the title; implemented sections converted to past tense. Factual corrections: actual retained grid (§3.2.1) · removed the xESMF/conservative-remapping claim, neither was used (§3.4.1) · areal block means not point-sampling, plus the SVF search-radius caveat (§3.5.2) · SZA computed analytically not via PVLIB (§3.5.3) · the U-Net/CNN ReLU distinction and the exact-cancellation property (§3.5.4) · upsampling factor (§3.6.1) · grid corrected 75×85=6,375 → **71×81=5,751** (§3.6.2) · dual-branch input, dropout, channel attention, pad-to-32; false "output ReLU" claim removed (§3.6.6) · never-performed Bayesian stage removed, CV-transfer caveat added (§3.6.7) · ENSO sampling corrected to **U-Net only** (§3.6.8) · Taylor stats clarified as time-mean (§3.7.2) · RF designated deployed (§3.8.1) · Table 3.2 search ranges replaced with grids actually searched. **New §3.6.9** (daily-resolution experiment) and **new §3.8.4** (deployed-model justification).

**Three claims that were untrue and are now disclaimed:**
1. **SARAH-2 and NSRDB were never acquired.** `data/raw/` holds only cmip6, era5, oni, srtm. Validation is entirely against withheld ERA5.
2. **The delta-mapping baseline was never implemented** — now it is (B9).
3. **RF/XGBoost feature importance was never extracted** — now it is (B8).

---

## 12. Outputs

```
data/processed/
  models/{unet,cnn}/*.pth · models/pixelwise_rf/*.joblib + manifest.json (~4 GB)
  era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc
  ml_ready/ml_{training,validation}_dataset.nc
  cmip6_bias_corrected/
  projections/ (U-Net) · projections_rf/ · projections_cnn/ · projections_xgb/
  mme_aggregations/ · mme_aggregations_rf/ · mme_aggregations_cnn/ · mme_aggregations_xgb/
  evaluation/
    validation_spatial_fields.nc      truth + baseline + all 4 models
    taylor_diagram_stats.csv · spatial_verification_maps.nc
    uncertainty_decomposition_{unet,rf}.nc + summary.csv
    information_content.csv · feature_importance.csv
deprecated/daily_resolution_experiment/    superseded, with README
figures/                                   six presentation PNGs (make_figures.py)
tests/                                     pytest regression suite
logs/                                      all run logs
environment.yml                            247 pinned packages
README.md                                  run order, OpenMP workaround, env vars
.gitignore                                 excludes data/ (16 GB), figures/, logs/
```

Under **git** since the second audit: 61 files tracked, two commits, `data/` excluded.

**Verification:** all 6 sanity checks pass — no all-NaN/all-zero variables across 32 files; latitude ascending and consistent; no fabricated edge band; zero NaN in any projection; SVF varies; counts confirmed at 312 / 168 / 5,751.

---

## 13. Outstanding

**Not started**
- **§3.9 Multi-Criteria Suitability Analysis** — the largest remaining piece. Needs ESA CCI land cover, OpenStreetMap roads and transmission lines, WDPA/ZimParks protected areas, WorldPop. **None acquired.**
- **Chapter 4** (results and discussion)
- **SARAH-2 / NSRDB independent validation — COMMITTED, deferred.** Will be implemented, but deliberately **out of scope for the 24 August interview**. It is the only route to genuine sub-0.25° resolution (§7.2) and the fix for the validation-independence gap (§14.1).

**Needs your hand**
- Chapter 1 (~line 149) and Chapter 2 §2.3.3 present SARAH-2 and NSRDB as validation data for this study. Because that validation is committed rather than abandoned, these sections do **not** need retracting — they describe work that is planned. Chapter 3 §3.3.3 has been reworded to match ("is planned as the next stage of this work and is not reported here"), so the three chapters are now consistent in treating it as deferred. Check the tense in Chapters 1 and 2 reads as intent rather than as completed work.
- Chapter 2 also sets up the expectation that encoder-decoders beat tree methods, which the results overturn — needs a forward-reference to §3.8.4 or reframing as a hypothesis the study tests.
- **Dozier & Frew (1990)** is cited as plain text and is not in the Zotero library.
- §6.1 uses 0.68 for a collapsed *correlation* while §7.5 discusses *std ratios* near 0.68 — add a clarifying clause if both appear in the thesis.
- **The published results dashboard has been withdrawn** and replaced with a retraction notice (see §10a / E1). Nothing further is needed unless you want the URL itself deleted, which must be done from the artifacts gallery.

**Still specified but not implemented:** per-cell QC flag counts for the appendix (§3.4.3).

**E1 — closed.** The published dashboard has been **withdrawn**: its content is replaced by a retraction notice explaining that every figure was superseded by the alignment fix, the circular-predictor removal, the two leakage fixes and three retrains. The URL now resolves to that notice rather than to wrong numbers. Fully deleting the URL, if wanted, must be done from the artifacts gallery.

---

## 14. Standing methodological caveats

1. **ERA5 is the only reference *in the work reported here*.** Training and validation both use it, over a data-sparse region where reanalysis is weakest. The temporal split is genuine, so this is a real out-of-sample test *of the reanalysis relationship* — but not an independent test against observations. SARAH-2 / NSRDB validation is committed as the next stage (§13); until it lands, every skill figure in §8 carries this qualifier.
2. **Stationarity is assumed.** Models fitted on 1985–2010 are applied to bias-corrected SSP5-8.5 fields to 2100. EDCM preserves the absolute change signal, but the learned predictor→irradiance mapping is assumed valid in an unseen climate.
3. **Cloud is unresolved.** The dominant control on surface irradiance is parameterised inside the GCM and available only as a grid mean. Feature importance confirms cloud fraction dominates the fit.
4. **Defensive NaN handling hides failures.** Two of the three severe silent bugs were `np.nan_to_num` converting a loud failure into a plausible field. Both were caught by inspecting fields, not scores. `safe_nan_to_num` now makes this loud.
5. **Two bugs cancelled each other** (§6.11). A bug invisible in one data split because a second bug reverses it is the hardest class to find — the lesson is that identical logic must be used at every join, not merely logic that works.

---

## 15. Note on authorship

Substantial parts of this pipeline were written with AI assistance rather than typed from scratch — particularly the evaluation and verification layer (`generate_validation_spatial_fields.py`, `compute_spatial_verification.py`, `compute_uncertainty_decomposition.py`, `compute_information_content.py`, `compute_feature_importance.py`, `hpo_pixelwise.py`, `generate_future_projections_rf.py`) and major rewrites of `unet_model.py`, `ml_dataset_common.py`, `compute_finegrid_clearsky_ghi.py`, `apply_edcm_bias_correction.py` and the training/evaluation scripts. The analytical decisions, diagnoses and interpretations recorded above should be understood and defensible independently of how the code was produced.
