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
| `apply_qc_bounds.py` | **§3.4.3 QC.** Enforces physical bounds on the bias-corrected fields (EDCM overshoots them at the tails) and writes the per-cell flag counts. Run with no arguments to report, `--apply` to enforce. Originals preserved in `cmip6_bias_corrected_preqc/`. |
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
| `compute_bootstrap_ci.py` | Paired year-block bootstrap: CIs on Table 3.3 and on model differences |
| `check_status_consistency.py` | Guards this document against stale numbers; run by `pytest` |

### Projection and verification
| Script | Role |
|---|---|
| `generate_future_projections{,_rf,_cnn,_xgb}.py` | U-Net / RF / CNN / XGBoost → downscaled GHI, 6 GCM-scenario combos each |
| `compute_mme_aggregations.py` | Ensemble mean + inter-model spread, 3 horizons. `PROJECTIONS_DIR` / `MME_OUTPUT_DIR` |
| `generate_validation_spatial_fields.py` | Full GHI fields: truth, baseline, all 4 models |
| `compute_spatial_verification.py` | Taylor statistics + per-pixel bias/RMSE maps |
| `compute_uncertainty_decomposition.py` | **Four-component** σ_GCM / σ_SSP / σ_arch / σ_DS. σ_DS **derived** from the validation fields; σ_arch from `ARCH_DIRS` (n = 4), kept separate from the deployed-model list. |

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
The study trains on ERA5 predictors and applies the relationship to bias-corrected CMIP6. That transfer was previously **asserted, never tested**. Test: feed bias-corrected CMIP6 *historical* predictors to the model and compare against ERA5 truth for the same period, on climatology (CMIP6 is free-running, so it reproduces the statistics of the period, not its specific months — a month-by-month comparison would be unfair by construction). Run for **both** pixel-wise models: `PP_MODEL=xgb` (deployed) and `PP_MODEL=rf`.

| Predictor source | XGBoost RMSE | XGBoost Spatial R | RF RMSE | RF Spatial R |
|---|---|---|---|---|
| ERA5 (as trained) | 0.59 | 0.9998 | 1.66 | 0.9988 |
| CMIP6 CNRM-CM6-1 | 5.40 | 0.9880 | 4.81 | 0.9903 |
| CMIP6 MPI-ESM1-2-HR | 6.71 | 0.9799 | 5.87 | 0.9848 |
| CMIP6 ACCESS-CM2 | 6.35 | 0.9818 | 5.49 | 0.9865 |
| **CMIP6 ensemble mean** | **4.82** | **0.9895** | **4.65** | **0.9903** |

**The assumption holds, and the deployed model is marginally the weaker of the two on this axis — stated because it cuts against the deployment.** Transfer costs XGBoost a factor of **8.15** (0.59 → 4.82) against RF's **2.80** (1.66 → 4.65). The ratio is the wrong statistic to read, though: it is large for XGBoost mainly because its ERA5-driven climatological error is nearly three times smaller to begin with (0.59 against 1.66), so the same absolute degradation divides into a bigger number. On the quantity that matters — the **absolute** error once driven by CMIP6 — the two are close, 4.82 against 4.65, and both sit well below their own validation RMSE (9.24 and 10.30), so transfer is a secondary error source for either. Spatial correlation holds at 0.99 for both.

**The honest reading of the difference:** XGBoost fits the ERA5 predictor distribution more tightly, and a tighter fit to one predictor source is exactly what should transfer slightly less well. The gap is 0.17 W m⁻², about 2% of σ_DS, and does not approach reversing §7.10 or §7.11 — but it is a real cost of the deployment and belongs in the same paragraph as the bias cost in §9.

**Two caveats, stated because they limit the claim.** First, this probes **climatological** transfer only. That is deliberate — CMIP6 is free-running, so a month-by-month comparison would be unfair by construction — but it means **nothing here tests whether *temporal* skill transfers**, and temporal skill is precisely what these models deliver (§7.2: the spatial pattern was already recoverable by interpolation). A model could reproduce the climatology under CMIP6 forcing while losing its month-to-month discrimination entirely, and this test would not detect it. Second, the EDCM correction is calibrated on this same period, so each variable's *marginal* distribution matches ERA5 almost by construction. The test is therefore optimistic on marginals. What it does genuinely probe — and what the projection actually depends on — is whether the learned multivariate mapping survives CMIP6's own inter-variable correlations, spatial covariance and temporal sequencing. Those are untouched by the correction.

### 7.8 Spectral diagnostic: the U-Net damps, the CNN does not (C2)
Ullrich et al. recommend power spectra because ML emulators typically damp high wavenumbers, making effective resolution coarser than the stated grid. **The models must be judged in CSI space, which is where their loss operates.** GHI is recovered as CSI × clear-sky, and because the CSI target was built by *dividing* by that same clear-sky field, its fine structure is nearly the inverse of the clear-sky field's — multiplying back cancels it. That is why the GHI target is smooth (0.02% of power beyond k=10) while the CSI target is not (0.25%).

| Field | CSI power beyond k=10 | vs truth | Verdict |
|---|---|---|---|
| Truth | 0.00249 | reference | reference |
| Baseline (bilinear) | 0.00247 | 0.99× | matches truth |
| Random Forest | 0.00245 | 0.98× | matches truth |
| XGBoost | 0.00251 | 1.01× | matches truth |
| CNN | 0.00259 | 1.04× | matches truth |
| **U-Net** | **0.00021** | **0.08×** | **DAMPED** |

**The U-Net damps CSI power beyond k=10 by a factor of twelve** — the Ullrich failure mode, tied to loss and architecture. The CNN sits within 4% of the target: its explicit spatial-gradient penalty, which acts on CSI, is doing its job.

**Correction to an earlier version of this document.** It reported all four models *injecting* ~30× the truth's fine-scale power, measured in GHI space, and named the CNN the worst offender. That was an artefact of the measurement space. GHI-space excess is a *symptom* of getting CSI wrong: a model that reproduces CSI's fine structure recovers the cancellation and yields a smooth GHI, while one that smooths CSI destroys the structure that would have cancelled and lets the clear-sky field's own structure survive. The U-Net's GHI excess is therefore caused by its CSI damping, not by invented detail — and the CNN's apparent excess is the smallest real discrepancy of the group.

**No effective-resolution figure is quoted, deliberately.** The conventional definition — the wavelength at which power falls to half the reference — is one-sided and degenerates on this GHI field, whose reference has almost no short-wavelength power for a ratio to be taken against. A cumulative-power substitute was tried and discarded: on a 71 × 81 grid the available wavelengths are the discrete set 788/k km (788, 394, 263, 197, 158, 131, 113…), so the statistic is quantised into a handful of bins. It took only two distinct values across all six fields and put the CNN and U-Net in the *same* bin while their power ratios were 1.04× and 0.08× — opposite verdicts, identical number. Read cold it also appeared to say the damped model resolved finer scales than the truth. The power ratio carries the result on its own.

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

### 7.10 Confidence intervals: which differences survive resampling
Every metric had been a point estimate, including the gaps §3.8.4 turns on. A **paired year-block bootstrap** (2,000 replicates, resampling whole calendar years so spatial and seasonal correlation are preserved within each unit, every model scored on the same resampled years) gives:

| Model | RMSE | 95% CI |
|---|---|---|
| **XGBoost** | 9.24 | [8.55, 9.87] |
| U-Net | 10.11 | [9.63, 10.65] |
| CNN | 10.14 | [9.54, 10.74] |
| Random Forest | 10.30 | [9.37, 11.25] |

Individual intervals overlap heavily — but that is the wrong comparison. Because the bootstrap is paired, the interval on each *difference* removes the year-to-year variation common to all models:

| Comparison | ΔRMSE | 95% CI | Verdict |
|---|---|---|---|
| Random Forest − XGBoost | +1.058 | [+0.530, +1.671] | **distinguishable** |
| CNN − XGBoost | +0.900 | [+0.561, +1.228] | **distinguishable** |
| U-Net − XGBoost | +0.866 | [+0.412, +1.281] | **distinguishable** |
| CNN − Random Forest | −0.158 | [−0.907, +0.508] | not distinguishable |
| Random Forest − U-Net | +0.193 | [−0.519, +0.970] | not distinguishable |
| CNN − U-Net | +0.035 | [−0.287, +0.388] | not distinguishable |

**XGBoost's aggregate advantage is real**: its RMSE deficit against all three others excludes zero. The comparison §3.8.4 turned on — Random Forest against XGBoost, +1.058 W m⁻² — survives at [+0.530, +1.671].

**Two consequences worth stating.** First, the deployment argument could no longer be carried on aggregate grounds: RF would have to be preferred *despite* a statistically distinguishable deficit, so the spatial-fidelity case had to carry that weight explicitly — and the next subsection shows it cannot. Second, **RF, CNN and U-Net are not distinguishable from one another on RMSE** — the ordering among those three is noise at this sample size, and any narrative ranking them should say so.

**Reconciling that with the sections that do rank them.** §7.5, §7.8 and §9 all order these three, and Table 3.3 lists them in an RMSE order (U-Net 10.11, CNN 10.14, RF 10.30) that the bootstrap says is not real. Both are correct because **they are rankings on different axes, and only one of the two axes supports a ranking**:

| Axis | RF vs CNN | CNN vs U-Net | RF vs U-Net | Ordering established? |
|---|---|---|---|---|
| Aggregate RMSE | −0.158 [−0.907, +0.508] | +0.035 [−0.287, +0.388] | +0.193 [−0.519, +0.970] | **no — none of the three** |
| Centred RMSE | −2.525 [−2.758, −2.217] | −0.299 [−0.517, −0.070] | −2.824 [−3.024, −2.486] | **yes — all three pairs** |
| Spatial R | +0.080 [+0.066, +0.095] | +0.018 [+0.006, +0.031] | +0.098 [+0.082, +0.115] | **yes — all three pairs** |

**RF > CNN > U-Net is fully established on both spatial axes, including the narrow CNN-over-U-Net margin, and is established on none of the aggregate ones.** So the rule for the whole document is: *rank these three on spatial fidelity, never on RMSE.* Every ranking §7.5, §7.8 and §9 make is a spatial-fidelity ranking — the U-Net's ±15 W m⁻² per-cell bias range, its 0.08× spectral damping, the bottleneck argument — and each survives. Table 3.3's row order is an artefact of sorting by a column that does not separate them; read the Taylor table for the ordering that holds.

Mean bias is the weakest column: **every model's MBE interval spans zero**, including XGBoost's +1.20 [−0.14, +2.41]. The fivefold RF-vs-XGBoost bias ratio discussed in §9 is a point-estimate ratio between two quantities that are individually indistinguishable from zero, and is presented there as indicative rather than established.

### The spatial axis, tested — and the consequence for §3.8.4

The same paired bootstrap applied to the two spatial quantities §3.8.4 actually relies on:

| Comparison | Δ centred RMSE | 95% CI | Δ spatial R | 95% CI |
|---|---|---|---|---|
| **RF − XGBoost** | **−0.1445** | **[−0.3050, +0.0101]** | **+0.0016** | **[−0.0006, +0.0042]** |
| RF − U-Net | −2.8237 | [−3.0244, −2.4857] | +0.0980 | [+0.0822, +0.1150] |
| RF − CNN | −2.5250 | [−2.7584, −2.2169] | +0.0802 | [+0.0660, +0.0947] |
| XGBoost − U-Net | −2.6791 | [−2.8978, −2.3699] | +0.0964 | [+0.0807, +0.1139] |

**The pixel-wise/shared-weight distinction is established and large.** RF and XGBoost both beat both networks on spatial error structure by margins whose intervals are nowhere near zero. Everything §7.5 and §7.8 say about the U-Net stands.

**The RF-vs-XGBoost spatial distinction is not established.** Both intervals span zero. The two pixel-wise models are indistinguishable from each other on centred RMSE and on spatial correlation.

**This was decisive for the deployment argument, and §3.8.4 was rewritten accordingly (§9).** The composite criterion rested on two axes:

| Axis | RF vs XGBoost | Status |
|---|---|---|
| Systematic offset (mean bias) | ratio 4.8× | **not established** — both MBEs span zero (§7.10 above) |
| Spatial error structure (centred RMSE) | −0.1445 | **not established** — CI spans zero |
| *Aggregate error (RMSE)* | *+1.0581* | ***established*** — *against RF* |

Random Forest is **significantly worse on the one axis that is established, and not significantly better on either axis the deployment case invoked.** The composite criterion does not select RF over XGBoost; it fails to separate them, and the tie-break falls to the metric that does separate them, which favours XGBoost. §7.11 then removed any residual case for RF on independent grounds. **XGBoost is deployed (§9).**

### 7.11 Scenario discrimination: the deployment test that validation cannot perform
Validation measures how well a model reproduces 2011–2024. The product is a projection to 2100 under two emission scenarios, and nothing in Table 3.3 tests whether a model can tell those scenarios apart. It can be tested directly: the SSP5-8.5 minus SSP2-4.5 difference should be positive and should **grow** with lead time.

| Model | Near-term | Mid-term | Long-term | Grows? | Cells with SSP5-8.5 > SSP2-4.5 |
|---|---|---|---|---|---|
| **Random Forest** | +0.465 | +0.307 | **+0.167** | **NO — shrinks** | **71.7%** |
| XGBoost | +0.754 | +0.741 | +1.444 | yes, overall | 96.7% |
| CNN | +1.397 | +4.059 | +9.654 | yes | 100.0% |
| U-Net | +1.076 | +2.001 | +4.639 | yes | 100.0% |

**Random Forest's scenario separation shrinks as forcing grows** — the opposite of the physical expectation — and it inverts outright on the long-term change signal (+1.648 under SSP2-4.5 against +1.349 under SSP5-8.5).

**The mechanism is tree extrapolation.** A tree predicts a constant beyond the range it was trained on, so its response saturates once predictors leave the training envelope. Under SSP5-8.5 they increasingly do: temperature falls outside the 1985–2010 range **4.95%** of the time against 0.79% under SSP2-4.5, specific humidity 3.56% against 0.92%. The scenario that should produce the larger response is precisely the one where a tree's response is most clipped. XGBoost shares the limitation and shows it mildly; the neural models extrapolate through their linear layers and do not.

This is the concrete form of the stationarity caveat in §14.2, and it matters more for a projection product than any validation metric: a suitability map that cannot distinguish emission pathways fails at the task it exists for.

**A caution against over-reading it in the other direction.** All four scenario separations are small beside σ_DS ≈ 10 W m⁻². The CNN's +9.654 is not obviously *better* for being larger — it is comparable to the model's own error, and its long-term change of +16.6 W m⁻² is implausibly large for a 75-year irradiance trend. The honest reading is that RF is disqualified on this axis, XGBoost is adequate, and the neural models' larger responses are unverifiable rather than demonstrably right.

### 7.12 §3.4.3's quality control did not exist, and the null result is the interesting part

**What was claimed and what was there.** §3.4.3 stated that physically unrealistic CMIP6 values — negative rsds, cloud fraction above 100%, negative humidity — "were replaced with the model climatology using a 31-day centred moving window". No such step existed anywhere in the pipeline. The section also said per-cell flag counts "were not separately tabulated".

**The violations are real.** Equidistant CDF matching is tail-sensitive: matching a *bounded* variable against an empirical distribution drives a fraction of corrected values past the bound.

| Variable | Flagged | Of total | Worst excursion | Cells affected |
|---|---|---|---|---|
| `clt` | 38,843 | 0.6406% | **−5.93%** cloud fraction | **751 / 957** |
| `od550aer` | 107 | 0.0018% | −0.0046 optical depth | 69 / 957 |

A negative cloud fraction is not a small error in kind, and it was reaching models trained on a strictly non-negative predictor. 751 of 957 coarse cells are affected at least once, up to 277 time steps at a single cell — spread across the domain, not confined to a corner.

**The ERA5 half, by contrast, is exactly clean.** The clear-sky index is clipped to [0, 1.1], and the clip never binds: across 1,794,312 training and 966,168 validation values the maximum is 0.6323. **The flag count is identically zero at every cell** — so §3.4.3's appendix table is now a determinate result rather than an untabulated unknown.

**The fix.** `apply_qc_bounds.py` enforces the bounds by **truncation**, not by the climatological replacement §3.4.3 specified. The violations are small overshoots at the distribution tails; truncation preserves the corrected value everywhere else, whereas climatological replacement would discard a correctly corrected value to repair a boundary artefact. §3.4.3 has been rewritten to describe what the code does. Originals are preserved in `cmip6_bias_corrected_preqc/`, and a regression test asserts no predictor is physically impossible — verified to fail against those originals with 11 offenders.

**Impact on every published number: essentially none — and *why* is a genuine corroboration of §7.11.** All four models' projections, all four MME aggregations and the decomposition were regenerated from the corrected inputs.

| Quantity | Pre-QC | Post-QC |
|---|---|---|
| RF long-term change (SSP2-4.5 / SSP5-8.5) | +1.648 / +1.349 | **+1.648 / +1.349** (identical) |
| XGBoost long-term change | +4.081 / +4.771 | **+4.081 / +4.771** (identical) |
| CNN long-term change | +8.309 / +16.566 | +8.307 / +16.558 |
| U-Net long-term change | +5.432 / +8.995 | +5.432 / +8.988 |
| Scenario ordering, cells correct | RF 71.7%, XGB 96.7% | **unchanged** |
| σ_arch share, long term (deployed) | 27.20% | 27.18% |
| C1 transfer, XGBoost ensemble | 4.8226 | **4.8226** (identical to 4 dp) |

**The two tree models are bit-identical; only the two networks moved.** That is exactly what §7.11 predicts and is measured here independently. The out-of-bound cloud values were *already* outside the 1985–2010 training range, and a tree returns the same leaf for every input beyond that range — so moving the input from −5.93 to 0 moves the prediction not at all. The networks extrapolate through their linear layers, so the same input change propagates, in the third decimal. **The QC's null effect on the deployed model is the saturation mechanism showing up in a completely different experiment.**

**This does not make the fix optional.** A defect that happens not to move the current answer is still a defect: it would matter for any model that extrapolates, for any rerun with different data, and for a reader entitled to expect that a documented QC step exists. The honest statement is that §3.4.3 was wrong, is now correct, and that correcting it changed no conclusion in this study.

### 7.13 RQ2 answered against its own comparator: the attribution ladder
§7.2 measures sub-grid information against the product's **own coarse grid** and finds essentially none. RQ2 asks a different question — what is resolved *that is absent in raw GCM output* — and against that comparator the answer is positive. Both were measured; only one had been reported.

Climatological RMSE against ERA5 truth, 1985–2010, all on the 0.1° grid:

| Product | RMSE | Spatial R | Bias |
|---|---|---|---|
| Raw CMIP6 `rsds`, uncorrected | 11.29 | 0.7667 | +6.52 |
| Bias-corrected, interpolated to 0.1° | 6.24 | 0.9998 | — |
| **Deployed XGBoost** | **4.82** | 0.9895 | −0.13 |

Per-GCM raw RMSE spans 10.35 (CNRM-CM6-1) to 16.44 (MPI-ESM1-2-HR), with spatial correlations of 0.49, 0.68 and 0.89.

**The chain cuts climatological error by 57% and lifts spatial correlation from 0.77 to 0.99. The attribution is shared and not mostly the ML's:** bias correction does 11.29 → 6.24 and essentially all of the pattern gain; the ML adds 6.24 → 4.82, a further 23% on magnitude. **The interpolated bias-corrected GCM has a marginally *higher* spatial correlation than the deployed model** (0.9998 vs 0.9895) — the ML improves the level and very slightly softens the pattern, for the same reason the interpolation baseline reaches 0.233 (§7.1).

What bias correction cannot do is discriminate month to month: it operates on distributions. The skill score of 0.5155 against a 19.08 climatology is entirely the ML's.

### 7.14 The irradiance field is atmospheric, not topographic
RQ2's third clause asks how the patterns correspond to Zimbabwe's physiographic zones. Time-mean GHI and deployed-model error by elevation band:

| Zone | Elevation | Cells | Mean GHI | Spatial sd | XGB bias | XGB RMSE |
|---|---|---|---|---|---|---|
| Lowveld | < 600 m | 936 | 230.67 | 8.36 | +1.73 | 1.86 |
| Middleveld | 600–1200 m | 3,858 | 240.84 | 7.38 | +1.04 | 1.27 |
| Highveld | 1200–1500 m | 904 | 240.32 | 3.43 | +1.31 | 1.44 |
| Eastern Highlands | ≥ 1500 m | 53 | 239.01 | 3.32 | +1.56 | 1.58 |

The Lowveld sits ~10 W m⁻² below the rest of the country and is the most spatially variable zone; it is also where the deployed model is least accurate (RMSE 1.86 against 1.27 in the Middleveld). **Elevation explains only 22.1% of the spatial variance in time-mean GHI (r = 0.4702).**

**This unifies three findings that otherwise read as separate weaknesses.** Elevation explains 22% of the spatial variance; the topographic predictors carry *exactly* zero importance for the pixel-wise models (§7.4); and the product holds almost no sub-0.25° information (§7.2). All three say the same thing: over this domain, at this resolution, irradiance is governed by large-scale atmospheric structure rather than terrain — which is what the feature importances independently show, with cloud fraction and humidity dominating. It is a result about the physics of the domain, not a deficiency of the method.

### 7.6 Cross-validation selected worse hyperparameters
A 150-cell subsampled CV search picked configurations for both tree models that underperformed the untuned defaults on the full 5,751-cell holdout. Defaults retained. The search's own scores gave no warning.

---

## 8. Current results

### Table 3.3 — validation 2011–2024 (W m⁻²)

| Model | RMSE | MAE | Pearson R | MBE | SS vs climatology | R² |
|---|---|---|---|---|---|---|
| **XGBoost (deployed)** | **9.24** | **6.85** | **0.9707** | +1.20 | **0.5155** | **0.9410** |
| U-Net | 10.11 | 7.64 | 0.9640 | +0.0003 | 0.4701 | 0.9294 |
| CNN | 10.14 | 7.71 | 0.9646 | +1.20 | 0.4682 | 0.9289 |
| Random Forest | 10.30 | 7.73 | 0.9636 | +0.25 | 0.4601 | 0.9267 |

Regenerate with `compute_table33.py` — a single canonical script scoring every model against
identical references, so the table cannot drift between scripts and needs no retraining.

**Only one reference is an admissible skill measure.** Climatology (per-cell, per-calendar-month
mean of the **1985–2010 training** record, RMSE 19.08) is constructible without sight of the
evaluation period. Interpolation (0.233) and delta-mapping (6.39) are **circularity diagnostics**,
not competitors: both are built from `rsds`, a reuse of the target's own source field, and the
first is very nearly an identity. Scoring against an identity measures the circularity of the
task, not model quality.

Targets are R > 0.90 and |MBE| < 5. **All four models clear both for the first time in the project.**

### The raw-magnitude objection, and the honest answer

**The objection.** Domain-mean GHI over the validation period is **239.1 W m⁻²**, so an RMSE of 9.24 is 3.9% of the signal — which sounds excellent and *is the wrong denominator*. Most of that 239 is the deterministic solar-geometry component that the clear-sky normalisation removes before the model sees anything. Quoting error against it credits the model for astronomy.

**Four denominators, in increasing order of honesty:**

| Denominator | Value | XGBoost RMSE as % | What it means |
|---|---|---|---|
| Domain-time mean GHI | 239.1 | **3.9%** | flattering — mostly solar geometry |
| Seasonal range of the domain mean | 103.9 | 8.9% | still largely deterministic |
| Temporal std of GHI | 38.04 | **24.3%** | the fair aggregate figure |
| Climatology RMSE | 19.08 | 48.4% (SS = **0.5155**) | **the defensible headline** |

**Skill against climatology is the number to quote: the model halves the error of a per-cell, per-calendar-month mean built without sight of the evaluation period.** That is a real result and it is the one §8 leads with.

**The sharpest form of the objection, and why it does not land.** The spatial standard deviation of the time-mean GHI field is only **7.98 W m⁻²** — so the aggregate RMSE of 9.24 *exceeds the entire spatial variability of the map being produced*. Read cold, that says the error is bigger than the signal. It is a category error, but a fair one to raise: the 9.24 includes month-to-month temporal error, while a suitability map is built from the **time mean**, whose error is the centred RMSE of **0.742 W m⁻² — 9.3% of that 7.98 spatial std**. The two numbers answer different questions. The correct statement is that the model resolves the *spatial pattern* of the resource to about a tenth of its variation, and resolves *individual months* to about a quarter of their variation.

**What this does not rescue.** §7.2 established that the spatial pattern was already largely recoverable by interpolation, so the 0.742 is not evidence of skill at generating sub-grid structure. The genuine contribution is temporal, and the temporal figure is the 24.3% one.

### Taylor statistics — time-mean spatial pattern (reference spatial std 7.98 W m⁻²)

| Model | Spatial R | Std ratio | Centred RMSE | as % of ref std |
|---|---|---|---|---|
| Baseline (bilinear) | 0.9998 | 0.9949 | 0.161 | 2.0% |
| **Random Forest** | **0.9978** | 0.9725 | **0.571** | **7.2%** |
| XGBoost | 0.9963 | 0.9608 | 0.742 | 9.3% |
| CNN | 0.9176 | 0.9937 | 3.229 | 40.5% |
| U-Net | 0.8995 | 0.9762 | 3.539 | 44.4% |

The pixel-wise models reproduce the spatial climatology far more faithfully than the shared-weight ones. **The gap between the two pixel-wise models is not one of them**: RF−XGBoost is −0.145 [−0.305, +0.010] on centred RMSE and +0.0016 [−0.0006, +0.0042] on spatial R, both spanning zero (§7.10). Read the bold on the RF row as the best point estimate, not as an established difference from XGBoost.

### Uncertainty decomposition — four components

Columns are **RMS over the domain**, so they square and sum to σ_total exactly and the
percentages follow from them. An earlier version tabulated the spatial *mean*, which could not be
reconciled with the shares — except for σ_DS, which is spatially constant and so did reconcile,
which is what made the mismatch confusing.

| Model | Period | σ_GCM | σ_SSP | σ_arch | σ_DS | σ_total | % var DS | % var arch |
|---|---|---|---|---|---|---|---|---|
| **XGB (deployed)** | Near-term | 1.92 | 0.43 | 2.34 | 9.24 | 9.74 | 90.10% | 5.79% |
| **XGB (deployed)** | Mid-term | 2.20 | 0.48 | 3.90 | 9.24 | 10.28 | 80.81% | 14.41% |
| **XGB (deployed)** | Long-term | 2.54 | 0.83 | 5.88 | 9.24 | 11.27 | 67.21% | **27.18%** |
| U-Net | Near-term | 2.03 | 0.56 | 2.34 | 10.11 | 10.59 | 91.16% | 4.90% |
| U-Net | Mid-term | 3.20 | 1.02 | 3.90 | 10.11 | 11.35 | 79.40% | 11.83% |
| U-Net | Long-term | 4.00 | 2.38 | 5.88 | 10.11 | 12.58 | 64.53% | **21.82%** |
| RF | Near-term | 1.24 | 0.26 | 2.34 | 10.30 | 10.64 | 93.73% | 4.85% |
| RF | Mid-term | 1.42 | 0.18 | 3.90 | 10.30 | 11.11 | 86.00% | 12.35% |
| RF | Long-term | 1.58 | 0.29 | 5.88 | 10.30 | 11.97 | 74.08% | **24.12%** |

The Random Forest rows are retained because §3.8.4 deployed it until the bootstrap (§7.10) and the
scenario test (§7.11) reversed that choice; keeping them makes the reversal auditable.

**σ_arch rests on all four benchmarked architectures (n = 4)** — RF, XGBoost, CNN, U-Net — which exceeds σ_GCM's n = 3, so the comparison between them no longer favours the GCM term on sample size. Reaching n = 4 required persisting XGBoost (~400 MB at 200 fixed rounds, an order of magnitude below RF's ~4 GB) and projecting it — the same persisted models that now serve the deployment.

**By 2076–2100 architecture choice accounts for 27.18% of projection variance against the GCM's 5.06%** — a factor of about five. The estimate has been stable as members were added (14.2% at n=2, 27.8% at n=3, 24–27% at n=4), which is itself reassuring: the conclusion is not an artefact of which two models happened to be compared.

**σ_SSP was quietly reporting the scenario defect all along.** It is half the |SSP5-8.5 − SSP2-4.5|
ensemble-mean difference — that is, it *is* the scenario separation. At long term the deployed
XGBoost gives 0.83 and the U-Net 2.38, while the Random Forest gives 0.29: a model that cannot
tell the pathways apart contributes almost no scenario uncertainty, which reads as confidence and
is actually saturation (§7.11). Nothing flagged this until the scenarios were compared directly.

**σ_DS is constant across horizons** — it is the validation RMSE carried forward, not something
that varies with lead time. Its share falls from 90.10% to 67.21% only because σ_GCM and σ_arch
*grow*; the downscaling error itself does not improve. σ_DS is also a different **kind** of
quantity: a historical error measured against a reference, where the other three are spreads
across possible futures.



### Projected change, and how much it depends on architecture

| Long-term minus near-term, domain-mean GHI | SSP2-4.5 | SSP5-8.5 |
|---|---|---|
| **XGBoost (deployed)** | +4.08 W m⁻² | +4.77 W m⁻² |
| Random Forest | +1.65 W m⁻² | +1.35 W m⁻² |
| U-Net | +5.43 W m⁻² | +8.99 W m⁻² |
| CNN | +8.31 W m⁻² | +16.56 W m⁻² |

The magnitude spans a factor of four to twelve across architectures, which is what σ_arch measures.
All of it is small beside σ_DS ≈ 9–10 W m⁻²: **no projected change in this study is large relative
to its own error bar.**

**The change *pattern* is less robust than the magnitude, and this is the weaker part of the
result.** Spatial correlation between architectures' change fields:

| | SSP2-4.5 | SSP5-8.5 |
|---|---|---|
| XGBoost vs Random Forest | 0.43 | 0.18 |
| XGBoost vs U-Net | 0.48 | 0.13 |
| XGBoost vs CNN | 0.12 | 0.11 |
| Random Forest vs U-Net | 0.67 | 0.71 |

No pair exceeds 0.71, and the deployed model agrees with the others at 0.11–0.48. The models
concur closely on the *historical* spatial climatology (spatial R ≥ 0.90 for all four, ≥ 0.996 for
the pixel-wise pair) and diverge on *where* change occurs. Agreement on the present is therefore
not evidence of agreement on the future, and **the projected change pattern should not be read at
grid-cell resolution** — a caveat that holds whichever model is deployed, and one this study can
state only because four architectures were carried through to projection rather than one.

---

## 9. Deployed model and its justification (Chapter 3 §3.8.4)

**The pixel-wise XGBoost is deployed.** RF, CNN and U-Net projections are retained to quantify architecture sensitivity (σ_arch, n = 4).

**This reverses an earlier decision, and the reversal is the point.** §3.8.4 previously deployed the Random Forest on a **composite criterion** — mean bias plus spatial error structure — which Chapter 3 disclosed as having been adopted **after** the corrected-alignment run reversed the ranking (§10a, A10). Two tests then removed its basis:

1. **The composite criterion does not separate the two models (§7.10).** RF−XGBoost is −0.145 [−0.305, **+0.010**] on centred RMSE and +0.0016 [−0.0006, +0.0042] on spatial R. Neither interval excludes zero, so neither leg of the criterion is established. The aggregate deficit it was meant to outweigh *is* established: +1.058 [+0.530, +1.671], confirmed by XGBoost winning all four rolling-origin folds (§7.9).
2. **RF cannot separate the emission scenarios (§7.11).** Its SSP5-8.5 minus SSP2-4.5 separation *shrinks* with lead time (+0.465 → +0.307 → +0.167), it inverts on the long-term change signal, and only 71.7% of cells order the pathways correctly, against XGBoost's 96.7%. The mechanism is tree extrapolation: temperature leaves the 1985–2010 training range 4.95% of the time under SSP5-8.5 against 0.79% under SSP2-4.5, and a tree's prediction saturates outside the range it was fitted on.

3. **The ranking is not an artefact of the chosen split (§7.9).** This is the consequence of the rolling-origin result, and it belongs here rather than only in §7. The entire deployment argument rests on a comparison measured over one 1985–2010 / 2011–2024 division of the record. A rolling-origin design, refitting on an expanding window and testing on the block immediately after it, puts XGBoost ahead in **all four** forward-in-time folds by margins of 0.89 to 2.26 W m⁻². The preference is therefore a property of the models rather than of the validation period, and no fold reverses it. The negative form matters more than the positive one: had the ranking flipped between folds, *any* selection rule — the original single-metric one included — would have been arbitrating noise, and the honest conclusion would have been that the four models are not separable at this sample size. That is precisely the conclusion §7.10 forces for RF against CNN against U-Net, whose RMSE differences are not distinguishable. It is not the conclusion for XGBoost, which is separable from all three and stays separable in every fold.

**XGBoost is the only model no other model is established to beat on any tested axis**, while it is established to beat all three on RMSE. Deploying it also **returns the study to §3.8.1's original pre-registered criterion** — lowest validation RMSE — which removes the post-hoc criterion change (A10) as an attack surface rather than defending it.

**Why the two-axis analysis still matters.** Centred RMSE of the time-mean field and the standard deviation of the per-cell bias are *the same quantity* — expanding the centred error gives var(m − r) — verified identical to 6 decimal places. Since RMSE² = bias² + centredRMSE², the genuinely independent axes are the **systematic offset** and the **spatial error structure**. That analysis is what makes §7.10 testable, and it is what showed the criterion could not do the work asked of it. It also still holds against the shared-weight models, where the margins *are* established: RF and XGBoost both beat CNN and U-Net on centred RMSE by 2.4–2.8 W m⁻², intervals nowhere near zero. **The pixel-wise/shared-weight distinction stands; only the separation within the pixel-wise pair failed.**

**The cost of the switch, stated plainly — two items, both against XGBoost.** Its mean bias is +1.20 W m⁻² against RF's +0.25, a factor of nearly five, the one validation axis on which RF is genuinely preferable (both intervals span zero, and +1.20 is an eighth of σ_DS). And it transfers marginally less well from ERA5 to CMIP6 predictors: 4.82 against RF's 4.65 in climatological RMSE (§7.7), a 0.17 W m⁻² gap that follows from XGBoost fitting the ERA5 predictor distribution more tightly. Set against a scenario-discrimination failure in the product's core function, both are the better trade; a reader who weights systematic offset and transfer robustness above scenario response should prefer RF, and should then also accept §7.11's consequence — that the resulting projections cannot reliably distinguish the pathways they are labelled with.

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

**A10 — the criterion was changed after seeing the results, and has since been changed back.** Traced through the backups: `_preupdate` (Jun 28) *"lowest test-period RMSE"*; `_pre_rf_deploy` (Aug 6) *"the Random Forest, which achieved the lowest validation-period RMSE"* — and RF genuinely won it then (18.72 vs 23.15); `_pre_384_rewrite` (Aug 10) still single-metric; the composite criterion appears only in the Aug 11 file, **after** the corrected-alignment run reversed the ranking that morning.

**This is now resolved rather than merely disclosed.** The composite criterion was tested against sampling uncertainty (§7.10) and failed to separate the two models on either of its legs, and §7.11 then found an independent defect in RF. §3.8.4 was rewritten to deploy XGBoost under §3.8.1's **original** pre-registered rule, lowest validation RMSE. The full history — original rule, post-hoc criterion, and the evidence that retired it — is stated in §3.8.4 rather than removed, so the reversal is auditable. A10 is no longer a live objection: the study is back on the criterion it started with, and the deviation is on the record.

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

"PROPOSED" dropped from the title; implemented sections converted to past tense. Factual corrections: actual retained grid (§3.2.1) · removed the xESMF/conservative-remapping claim, neither was used (§3.4.1) · areal block means not point-sampling, plus the SVF search-radius caveat (§3.5.2) · SZA computed analytically not via PVLIB (§3.5.3) · the U-Net/CNN ReLU distinction and the exact-cancellation property (§3.5.4) · upsampling factor (§3.6.1) · grid corrected 75×85=6,375 → **71×81=5,751** (§3.6.2) · dual-branch input, dropout, channel attention, pad-to-32; false "output ReLU" claim removed (§3.6.6) · never-performed Bayesian stage removed, CV-transfer caveat added (§3.6.7) · ENSO sampling corrected to **U-Net only** (§3.6.8) · Taylor stats clarified as time-mean (§3.7.2) · deployed model designated in §3.8.1 (RF at the time; now XGBoost, §9) · Table 3.2 search ranges replaced with grids actually searched. **New §3.6.9** (daily-resolution experiment) and **new §3.8.4** (deployed-model justification).

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

**Needs your hand — one item, and it needs the Zotero desktop app**
- **Dozier & Frew (1990) is not in the Zotero library.** Verified precisely: it appears once, as plain text at Chapter 3 paragraph 75 ("the Dozier and Frew (1990) sky-view integral"), and is in **none** of the 26 `ZOTERO_ITEM` citation fields. Chapter 3's bibliography is Zotero-generated (`ZOTERO_BIBL` field present), so **the reference will not appear in the reference list** as things stand. I cannot add it — the library is the desktop application's own database. Add this item, then re-cite the plain text as a live field:

  > Dozier, J. and Frew, J. (1990). Rapid calculation of terrain parameters for radiation modeling from digital elevation data. *IEEE Transactions on Geoscience and Remote Sensing*, 28(5), 963–969. doi:10.1109/36.58986

- **The published results dashboard has been withdrawn** and replaced with a retraction notice (see §10a / E1). Nothing further is needed unless you want the URL itself deleted, which must be done from the artifacts gallery.

**Closed in the Round 5 pass**
- **Chapter 1 needed no change.** Its SARAH-2/NSRDB passage (paragraph 35) describes those products' *limitation* — that they carry no future information — and never claims them as this study's validation. Checked rather than assumed.
- **Chapter 2 §2.3.3 tense — fixed.** Paragraph 69 claimed SARAH-2 was "a suitable independent validation dataset for the downscaled products produced in this study"; paragraph 71 said "NSRDB **serves as** a secondary independent validation source". Both now state the validation is planned and not reported here, pointing to §3.3.3, so Chapters 1, 2 and 3 agree.
- **Chapter 2 forward reference — added.** Paragraph 128's "Benchmarking studies **confirm** that deep learning models outperform classical baselines" is softened to "**report**", and the paragraph now closes by framing this as a hypothesis the study tests rather than assumes, states that it is not borne out here, and forward-references §3.8.4.
- **The 0.68 ambiguity does not exist in the thesis.** Searched all three chapters: **0.68 appears nowhere**. The collision is between two *status-document* sections (§6.1's collapsed correlation and §7.5's std ratios), not between two thesis passages, so no clarifying clause is needed. Resolved as not applicable rather than left open.
- **Per-cell QC flag counts — implemented, and they found a real defect.** See §7.12.

**E1 — closed.** The published dashboard has been **withdrawn**: its content is replaced by a retraction notice explaining that every figure was superseded by the alignment fix, the circular-predictor removal, the two leakage fixes and three retrains. The URL now resolves to that notice rather than to wrong numbers. Fully deleting the URL, if wanted, must be done from the artifacts gallery.

---

## 14. Standing methodological caveats

1. **ERA5 is the only reference *in the work reported here*.** Training and validation both use it, over a data-sparse region where reanalysis is weakest. The temporal split is genuine, so this is a real out-of-sample test *of the reanalysis relationship* — but not an independent test against observations. SARAH-2 / NSRDB validation is committed as the next stage (§13); until it lands, every skill figure in §8 carries this qualifier.
2. **Stationarity is assumed.** Models fitted on 1985–2010 are applied to bias-corrected SSP5-8.5 fields to 2100. EDCM preserves the absolute change signal, but the learned predictor→irradiance mapping is assumed valid in an unseen climate.
3. **Cloud is unresolved.** The dominant control on surface irradiance is parameterised inside the GCM and available only as a grid mean. Feature importance confirms cloud fraction dominates the fit.
4. **Defensive NaN handling hides failures.** Two of the three severe silent bugs were `np.nan_to_num` converting a loud failure into a plausible field. Both were caught by inspecting fields, not scores. `safe_nan_to_num` now makes this loud.
5. **Two bugs cancelled each other** (§6.11). A bug invisible in one data split because a second bug reverses it is the hardest class to find — the lesson is that identical logic must be used at every join, not merely logic that works.
6. **Prose drifts from tables.** Four audit rounds found the same failure: a number corrected in a table while the sentence beneath kept the old value, every time caught by an external reader. The tables had a defence — `compute_table33.py` as the single canonical source — and the prose had none. `check_status_consistency.py` now supplies one, checking that canonical CSV values appear and that superseded values do not reappear outside an explanatory context. It runs under `pytest`. Its own first version had exactly the bug it exists to prevent: it scanned a whole paragraph for any explanatory word, so a stale figure passed because an unrelated clause elsewhere in the same line said "no longer". It now searches a 90-character window around the value.

---

## 15. Note on authorship

Substantial parts of this pipeline were written with AI assistance rather than typed from scratch — particularly the evaluation and verification layer (`generate_validation_spatial_fields.py`, `compute_spatial_verification.py`, `compute_uncertainty_decomposition.py`, `compute_information_content.py`, `compute_feature_importance.py`, `hpo_pixelwise.py`, `generate_future_projections_rf.py`) and major rewrites of `unet_model.py`, `ml_dataset_common.py`, `compute_finegrid_clearsky_ghi.py`, `apply_edcm_bias_correction.py` and the training/evaluation scripts. The analytical decisions, diagnoses and interpretations recorded above should be understood and defensible independently of how the code was produced.
