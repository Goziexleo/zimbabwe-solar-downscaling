# Project status and work record

**Project:** Machine Learning-Based Downscaling of GCM Outputs for Solar Energy Suitability Mapping over Zimbabwe
**Author:** Chiagozie Raphael Madukwe (FSR2526653), University of Zimbabwe MSc
**Project root:** `/Users/gozie/Documents/MCSM PROJECT AGY` (not a git repository)
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
| `train_pixelwise_rf.py` | 5,751 forests. `PERSIST_RF_MODELS=1` writes per-cell joblib (~14 GB) + `manifest.json` |
| `train_pixelwise_xgb.py` | 5,751 boosters, never persisted (~1 min to refit) |
| `train_cnn_downscaler.py` | CNN, MSE + spatial-gradient penalty, 100 epochs |
| `train_unet_downscaler.py` | U-Net, ENSO-stratified batches, flip augmentation, early stopping |
| `evaluate_unet.py` / `evaluate_cnn.py` | Table 3.3 metrics. All evaluations report skill against **three** references: interpolation, climatology, delta-mapping |
| `hpo_pixelwise.py` | 5-fold temporal CV grid search on a 150-cell subsample |
| `compute_information_content.py` | Sub-0.25° information test with elevation positive control |
| `compute_feature_importance.py` | RF MDI + permutation, XGBoost gain + cover |

### Projection and verification
| Script | Role |
|---|---|
| `generate_future_projections.py` / `..._rf.py` | U-Net / RF → downscaled GHI, 6 GCM-scenario combos |
| `compute_mme_aggregations.py` | Ensemble mean + inter-model spread, 3 horizons. `PROJECTIONS_DIR` / `MME_OUTPUT_DIR` |
| `generate_validation_spatial_fields.py` | Full GHI fields: truth, baseline, all 4 models |
| `compute_spatial_verification.py` | Taylor statistics + per-pixel bias/RMSE maps |
| `compute_uncertainty_decomposition.py` | **Four-component** σ_GCM / σ_SSP / σ_arch / σ_DS. σ_DS is now **derived** from the validation fields, not hardcoded. |

`deprecated/daily_resolution_experiment/` holds the superseded daily-resolution scripts and outputs, with a README explaining why.

---

## 6. Bugs found and fixed

### 6.11 One-month `rsds` misalignment — the most consequential bug in the project
`merge_predictor_stack.py` aligned `rsds` with `reindex(method="nearest")`. The predictor files stamp each month at its **first** day (`2011-02-01`); the `ssrd` files stamp it at its **last** (`2011-01-31`). Nearest-neighbour matched them one day apart, writing **January's irradiance into February's slot** — validation `rsds` lagged exactly one month (`rsds[t] == ssrd[t-1]`, bit-exact, `corr = 1.000000` at lag +1).

Invisible in training because `build_ml_features_and_targets.py` used the *same* unsafe match in the opposite direction, cancelling it. Two wrongs made a right in one split and not the other.

**Consequences, all corrected:** models were fed the wrong month's irradiance as a high-weight feature at validation only, degrading every Table 3.3 figure (XGBoost RMSE 22.94 → 5.54 on fixing this alone). The baseline was computed from the same lagged field, inflating its RMSE from 0.233 to 32.38 and turning deeply negative skill scores into apparently positive ones. Both build scripts now match on calendar month.

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
`rsds` was populated by copying ERA5 `ssrd` — the field the target is built from. With correct alignment, **bilinear interpolation of `rsds` reproduces the target at RMSE 0.233 W m⁻², correlation 0.999982**, and delta-mapping does better still at **0.131**. Every model scores ≈ −38 to −43 against these. Two independent baselines confirm it is not an artefact of one interpolation choice.

This was first seen at daily resolution and misdiagnosed as daily-specific; it was masked at monthly resolution by bug §6.11. The response was to drop `rsds` from the model inputs, making the task a genuine perfect-prognosis problem.

### 7.2 The product contains no information finer than its input grid
Degrading the 0.1° GHI product to 0.25° and interpolating back recovers it at **correlation 0.999991**, losing **0.0019%** of variance. Elevation, a genuinely fine-scale field, loses 0.898% under the same test — so the test does detect sub-grid structure when present. **The product is a bias-and-variability correction evaluated on a finer mesh, not spatial super-resolution.** Only a genuinely high-resolution target (SARAH-2, NSRDB) could change this.

### 7.3 The alignment fix, not the predictor removal, cleared R > 0.90
Ablation (XGBoost, all else constant):

| Configuration | RMSE | Pearson R | SS vs climatology |
|---|---|---|---|
| A. Lagged `rsds`, 6 predictors (original) | 22.94 | 0.817 | — |
| B. Lag fixed, `rsds` **kept** | **5.54** | **0.9894** | 0.6922 |
| C. Lag fixed, `rsds` **dropped** (deployed) | 9.11 | 0.9715 | 0.4932 |

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
| XGBoost | +1.258 | 0.721 | −0.91 to +3.41 |
| **U-Net** | **+0.000** | **3.539** | **−14.81 to +10.18** |

It over-predicts some districts by +10 and under-predicts others by −15; they average to nothing. For a map read cell by cell to choose sites, this is the most damaging error structure available.

### 7.6 Cross-validation selected worse hyperparameters
A 150-cell subsampled CV search picked configurations for both tree models that underperformed the untuned defaults on the full 5,751-cell holdout. Defaults retained. The search's own scores gave no warning.

---

## 8. Current results

### Table 3.3 — validation 2011–2024 (W m⁻²)

| Model | RMSE | MAE | Pearson R | MBE | SS vs clim. | R² |
|---|---|---|---|---|---|---|
| **XGBoost** | **9.11** | **6.73** | **0.9715** | +1.26 | **0.4932** | **0.9426** |
| U-Net | 10.11 | 7.64 | 0.9640 | +0.0003 | 0.4378 | — |
| CNN | 10.14 | 7.71 | 0.9646 | +1.20 | 0.4359 | 0.9289 |
| **Random Forest (deployed)** | 10.30 | 7.73 | 0.9636 | +0.25 | 0.4272 | 0.9267 |

Targets are R > 0.90 and |MBE| < 5. **All four models clear both for the first time in the project.**

Three reference forecasts are reported. Skill against **interpolation** (ref RMSE 0.233) and **delta-mapping** (0.131) is ≈ −38 to −76 for every model and measures the circularity of §7.1, not model quality. Skill against **climatology** (17.98) is the meaningful figure.

### Taylor statistics — time-mean spatial pattern (reference spatial std 7.98 W m⁻²)

| Model | Spatial R | Std ratio | Centred RMSE | as % of ref std |
|---|---|---|---|---|
| Baseline (bilinear) | 0.9998 | 0.9949 | 0.161 | 2.0% |
| **Random Forest** | **0.9978** | 0.9725 | **0.571** | **7.2%** |
| XGBoost | 0.9965 | 0.9612 | 0.721 | 9.0% |
| CNN | 0.9176 | 0.9937 | 3.229 | 40.5% |
| U-Net | 0.8995 | 0.9762 | 3.539 | 44.4% |

The pixel-wise models reproduce the spatial climatology far more faithfully than the shared-weight ones.

### Uncertainty decomposition — four components

| Model | Period | σ_GCM | σ_SSP | σ_arch | σ_DS | σ_total | % var DS | % var arch |
|---|---|---|---|---|---|---|---|---|
| RF | Near-term | 1.14 | 0.24 | 1.63 | 10.30 | 10.57 | 94.99% | 3.57% |
| RF | Mid-term | 1.35 | 0.16 | 2.56 | 10.30 | 10.80 | 90.76% | 7.49% |
| RF | Long-term | 1.53 | 0.23 | 3.87 | 10.30 | 11.24 | 83.74% | 14.22% |
| U-Net | Near-term | 1.89 | 0.54 | 1.63 | 10.11 | 10.51 | 92.40% | 3.61% |
| U-Net | Mid-term | 3.05 | 1.00 | 2.56 | 10.11 | 11.05 | 83.60% | 7.16% |
| U-Net | Long-term | 3.89 | 2.32 | 3.87 | 10.11 | 11.89 | 72.04% | 12.70% |

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

Selection uses a **composite criterion**, stated as such because applying it changes the answer: spatial pattern fidelity and local-bias distribution weigh alongside aggregate error, since a suitability map is read cell by cell. On aggregate RMSE alone the choice would be XGBoost (9.11 vs 10.30, a 12% gap). On the composite criterion RF wins — lowest centred RMSE (0.571 vs 0.721) and the smallest, tightest local bias (+0.25, std 0.57 vs XGBoost's +1.26). The 12% penalty is small beside σ_DS ≈ 10.

**Why U-Net fails hardest** — three measured architectural causes: 17.8M parameters against 312 training fields (≈57,000 per field); the padded 96×96 domain compressed to **3×3** at the bottleneck; and one shared kernel set forced to fit a single predictor→irradiance relation from Lowveld to Eastern Highlands, so it learns the domain-*average* relation (hence competitive aggregate error) and fails where local relations depart. RF fits 5,751 independent local relations and structurally cannot trade one district against another. CNN sits between — shared weights but only 79k parameters plus a spatial-gradient penalty — and its scores sit between accordingly.

This runs against Chapter 2's expectation (Rampal et al. 2024) that encoder-decoders beat tree methods. The explanation is domain-specific: that literature comes from targets carrying genuine sub-grid information, which this ERA5-derived target does not (§7.2). It is a statement about this problem, not a refutation of the literature.

### Final configurations
| Model | Configuration |
|---|---|
| Random Forest | `n_estimators=500, max_features=sqrt, min_samples_leaf=5` |
| XGBoost | `max_depth=6, eta=0.05, subsample=0.8, min_child_weight=3` |
| CNN | `lr=0.0005, lambda_gp=0.01` |
| U-Net | `lr=2e-4`, dropout 0.3, weight decay 1e-4, batch 16, patience 20 |

Tree defaults were retained after HPO (§7.6). Deployed values are now the **script defaults** — running any training script with no environment variables reproduces the deployed configuration.

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
| B9 delta-mapping baseline | **Done** — third reference; ref RMSE 0.131 |
| B10 documentation | **Partly** — see below |

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
  models/{unet,cnn}/*.pth · models/pixelwise_rf/*.joblib + manifest.json (~14 GB)
  era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc
  ml_ready/ml_{training,validation}_dataset.nc
  cmip6_bias_corrected/
  projections/ (U-Net) · projections_rf/ (RF)          6 GCM-scenario combos each
  mme_aggregations/ · mme_aggregations_rf/             3 horizons × 2 scenarios
  evaluation/
    validation_spatial_fields.nc      truth + baseline + all 4 models
    taylor_diagram_stats.csv · spatial_verification_maps.nc
    uncertainty_decomposition_{unet,rf}.nc + summary.csv
    information_content.csv · feature_importance.csv
deprecated/daily_resolution_experiment/    superseded, with README
logs/                                      all run logs
```

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
- **The published results dashboard is stale** (predates three retrains): `https://claude.ai/code/artifact/0998da0d-86ae-45b5-810a-a8d616f9323a`

**Still specified but not implemented:** per-cell QC flag counts for the appendix (§3.4.3).

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
