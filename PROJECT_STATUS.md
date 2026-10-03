# Project status and work record

**Project:** Machine Learning-Based Downscaling of GCM Outputs for Solar Energy Suitability Mapping over Zimbabwe
**Author:** Chiagozie Raphael Madukwe (FSR2526653), University of Zimbabwe MSc
**Project root:** `/Users/gozie/Documents/MCSM PROJECT AGY` (git repository since the second audit; `data/` untracked)
**Chapter 3:** `.../UNI ZIM/PROJECT CHAPTERS/CR_Madukwe_Chapter3_Final.docx`

**Chapter 3 backups (same folder), oldest first:**
`..._BACKUP_preupdate.docx` (before the proposal→implemented rewrite) ·
`..._BACKUP_pre_rf_deploy.docx` (before §3.8.4 was first added) ·
`..._BACKUP_pre_384_rewrite.docx` (before §3.8.4 was rewritten for the corrected predictor set) ·
`..._BACKUP_pre_validation_scope.docx` (before §3.3.3 was reworded to treat SARAH as planned) ·
`..._BACKUP_pre_audit2.docx` · `..._BACKUP_pre_xgb_deploy.docx` (before the deployment moved to XGBoost) ·
`..._BACKUP_pre_bca.docx` (before §3.8.4 took the BCa intervals) ·
`..._BACKUP_pre_both_legs.docx` (before the composite criterion was conceded on both legs) ·
`..._BACKUP_pre_perfect_prognosis.docx` (before the §3.4.4 perfect-prognosis paragraph) ·
`..._BACKUP_pre_resolution_fix.docx` · `..._BACKUP_pre_factor_fix.docx` ·
`..._BACKUP_pre_ablation.docx` (before §3.7.4 and the five-predictor corrections)

This is the authoritative record of what the pipeline does, what was built and fixed, the current numbers, the honest limitations, and what remains. All figures below are from the final corrected-alignment / 5-predictor run and are mutually consistent.

---

## 1. What the project does

Statistical downscaling of coarse CMIP6 global-climate-model output to 0.1° over Zimbabwe. **The resolution chain has three steps, not two, and the middle one is easy to omit:** CMIP6 is regridded *linearly* from its native 0.94°–1.88° grid onto the 0.25° ERA5 grid, bias corrected there, and only then does the ERA5-learned 0.25° → 0.1° relationship apply. The models never see native CMIP6; their step is a factor of 2.5, while the gap from GCM to product is 9.3–18.8 per axis, most of which regridding and bias correction close rather than the machine learning. to produce a solar-resource product for siting analysis. Four architectures are benchmarked on identical inputs:

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
*Chapter 1's RQ1 previously asked how well the models downscale "to ERA5 resolution". The product is 0.1°, **finer** than ERA5's 0.25°, so the question understated its own target. Reworded to "to 0.1° resolution … against an ERA5-derived target", which is what the study does.*

*Chapter 3 §3.2.1 previously described this as "a spatial upscaling factor of approximately 23x" — the nominal 250 km label divided by 11 km. Corrected: ACCESS-CM2's actual spacing is 1.25° × 1.875°, and the factor the ML bridges is 2.5, not 23. The rest is regridding and bias correction.*

| CMIP6 native grid | 0.94°–1.88° (100–250 km nominal): CNRM-CM6-1 1.40°, MPI-ESM1-2-HR 0.94°, ACCESS-CM2 1.25° × 1.88°. **24 to 64 cells over this domain.** |
| Coarse (predictor) grid — **ERA5's, and what the models see** | 0.25°, 29 × 33 = 957 |
| Fine (target) grid | 0.1°, 71 × 81 = **5,751 cells** |
| Domain retained | 22.0°S–15.0°S, 25.0°E–33.0°E |
| Upsampling factor | 2.5 by resolution; ≈2.45 along each axis (71/29 = 2.448, 81/33 = 2.455); **6.01 by total cell count** (5,751/957) |
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

**Target:** `CSI = ssrd_fine / clearsky_ghi`, clipped to [0, 1.1]. Recovered as `GHI = CSI × clearsky_ghi`. Observed CSI spans 0.246–0.632 across both splits (0.2464 training, 0.2564 validation), so **the clip never binds** and the normalisation is an exact inverse — verified to machine precision (max difference 1.14 × 10⁻¹³ W m⁻²), which is what matters for the product: the clear-sky field **cancels entirely from the GHI**.

**The intermediate CSI is not a clear-sky index in the physical sense, and should not be described as one (§7.15).** The denominator averages thirteen daytime hours while the ERA5 `ssrd` numerator is a 24-hour monthly mean, so the ratio runs a factor of **1.724** below a true clear-sky index. That is also why the observed range tops out at 0.632 rather than approaching 1 in the dry season — the range is evidence about the denominator, not only about the clip.

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
| `build_chapter5.py` | Generates Chapter 5 from the same CSVs as Chapter 4 |
| `build_chapter4.py` | Generates Chapter 4 from the evaluation CSVs. Prose drifting from tables is this project's most frequent failure, so the chapter is generated rather than typed |
| `optimise_unet.py` | Controlled U-Net sweep. Fits on 1985–2004, selects on 2005–2010, touches the withheld record once per variant. Source of §7.18 |
| `make_suitability_maps.py` | §3.9 map figures 07–10, exclusions drawn off the score ramp |
| `compare_sarah_era5.py` | SARAH against the ERA5-derived GHI over 1985–2024, on the target grid. Resumable: caches each month as it interpolates, after an OS update killed a 25-minute run that held everything in memory |
| `build_suitability_layers.py` | §3.9 criterion layers and exclusion mask on the 0.1° grid |
| `compute_suitability.py` | §3.9 criterion weighting, WLC, five-tier classification, four-scheme sensitivity |
| `compare_sarah_clearsky.py` | Compares the PVLIB clear-sky ceiling against SARAH's `SISC` by calendar month. The near-constant 1.724 ratio is the evidence for §7.15 |
| `compute_bca_centred_rmse.py` | **BCa intervals** on the paired differences. Required because the percentile interval is invalid for centred RMSE, whose bootstrap distribution is biased by construction. Reports percentile, basic and BCa side by side. |
| `check_status_consistency.py` | Guards this document against stale numbers; run by `pytest` |
| `check_brief_consistency.py` | Guards the interview brief. Checks table cells against the canonical CSVs (cell-level, not substring — `9.24` appears 17 times), a list of **superseded sentences** that must not reappear, and that the markdown is regenerable from the HTML. Run by `pytest`. |
| `check_chapter3_consistency.py` | Guards **Chapter 3**, which lives outside the repo. Canonical values from the CSVs, superseded figures and design claims, **Section 3.7.4's ablation table cell by cell**, and **Zotero field integrity** — begin/end pairs must match, since the edits here are run-level surgery on live citation fields. Skips cleanly if the chapter is absent; `CHAPTER3_PATH` overrides the location. Run by `pytest`. |
| `brief/build_brief_formats.py --sync` | Copies the deliverables **including `PROJECT_STATUS.md`** into `PROJECT CHAPTERS`. The status document previously lived only in the repo, so the project folder held a stale picture. Run it with every rebuild. |
| `brief/build_brief_formats.py` | `brief/viva-brief.html` is the single source; this generates the print HTML, the PDF (`--pdf`) and the markdown. Print CSS lives in `brief/print.css` so a clean checkout can build. |

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

## 6.14 The neural hyperparameter search was the last live leak — now closed

§3.6.7's grid search for the CNN and U-Net was scored on "validation MSE": the withheld
2011–2024 record. That is selection on the evaluation set, the same defect corrected for
XGBoost's boosting rounds (§6.12) and both networks' epoch counts (§6.13). It was the one
instance still live, flagged in §12j and recommended in Chapter 5 §5.6.

**`hpo_neural.py` re-runs Table 3.2's grids entirely inside the training period.** Each
candidate is fitted on 1985–2004 and scored on an inner selection split of 2005–2010,
through the real training scripts under `PHASE_A_ONLY=1` rather than a reimplementation —
the U-Net sweep had already shown that a reimplemented baseline does not reproduce the
deployed model. Seven candidates, ~70 minutes.

| model | lr | λ_gp | inner-select MSE | |
|---|---|---|---|---|
| CNN | 0.0005 | 0.001 | 0.000448 | |
| CNN | 0.0005 | 0.01 | 0.000424 | **deployed before** |
| CNN | 0.001 | 0.001 | 0.000400 | |
| CNN | 0.001 | 0.01 | 0.000418 | |
| U-Net | 0.0002 | — | 0.128886 | **deployed before** |
| U-Net | 0.0005 | — | 0.126946 | |
| U-Net | 0.001 | — | 0.124575 | |

**Both configurations changed.** The CNN moves to lr 1e-3 with λ_gp 1e-3; the U-Net to
lr 1e-3. Neither had been reachable by the old search, which was comparing candidates on
the wrong data.

**What it was worth on the withheld record**, with the two corrections separated because
they pull opposite ways:

| | leaky checkpoint + leaky HPO | honest checkpoint, leaky HPO (§6.13) | honest throughout |
|---|---|---|---|
| CNN | 10.14 | 11.09 | **10.08** |
| U-Net | 10.11 | 11.03 | **10.71** |

**The CNN's honest configuration reproduces its leaked result almost exactly** (10.08
against 10.14). Its apparent accuracy was not bought by the leakage so much as by a
configuration the leakage happened to find, and an honest search finds an equally good one.
The U-Net ends at 10.71 against 10.11, so for that architecture part of the original
figure genuinely was selection on the test set.

**The ordering changed again, and again the evidence did not.** The order is now XGBoost
9.24, CNN 10.08, RF 10.30, U-Net 10.71 — the CNN has overtaken the Random Forest.
**That reordering is not established**: RF − CNN is +0.217 [-0.368, +1.105] and RF − U-Net is
-0.413 [-1.052, +0.394], both spanning zero. What *is* established is the deployed model's
margin over both networks: XGBoost − CNN -0.841 [-1.090, -0.353] and XGBoost − U-Net
-1.471 [-1.893, -1.022]. **The deployment is unaffected** — it turned on scenario
discrimination, and the Random Forest still fails that screen.

**σ_arch is 59.39%** of long-term variance, with σ_DS at 9.08%.
Improving two of the four members narrowed the spread between architectures, which is
exactly what that term measures — the mirror image of what the §6.13 correction did to it.
Chapter 5's methodological point stands and is now much stronger: architecture choice
dominates GCM choice (21.64%) by a factor of about 3.1.

**Corrected after the examiner's critique (1 Oct 2026).** Two errors were fixed here.
σ_DS carried the MONTHLY validation RMSE (9.24 W m-2) into a budget whose other terms are
spreads of 25-year mean changes; random monthly error largely averages out of a 300-month
mean, so the term is now the part that survives it (1.47 W m-2, from a systematic 1.41 and
N_eff 223 of 300). And σ_arch spanned all four architectures including the Random Forest,
which the §4.4 screen disqualifies — a member cannot be both inadmissible and a measure of
the uncertainty in choosing between members. The shares are also now computed over the
3,291 cells inside Zimbabwe rather than all 5,751 in the analysis box. Long-term σ_DS fell
from 67.79% to 7.11% and σ_arch rose from 26.56% to 68.04%, which does not weaken the
thesis: it removes an artefact that was suppressing its own central result.

**Table 4.3 is unaffected, and this was verified against the completed rerun rather than
argued.** It prints only the Random Forest and XGBoost columns, and neither model was
refitted. All four folds reproduce to the decimal the chapter prints: 11.7272/9.4716,
12.8372/11.2859, 9.0493/8.1634, 10.8254/9.6165 against 11.73/9.47, 12.84/11.29, 9.05/8.16
and 10.83/9.62. The rerun mattered only for the CNN and U-Net CSI column, which appears in
this document and in no chapter — which is why the chapters could be finished while it ran.

**The rerun completed without deadlocking, including on fold 3**, the fold that hung under
`n_jobs=-1`, so the capped pool is the fix rather than a coincidence.

**Two bugs found while regenerating.** `compute_rolling_origin.py` deadlocked on fold 3
(`Parallel(n_jobs=-1)` lost workers to memory pressure and the parent waited forever at
zero CPU); it now caps the pool, dispatches in bounded batches and carries a timeout. And
its neural leg had been silently returning `nan` since §6.13 renamed the line it greps for
— the CSI values in the table on disk came from the superseded code path — while a fixed
`INNER_SPLIT_YEAR=2005` left the 1985–1998 fold with an empty selection split and killed
the CNN with `ZeroDivisionError`. Both fixed; the split now follows each fold's own window.

**One gap in my own cascade, caught by checking mtimes rather than trusting it.**
`compute_spatial_verification.py` was omitted, so `taylor_diagram_stats.csv` and
`spatial_verification_maps.nc` still held pre-retrain values after the cascade reported
success. Regenerated: CNN spatial correlation 0.8937 → 0.9228, U-Net 0.9028 → 0.8847.

### 6.12 XGBoost was early-stopped on the validation set
`train_pixelwise_xgb.py` fitted each cell with `eval_set=[(X_val, y_val)]` and `early_stopping_rounds=50`, so the number of boosting rounds — each cell's effective capacity — was chosen by watching the data the model was then scored against. 5,751 hyperparameters fitted on the evaluation set.

Measured cost: **9.1145 leaky against 9.2422 clean**, about 1.4% of RMSE. Early stopping on an inner 80/20 split of the training record was also tried and scored *worse* (10.0157), because holding back a fifth of an already-small 312-month record costs more than the stopping rule gains. Fixed 200 rounds uses all training data and no validation information, and is now the default.

**The ranking is unchanged** — at 9.2422 XGBoost still has the lowest aggregate RMSE, so the §3.8.4 selection argument is unaffected. But it was not an out-of-sample number, and it was the number that put XGBoost ahead of RF in the first place.

### 6.13 The CNN and U-Net selected their checkpoints on the evaluation record — FIXED

Both neural training scripts scored `ml_validation_dataset.nc`, the withheld 2011–2024 record, every epoch and saved the weights that scored best on it. The U-Net additionally early-stopped and scheduled its learning rate on it. This is §6.12's defect — found and fixed for XGBoost, which is fitted for a fixed 200 rounds — left live in the other two for the whole project.

**The fix keeps the comparison fair.** Phase A fits on 1985–2004 and scores on 2005–2010 to choose the epoch count; Phase B refits from scratch on the **full** 1985–2010 record for that many epochs. Both models still see all 312 training months, matching RF and XGBoost, and the evaluation record is never read. Chosen epochs: **CNN 77, U-Net 24**. The U-Net previously early-stopped at 48 against the test set, so it had been training roughly twice as long as an honest signal supports.

**The bias, measured.** Prior checkpoints and all six affected CSVs are preserved under `data/processed/models/_pre_honest_selection/`.

| | before | after | change |
|---|---|---|---|
| U-Net RMSE | 10.11 | **11.03** | **+0.92** |
| CNN RMSE | 10.14 | **11.09** | **+0.95** |
| U-Net skill | 0.4701 | 0.4218 | −0.048 |
| CNN skill | 0.4682 | 0.4186 | −0.050 |
| XGBoost RMSE | 9.24 | 9.24 | unchanged |
| Random Forest RMSE | 10.30 | 10.30 | unchanged |

Roughly **9% of each neural model's reported accuracy was selection on the test set.** XGBoost and RF are unchanged to four decimals, which is the determinism check: they were not retrained, and nothing else moved them.

**Three consequences.**

**The Random Forest rises from fourth to second on aggregate error** — the order is now XGBoost 9.24, RF 10.30, U-Net 11.03, CNN 11.09. **That reordering is not statistically established.** RF − U-Net is −0.731 with a BCa interval of [−1.36, +0.03], and RF − CNN is −0.792 [−1.55, +0.30]; both span zero. The table order changed, the evidence did not.

**XGBoost's advantage over both neural models roughly doubled** and remains established: −1.850 [−2.19, −1.20] against the CNN and −1.789 [−2.18, −1.21] against the U-Net.

**σ_arch rose from 27.18% to 35.40% of long-term variance**, against the GCM's 4.49%. Correcting two of the four members' inflated accuracy widened the spread between architectures, which is precisely what that term measures. Chapter 5's lead methodological contribution is strengthened, not weakened, by the correction.

**What did not move: the U-Net's spectral damping, 0.085 → 0.090** (the deployed model, at k≥11). Dropout is unchanged by this retrain, and §7.18 attributes the damping to dropout. The damping persisting through a retrain that changed everything else is independent corroboration of that attribution.

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

This was first seen at daily resolution and misdiagnosed as daily-specific; it was masked at monthly resolution by bug §6.11. The response was to drop `rsds` from the model inputs. **Note what that does and does not change.** It does not change whether this is perfect prognosis — training is ERA5-to-ERA5 with or without `rsds`, so both configurations qualify, and describing the exclusion as what *makes* it perfect prognosis is wrong. What it changes is whether the ML step does anything: with `rsds` present the learned relationship is essentially interpolation, so at projection the product reduces to bias-corrected GCM `rsds` regridded — the §3.7.3 baseline, reached at considerably more expense.

### 7.2 The product contains no information finer than its input grid
Degrading the 0.1° GHI product to 0.25° and interpolating back recovers it at **correlation 0.999991**, losing **0.0019%** of variance. Elevation, a genuinely fine-scale field, loses 0.898% under the same test — so the test does detect sub-grid structure when present. **The product is a bias-and-variability correction evaluated on a finer mesh, not spatial super-resolution.** Only a genuinely high-resolution target could change this. **SARAH is now held** (§13), so this is actionable rather than hypothetical.

### 7.3 The alignment fix, not the predictor removal, cleared R > 0.90
Ablation (XGBoost, all else constant):

| Configuration | RMSE | Pearson R | SS vs climatology |
|---|---|---|---|
| A. Lagged `rsds`, 6 predictors (original) | 22.94 | 0.817 | −0.203 |
| B. Lag fixed, `rsds` **kept** | **5.54** | **0.9894** | 0.7096 |
| C. Lag fixed, `rsds` **dropped** (deployed) | 9.24 | 0.9707 | 0.5155 |

Skill is against the corrected 19.08 climatology, so configuration C agrees with §8 (0.5155). An earlier version used the leaky 17.98 reference and gave 0.6922 and 0.4932 — putting the same model at two different skill scores in two sections of this document.

A→B is the fix alone and accounts for essentially all the improvement. B→C shows **dropping `rsds` made the models measurably worse** — it merely left them above threshold. The ~20 points of climatology skill lost (0.69 → 0.49) is precisely the circular portion. *"We removed a predictor and R improved" is the wrong causal claim* and an examiner comparing B and C would catch it.

**This table is now in Chapter 3 as §3.7.4 (Table 3.4)**, immediately after §3.7.3's discussion of the interpolation baseline. Configurations A and B are one-off refits that no script re-emits, so `check_chapter3_consistency.py` anchors their five figures as literals transcribed from here — the one place in that guard where a value does not come from a CSV, and therefore the one place where this document and the chapter can silently diverge.

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

It over-predicts some districts by +10 and under-predicts others by −15; they average to nothing.

**The principle that reconciles this with §7.10, where mean bias is conceded as a real loss for XGBoost.** The U-Net beats XGBoost on mean bias by **1.200**, *larger* than the Random Forest's 0.953, and the U-Net passes the scenario screen — so why is one a concession and the other a trap? **Mean bias counts as evidence only when centred RMSE points the same way.** Random Forest: better bias (0.25) *and* better centred RMSE (0.571 vs 0.742) — they agree, so the advantage is real. U-Net: better bias (0.0003) *and* far worse centred RMSE (3.539), per-cell range −14.81 to +10.18 — they contradict, which is the signature of cancellation. A domain-mean bias near zero means something only if the per-cell biases are also small. For a map read cell by cell to choose sites, this is the most damaging error structure available.

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

**The figure said the opposite of this section until 19 August.** `figures/05_power_spectra.png` plotted GHI only and titled its ratio panel *"Models INJECT small-scale power the target does not contain"* — the reading this section overturned. It survived the deployment switch because `fig_uncertainty` and `fig_error_maps` were repointed to XGBoost and `fig_spectra` was not re-read. Now three panels: CSI spectrum, CSI ratio carrying the 1.04× / 0.08× result, and GHI ratio labelled as a symptom. `compute_power_spectra.py` was extended to emit CSI spectra per wavenumber, which it computed but never saved. **Figure titles are prose and drift like prose; no guard covered them.** Retiring the old title then exposed a weakness in the guards themselves: the suppression rule asked whether an explanatory word sat *near* the phrase, and "superseded sentences" — describing what the guards do — sat 48 characters from a deliberately injected recurrence, so a real one was silently allowed. A two-sentence lookback failed the same way. The rule now turns on **quoting**: legitimate discussion quotes a retired phrase, a recurrence asserts it bare, and that distinction does not depend on how near some other word happens to fall. Verified against four cases — bare assertion, bare assertion beside the word "superseded", quoted discussion, and unquoted-but-explained.

| Field | CSI power beyond k=10 | vs truth | Verdict |
|---|---|---|---|
| Truth | 0.00249 | reference | reference |
| Baseline (bilinear) | 0.00247 | 0.99× | matches truth |
| Random Forest | 0.00245 | 0.98× | matches truth |
| XGBoost | 0.00251 | 1.01× | matches truth |
| CNN | 0.00259 | 1.04× | matches truth |
| **U-Net** | **0.00021** | **0.08×** | **DAMPED** |

**The U-Net damps CSI power beyond k=10 by a factor of twelve** — the Ullrich failure mode.

**The cause is the dropout setting, not the loss or the architecture (§7.18).** An earlier version of this section attributed the damping to loss and architecture. A controlled sweep (`optimise_unet.py`) shows otherwise: removing `Dropout2d(0.3)`, which the deployed configuration applies eight times per level, lifts the spectral ratio from **0.085 to 0.771** while simultaneously improving held-out RMSE from 10.98 to 8.63 and spatial correlation from 0.894 to 0.970. Varying the gradient penalty instead moves the ratio hardly at all. Aggressive spatial dropout smooths the output field; that is the mechanism.

**The two verdicts in that table are not equally robust, and the table alone does not show it.** `k=10` is one arbitrary cut. Sweeping it (`compute_power_spectra.py` → `effective_resolution_cut_sensitivity.csv`):

| Field | k≥3 | k≥5 | k≥8 | **k≥11** | k≥15 | k≥20 | verdict |
|---|---|---|---|---|---|---|---|
| Baseline | 0.99 | 0.97 | 1.00 | 0.99 | 1.00 | 1.03 | stable |
| Random Forest | 1.02 | 0.95 | 0.95 | 0.98 | 1.01 | 1.03 | stable |
| XGBoost | 1.03 | 0.97 | 0.98 | 1.01 | 1.08 | 1.11 | stable |
| **CNN** | 1.13 | 1.21 | 1.19 | **1.04** | 0.91 | 1.29 | **CUT-DEPENDENT** |
| **U-Net** | 0.72 | 0.45 | 0.20 | **0.08** | 0.04 | 0.01 | **damped at every cut** |
| *truth's share of CSI power* | *10.0%* | *2.7%* | *0.6%* | *0.25%* | *0.10%* | *0.03%* | |

**The U-Net result is robust**: damped at every cut, deepening monotonically, and already halved at k≥5 where the truth still holds 2.7% of its variance. That is a real finding about loss and architecture.

**The CNN's "1.04×, within 4%" is not.** Across the sweep it runs 0.91 to 1.29, and **k=10 is the closest point to unity of any cut tested** — the most flattering choice, arrived at innocently but flattering nonetheless. The tail also carries too little variance for the ratio to be stable: 0.25% at k≥11, 0.03% at k≥20.

**So state the robust half and stop.** The CNN **does not damp** — it never falls below 0.91 and never approaches the U-Net's collapse, so the spatial-gradient penalty is doing its job. But *how close* the CNN sits to truth is not resolved by this test, and "matches truth to within 4%" claims a precision the measurement does not support.

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
| CNN | 1.08 | 1.00 | 1.00 | 0.71 | 1.53 |
| U-Net | 0.96 | 0.80 | 1.00 | 0.67 | 1.49 |

**The CNN and U-Net columns are CSI validation MSE, not RMSE, and the two networks report it in different units** — the CNN on the raw clear-sky index, the U-Net on a standardised anomaly — so they are comparable to themselves across folds and to nothing else. These had been silently `nan` since §6.13 renamed the line the parser greps for (§6.14); the values below are the first honest ones:

| Model | 1999–2004 | 2005–2010 | 2011–2016 | 2017–2024 |
|---|---|---|---|---|
| CNN | 0.000451 | 0.000419 | 0.000418 | 0.000295 |
| U-Net | 0.125600 | 0.105000 | 0.131200 | 0.087800 |

All four sit within a spread of ~1.4, and all four find 2005–2010 hardest and 2011–2016 easiest — a property of the periods, not of any model.

**A units caveat, recorded because the raw output invites the error.** The two networks report validation MSE in different units: the CNN trains on raw CSI, the U-Net on a standardised anomaly `(CSI − climatology)/anomaly_std`. With `anomaly_std = 0.0578`, the scale factor is 1/std² ≈ 299, which accounts for essentially the whole of the ~278× gap between their raw numbers. **Their absolute MSEs are not comparable**; only each model against itself across folds is, which is what the relative table above reports.

### 7.10 Confidence intervals: which differences survive resampling
Every metric had been a point estimate, including the gaps §3.8.4 turns on. A **paired year-block bootstrap** (2,000 replicates, resampling whole calendar years so spatial and seasonal correlation are preserved within each unit, every model scored on the same resampled years) gives:

| Model | RMSE | 95% CI |
|---|---|---|
| **XGBoost** | 9.24 | [8.55, 9.87] |
| U-Net | 11.03 | [10.25, 11.80] |
| CNN | 11.09 | [10.28, 11.86] |
| Random Forest | 10.30 | [9.37, 11.25] |

Individual intervals overlap heavily — but that is the wrong comparison. Because the bootstrap is paired, the interval on each *difference* removes the year-to-year variation common to all models:

| Comparison | ΔRMSE | 95% CI (percentile) | Verdict |
|---|---|---|---|
| Random Forest − XGBoost | +1.187 | [+0.600, +1.908] | **distinguishable** |
| CNN − XGBoost | +0.900 | [+0.561, +1.228] | **distinguishable** |
| U-Net − XGBoost | +0.866 | [+0.412, +1.281] | **distinguishable** |
| CNN − Random Forest | −0.158 | [−0.907, +0.508] | not distinguishable |
| Random Forest − U-Net | +0.193 | [−0.519, +0.970] | not distinguishable |
| CNN − U-Net | +0.035 | [−0.287, +0.388] | not distinguishable |

**XGBoost's aggregate advantage is real**: its RMSE deficit against all three others excludes zero. The comparison §3.8.4 turned on — Random Forest against XGBoost, +1.183 W m⁻² — survives at [+0.530, +1.671].

**Two consequences worth stating.** First, the deployment argument could no longer be carried on aggregate grounds: RF would have to be preferred *despite* a statistically distinguishable deficit, so the spatial-fidelity case had to carry that weight explicitly — and the next subsection shows it cannot. Second, **RF, CNN and U-Net are not distinguishable from one another on RMSE** — BCa: RF−CNN +0.155 [−0.447, +1.007], RF−U-Net +0.191 [−0.439, +1.110], CNN−U-Net +0.035 [−0.258, +0.417], all spanning zero — the ordering among those three is noise at this sample size, and any narrative ranking them should say so.

**Reconciling that with the sections that do rank them.** §7.5, §7.8 and §9 all order these three, and Table 3.3 lists them in an RMSE order (U-Net 11.03, CNN 10.14, RF 10.30) that the bootstrap says is not real. Both are correct because **they are rankings on different axes, and only one of the two axes supports a ranking**:

| Axis | RF vs CNN | CNN vs U-Net | RF vs U-Net | Ordering established? |
|---|---|---|---|---|
| Aggregate RMSE | −0.158 [−0.907, +0.508] | +0.035 [−0.287, +0.388] | +0.193 [−0.519, +0.970] | **no — none of the three** |
| Centred RMSE | −2.525 [−2.758, −2.217] | −0.299 [−0.517, −0.070] | −2.824 [−3.024, −2.486] | **yes — all three pairs** |
| Spatial R | +0.080 [+0.066, +0.095] | +0.018 [+0.006, +0.031] | +0.098 [+0.082, +0.115] | **yes — all three pairs** |

*Now measured under BCa for all six pairs rather than argued from the percentile intervals.* Centred RMSE: RF−CNN −2.658 [−2.923, −2.536], RF−U-Net −2.968 [−3.094, −2.868], CNN−U-Net −0.310 [−0.537, −0.090]. Spatial correlation: +0.080 [+0.066, +0.095], +0.098 [+0.084, +0.116], +0.018 [+0.007, +0.032]. **All six exclude zero**, so the ordering is established rather than assumed — though the CNN−U-Net gap on centred RMSE clears zero only just.

**RF > CNN > U-Net is fully established on both spatial axes, including the narrow CNN-over-U-Net margin, and is established on none of the aggregate ones.** So the rule for the whole document is: *rank these three on spatial fidelity, never on RMSE.* Every ranking §7.5, §7.8 and §9 make is a spatial-fidelity ranking — the U-Net's ±15 W m⁻² per-cell bias range, its 0.08× spectral damping, the bottleneck argument — and each survives. Table 3.3's row order is an artefact of sorting by a column that does not separate them; read the Taylor table for the ordering that holds.

Mean bias needs the paired/marginal distinction stated carefully, because an earlier version of this section drew the wrong inference from the right numbers. **Every model's *marginal* MBE interval spans zero**, including XGBoost's +1.20 [−0.14, +2.41] — those intervals are correct and worth keeping as marginal intervals. **But the *paired* RF-minus-XGBoost difference is established**: −0.953 on |MBE| with a BCa interval of [−1.499, −0.475]. Pairing cancels the year-to-year variation that moves both models together and dominates each marginal interval, which is the same argument this section makes for RMSE two paragraphs above. Concluding from two zero-spanning marginals that the difference is unestablished is precisely the fallacy §7.10 exists to refute.

### The spatial axis, tested — and the consequence for §3.8.4

The same paired bootstrap applied to the two spatial quantities §3.8.4 actually relies on:

| Comparison | Δ centred RMSE | 95% CI (percentile) | Δ spatial R | 95% CI (percentile) |
|---|---|---|---|---|
| **RF − XGBoost** | **−0.1445** | **[−0.3050, +0.0101]** | **+0.0016** | **[−0.0006, +0.0042]** |
| RF − U-Net | −2.8237 | [−3.0244, −2.4857] | +0.0980 | [+0.0822, +0.1150] |
| RF − CNN | −2.5250 | [−2.7584, −2.2169] | +0.0802 | [+0.0660, +0.0947] |
| XGBoost − U-Net | −2.6791 | [−2.8978, −2.3699] | +0.0964 | [+0.0807, +0.1139] |

**A reading caveat, and a correction it forced.** The differences above are **bootstrap resample means**; Taylor reports the **plug-in** full-sample statistic. For centred RMSE they differ: RF−XGBoost is −0.1706 plug-in against −0.1445 as a resample mean, and CNN−XGBoost 2.4873 against 2.3804. Each replicate recomputes the time-mean map from resampled years, so its centred RMSE carries sampling noise added in quadrature (≈ √(c² + n²)), and since √(a²+n²) − √(b²+n²) < a − b, replicate differences compress toward zero. Spatial correlation, bounded near 1, is barely affected — which is why both of its differences reproduce Taylor to 4 dp. The `estimate` column of `bootstrap_ci.csv` is the plug-in and matches Taylor exactly; the `mean_difference` column of `bootstrap_differences.csv` does not, and that distinction is the whole of the discrepancy.

**Naming the bias forced the obvious next step, and it reversed a verdict.** A percentile interval assumes an approximately unbiased, symmetric bootstrap distribution. For centred RMSE that assumption fails *by construction*, so the percentile interval was the wrong tool. `compute_bca_centred_rmse.py` reports BCa, which corrects for bias and skewness:

| RF − XGBoost, centred RMSE | Interval | Verdict |
|---|---|---|
| Percentile (as originally reported) | [−0.3050, +0.0101] | spans zero |
| Basic / reverse percentile | [−0.3513, −0.0362] | excludes zero |
| **BCa** | **[−0.3603, −0.0408]** | **excludes zero** |

**The composite criterion survives on BOTH of its legs.** The criterion was *mean bias plus spatial error structure*, and BCa establishes the Random Forest as better on each:

| RF − XGBoost | Plug-in | BCa 95% CI | |
|---|---|---|---|
| Mean bias, \|MBE\| | −0.953 | [−1.499, −0.475] | established |
| Centred RMSE | −0.171 | [−0.360, −0.041] | established |
| Spatial correlation | +0.0015 | [−0.0008, +0.0040] | not established — and never a leg of the criterion |

**This document has now been wrong about this twice, in the same way.** First it claimed neither leg survived, on a percentile interval that is invalid for centred RMSE. Then it claimed one leg survived, having dismissed the bias leg on *marginal* MBE intervals — the very fallacy this section refutes two paragraphs earlier. Both errors are the same habit: reaching for whichever interval was to hand rather than the one the quantity required. The paired MBE difference reduces to mean(a) − mean(b), since the truth cancels, so it is precisely the quantity a paired bootstrap is built for.

**Nothing about the deployment changes, and that is the point.** A criterion that fully survives and still loses to a categorical disqualification is a stronger result than one that half-survives, because it removes any suspicion the scorecard was arranged. The Random Forest reproduces the historical field better in both level and spatial error structure. It still cannot produce a projection to 2100, because it inverts the scenario signal (§7.11).

**What this does not touch.** RF's aggregate deficit is established *more* strongly under BCa, +1.0576 [+0.5976, +1.7684], than under the percentile interval. The two comparisons previously asserted rather than shown are also settled: CNN−XGBoost +0.9023 [+0.5264, +1.2032] and U-Net−XGBoost +0.8671 [+0.3995, +1.2729], both excluding zero under all three methods.

**The pixel-wise/shared-weight distinction is established and large.** RF and XGBoost both beat both networks on spatial error structure by margins whose intervals are nowhere near zero. Everything §7.5 and §7.8 say about the U-Net stands.

**The RF-vs-XGBoost spatial distinction is not established.** Both intervals span zero. The two pixel-wise models are indistinguishable from each other on centred RMSE and on spatial correlation.

**This was decisive for the deployment argument, and §3.8.4 was rewritten accordingly (§9).** The composite criterion rested on two axes:

| Axis | RF vs XGBoost | Status |
|---|---|---|
| Systematic offset (mean bias) | −0.953 on \|MBE\| | **ESTABLISHED** — BCa [−1.499, −0.475]. The earlier "both MBEs span zero" reasoning used *marginal* intervals, the fallacy §7.10 itself refutes |
| Spatial error structure (centred RMSE) | −0.1706 plug-in | **ESTABLISHED** — BCa [−0.3603, −0.0408]; the percentile interval that spanned zero was the wrong tool |
| *Aggregate error (RMSE)* | *+1.0581* | ***established*** — *against RF* |

Random Forest is **significantly worse on the one axis that is established, and not significantly better on either axis the deployment case invoked.** The composite criterion does not select RF over XGBoost; it fails to separate them, and the tie-break falls to the metric that does separate them, which favours XGBoost. §7.11 then removed any residual case for RF on independent grounds. **XGBoost is deployed (§9).**

### 7.11 Scenario discrimination: the deployment test that validation cannot perform
Validation measures how well a model reproduces 2011–2024. The product is a projection to 2100 under two emission scenarios, and nothing in Table 3.3 tests whether a model can tell those scenarios apart. It can be tested directly: the SSP5-8.5 minus SSP2-4.5 difference should be positive and should **grow** with lead time.

| Model | Near-term | Mid-term | Long-term | Grows? | Cells with SSP5-8.5 > SSP2-4.5 |
|---|---|---|---|---|---|
| **Random Forest** | +0.465 | +0.307 | **+0.167** | **NO — shrinks** | **71.7%** |
| XGBoost | +0.754 | +0.741 | +1.444 | yes, overall | 96.7% |
| CNN | +1.272 | +3.851 | +9.478 | yes | 99.9% |
| U-Net | +1.034 | +2.039 | +4.703 | yes | 100.0% |

**Random Forest's scenario separation shrinks as forcing grows** — the opposite of the physical expectation — and it inverts outright on the long-term change signal (+1.648 under SSP2-4.5 against +1.349 under SSP5-8.5).

**The out-of-range rates, resolved by horizon — this is the sharper form of the evidence.** Percentage of predictor values falling outside the 1985–2010 ERA5 training range, post-QC-clamp:

| Predictor | Scenario | Near | Mid | Long |
|---|---|---|---|---|
| `tas` | SSP2-4.5 | 0.32% | 0.72% | 1.34% |
| `tas` | **SSP5-8.5** | 0.36% | 3.01% | **11.48%** |
| `huss` | SSP2-4.5 | 0.31% | 1.09% | 1.37% |
| `huss` | **SSP5-8.5** | 0.49% | 2.53% | **7.65%** |
| `clt` | SSP2-4.5 | 0.30% | 0.46% | 0.52% |
| `clt` | **SSP5-8.5** | 0.61% | 1.08% | **2.03%** |
| `ps` | both | 0.00% | 0.00% | 0.00% |

**The clipping grows with lead time under SSP5-8.5 and barely grows under SSP2-4.5** — temperature from 0.36% to 11.48% against 0.32% to 1.34%. That is precisely why the Random Forest's scenario separation *shrinks* with horizon rather than merely being too small: the higher-emission pathway is progressively more clipped, so its response saturates while the lower pathway's does not.

**Cloud fraction is included because its absence would be conspicuous, and the answer is two-sided.** `clt` is the dominant predictor by importance (0.33 / 0.39 / 0.47) and its out-of-range rate *is* differentially higher under SSP5-8.5 — 2.03% against 0.52% at long term, a factor of 3.9 — so it supports the mechanism. But its absolute rate is low: at long term the dominant *source of clipping* is temperature at 11.48% and humidity at 7.65%, not cloud at 2.03%. **The dominant predictor is not the dominant source of extrapolation failure**, and saying so is more accurate than implying the two coincide.

**The mechanism is tree extrapolation.** A tree predicts a constant beyond the range it was trained on, so its response saturates once predictors leave the training envelope. Under SSP5-8.5 they increasingly do: temperature falls outside the 1985–2010 range **4.95%** of the time against 0.79% under SSP2-4.5, specific humidity 3.56% against 0.92%. The scenario that should produce the larger response is precisely the one where a tree's response is most clipped. XGBoost shares the limitation and shows it mildly; the neural models extrapolate through their linear layers and do not.

This is the concrete form of the stationarity caveat in §14.2, and it matters more for a projection product than any validation metric: a suitability map that cannot distinguish emission pathways fails at the task it exists for.

**A caution against over-reading it in the other direction.** All four scenario separations are small beside σ_DS ≈ 10 W m⁻². The CNN's +9.478 is not obviously *better* for being larger — it is comparable to the model's own error, and its long-term change of **+26.3 W m⁻²** is implausibly large for a 75-year irradiance trend. *(Both figures were wrong here until the table audit: the separation was the pre-retrain +9.643, and the +16.6 W m⁻² quoted as the CNN's change is the U-Net's — the CNN's was always larger, so the point held for the wrong reason.)* The honest reading is that RF is disqualified on this axis, XGBoost is adequate, and the neural models' larger responses are unverifiable rather than demonstrably right.

### 7.12 §3.4.3's quality control did not exist, and the null result is the interesting part

**What was claimed and what was there.** §3.4.3 stated that physically unrealistic CMIP6 values — negative rsds, cloud fraction above 100%, negative humidity — "were replaced with the model climatology using a 31-day centred moving window". No such step existed anywhere in the pipeline. The section also said per-cell flag counts "were not separately tabulated".

**The violations are real.** Equidistant CDF matching is tail-sensitive: matching a *bounded* variable against an empirical distribution drives a fraction of corrected values past the bound.

| Variable | Flagged | Of total | Worst excursion | Cells affected |
|---|---|---|---|---|
| `clt` | 38,843 | 0.6406% | **−5.93%** cloud fraction | **751 / 957** |
| `od550aer` | 107 | 0.0018% | −0.0046 optical depth | 69 / 957 |

A negative cloud fraction is not a small error in kind, and it was reaching models trained on a strictly non-negative predictor. 751 of 957 coarse cells are affected at least once, up to 277 time steps at a single cell — spread across the domain, not confined to a corner.

**The ERA5 half, by contrast, is exactly clean.** The clear-sky index is clipped to [0, 1.1], and the clip never binds: across 1,794,312 training and 966,168 validation values the maximum is 0.6323. **The flag count is identically zero at every cell** — so §3.4.3's appendix table is now a determinate result rather than an untabulated unknown.

**The cause, determined rather than guessed.** Not interpolation overshoot — the regridding before EDCM uses `method="linear"`, a convex combination, which cannot exceed the input range. The cause is that **EDCM is an additive correction with no range constraint**: `edcdf_matching_1d` returns `sim_fut + F⁻¹_obs,h(p) − F⁻¹_sim,h(p)`, and both inverse CDFs are built with `fill_value="extrapolate"`. Where the GCM's cloud fraction is already near the bottom of its distribution and the ERA5-minus-GCM quantile offset at that probability is negative and larger in magnitude, the sum goes below zero. Nothing in the function clamps the result to the variable's physical range.

Confirmed by inspection: in `ACCESS-CM2_ssp245`, the negative values at the worst-affected cell are **ranks 1–33 of that cell's 900 months** — the extreme low tail, exactly as an additive offset predicts, and not scattered at spatial gradients as an interpolation artefact would be.

**The fix.** `apply_qc_bounds.py` enforces the bounds by **truncation**, not by the climatological replacement §3.4.3 specified. The violations are small overshoots at the distribution tails; truncation preserves the corrected value everywhere else, whereas climatological replacement would discard a correctly corrected value to repair a boundary artefact. §3.4.3 has been rewritten to describe what the code does. Originals are preserved in `cmip6_bias_corrected_preqc/`, and a regression test asserts no predictor is physically impossible — verified to fail against those originals with 11 offenders.

**Impact on every published number: essentially none — and *why* is a genuine corroboration of §7.11.** All four models' projections, all four MME aggregations and the decomposition were regenerated from the corrected inputs.

| Quantity | Pre-QC | Post-QC |
|---|---|---|
| RF long-term change (SSP2-4.5 / SSP5-8.5) | +1.648 / +1.349 | **+1.648 / +1.349** (identical) |
| XGBoost long-term change | +4.081 / +4.771 | **+4.081 / +4.771** (identical) |
| CNN long-term change | +8.309 / +16.566 | +8.307 / +16.558 |
| U-Net long-term change | +5.432 / +8.995 | +5.432 / +8.988 |
| Scenario ordering, cells correct | RF 71.7%, XGB 96.7% | **unchanged** |
| σ_arch share, long term (deployed) | 27.20% | 35.40% |
| C1 transfer, XGBoost ensemble | 4.8226 | **4.8226** (identical to 4 dp) |

**Scope, because the scope is what makes this evidence.** EDCM is applied to CMIP6 only, so the ERA5 training and validation pipeline was never touched — ERA5 `clt` has zero negative values, minimum 0.1123% — and Table 3.3 is unaffected *by construction*, not by any property of the models. The comparison below is on the **CMIP6-driven projections and the transfer test**, which is where the defect actually lived.

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

### 7.15 The clear-sky denominator is on the wrong temporal basis

Found by acquiring SARAH, which ships its own clear-sky field (`SISC`) beside all-sky `SIS` on the same grid and the same monthly basis, making the comparison direct rather than inferential.

`compute_finegrid_clearsky_ghi.py` builds the clear-sky ceiling as the mean over **thirteen daytime hours**, `np.linspace(6.0, 18.0, 13)`. The ERA5 `ssrd` it divides is a **24-hour monthly mean**. Numerator and denominator are therefore on different temporal bases and the quotient is not a clear-sky index.

| Month | PVLIB ceiling | SARAH `SISC` | ratio | SARAH CSI |
|---|---|---|---|---|
| Jan | 598.11 | 349.81 | 1.710 | 0.740 |
| Apr | 464.63 | 270.96 | 1.715 | 0.860 |
| Aug | 442.86 | 252.17 | 1.756 | 0.947 |
| Dec | 599.84 | 351.03 | 1.709 | 0.735 |

Across all twelve months the ratio is **1.724, sd 0.0326, range 1.670–1.780** (`compare_sarah_clearsky.py` → `sarah_clearsky_comparison.csv`).

**The flatness is the evidence.** Two clear-sky *models* disagreeing would diverge seasonally, through air mass and turbidity. A near-constant multiplicative offset is what an averaging-window mismatch produces, and 06:00–18:00 against 24 hours predicts roughly a factor of two. SARAH's own CSI meanwhile shows the seasonality that ought to be there — 0.74 in the January wet season rising to 0.95 in the August dry season — which a field capped at 0.632 cannot express.

**What this does not affect: anything in the product.** `csi_to_ghi` multiplies the same field back, an exact inverse to 1.14 × 10⁻¹³ W m⁻². Table 3.3, the projections, the uncertainty decomposition and the deployment argument are all untouched, and nothing needs recomputing. The models learn a well-defined quantity; it is only mis-named.

**What it does affect** is two statements. That the clear-sky specification "fixes the physical meaning of the intermediate CSI" — it does not. And the use of the 0.246–0.632 range as evidence about the clip alone, when it is at least as much evidence about the denominator. An examiner who knows solar resource is likely to ask why a clear-sky index never exceeds 0.63; the answer is that the normalisation is a fixed rescaling that cancels exactly, so the intermediate's absolute level carries no meaning.

**Not yet done:** rebuilding the ceiling on a 24-hour basis. It would change no reported number, since it cancels, so it is cosmetic for the product and worth doing only when §3.5.4 is next revised. **SARAH's `SISC` is now the better reference** if it is rebuilt.

### 7.16 ERA5 sits 3.0% below SARAH, and the gap has a seasonal cycle

The first quantity in this project measured against something that is not ERA5. Full record, 479 months over 1985–2024, all 5,751 cells (`compare_sarah_era5.py` → `sarah_era5_comparison.nc`).

| | |
|---|---|
| ERA5-derived mean | 237.92 W m⁻² |
| SARAH mean | 245.27 W m⁻² |
| **bias (ERA5 − SARAH)** | **−7.35 W m⁻² (−3.0%)** |
| RMSE | 13.37 |
| spatial-mean correlation | 0.9800 |
| bias range across cells | −20.88 to +14.40 W m⁻² |

**An earlier version of this section reported 13% low, from January 2021 alone.** That month was unrepresentative; the whole-record figure is 3.0%. The lesson is the ordinary one about single-month statistics, and it is recorded rather than quietly overwritten.

**The gap is seasonal, not a constant offset.** The ERA5/SARAH ratio runs 0.94–0.96 from January to July, rises through August and September, and **crosses above one in October and November** (1.011, 1.025). ERA5 understates the wet-season resource and slightly overstates the late dry season.

**Two coverage points, both about grid geometry rather than data.** SARAH is cell-centred (25.025, …) and the target grid is node-centred (25.0, …), so the outer ring of 300 cells falls 0.025° beyond the range of SARAH cell centres and linear interpolation will not extrapolate. That is not missing data — the SARAH cell centred at 25.025 spans 25.00–25.05 and physically contains the target node — so those cells are **clamped to the nearest interior value**, which for a cell-centred field is the correct answer rather than an approximation. Half a degree of margin on the CM SAF order would have avoided the question. Separately, **1985-02 is dropped**: 23.5% NaN at native resolution, built from about 21 daily averages, with real interior gaps. Left in, that one month dragged apparent whole-record coverage from 94.8% to 72.5% through an any-NaN test.

### 7.17 The present-day suitability layer must be SARAH, and the reason is not the bias

§3.9 needs a present-day irradiance layer, and the choice looked presentational. The argument for ERA5 is that the projections come from an ERA5-trained chain, so present-to-future *change* must stay inside one measurement system or it differences two instruments and calls the difference climate. That argument is correct — **for change**.

It does not carry to a present-day siting map, and the expected escape route turns out to be closed. **The offset is spatially near-uniform** — ratio sd 0.0142, range 0.919 to 1.067 — which suggested it would cancel under §3.9.3's min-max normalisation and leave the ranking untouched. **It does not.** Min-max rescales by the *range*, so it amplifies exactly the small spatial structure the ratio sd conceals:

**SUPERSEDED — never written to disk, and they do not reproduce on the current layers (§12j).** Kept as the record of what the decision was actually made on:

| | |
|---|---|
| Spearman ρ of the min-max GHI score | **0.860** |
| mean absolute score difference | 0.170 |
| cells crossing the 0.75 tier threshold | **1,485 (27.2%)** |
| cells crossing 0.60 | 2,503 (45.9%) |

**CURRENT**, from `layer_choice_sensitivity.csv`, on the 2,386 assessed cells:

| | |
|---|---|
| Spearman ρ of the min-max GHI score | **0.875** |
| mean absolute score difference | 0.087 |
| cells crossing the 0.75 tier threshold | **371 (15.5%)** |
| cells crossing 0.60 | 213 (8.9%) |

About one assessed cell in six changes tier on the irradiance criterion alone — weaker than the "nearly half the domain" the superseded figures supported, and still enough to make the layer choice consequential rather than presentational. **So the layer choice is a methodological decision, not a presentational one**, and defaulting to ERA5 without measuring it would have put an unexamined choice under every suitability map.

**Resolution: SARAH for the present-day map** — an independent retrieval at 0.05°, finer than the target grid, over a region where reanalysis is weakest. **ERA5-derived for present-to-future change**, so the change signal stays within one measurement system. `build_suitability_layers.py` writes both and `compute_suitability.py` prefers SARAH.

### 7.18 The U-Net's damping is a dropout artefact, and both neural models select on the test set

Two findings from `optimise_unet.py`, a controlled sweep that fits on 1985–2004, selects on 2005–2010, and touches the withheld record once per variant.

**Dropout costs accuracy. It has NOT been shown to cause the damping — that attribution is WITHDRAWN (§12h).** The deployed configuration applies `Dropout2d(0.3)` eight times per encoder/decoder level. Removing it, within the sweep:

| | sweep baseline (dropout 0.3) | dropout removed |
|---|---|---|
| held-out RMSE | 10.98 | **8.63** |
| spatial correlation | 0.894 | **0.970** |
| centred RMSE | 3.59 | **1.97** |
| spectral ratio (k≥10) | 1.24 | **0.771** |

**The left column is the sweep's baseline, not the deployed model**, which is 11.03 / 0.9028 / 3.479 and has a spectral ratio of 0.090 at k≥11. The table previously labelled that column "deployed" and put **0.085** in its spectral row — a pre-retrain deployed-model figure sitting beside three sweep figures.

**Why the attribution fails.** The sweep's baseline carries the same dropout rate as the deployed model and shows a spectral ratio of **1.24 — an excess of fine-scale power, not a deficit**. It never reproduced the damping it was built to explain, so it cannot isolate its cause. Removing dropout moves the ratio to 0.771, which is no closer to unity: 0.23 against 0.24 in absolute deviation.

**And the supporting claim was backwards.** "Varying the gradient penalty barely moves the ratio, so the smoothness-prior hypothesis is refuted" fails on both halves: the penalty variants span 1.01 to 1.24, and removing the penalty gives **1.006 — the closest to unity of any variant tested**, which points toward a smoothness prior rather than away from it. Three other results from the same sweep: a per-cell climatology gives the best spatial correlation of any variant (0.993) and the worst RMSE (18.68), because it hands the network the pattern and costs it the level; reducing capacity to 1.1M parameters costs little; and every combination containing the per-cell climatology inherits its error.

**`train_unet_downscaler.py` and `train_cnn_downscaler.py` both select their saved checkpoint on `ml_validation_dataset.nc`** — the withheld 2011–2024 record. The U-Net additionally early-stops and schedules its learning rate on it. This is §6.12's defect, found and fixed for XGBoost, still live in the other two. The deployed U-Net's log shows the validation curve bouncing between 0.109 and 0.140 with the checkpoint saved at the 0.1085 minimum: selecting a favourable fluctuation from ~28 draws.

**Consequence.** **RESOLVED (§6.13).** Both were retrained under honest selection; the bias was +0.92 W m⁻² for the U-Net and +0.95 for the CNN. The deployment decision is unaffected — it turned on scenario discrimination, not RMSE — but the four-way comparison needs the caveat, and Chapter 4 §4.2 now carries it.

**Done (§6.13).** Both were retrained under honest selection and the whole downstream cascade regenerated. The measured bias was +0.92 W m⁻² for the U-Net and +0.95 for the CNN.

### 7.6 Cross-validation selected worse hyperparameters
A 150-cell subsampled CV search picked configurations for both tree models that underperformed the untuned defaults on the full 5,751-cell holdout. Defaults retained. The search's own scores gave no warning.

---

## 8. Current results

### Table 3.3 — validation 2011–2024 (W m⁻²)

| Model | RMSE | MAE | Pearson R | MBE | SS vs climatology | R² |
|---|---|---|---|---|---|---|
| **XGBoost (deployed)** | **9.03** | **6.64** | **0.9721** | +1.42 | **0.5264** | **0.9436** |
| CNN | 10.04 | 7.51 | 0.9667 | +2.11 | 0.4738 | 0.9304 |
| Random Forest | 10.22 | 7.65 | 0.9639 | +0.22 | 0.4644 | 0.9278 |
| U-Net | 12.73 | 9.47 | 0.9447 | +1.63 | 0.3329 | 0.8881 |

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
| CNN | 0.9228 | 0.9368 | 3.076 | 38.5% |
| U-Net | 0.8847 | 0.9061 | 3.723 | 46.7% |

The pixel-wise models reproduce the spatial climatology far more faithfully than the shared-weight ones. **The gap between the two pixel-wise models splits, and §7.10 settles it under BCa.** On **centred RMSE** the Random Forest's advantage **is established**: −0.171 [−0.360, −0.041]. On **spatial correlation** it is not: +0.0015 [−0.0008, +0.0040]. So read the bold on the RF row as an established advantage on centred RMSE and a point estimate only on spatial R. An earlier version of this note said both spanned zero, on percentile intervals that are invalid for centred RMSE — see §7.10.

### Uncertainty decomposition — four components

Columns are **RMS over the domain**, so they square and sum to σ_total exactly and the
percentages follow from them. An earlier version tabulated the spatial *mean*, which could not be
reconciled with the shares — except for σ_DS, which is spatially constant and so did reconcile,
which is what made the mismatch confusing.

| Model | Period | σ_GCM | σ_SSP | σ_arch | σ_DS | σ_total | % var DS | % var arch |
|---|---|---|---|---|---|---|---|---|
| **XGB (deployed)** | Near-term | 1.74 | 0.49 | 2.14 | 1.47 | 3.16 | 21.59% | 45.80% |
| **XGB (deployed)** | Mid-term | 2.09 | 0.55 | 3.11 | 1.47 | 4.06 | 13.09% | 58.64% |
| **XGB (deployed)** | Long-term | 2.56 | 0.99 | 4.54 | 1.47 | 5.50 | 7.11% | **68.04%** |
| U-Net | Near-term | 1.64 | 0.55 | 2.14 | 3.68 | 4.59 | 64.12% | 21.68% |
| U-Net | Mid-term | 2.33 | 0.92 | 3.11 | 3.68 | 5.42 | 45.92% | 32.79% |
| U-Net | Long-term | 3.01 | 1.73 | 4.54 | 3.68 | 6.79 | 29.26% | **44.65%** |
| RF | Near-term | 1.10 | 0.28 | 2.14 | 0.80 | 2.55 | 9.73% | 70.28% |
| RF | Mid-term | 1.31 | 0.20 | 3.11 | 0.80 | 3.47 | 5.26% | 80.21% |
| RF | Long-term | 1.55 | 0.28 | 4.54 | 0.80 | 4.87 | 2.67% | **86.85%** |

The Random Forest rows are retained because §3.8.4 deployed it until the bootstrap (§7.10) and the
scenario test (§7.11) reversed that choice; keeping them makes the reversal auditable.

**σ_arch rests on all four benchmarked architectures (n = 4)** — RF, XGBoost, CNN, U-Net — which exceeds σ_GCM's n = 3, so the comparison between them no longer favours the GCM term on sample size. Reaching n = 4 required persisting XGBoost (~400 MB at 200 fixed rounds, an order of magnitude below RF's ~4 GB) and projecting it — the same persisted models that now serve the deployment.

**By 2076–2100 architecture choice accounts for 35.40% of projection variance against the GCM's 4.49%** — a factor of about eight. *This rose from 27.18% / 5.06% when the CNN and U-Net were retrained under honest selection (§6.13): correcting their inflated accuracy widened the spread between architectures, which is the quantity σ_arch measures.*

**The attack on this, and the answer.** σ_arch is the spread across *all four* architectures, including the two this study argues against. Removing them shrinks it:

| Members | σ_arch | arch % var | GCM % var |
|---|---|---|---|
| **All four, as reported** | **7.12** | **35.40%** | 4.49% |
| Drop RF (fails the scenario screen) | 4.89 | 20.53% | 5.52% |
| Drop RF and CNN (CNN change implausible) | 2.56 | 6.63% | 6.49% |

**Excluding architectures on grounds of implausibility is circular:** it uses an unvalidated judgement about the future to shrink an estimate of how uncertain the future is. Nothing here validates projection magnitude — which is precisely why §7.11 refuses to *rank* the models that pass the scenario screen by the size of their response. The same refusal must apply to σ_arch, or the two positions contradict each other. The full spread is both the conservative and the only non-circular choice.

**And the finding should be stated in the form the evidence supports.** Not "architecture choice contributes 27% and here is the right architecture", but **"a single-architecture study would have reported zero architecture uncertainty and been wrong by 27%"** — which holds whichever architecture that study picked. The sensitivity is volunteered rather than hidden: 20.5% across the three that pass the screen, still four times the GCM term; 6.6% across the two whose projections are credible, comparable to it. The estimate has been stable as members were added (14.2% at n=2, 27.8% at n=3, 24–27% at n=4), which is itself reassuring: the conclusion is not an artefact of which two models happened to be compared.

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
| **XGBoost (deployed)** | **9.03** | **6.64** | **0.9721** | +1.42 | **0.5264** | **0.9436** |
| Random Forest | +1.65 W m⁻² | +1.35 W m⁻² |
| U-Net | +5.43 W m⁻² | +8.99 W m⁻² |
| CNN | +8.31 W m⁻² | +16.56 W m⁻² |

The magnitude spans a factor of five to twelve (8.31/1.65 = 5.04, 16.56/1.35 = 12.27) across architectures, which is what σ_arch measures.
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

1. **Both legs of the composite criterion survive; they are outweighed, not absent (§7.10).** The criterion was *mean bias plus spatial error structure*, and under BCa the Random Forest is established better on each: **mean bias by −0.953 [−1.499, −0.475]** on |MBE|, the larger of the two, and **centred RMSE by −0.171 [−0.360, −0.041]**. Spatial *correlation* is the one axis where it is not established, +0.0015 [−0.0008, +0.0040], and that was never a leg of the criterion. This section has been wrong about this twice — first claiming neither leg survived, on a percentile interval invalid for centred RMSE; then claiming one did, having dismissed the bias leg on *marginal* intervals. Both errors were the same habit: reaching for whichever interval was to hand rather than the one the quantity required.
   **Grant the whole criterion, and answer it on relevance rather than size.** The Random Forest reproduces the historical field better in both its level and its spatial error structure. It still cannot produce the deliverable, which is a projection to 2100, because it inverts the scenario signal (§7.11) — a **categorical disqualification, not a magnitude trade**. Resist the shortcut of comparing 0.171 against +1.058 and calling it six times smaller: that treats a centred error on a climatological field as commensurable with a space-time aggregate, which is exactly what the composite criterion denied and what §7.14 and §8 warn against elsewhere. Secondarily, and with both quantities named as different kinds: 0.171 W m⁻² of centred error against +1.058 [+0.598, +1.768] of space-time aggregate error, plus four losses from four rolling-origin folds (§7.9).
2. **RF cannot separate the emission scenarios (§7.11).** Its SSP5-8.5 minus SSP2-4.5 separation *shrinks* with lead time (+0.465 → +0.307 → +0.167), it inverts on the long-term change signal, and only 71.7% of cells order the pathways correctly, against XGBoost's 96.7%. The mechanism is tree extrapolation: temperature leaves the 1985–2010 training range 4.95% of the time under SSP5-8.5 against 0.79% under SSP2-4.5, and a tree's prediction saturates outside the range it was fitted on.

3. **The ranking is not an artefact of the chosen split (§7.9).** This is the consequence of the rolling-origin result, and it belongs here rather than only in §7. The entire deployment argument rests on a comparison measured over one 1985–2010 / 2011–2024 division of the record. A rolling-origin design, refitting on an expanding window and testing on the block immediately after it, puts XGBoost ahead in **all four** forward-in-time folds by margins of 0.89 to 2.26 W m⁻². The preference is therefore a property of the models rather than of the validation period, and no fold reverses it. The negative form matters more than the positive one: had the ranking flipped between folds, *any* selection rule — the original single-metric one included — would have been arbitrating noise, and the honest conclusion would have been that the four models are not separable at this sample size. That is precisely the conclusion §7.10 forces for RF against CNN against U-Net, whose RMSE differences are not distinguishable. It is not the conclusion for XGBoost, which is separable from all three and stays separable in every fold.

**The scorecard, stated as a scorecard rather than a clean sweep.** XGBoost is beaten on two tested axes — mean bias by 0.953 W m⁻² and centred RMSE by 0.171, both established under BCa — and it beats all three rivals on aggregate RMSE with BCa intervals excluding zero, wins all four rolling-origin folds, and is one of only three models that pass the scenario screen. It also carries one unestablished soft spot: the largest spatial-variance damping in the Taylor table, std-ratio deviation 0.0392, though every pairwise BCa interval on that metric spans zero. An earlier version of this section claimed no model was established to beat XGBoost on any axis; that was an absolute claim requiring only one counterexample, and §7.10 now supplies it. Deploying XGBoost still **returns the study to §3.8.1's original pre-registered criterion** — lowest validation RMSE — which removes the post-hoc criterion change (A10) as an attack surface rather than defending it.

**And the surviving leg is answered on relevance, not size.** The Random Forest draws a marginally better historical map. The deliverable is a projected irradiance layer to 2100, which it cannot produce because it inverts the scenario signal — a categorical disqualification rather than a magnitude trade. Comparing 0.171 against 1.058 is tempting but treats a centred error on a climatological field as commensurable with a space-time aggregate, which is the comparison §7.14 and §8 both warn against elsewhere. Keep it as a secondary remark with both quantities named.

**Why the two-axis analysis still matters.** Centred RMSE of the time-mean field and the standard deviation of the per-cell bias are *the same quantity* — expanding the centred error gives var(m − r) — verified identical to 6 decimal places. Since RMSE² = bias² + centredRMSE², the genuinely independent axes are the **systematic offset** and the **spatial error structure**. That analysis is what makes §7.10 testable, and it is what showed the criterion could not do the work asked of it. It also still holds against the shared-weight models, where the margins *are* established: RF and XGBoost both beat CNN and U-Net on centred RMSE by 2.4–2.8 W m⁻², intervals nowhere near zero. **The pixel-wise/shared-weight distinction stands; only the separation within the pixel-wise pair failed.**

**The cost of the switch, stated plainly — two items, both against XGBoost.** Its mean bias is +1.20 W m⁻² against RF's +0.25, a factor of nearly five, and RF is **established** preferable on this axis: the paired difference is −0.953 [−1.499, −0.475] under BCa. For scale, +1.20 is an eighth of σ_DS. And it transfers marginally less well from ERA5 to CMIP6 predictors: 4.82 against RF's 4.65 in climatological RMSE (§7.7), a 0.17 W m⁻² gap that follows from XGBoost fitting the ERA5 predictor distribution more tightly. Set against a scenario-discrimination failure in the product's core function, both are the better trade; a reader who weights systematic offset and transfer robustness above scenario response should prefer RF, and should then also accept §7.11's consequence — that the resulting projections cannot reliably distinguish the pathways they are labelled with.

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
| B15 σ_arch from more members | **Done — n = 4**, all benchmarked architectures. CNN projected from its existing checkpoint; XGBoost persisted (~400 MB) and projected. Exceeds σ_GCM's n = 3. Long-term variance share settled at 27.18% for the deployed XGBoost (24.12% is the Random Forest row, a relic from when RF was deployed) (14.2% at n=2, 27.8% at n=3). |
| B16 U-Net R² | **Done** — 0.9294 |
| C1 perfect prognosis | **Done** — §7.7. Required extending EDCM to emit a historical pseudo-scenario (`EDCM_INCLUDE_HISTORICAL=1`) |
| C2 power spectra | **Done** — §7.8 |
| C3 multiple temporal splits | **Done** — `compute_rolling_origin.py`, 4 expanding-origin folds. XGBoost lowest RMSE in all four; all models within a 1.4 spread relative to their own deployed-split fold. See §7.9. |
| C4 join logic | **Done** — no nearest-neighbour temporal joins survive outside the guarded, warned carry-forward inside `align_to_months` |
| D1 version control | **Done** — git initialised, 61 files committed, `data/` (16 GB) excluded; staged content scanned for credentials before committing |
| D2 regression tests | **Done** — `pytest tests/` passes; reintroducing the §6.11 lag turns it red, restoring turns it green. *No count is quoted: it changes whenever a test is added, and quoting it produced three different numbers across three documents.* |
| D3 pin environment | **Done** — `environment.yml`, 247 packages |
| D4 OpenMP workaround | **Done** — documented in README with the actual cause and the proper fix |
| E1 dashboard | **Done** — withdrawn and replaced with a retraction notice |
| E2–E6 | Carried forward below |
| Part F figures | **Done** — six PNGs in `figures/` from `make_figures.py` |

---

## 11. Chapter 3 edits

Edited at run level with `python-docx` so all **26 live Zotero citation fields survived** (162 field characters, verified before and after every edit).

"PROPOSED" dropped from the title; implemented sections converted to past tense. Factual corrections: actual retained grid (§3.2.1) · removed the xESMF/conservative-remapping claim, neither was used (§3.4.1) · areal block means not point-sampling, plus the SVF search-radius caveat (§3.5.2) · SZA computed analytically not via PVLIB (§3.5.3) · the U-Net/CNN ReLU distinction and the exact-cancellation property (§3.5.4) · upsampling factor (§3.6.1) · grid corrected 75×85=6,375 → **71×81=5,751** (§3.6.2) · dual-branch input, dropout, channel attention, pad-to-32; false "output ReLU" claim removed (§3.6.6) · never-performed Bayesian stage removed, CV-transfer caveat added (§3.6.7) · ENSO sampling corrected to **U-Net only** (§3.6.8) · Taylor stats clarified as time-mean (§3.7.2) · deployed model designated in §3.8.1 (RF at the time; now XGBoost, §9) · Table 3.2 search ranges replaced with grids actually searched. **New §3.6.9** (daily-resolution experiment) and **new §3.8.4** (deployed-model justification).

**The five-predictor correction (§3.5.1, §3.6.5, new §3.7.4).** Two claims still described the *six*-predictor configuration — configuration B of the §7.3 ablation, the circular one — as the deployed design: §3.5.1 said all six CMIP6 variables "were used directly as atmospheric predictor features" and listed "direct radiation flux (rsds)" among them, and §3.6.5 gave the CNN input as **C = 6**. The deployed models take five; `rsds` is excluded. Both are corrected, §3.5.1 now states the exclusion and why, and **new §3.7.4** carries the ablation table that justifies it. The chapter's own MCE table was renumbered **3.4 → 3.5** to make room — Table 3.3 keeps its number, which matters because `table_3_3.csv`, the brief and the interview deck all reference it. All three claims are on the guard's retired list; the ablation table is checked cell by cell, because its figures also appear in the paragraph beneath it and a whole-document substring check passes a corrupted cell.

**NSRDB removed from the study (all three chapters).** It was never acquired and is now out of scope; SARAH is the sole satellite reference. Chapter 1 ¶35, Chapter 2 §2.3.3 (heading and the whole NSRDB paragraph), Chapter 3 §3.3.3 and the §3.10 data-availability list. Two live **Sengupta et al. (2018)** citations went with it, so Chapter 2 is now 24 `ZOTERO_ITEM` fields and Chapter 3 is 25, both still balanced at 25 and 26 begin/end pairs. Their bibliography entries were deleted by hand as well — **refresh the Zotero bibliography in Word to make that authoritative**, since the reference lists are Zotero-generated and a manual deletion is only cosmetic until it is regenerated. Backups: `*_BACKUP_pre_nsrdb.docx` beside each chapter.

**Three claims that were untrue and are now disclaimed:**
1. **No satellite product was acquired at the time this was written; SARAH now is (§13).** The results reported here are validated entirely against withheld ERA5. `data/raw/` now also holds `osm/` and `worldpop/` for §3.9, but neither is a validation product and neither touches the ML stage.
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
environment.yml                            261 pinned packages (note below)
README.md                                  run order, OpenMP workaround, env vars
.gitignore                                 excludes data/ (16 GB), figures/, logs/
```

**On `environment.yml`.** A faithful `conda env export --no-builds` with the machine-specific `prefix:` line stripped, so it reproduces the environment exactly rather than a curated subset. Three groups were added while acquiring the §3.9 and SARAH data, and they are not all pipeline dependencies:

| Group | Packages | Needed to run the pipeline? |
|---|---|---|
| Geospatial | `fiona`, `geopandas`, `pyogrio` | yes, for §3.9 — and note **`fiona`'s bundled GDAL has no OSM driver while `pyogrio`'s does**, so the `.osm.pbf` must be read through pyogrio |
| Deliverables | `python-pptx`, `xlsxwriter` | yes, for `brief/build_interview_deck.py` |
| Tooling only | `markitdown` and its chain (`magika`, `onnxruntime`, `protobuf`, `flatbuffers`, `beautifulsoup4`, `soupsieve`, `markdownify`, `python-dotenv`), `defusedxml`, `eumdac`, `paho-mqtt` | no — document QA, plus an EUMETSAT Data Store client that went unused once the CM SAF WUI route was taken. Retained because the export is faithful |

`rasterio`, `pyproj`, `fiona` and `pyogrio` each bundle their own GDAL — 3.10.3, 3.9.2 and 3.12.4. All five geospatial packages import together in one process, verified, but this is the first thing to suspect if a segfault or a projection oddity appears.

Under **git** since the second audit: 61 files tracked, two commits, `data/` excluded.

**Verification:** all 6 sanity checks pass — no all-NaN/all-zero variables across 32 files; latitude ascending and consistent; no fabricated edge band; zero NaN in any projection; SVF varies; counts confirmed at 312 / 168 / 5,751.

---

## 12a. Section 3.9 suitability analysis — first end-to-end run

Both scripts now run. `build_suitability_layers.py` writes 19 layers on the 71 x 81 grid; `compute_suitability.py` does the AHP weighting, the overlay, the five-tier classification and the four-scheme sensitivity. The weights are Table 3.5's throughout; the reconstructed Saaty matrix reproduces them to within **0.008** on every criterion but is a diagnostic only, and **its consistency ratio is not a reportable result** (§12d). Irradiance layer: **SARAH** (§7.17).

**Exclusions.** One rule for every areal category — a cell is excluded when more than half its area falls in it. The first version used `> 0.5` for protected areas and water but `> 0.0` for urban, which was an accident rather than a decision; the threshold is justifiable at this resolution because a cell is ~121 km² and a utility-scale plant needs 2–5 km², so a minority of excluded land belongs in the score rather than in a disqualification. It matters: **urban at `>0%` excludes 537 cells, at `>50%` it excludes 22.**

| Category | Cells | % |
|---|---|---|
| outside Zimbabwe | 2,460 | 42.8 |
| protected area | 895 | 15.6 |
| water | 110 | 1.9 |
| urban (incl. 1 km buffer) | 22 | 0.4 |
| **slope > 15°** | **2** | **0.0** |
| **retained** | **2,386** | **41.5** |

**The 1 km urban buffer was specified in §3.9.2 and had never been implemented.** It is applied at the 300 m land-cover resolution, where 1 km is a radius of about 3.3 pixels and means something; applying it after aggregation to 0.1° would be meaningless, since a cell is 11 km across. Urban coverage rises from 0.201% to 0.802% of pixels.

**The slope exclusion does not bind, and the threshold was deliberately not retuned.** At 0.1° a cell is ~121 km², so averaging 90 m slope smooths every face: the 95th percentile of cell-mean slope inside Zimbabwe is 7.3°, and 2 cells exceed 15°. §3.9.2 claimed the exclusion "disproportionately affects the Eastern Highlands" — it does not. Lowering the threshold after seeing it barely bites would be tuning the method toward a wanted outcome, so **the chapter's claim was corrected instead of the rule**. The relief is in the data rather than absent from it: 81 cells have more than 20% of their area above 15°, and 80% of those lie east of 32°E. The Eastern Highlands are penalised through the slope *criterion* at weight 0.20, not through the exclusion.

**The classification is far more weight-sensitive than §3.9.6 anticipated.**

**SUPERSEDED — first-run values, kept as the record of that run.** Every figure in
this table predates the §3.9.3 decay resolution (§12b), which changed the road,
grid and settlement scores and therefore every scheme's classification. The
current values are below.

| Scheme | mean SI | very high | high | Cohen's κ vs primary |
|---|---|---|---|---|
| AHP (primary) | 0.571 | 21 | 986 | — |
| irradiance-dominant | 0.593 | 47 | 1,264 | 0.697 |
| infrastructure-dominant | 0.512 | 27 | 391 | 0.317 |
| **equal** | 0.417 | 2 | 40 | **−0.111** |

**CURRENT**, read from `suitability_schemes.csv`, which exists because these
numbers used to live only in a print statement and three of them went stale
inside Chapter 4 (§12e):

| Scheme | mean SI | very high | high | Cohen's κ vs primary |
|---|---|---|---|---|
| primary | 0.602 | 54 | 1,282 | — |
| irradiance-dominant | 0.617 | 111 | 1,380 | 0.725 |
| infrastructure-dominant | 0.569 | 93 | 783 | 0.476 |
| **equal** | 0.467 | 3 | 144 | **-0.091** |

**Only 42 cells (1.8% of retained) are very high or high under all four schemes; 90.8% change tier under at least one.** Equal weighting agrees with the AHP classification *worse than chance*. §3.9.6 was written to measure this and did not anticipate the answer being this stark. **The defensible output is the 42-cell robust set, not the headline five-tier map** — which is exactly what §3.9.6's "robustly suitable" designation exists to produce, and it should be presented that way in Chapter 4 rather than as a caveat to a map.

**HydroSHEDS is now held and wired in, and it changes one cell.** HydroRIVERS v10 (Africa, 103 MB) gives 40,016 reaches in the box, 3,049 of flow order ≤ 5. A 200 m corridor — 100 m either side of those centrelines — covers **0.239% of the mean cell, at most 4.76%**, so under the areal-majority rule it excludes nothing on its own and moves the water total from 110 to 111. That is the honest outcome and it makes §3.9.2 true rather than aspirational: at 0.1° a 200 m corridor is 1.8% of a cell width, so riparian exclusion is a site-selection constraint, not a screening one. Same shape of finding as the slope exclusion, and for the same reason — the rule is sound at its native scale and cannot bind at 11 km.

*The download needed a User-Agent header: `data.hydrosheds.org` answers HEAD but returns 403 to `urllib`'s default UA while serving the identical URL to curl. `fetch()` now identifies itself, which helps every dataset in that script.*

**The maps.** `make_suitability_maps.py` writes four figures: `07_suitability_criteria` (seven layers plus the exclusion mask), `08_suitability_primary` (SI and the five tiers), `09_suitability_schemes` (four weightings and the robust set), `10_suitability_periods` (SI by period and ΔSI). Excluded land is drawn in flat grey rather than on the score ramp — putting exclusions on the same colour scale as poor sites is the commonest way a suitability map misleads.

The pattern is geographically coherent: high suitability follows the central watershed where the Harare–Bulawayo road and grid corridors run, with the Eastern Highlands and the south-east scoring low. **Best cell SI 0.838 at 29.8°E, 18.9°S** — the Kadoma–Chegutu stretch of that corridor — and the top twelve cluster along the central-western axis between 26.2° and 30.1°E.

### The future-period maps are weaker than they look, and should be labelled as such

All six projections improve **2,386 of 2,386 retained cells**. Not most — every one, in every scenario. That is not a bug, and that is the problem: **only the GHI layer varies by period.** Slope, land cover, roads, grid, settlements and population are all frozen at present values, so the change reduces to

    ΔSI = 0.358 × Δ(GHI score)

and since projected irradiance rises almost everywhere, suitability rises everywhere. The future map carries no information the irradiance projection did not already contain, rescaled by the GHI weight.

| Period | mean GHI | mean SI | very high | high |
|---|---|---|---|---|
| present (SARAH) | 242.83 | 0.5712 | 21 | 986 |
| present (ERA5 basis) | 235.75 | 0.5136 | 2 | 370 |
| SSP2-4.5 near | 240.48 | 0.5521 | 14 | 703 |
| SSP2-4.5 long | 244.87 | 0.5878 | 37 | 1,140 |
| SSP5-8.5 near | 241.37 | 0.5594 | 20 | 780 |
| SSP5-8.5 long | 246.83 | 0.6038 | 65 | 1,274 |

Two consequences for Chapter 4. **The rise in "very high" counts from 21 to 65 is threshold crossing, not relocation** — the ordering of cells barely changes, everything drifts upward together. And presenting these as "where Zimbabwe should site solar in 2100" would overclaim, because it assumes the 2100 grid, road network and population are those of 2020, in a country expected to add substantially to all three. **The defensible claim is how much the resource improves at today's viable sites, not where tomorrow's best sites will be** — the second question needs infrastructure and population scenarios this study does not have.

Note also the present row: **SI 0.5712 on SARAH against 0.5136 on the ERA5 basis, with 21 very-high cells against 2.** The §7.17 layer decision shows up directly in the headline numbers, which is further confirmation it was consequential rather than cosmetic.

All periods are standardised on the **present** min-max range rather than each on its own, deliberately: rescaling each period by its own range would hide the change entirely, since a uniformly brighter future would normalise back to identical scores.

**One scoping fact worth stating in Chapter 4.** Only **57.2%** of the rectangular analysis box lies inside Zimbabwe; the rest is Zambia, Mozambique and Botswana. The effective domain is ~3,291 cells, not 5,751, which is why "outside Zimbabwe" is the largest single exclusion.

---

## 12b. Document audit

A full cross-check of Chapters 1–4, PROJECT_STATUS, UPDATE_BRIEF and the brief against the canonical CSVs and against the code. Every headline value agrees across every document — XGBoost RMSE 9.24, RF 10.30, U-Net 10.11, CNN 10.14, skill 0.5155, R 0.9707, architecture variance 27.18%, 42 robust cells — and §3.9's full specification matches the implementation exactly: all seven Table 3.5 weights, both reference distances, all four tier boundaries, the 15° slope threshold, the 1 km urban buffer, and the five land-cover scores. What follows is what did not check out.

**§3.9.3's decay function is ambiguous, and it moves the headline.** The section prints `Score = e^(−d/d_ref)` as an equation object, and separately states that the score falls below 0.14 beyond d_ref. These disagree: e⁻¹ = 0.368. The implementation follows the 0.14 statement, which needs a factor of two in the exponent. Under the formula as printed:

| | `e^(−2d/d_ref)` (implemented) | `e^(−d/d_ref)` (as printed) |
|---|---|---|
| robust set | **42 cells (1.8%)** | **145 cells (6.1%)** |
| very high | 21 | 54 |
| grid proximity factor | 12.5× | 6.1× |
| weight-sensitive | 90.8% | 87.8% |

**RESOLVED in favour of the equation as printed.** §3.9.3's prose now states 0.37 at d_ref and below 0.14 at twice it, `DECAY_EXPONENT_FACTOR` is 1.0, and every downstream number has been regenerated. The grounds: a site 10 km from an existing line is routinely connectable for utility-scale development and should not score as though it were remote. **The robust set is therefore 145 cells, not 42**, and Chapters 4 and 5 use that throughout.

**Chapter 4 promised citations it does not contain.** Its closing note said the bracketed citations needed converting to Zotero fields. There are none: 0 citations, 0 `et al.`, 0 figure references, against 10 figures that exist. The note now says so plainly instead of implying the work is done.

**An asserted mechanism, now backed.** §4.7 said the irradiance rise "is consistent with a projected reduction in cloud cover" without checking. It checks out — domain-mean cloud fraction falls from 38.87% (1985–2010) to 33.85% under SSP2-4.5 and 31.26% under SSP5-8.5 by 2076–2100, larger under the higher pathway and monotonic with lead time — and the numbers are now in the text rather than the assertion.

**Three overclaims in Chapter 1, corrected.** The climatology was dated 1985–**2025**; the record ends 2024. "A validated high-resolution climatology" is now "a 0.1° climatology, validated against a withheld ERA5 record", since §7.2 establishes it resolves nothing below 0.25° and §4.9 establishes the validation is not independent. And §1.5's undertaking to deliver priority zones "with quantified uncertainty" is not met — the analysis delivers a weighting sensitivity analysis and a robust subset, and no uncertainty is propagated onto the index from §4.7, §4.6 or the infrastructure layers. Chapter 1 now says what is delivered; §4.9 records the shortfall.

**One claim in Chapter 2 was refuted by the results.** §2.4 said higher downscaling fidelity "confirm[s] that the downscaling quality achieved in this study will propagate directly into the suitability outputs". §4.8 finds it largely does not: irradiance separates the robust set from the domain by 1.4%, grid distance by a factor of 12.5. The sentence now frames it as the empirical question it is and points to the answer.

**A code comment that misdescribed its own function.** `decay()`'s docstring claimed the score is 0.135 at 2·d_ref; the implementation gives 0.135 at d_ref and 0.018 at 2·d_ref. Fixed, with the ambiguity documented at the call site.

**Not defects, checked and cleared.** Chapter 3's "missing" formulas are OMML equation objects that `python-docx` cannot read, not gaps. The super-resolution framing in Chapters 1 and 2 describes the literature, not this study. And `clt` is genuinely the dominant predictor on all four importance measures (0.333 / 0.395 / 0.470 / 0.178), though the margin on XGBoost cover is slim.

---

## 12c. Both document guards could skip silently, and one did

Running the three guards under the wrong interpreter (`/opt/anaconda3/bin/python`
rather than the `climate_stack` environment) produced this:

```
SKIPPED - chapter not found at:
  .../CR_Madukwe_Chapter3_Final.docx
```

The chapter was there. What was missing was `python-docx`. `load()` returned the
same `None, None, None` for an absent dependency as for an absent file, and
`main()` printed the file-not-found message for both and **exited 0**. A guard
that reports a clean bill of health for a document it never opened is worse than
no guard, because the exit status is what a person or a hook actually reads.

`check_brief_consistency.py` had the same defect in a quieter form: without
`python-pptx`, `deck_prose()` returned `""`, so every deck assertion passed
vacuously and the summary still printed *"Every checked table cell matches the
canonical evaluation CSVs."* Nothing announced that the fourth guarded document
had not been read.

Both are fixed. The two causes are now distinguished and named, a missing
dependency exits **1** while a genuinely absent document still exits 0, and the
brief guard prints an explicit warning naming the interpreter when the deck was
not checked. Verified both ways: under `climate_stack` all three guards pass and
exit 0; under base, the Chapter 3 guard now reports the real cause and exits 1.

**Everything reported in Section 12b and since was produced under
`climate_stack`, where `docx`, `pptx`, `pandas` and `numpy` are all present, so
no previously reported pass was one of these vacuous ones.**

## 12d. The AHP consistency ratio was a fabricated validation statistic

Chapter 4 Section 4.8 stated:

> The consistency ratio of the pairwise comparison matrix is 0.0076, below the
> 0.10 acceptability threshold.

That number came from `compute_suitability.PAIRWISE`, a Saaty matrix **fitted to
reproduce Table 3.5's weights**. Its consistency ratio measures how well the fit
succeeded, not whether the researcher's judgements were coherent. Publishing it
as the thesis's CR is the same circularity as training on a transform of the
target: the quantity offered as validation was derived from the answer it claims
to validate. The tell was in the number itself - 0.0076 is an order of magnitude
below what a real elicitation produces, typically 0.03 to 0.08.

Chapter 3 paragraph 180 was careful: it says a CR "is computed and verified to be
below 0.10" without quoting a value. Chapter 4 invented the value. **Chapter 4 no
longer quotes any CR**; it states that none is quotable, says why, and points to
Section 4.8.1, where the weighting sensitivity analysis answers the question a CR
is meant to answer - and answers it far less flatteringly: **87.8% of assessed
cells change suitability tier under a defensible reweighting.** A CR of 0.008
printed beside that figure would have been the more reassuring of the two and the
less true.

**Nothing downstream moved, and this was checked rather than assumed.** Every
scheme weights from `PRIMARY_WEIGHTS` (Table 3.5); `ahp()`'s vector is printed and
discarded. Re-running `compute_suitability.py` leaves both suitability CSVs
byte-identical, and a token-level diff of the rebuilt Chapter 4 shows the only
numbers removed are `0.0076` and its `0.10` threshold.

**Guarded.** `PAIRWISE_IS_ELICITED = False` gates the diagnostic, which now
prints "NOT REPORTABLE" rather than "acceptable", and
`test_reconstructed_ahp_cr_is_never_published` fails if either chapter generator
hardcodes the value or quotes a consistency ratio. Injection-tested: restoring
the old sentence fails the test on both the value and the phrase.

**Chapter 3 paragraph 180 - RESOLVED by softening, on the author's instruction**
("soften ¶180, I don't have the matrix"). It previously claimed the matrix was
"completed by the researcher in consultation with the thesis supervisors" and its
CR "computed and verified to be below the AHP acceptability threshold of 0.10,
confirming that the weight assignments are internally consistent". Both halves
were unevidenced once the matrix could not be produced. Section 3.9.4 now states
that the weights are assigned by the researcher, reflect the National Renewable
Energy Policy priorities, follow the AHP's ordinal logic, and that **no formal
pairwise comparison matrix was elicited or retained and no consistency ratio is
therefore reported** - then points at Section 3.9.6, whose four-scheme
robustness test is the claim the chapter can actually support.

**The heading went with it.** Section 3.9.4 was titled "Criterion Weighting Using
AHP", which would have advertised, one line above the new paragraph, the
procedure that paragraph says was not performed. It is now "Criterion Weighting".
Section numbering is untouched and Chapter 3 carries no TOC field, so nothing
needs refreshing in Word.

**The label was kept briefly and then dropped.** Table 3.5's caption, that table's
"Initial AHP Weight" column, paragraphs 190 and 196 and Chapter 4 Section 4.8 all
went on using AHP as the *name* of the primary scheme. That was arguably
defensible once Section 3.9.4 stated plainly what was and was not done, but it
invited the obvious question, so it is **now renamed throughout (Section 12e)** -
and the rename turned up three stale Cohen's kappa values in Chapter 4.

**Guarded.** Four literal anchors in `check_chapter3_consistency.RETIRED` cover
the elicitation claim, the verified-CR claim, the internal-consistency conclusion
and the old heading. Injection-tested on a copy: restoring the original wording
fires all four and exits 1.

Zotero integrity held through both edits - 25 `ZOTERO_ITEM` fields, 1
`ZOTERO_BIBL`, 156 `fldChar`, 222 paragraphs, 5 tables, unchanged before and
after. Backup: `CR_Madukwe_Chapter3_Final_BACKUP_pre_ahp_soften.docx`.

---

## 12e. The AHP label is gone, and removing it exposed three stale kappa values

**Renamed throughout, on the author's instruction** ("rename it throughout").
With §3.9.4 no longer claiming a pairwise elicitation, "AHP" named machinery the
thesis does not have. The primary scheme is now just **primary**:

| Where | Was | Now |
|---|---|---|
| `compute_suitability.py` | `AHP_WEIGHTS`, scheme key `AHP (primary)` | `PRIMARY_WEIGHTS`, key `primary` |
| `suitability_index.nc` | `si_AHP`, `tier_AHP` | `si_primary`, `tier_primary` |
| `suitability_weights.csv` | row `AHP (primary)` | row `primary` |
| Chapter 3 ¶164, ¶166, Table 3.5 column | "initial AHP weight(s)" | "initial weight(s)" |
| Chapter 3 ¶190, ¶196 | "primary AHP weights/classification" | "primary weights of Table 3.5" / "primary classification" |
| Chapter 4 §4.8 | "Analytic Hierarchy Process weights (Saaty, 1980)" | "the primary weighting scheme of Table 3.5" |
| Brief, deck | "AHP weights", "AHP pairwise comparison matrix" | researcher-assigned weights, no CR claimed |

Chapter 3 keeps **one** mention, spelled out at §3.9.4: the weights follow the
AHP's ordinal logic. That is honest provenance rather than a claimed procedure,
and it is the only place the concept is invoked.

**Saaty (1980) drops off the must-add list.** Chapter 4 no longer cites it, so the
list at the end of that chapter is now **three** items, not four: Cohen (1960),
Efron (1987), Dozier & Frew (1990). Saaty is needed only if the author chooses to
cite it at §3.9.4, and Chapter 4 says so explicitly rather than leaving it to be
discovered.

**What the rename exposed.** Re-running `compute_suitability.py` printed Cohen's
kappa as **0.725 / 0.476 / −0.091**. Chapter 4 said **0.697 / 0.317 / −0.111** —
hardcoded literals, because kappa was never written to disk. They were correct for
the first run and went stale at the §3.9.3 decay resolution, which changed the
road, grid and settlement scores and so every scheme's tiers. The generator whose
stated purpose is that "no figure in the prose can drift from the analysis" was
itself carrying three drifted numbers, and it went unnoticed through the document
audit because nothing on disk contradicted it.

Fixed at the root rather than by correcting the literals:
`compute_suitability.py` now writes **`suitability_schemes.csv`** (mean SI, very
high, high, kappa per scheme) and **`suitability_robustness.csv`** (assessed,
robust, weight-sensitive, irradiance layer), and `build_chapter4.py` reads kappa
from the first. `test_scheme_kappa_is_read_from_disk_not_hardcoded` fails if
either the superseded or the current values appear as literals in the generator —
the latter because a correct literal today is a stale literal after the next
regeneration.

**And the same defect was in the figure, where it reached the reader most
directly.** Figure 09's own plotted title read "90.8% of retained cells change
tier" with a panel labelled "under ALL four schemes (42 cells)" - both baked into
`make_suitability_maps.py` - while the generated caption printed beneath it in
Chapter 4 said 87.8% and 145. That figure appears in **both Chapter 4 and Chapter
5**. It was caught by looking at the rendered PNG after the rename, not by any
check: every guard in this project reads text, and nothing reads a figure. The
titles now come from `suitability_robustness.csv`, and the same test rejects any
robustness number baked into that script.

*The first version of that test failed on its own explanatory comment, which
names the stale figures in order to record why they are read from disk. It now
strips comments before scanning: a guard that cannot tell code from prose
punishes documentation.*

The same three stale values had reached the interview brief, where the surrounding
answer also still told the author to concede that the sensitivity analysis "has
not been run". It has. Both are corrected, and the brief now advises volunteering
the withdrawn consistency ratio rather than waiting to be asked for it.

Everything else held: `suitability_by_period.csv` is byte-identical across the
rename, and the only change in `suitability_weights.csv` is the row label.

---

## 12f. Figure audit: three more baked results, and one the retrain falsified

Prompted by the question "are there other hardcoded numbers in the figures?"
after figure 09 was caught by eye. Method: a static scan of both figure
generators for digit-bearing string literals reaching a rendered label
(`set_title`, `suptitle`, `text`, `legend`, axis labels), then **viewing every
one of the ten PNGs** and checking what they assert against the canonical CSVs.
The scan alone would have missed the third finding, which carries no digits.

| Figure | Said | Should have said |
|---|---|---|
| 05 power spectra | U-Net ratio "0.72 to 0.01", CNN "0.91-1.29" | **0.86 to 0.01**, **0.92-1.59** |
| 01 per-cell bias | "U-Net's zero mean is cancellation, not accuracy" | U-Net's mean bias is **+2.50** |
| 09 schemes | "90.8%", "42 cells" | 87.8%, 145 (fixed in §12e) |

**Figure 05 contradicted itself.** The annotations beside the panel were computed
from the cut sweep while the title was hardcoded, so after the honest retrain the
figure displayed "0.86-0.01x over cuts" next to a title reading "0.72 to 0.01".

**Figure 01's defect carries no number at all.** "U-Net's zero mean is
cancellation, not accuracy" was true of the leakage-selected U-Net. The honest
retrain moved its mean bias to +2.50, which the panel beside it printed correctly
the whole time. The claim survives in a form the data can carry, and is now
computed: *"CNN spans 31 W m-2 across cells, Random Forest only 3"* - the point
was always the spread, not the mean.

**Cleared.** Figures 02, 03, 04, 06, 07 and 10 carry no baked results: every
number in them is read from the CSVs, and the non-numeric claims check out - clt
really is top on both importance measures, the topographic importances really are
exactly zero (now derived, with a visible fallback if they ever stop being),
architecture variance really does overtake GCM by the long horizon, and figure
07's "1 km buffer" and 200 m river corridor match `build_suitability_layers.py`.

**One layout defect fixed in passing.** Figure 02's legend sat across the second
line of its own title and `set_xlabel` on a polar axis landed on top of the
radial tick labels. Both are placed in figure coordinates now.

`test_figure_titles_carry_no_baked_results` rejects the three superseded strings
and any current cut-sweep value baked back in; injection-tested. It strips
comments first, like the kappa guard.

**The standing gap this leaves.** Every automated check in this project reads
text - chapters, status, brief, deck, CSVs. **Nothing reads a figure.** All three
findings here needed a human-equivalent look at a PNG. The guards added can only
reject known-bad strings in the generators; they cannot tell that a rendered
number disagrees with the analysis. Figures should be re-viewed after any retrain
or regeneration, and that is a procedure, not a test.

---

## 12g. Table audit: Table 4.5 had no source at all

Same method as the figures - a static scan of `build_chapter4.py` for string
constants inside `TBL(...)` rows carrying a bare number, then every rendered cell
of all fifteen tables checked against the canonical CSVs or recomputed from the
layers.

**Nine of Chapter 4's ten tables are clean**, and were verified rather than
assumed: 4.1 against `table_3_3.csv`, 4.2 against the Taylor statistics, 4.3
against `rolling_origin.csv` including the difference column, 4.4 against
`bca_intervals.csv` (the "mean bias magnitude" row correctly takes the `|MBE|`
entry, not `MBE`), 4.6 against the cut sweep, 4.7 recomputed from
`mme_aggregations_xgb` against the present field, 4.8 against the variance
decomposition, 4.9 recomputed cell by cell from `criterion_layers.nc`, and 4.10
against `suitability_by_period.csv`. Chapter 3's two numeric tables are clean
too: the ablation table already has a cell-level guard, and Table 3.5's weights
match `PRIMARY_WEIGHTS` exactly. Chapter 5 has no tables.

**Table 4.5 was the exception, and it was the worst case available.** Twelve
separations and four ordering percentages, typed as literals, with **no CSV
behind them anywhere in the project**. The Random Forest and XGBoost rows were
right, because neither model was ever retrained. Every CNN and U-Net figure was
stale:

| | Chapter said (superseded) | Actual |
|---|---|---|
| CNN | previously +1.393 / +4.053 / +9.643, 100.0% — stale | **+1.272 / +3.851 / +9.478, 99.9%** |
| U-Net | previously +1.075 / +1.998 / +4.631, 100.0% — stale | **+1.034 / +2.039 / +4.703, 100.0%** |

The honest retrain regenerated both models' projections - the directory
timestamps show it - and nothing regenerated the table, because there was nothing
to regenerate. **This is the table that justifies the deployment decision.** The
conclusions are unaffected: the Random Forest still fails, XGBoost still grows,
both neural models still pass. But the evidence for a deployment decision cannot
be a number a reader has no way to re-derive.

`compute_scenario_discrimination.py` now writes
**`scenario_discrimination.csv`**, and Chapters 4 and 5 read from it - including
the "Grows?" verdict, which is evaluated from the separations rather than typed,
and which reproduces the original labels exactly (XGBoost dips before it rises,
so it is "Yes, overall" rather than "Yes"). The same values were stale in
PROJECT_STATUS 7.11, the viva brief and the interview deck's chart series; all
three are corrected, and the six superseded figures are now retired phrases in
both the status and brief guards.

**One further error the audit found in 7.11's prose.** The caution paragraph
previously read *"The CNN's +9.643 ... its long-term change of +16.6 W m⁻² is
implausibly large"*, and both of those numbers were wrong: the separation was the
stale pre-retrain value, and **+16.6 is the U-Net's change, not the CNN's**. The CNN's is +26.3, so the point stood for
the wrong reason. Corrected, with the substitution noted in place.

`test_scenario_discrimination_is_read_from_disk` rejects the superseded values and
today's values as literals in either generator, and asserts the two verdicts §4.4
leans on - that the Random Forest fails the screen and XGBoost passes it - so a
regenerated CSV that quietly reversed either would fail rather than silently
rewrite the chapter's argument. Injection-tested.

---

## 12h. Prose audit: a mechanism the data does not support

Same method again - scan both generators for numeric literals inside `P()` that
are not filled at build time, then check each surviving claim against the data.
Most hits were section numbers. Four were real, and the last is not a stale
number but a wrong inference.

**Section 4.7 contradicted itself one paragraph apart.** The paragraph computing
the variance decomposition printed architecture at **35.4 per cent** from the
CSV; the very next paragraph said *"architecture contributes 27 per cent"* and
repeated it. 27.18% was the pre-retrain figure. Both are computed now.

**Its two supporting correlations were stale as well**, and no CSV held them: *"no
pair of architectures correlates above 0.71"* is really **0.58**, and *"the
deployed model agrees with the others at 0.11 to 0.48"* is **0.12 to 0.28**. Both
now come from `architecture_agreement.csv`, written by
`compute_scenario_discrimination.py`. The corrected figures make Section 4.7's
point more strongly than the printed ones did.

**The dropout attribution is withdrawn.** Chapters 4 and 5 and Section 7.18 all
stated that removing spatial dropout *"raises the spectral ratio beyond wavenumber
10 from 0.085 to 0.771"*, and concluded that dropout causes the U-Net's damping.
Checking it against `unet_optimisation.csv`:

- **0.085 was a pre-retrain deployed-model figure** (now 0.090) sitting beside
  0.771, which is a **sweep** figure. Two different models, one comparison.
- **The sweep's own baseline has a spectral ratio of 1.24** - an *excess* of
  fine-scale power. It carries the same `Dropout2d(0.3)` as the deployed model
  and shows no damping at all, so it never reproduced the failure it was built to
  explain. A sweep whose baseline lacks the defect cannot isolate its cause.
- **Removing dropout moves the ratio to 0.771, which is no closer to unity**:
  0.23 against 0.24 in absolute deviation. Essentially unchanged.
- **The supporting claim was backwards.** *"Varying the gradient penalty barely
  moves the ratio, so the smoothness-prior hypothesis is refuted"* fails twice:
  the penalty variants span 1.01 to 1.24, and removing the penalty gives **1.006,
  the closest to unity of any variant tested** - evidence *for* a smoothness
  prior, not against it.

**What survives is the accuracy result**, which is solid and unchanged: removing
dropout improves held-out RMSE from 10.98 to 8.63, spatial correlation from 0.894
to 0.970, and centred error from 3.59 to 1.97. Dropout is costly. What it does to
the spectra is now an open question, and both chapters say so.

**CONFIRMED BY THE AUTHOR** ("leave it withdrawn"), after reading Section 4.5 in
full. The withdrawal stands and should not be re-litigated without new evidence.
The route back, if anyone wants one, is to fix the sweep so its baseline
reproduces the deployed model's spectra and re-run it, not to restore the claim.

**This was a change to a scientific claim, not a typo, and it needed the author's
review.** Section 4.5 now reports the withdrawal and the inconvenient
gradient-penalty result explicitly rather than dropping the paragraph; Section 5.6
asks for a sweep whose baseline reproduces the deployed model's spectra. Nothing
downstream depends on the attribution - the deployment decision turned on scenario
discrimination - so no result moves.

*Cleared in the same pass, checked rather than assumed:* the 19.08 climatology
(skill is RMSE-based, so 9.242/(1−0.5155) = 19.08 - my first check assumed an
MSE-based skill and was wrong, not the chapter), mean bias below 5 W m⁻² for all
four (max 4.553), 479 usable SARAH fields, the 3 per cent SARAH offset (3.23%),
168 evaluation months and 312 training months, the 214–258 W m⁻² assessed span
(214.1–257.1), two cells excluded by slope, and the +0.92/+0.95 honest-selection
cost.

---

## 12i. Chapters 1 and 2, which have no guard

The prose pass covered the generated chapters. Chapters 1 and 2 are hand-written
and nothing checks them, so they were scanned separately for result-like decimals.

**Neither chapter reports a result of this study.** Every number in them is a
literature value, a resolution, a policy target or a section cross-reference —
0.25° for ERA5, 0.05° for SARAH, 16.5% and 26.5% from the NREP, Maposa's 0.92.
Nothing there can go stale when this study's numbers move, which is why the
honest retrain left them untouched. Two exceptions:

**Chapter 2 told the reader the suitability analysis had not been done.**
Paragraph 140 closed the gap analysis with the integration of the irradiance
layer *"specified in Section 3.9 and remains to be executed"*. It has been
executed and is reported in Section 4.8. The paragraph now says so, and points at
the weighting sensitivity analysis as the result rather than the five-tier map.

**Chapter 2 overstated the warming between the two periods.** Paragraph 119 said
the validation period is *"approximately 0.5 to 0.8 degrees Celsius warmer"*.
Measured on `tas` in the aligned stack, the domain mean is **+0.501 °C** and the
per-cell range is **+0.196 to +0.777**. The printed range described the upper part
of the spatial spread as though it were the typical shift. Now stated as 0.5 in
the domain mean, 0.2 to 0.8 across cells. The argument is unaffected — a warmer
validation period is still a within-sample stationarity test — but a reader
sizing the test would have taken the shift as larger than it is.

Zotero integrity held: unchanged field and paragraph counts before and after.
Backup: `CR_Madukwe_Chapter2_BACKUP_pre_prose_audit.docx`.

---

## 12j. Chapter 3 prose: method descriptions that outlived the code

Every numeric claim in Chapter 3's prose was extracted and checked against the
code or the data — 61 paragraphs carrying numbers. **Most of the chapter is
exactly right**, including several things that looked wrong until checked
properly:

| Checked | Result |
|---|---|
| QC flag counts (38,843 of 6,063,552, 0.64%, 107 aerosol, 751 of 957 cells, max 277) | **all exact** |
| 1,794,312 / 966,168 values, CSI max 0.6323, constraint never binds | exact |
| Elevation 124 m to 1840 m given to PVLIB | exact (124–1839) |
| Clear-sky ratio 1.724, sd 0.033 | exact |
| Out-of-range tas, 4.95% under SSP5-8.5 against 0.79% | **exact** |
| Hwange 14,651, Mana Pools 2,196, Gonarezhou 5,053 km² | exact against WDPA `REP_AREA` |
| BCa intervals, centred RMSE by model, MBE, rolling-origin | all exact |
| Bilinear baseline 0.23 W m⁻² | exact (0.2331) |
| Decay 0.37/0.14, tiers, scheme weights, land-cover scores, ONI 0.5/20% | all exact |
| 71×81, 29×33, 2.45, 312/168, 9,496/5,114 daily, 1.25°×1.875° → 12.5/18.8 | all exact |
| RF 500 trees / leaf 5; XGB depth 6, eta 0.05, subsample 0.8, mcw 3; CNN 64 filters, 100 epochs, lr 5e-4, λ 0.01; U-Net 64–512 + 1×1 1024, dropout 0.3, lr 2e-4, batch 16 | all exact |

Two of my own checks were wrong before the chapter was: the elevation range
(I read it off the wrong grid) and Mana Pools (I matched the World Heritage
complex, 6,766 km², rather than the National Park). **The chapter was right both
times.**

**What was actually wrong: the methods chapter described two procedures that had
been removed as leakage.**

- **§3.6.4 said XGBoost's boosting rounds were "determined by early stopping on
  the validation split with a patience of 50 rounds".** That is §6.12's defect
  verbatim — the thing that was found and fixed. The model uses a **fixed 200
  rounds**; `early_stopping_rounds` is `None`. The text now says so, gives the
  measured cost of the fix (9.11 leaky against 9.24 clean) and why an inner-split
  variant scored worse (10.02).
- **§3.6.6 said the U-Net used "early stopping on the validation MSE with a
  patience of 20 epochs".** That is §6.13's defect. The script now runs two-phase
  honest selection — Phase A fits 1985–2004 and stops on a 2005–2010 inner split,
  Phase B refits on the full training period for that epoch count. The text now
  describes what the code does.

A methods chapter that documents a corrected defect as current practice is worse
than one that never mentioned it: it tells the examiner the leakage is still
there.

**Two claims that the project had overtaken.** §3.6.3 and §3.6.4 both said
predictor importances "were not extracted in the present implementation". All
four measures were extracted, live in `feature_importance.csv`, and carry a
figure and a section in Chapter 4. Both now point at §4.7.

**Two figures that could not be reproduced.** §3.9 gave the ERA5-versus-SARAH
layer comparison as ρ = 0.860 with 27.2% of cells crossing the top tier. Neither
reproduces on the current layers under any basis tried, and neither was ever
written to disk. Recomputed on the standardised score the overlay actually uses,
over the 2,386 assessed cells: **ρ = 0.875, 15.5% crossing**. Now written to
`layer_choice_sensitivity.csv` by `compute_suitability.py`, and read by Chapter 4.
The argument holds — one assessed cell in six still changes tier on the
irradiance criterion alone — but it is weaker than "nearly half the domain".

**One ambiguity tightened.** §3.7.3 compared the bilinear baseline "against a
target standard deviation near 19". The target's raw standard deviation is 38.04;
7.98 for the time-mean field; and about its seasonal climatology it is 17.98 — which
collides numerically with the retired A7 climatology reference and has nothing to do
with it. The intended
comparator is the 19.08 W m⁻² climatology RMSE, and the text now names it.

**One live defect left in place and flagged rather than fixed.** §3.6.7 states
that the neural hyperparameter search "was scored directly on the validation mean
squared error". That is true, it is selection on the withheld record, and unlike
the epoch count it has **not** been corrected. The paragraph now says so and
points to §5.6, where it is already the first recommendation. Correcting it means
re-running the grid search inside the training period — a real piece of work, not
an edit.

Six literal anchors added to the Chapter 3 guard; injection-tested. Zotero
integrity held through every edit: 25 `ZOTERO_ITEM`, 1 `ZOTERO_BIBL`, 156
`fldChar`, 222 paragraphs, 5 tables, unchanged.

*One near-miss worth recording: an intermediate version of this edit collapsed
paragraph 102's runs to apply a replacement, which destroyed a live Zotero field
(25 → 24 items, 156 → 150 `fldChar`). Caught by the integrity counter, restored
from backup, redone run-by-run leaving the field runs untouched. Run-level
surgery is the rule for a reason.*

---

## 12k. The dissertation is assembled, and the SARAH question is answered with numbers

**Why the product is not validated against SARAH, computed rather than argued.**
`validate_against_sarah.py` scores every model against both references over the same
168 withheld months and cells:

| | RMSE v ERA5 | RMSE v SARAH | bias v SARAH |
|---|---|---|---|
| XGBoost (deployed) | 9.24 | 13.67 | -6.95 |
| Random Forest | 10.30 | 14.42 | -7.91 |
| CNN | 10.04 | 13.89 | -5.33 |
| U-Net | 10.33 | 14.59 | -6.79 |
| Baseline (bilinear) | 0.24 | 13.83 | -8.16 |
| **the ERA5 target itself** | **0.00** | **13.84** | **-8.14** |

Three things follow, and together they are the justification. The ERA5 target's own
distance from SARAH is **13.84 W m⁻²** against a spread across the four
architectures of **1.09**, a factor of **12.7**. The bilinear baseline, which reproduces
the target to 0.24 and therefore contains no downscaling at all, scores
13.83 against SARAH: the test cannot distinguish a trained model from an
interpolation. And XGBoost's 13.67 is *lower* than the target it was trained to
reproduce, because its small positive bias against ERA5 partially cancels ERA5's
-8.14 against SARAH. A model beating its own training target is conclusive
evidence that the quantity measured is the reference, not the model.

So the answer is not that the validation is unnecessary. It is that **re-scoring an
ERA5-trained product against SARAH is not an observational validation**; it requires
refitting against SARAH as the target at 0.05°, which is Chapter 5 §5.6's first
recommendation and a separate study. Written up as Chapter 4 §4.6.1 and summarised in
§5.4.

**The dissertation is now one document.** `build_thesis.py` builds front matter in
University of Zimbabwe styling (crest from the University's own site, wordmark purple
`#2C1A70` and crest blue `#3251A1` sampled from it), merges Chapters 1 to 5 with
`docxcompose` so that Zotero fields, tables, figures and OMML equations survive, then
appends the consolidated references and two appendices. 648 paragraphs, 18 tables,
11 images, **60 in-text Zotero fields intact**, ~30,750 words.

- **Front matter:** title page, abstract (~600 words, every figure read from the
  result CSVs so it cannot drift), acknowledgements, and a Word contents field.
- **Page setup normalised.** The chapters arrived with three different left margins;
  every section is now A4 with a 3.5 cm binding edge, roman numerals for the front
  matter and arabic restarting at 1 for the body.
- **One reference list, not four.** Chapters 1, 2 and 3 each carried their own
  bibliography, two Zotero-generated. 82 paragraphs of per-chapter lists were removed
  and replaced by a single author-date list; the in-text fields were left untouched,
  since those are what Zotero needs. Chapter 2 cites numerically and the others by
  author-date, so the consolidated list is normalised to author-date, which is what a
  single style will emit on refresh.
- **Em-dashes removed** from the generators and from Chapter 3, so regeneration cannot
  reintroduce them. Spaced em-dashes became commas, unspaced ones hyphens. The
  dissertation contains zero. En-dashes in numeric ranges are left, being correct.
- **Appendix A** states the reproducibility case and names what the repository does
  not contain. **Appendix B** declares the use of AI for language, code generation and
  debugging, and states what it was not used for.

**Declaration page added**, between the title page and the abstract: originality,
signature blocks for the candidate and both supervisors, and a pointer to Appendix A
for the code. Its second paragraph declares the use of computational tools and points
at Appendix B, because a bare claim of "my own original work" sitting in the same
document as an AI declaration would be a contradiction the examiner has to resolve
rather than one the author has. The funder is named as **EACEA**, confirmed by the
author.

**One error caught in my own abstract, and it is the error this project keeps
finding.** The first draft reported the projected change as +4.0 W m⁻², differencing
the SSP5-8.5 projection against the **SARAH**-based present. The projections come from
an ERA5-trained chain, so that subtracts one instrument from another and calls the
difference climate, which is precisely what §7.17 exists to prevent. Recomputed on the
ERA5 basis, as Table 4.7 computes it: **+9.6 W m⁻²**.

**Repository prepared for publication.** README rewritten as a front door rather than
developer notes: the reproducibility argument, a dataset table naming every provider
and licence, the full pipeline in order with runtimes, how to run the guards, and the
AI disclosure. Plus `LICENSE` (MIT, explicitly not covering the data), `CITATION.cff`,
and a regenerated `environment.yml`. Chapter 3 §3.11 and Appendix A both state the
repository's existence and its limits, so the reproducibility claim is in the
dissertation and not only in the code.

---

## 12l. Three scope claims the analysis had overtaken

Found on a read-through by the author, who noticed that the thesis said the SARAH
validation was not part of it while Chapter 4 reported one. The cause is ordinary and
worth naming: **each was true when written, and each was falsified by work done later
in the same project.** Sections written early describe a plan; sections written late
describe what happened; nothing reconciles them unless someone looks.

| Where | Said | Why it was false |
|---|---|---|
| Ch 2 §2.3.3 | "Validation against SARAH-2 ... is not reported in this thesis, in which ERA5 is the sole reference" | §4.6 compares the references and §4.6.1 scores the product against SARAH |
| Ch 3 §3.3.3 | the SARAH record "is not analysed here" | it is the present-day suitability layer (§3.9.1) **and** is analysed in §4.6 and §4.6.1 |
| Ch 3 §3.1 | §3.9 "remains to be applied to the downscaled products" | §3.9 was applied; §4.8 reports it in full |

The third is the same defect found in Chapter 2 §2.6.3 during the prose audit
(§12i), in a chapter that had already been checked. The second is an internal
contradiction within Chapter 3 alone: one paragraph said SARAH was not analysed while
another adopted it as the present-day irradiance layer.

All three now state the position precisely, which is more useful than either the old
claim or a blunt correction: **ERA5 remains the sole training target and evaluation
reference; SARAH enters in three places without becoming the target** (present-day
suitability layer, reference comparison, product diagnostic); **what is absent is
refitting the chain against SARAH**, which is what an observational validation
requires. Chapter 4 §4.9's limitation now says the same thing rather than "a full
validation remains outstanding", which read as though nothing had been done.

Two literal anchors added to the Chapter 3 guard. The wider lesson is that the
guards check numbers and retired phrases but cannot check whether a statement about
what the study *does* is still true; those need a read-through, and this one came
from the author rather than from me.

---

## 12m. The citation styles were never unified, and the merged file had none

Raised by the author: why a different style from the Harvard the University uses.
The answer is that no style was chosen by me. Each chapter records its own in
`docProps/custom.xml`, and they disagreed:

| | recorded style |
|---|---|
| Chapter 1 | `elsevier-harvard` |
| Chapter 2 | **`taylor-and-francis-aip`** (numbered) |
| Chapter 3 | `elsevier-harvard` |
| merged dissertation | **none at all** |

The consolidated reference list was normalised to Elsevier Harvard because that is
what two of the three chapters already used, and it is a Harvard variant. The real
defect was never the list: it was **Chapter 2 sitting on a numbered physics style**,
which is why its citations render as `[5]` while every other chapter renders
`(Vandal et al., 2017)`.

**The merged dissertation was worse, and this was the more serious find.** Its front
matter is a fresh `python-docx` document, so the merged file had no
`docProps/custom.xml` at all: 60 live Zotero fields and no recorded style to render
them in. Opening it and pressing Refresh would have prompted for a style rather than
rebuilding the bibliography.

Both fixed, on the author's choice of Elsevier Harvard. Chapter 2's recorded style is
switched, and `build_thesis.py` now writes the preference block into the merged file
after composing, with the content-type override and package relationship the part
needs. `bibliographyStyleHasBeenSet` is cleared in both so Zotero rebuilds from
scratch rather than reusing a cached numbered list. Verified: the package passes a
CRC check, still opens, and all four files now report `elsevier-harvard`.

Changing to a different Harvard later is a one-line edit to `ZOTERO_PREF` in
`build_thesis.py`, or two clicks in Zotero's Document Preferences.

---

## 12n. I destroyed the author's edits, and what now prevents it

The author edited the merged dissertation on the Desktop and uploaded it. I read it
only to scan for SARAH statements, never diffed it against my own build, and then ran
`cp` over that exact path three times to "refresh the Desktop copy". Those copies
destroyed the edits. The file was recovered by the author from Word.

**The cause was treating a path as my output after it had become their working file.**
The Desktop directory is now theirs; the build writes to OneDrive and files are sent,
rather than copied into a directory the author edits in.

**`apply_author_edits.py`** pushes edits from a hand-edited merged document back to
the sources the build actually reads, which is the only place they survive a rebuild.
Three things it does carefully:

- **Word-level spans, not character-level.** A character diff widened to word
  boundaries produced overlapping replacements: one edit yielded both
  `("MSc.", "Master of Science")` and `("MSc.", "Science in")`, and applying both
  would have corrupted the sentence.
- **Only the changed spans are written, into the run that holds them.** Paragraphs are
  never rebuilt, because a citation occupies several runs and flattening them destroys
  the field, which has happened once in this project already.
- **Reference sections are excluded.** The matcher would otherwise pair each chapter's
  Zotero-generated bibliography with my consolidated list and rewrite a generated
  bibliography with ASCII-folded names. Caught in the dry run.

57 spans applied across Chapters 1 and 2; Chapter 3 needed none. Four edits sat exactly
on the boundary between a text run and a citation field, where a missing space or comma
belongs, and were applied there by hand. Citation field counts were verified before and
after every file and are unchanged.

**The supervisor names were in the project all along.** Chapter 1 opens with its own
cover page carrying "Supervisor: Prof E. Mashonjowa | Programme Coordinator: Prof T.D.
Mushore", and the degree name with it. I asked the author for information their own
Chapter 1 already contained. That cover page is now stripped during the merge, since a
dissertation with a title page does not need a second one, and Chapter 1 keeps it for
reading standalone.

Front-matter edits went into `build_thesis.py`, not the .docx: the abstract and
acknowledgement rewordings, the removal of the repeated University wordmark under the
crest, "Table of Content", the names and the October date. The abstract keeps every
figure as a computed placeholder, so the author's prose cannot drift from the results.

**One regression caught in the same pass.** Stripping Chapter 1's cover removed
paragraphs carrying a section break, collapsing three sections into one and silently
losing the roman/arabic page-numbering split. Such paragraphs are now emptied rather
than deleted.

Eight differences from the author's copy remain, and all eight are my later fixes that
their copy predates: the Cohen and Efron DOIs, the punctuation repair in Chapter 4's
to-add note, the script count, and the four scope corrections made in response to their
own inconsistency report.

---

## 12o. Chapter 2's superscripts are the citations, and 20 of its citations are not fields

The author asked whether Chapter 2's sub- and superscript reference markers could
simply be deleted. **No: the superscript `[n]` markers are the Zotero field results.**
Deleting them deletes the citations, not their formatting, and Chapter 2 would lose
all 24 source attributions with nothing to regenerate from. They also do not need
replacing with Zotero fields, because that is what they already are.

What was actually wanted is achieved by the style change already made (§12m): under
`elsevier-harvard` the field renders `(Wilby et al., 2002)` inline instead of a
superscript `[1]`. Word can keep manual run formatting across a refresh, so the
superscript and subscript properties have been cleared from all 142 affected runs in
Chapter 2 without touching the field structure. Field counts verified identical before
and after; zero sub/superscript runs remain in that chapter.

**The genuine subscripts survive.** Eight remain in the dissertation, all in Chapter
3's super-resolution formalism (H_low, W_low, H_high, W_high). Those are notation, not
citations, and the clearing pass was confined to Chapter 2 so they were never at risk.

**The larger finding, which the question uncovered.** Chapter 2 carries **24 live
Zotero fields and 20 in-text markers that are plain typed text**, covering 12 distinct
sources. On refresh the 24 will become author-date while the 20 stay as `[9]`, `[11]`
and so on, leaving the chapter visibly half-converted. These cannot be fixed from
outside Word: each must be re-inserted through Zotero at its position.
`CHAPTER2_CITATIONS_TO_RELINK.md` lists all 20 with the sentence each attaches to and
the reference it points to, so the work is mechanical rather than investigative.

Three of the twenty sit at the end of a sentence as a bare marker (paragraphs 115, 117
and 119), which is where a numbered style puts a citation and where an author-date
style will read oddly; those are worth re-siting inside the sentence as they are
re-cited.

---

## 12p. The citation markers carried three layers of stray formatting

The author reported that Chapter 2's citations looked wrong after the superscripts
were cleared. They were right, and clearing the superscript had exposed rather than
caused it. The AIP numbered style had rendered each marker as **9 pt, teal
`#1A5F7A`, superscript**. Removing only the superscript left 9 pt teal text sitting on
the baseline, which is worse than where it started.

All three layers are now cleared, on the citation runs only:

| | Chapter 1 | Chapter 2 | Chapter 3 |
|---|---|---|---|
| superscript/subscript | - | 142 | - |
| explicit 9 pt size | - | 140 | - |
| teal `#1A5F7A` | 1 | 144 | 23 |
| grey `#808080` | 2 | 3 | 5 |

Every run now inherits size and colour, which is what Chapters 1 and 3 already did for
their own citations and is therefore the target rather than a new convention. Field
counts were verified before and after each file and never changed.

**One of these was a genuine defect rather than an inconsistency.** In Chapter 2 the
teal colour ran past the end of a citation marker and into the body text after it, so
a sentence in paragraph 127 began `[13] This expectation is treated here...` with the
prose itself coloured. That is now black with the rest.

**Deliberately left alone.** Chapter 2's `Gap 1:` to `Gap 4:` labels are coloured
`#17A589` as a design choice, and the eight remaining subscripts are Chapter 3's
H_low/W_high notation. The residual explicit sizes (10, 11, 12, 16 pt) are the front
matter and figure captions this build writes on purpose.

---

## 13. Outstanding

**Not started**
- **§3.9 Multi-Criteria Suitability Analysis — DONE.** The analysis runs end to end (§12a), the maps are generated (`make_suitability_maps.py`, figures 07–10), the future-period suitability is computed, and it is written into Chapter 4 §4.8. The acquisition table below is kept as the provenance record; every row that mattered was resolved.

  | Criterion | Source | State |
  |---|---|---|
  | Annual mean GHI | this study | held |
  | Terrain slope | SRTM 90 m | held (§3.5.2) |
  | Proximity to roads | OSM `highway` | **acquired** — `data/raw/osm/zimbabwe-latest.osm.pbf`, 171 MB, verified |
  | Proximity to ZETDC grid | OSM `power` | **acquired**, same extract — but see the two caveats below |
  | Population density | WorldPop 100 m | **acquired** — `data/raw/worldpop/zwe_ppp_2020_constrained.tif`, 22.3 MB, verified |
  | Land cover class | ESA CCI 300 m | blocked: needs the `satellite-land-cover` and `vito-proba-v` licences accepted on the CDS account |
  | Proximity to settlements | OSM / GADM | GADM boundary not yet acquired |
  | *(exclusion mask)* | WDPA / ZimParks | blocked: protectedplanet.net requires accepting terms in a browser |

  **Two caveats on the OSM power layer.** GDAL's OSM driver does not promote `power` to a column — it lands in `other_tags` as an hstore string, so the grid layer needs either a string filter or a custom `osmconf.ini` that promotes `power` and `voltage`. And the country extract carries **cross-border** infrastructure: the first `power=line` found is tagged `operator=Zesco`, which is Zambia's utility. The grid layer must be clipped to the domain and operator-checked, or proximity is computed to a grid that cannot be connected to.

  **WorldPop is clipped to the national outline, not the domain box** — it spans 25.24–33.06°E, 22.42–15.61°S, falling short of the 25–33°E / 15–22°S grid by 0.24° west and 0.61° north, where the box lies in Zambia and Mozambique. This is correct behaviour for a Zimbabwe-only product, and it means **a national-boundary mask (GADM) is required** so out-of-country cells become clean exclusions rather than nodata holes in the weighted linear combination.
- **Chapters 4 and 5 now carry citations and figures.** Chapter 4 has **10 embedded figures** with numbered captions and **11 distinct in-text citations**; Chapter 5 has 1 figure and 7. (Counted from the built documents, not from the generator: Efron and Dozier & Frew appear only in Chapter 4's closing to-add list, not as in-text citations.) Figures are inserted by the generators from `figures/`, so regenerating a figure and rebuilding the chapter keeps image and caption in step. Citations are plain author-year text and **must be converted to live Zotero fields in Word** — most cite works already in the Chapter 2 and Chapter 3 bibliographies and need only re-citing. **Four are not in the library** and are listed at the end of Chapter 4: Saaty (1980) for AHP, Cohen (1960) for kappa, Efron (1987) for the BCa interval, and Dozier & Frew (1990), which was already outstanding.

- **Chapter 5 — FIRST DRAFT WRITTEN**, `CR_Madukwe_Chapter5_DRAFT.docx`, ~2,600 words, generated by `build_chapter5.py` from the same CSVs as Chapter 4 so the conclusions cannot quote figures that have drifted from the results. Answers all four RQs, and **RQ2 gets a largely negative answer** — the product resolves essentially nothing below its input grid — which is stated as the study's most important single finding rather than softened. Citations added (7 distinct) and one figure embedded; they remain plain author-year text pending the Zotero conversion.

- **Chapter 4 — FIRST DRAFT WRITTEN**, `CR_Madukwe_Chapter4_DRAFT.docx`, ~5,500 words, 10 tables, 10 figures, generated by `build_chapter4.py` from the evaluation CSVs so no figure in the prose can drift from the analysis. **Edit the generator, not the .docx.** **Figures are now inserted by the generator with numbered captions.** One manual step remains in Word: the author–year citations are plain text and need converting to live Zotero fields.

  §4.8.1 leads with the robust set rather than the five-tier map, and its central finding is that **the 145 robust cells are not the sunniest places in Zimbabwe** — mean irradiance 246.12 against 242.83 W m⁻², a difference of 1.4% — but the best-connected: 2.12 km from transmission against 26.54 km, a factor of **12.5**. Irradiance over Zimbabwe varies too narrowly to discriminate between sites after standardisation, while grid distance varies over two orders of magnitude and decides the outcome under every weighting. The planning implication is that **the binding constraint is grid access, not sunlight**.
- **SARAH independent validation — data now ACQUIRED, analysis not started.** 480 monthly SIS fields at 0.05° over the domain, 1985-01 to 2024-12, complete and gapless, MD5-verified against both order emails (`data/raw/sarah/`, orders ORD68669 CDR + ORD68670 ICDR, identical extraction spec so they join cleanly). Reproduce with `fetch_sarah_order.py`. Covers **both** the training and validation periods, so it serves both purposes: the validation-independence gap (§14.1) and use as an alternative high-resolution *target*, the only route to genuine sub-0.25° resolution (§7.2).

  Two things to know before building on it. **The grids do not coincide:** SARAH is cell-centred (25.025, 25.075, …) and the target grid is node-centred (25.0, 25.1, …), so *none* of the 81 longitudes or 71 latitudes match and the 0.05° → 0.1° step is interpolation, not block-averaging. **The perimeter needs one-sided treatment:** 300 of 5,751 cells (5.2%) sit 0.025° (~2.8 km) outside SARAH's cell-centre hull, because the order was placed with no margin. Clamping is adequate at that distance.

  Note the filename suffix differs across the join — `…UD1000101UD` for the CDR, `…UD10001I1UD` for the ICDR — so a naive glob silently picks up only one half.


**Needs your hand — one item, and it needs the Zotero desktop app**
- **Dozier & Frew (1990) is not in the Zotero library.** Verified precisely: it appears once, as plain text at Chapter 3 paragraph 75 ("the Dozier and Frew (1990) sky-view integral"), and is in **none** of the 26 `ZOTERO_ITEM` citation fields. Chapter 3's bibliography is Zotero-generated (`ZOTERO_BIBL` field present), so **the reference will not appear in the reference list** as things stand. I cannot add it — the library is the desktop application's own database. Add this item, then re-cite the plain text as a live field:

  > Dozier, J. and Frew, J. (1990). Rapid calculation of terrain parameters for radiation modeling from digital elevation data. *IEEE Transactions on Geoscience and Remote Sensing*, 28(5), 963–969. doi:10.1109/36.58986

- **The published results dashboard has been withdrawn** and replaced with a retraction notice (see §10a / E1). Nothing further is needed unless you want the URL itself deleted, which must be done from the artifacts gallery.

**Closed in the Round 5 pass**
- **Chapter 1 needed no change.** Its satellite-product passage (paragraph 35) describes their *limitation* — that they carry no future information — and never claims them as this study's validation. Checked rather than assumed.
- **Chapter 2 §2.3.3 tense — fixed.** Paragraph 69 claimed SARAH-2 was "a suitable independent validation dataset for the downscaled products produced in this study"; paragraph 71 made the same claim for a second product. Both now state the validation is planned and not reported here, pointing to §3.3.3, so Chapters 1, 2 and 3 agree.
- **Chapter 2 forward reference — added.** Paragraph 128's "Benchmarking studies **confirm** that deep learning models outperform classical baselines" is softened to "**report**", and the paragraph now closes by framing this as a hypothesis the study tests rather than assumes, states that it is not borne out here, and forward-references §3.8.4.
- **The 0.68 ambiguity does not exist in the thesis.** Searched all three chapters: **0.68 appears nowhere**. The collision is between two *status-document* sections (§6.1's collapsed correlation and §7.5's std ratios), not between two thesis passages, so no clarifying clause is needed. Resolved as not applicable rather than left open.
- **Per-cell QC flag counts — implemented, and they found a real defect.** See §7.12.

**E1 — closed.** The published dashboard has been **withdrawn**: its content is replaced by a retraction notice explaining that every figure was superseded by the alignment fix, the circular-predictor removal, the two leakage fixes and three retrains. The URL now resolves to that notice rather than to wrong numbers. Fully deleting the URL, if wanted, must be done from the artifacts gallery.

---

## 14. Standing methodological caveats

1. **ERA5 is the only reference *in the work reported here*.** Training and validation both use it, over a data-sparse region where reanalysis is weakest. The temporal split is genuine, so this is a real out-of-sample test *of the reanalysis relationship* — but not an independent test against observations. SARAH validation is committed as the next stage (§13), and the data is now held; until it lands, every skill figure in §8 carries this qualifier.
2. **Stationarity is assumed.** Models fitted on 1985–2010 are applied to bias-corrected SSP5-8.5 fields to 2100. EDCM preserves the absolute change signal, but the learned predictor→irradiance mapping is assumed valid in an unseen climate.
3. **Cloud is unresolved.** The dominant control on surface irradiance is parameterised inside the GCM and available only as a grid mean. Feature importance confirms cloud fraction dominates the fit.
4. **Defensive NaN handling hides failures.** Two of the three severe silent bugs were `np.nan_to_num` converting a loud failure into a plausible field. Both were caught by inspecting fields, not scores. `safe_nan_to_num` now makes this loud.
5. **Two bugs cancelled each other** (§6.11). A bug invisible in one data split because a second bug reverses it is the hardest class to find — the lesson is that identical logic must be used at every join, not merely logic that works.
6. **Prose drifts from tables — and the sentence half is the harder half.** Six audit rounds found the same failure in the interview brief: a table corrected while a sentence summarising it was left behind. "Neither leg survived", "one real point", "every severe bug was silent", "beaten by no model on any tested axis" — none is a wrong *number*, each is a wrong *claim built on a right number*, so no numeric check can see them. `check_brief_consistency.py` therefore carries a retired-**sentence** list alongside its cell checks, and the discipline is to re-read every counting or characterising sentence whenever a table changes. All four documents are now guarded — `check_status_consistency.py` and `check_brief_consistency.py` here and for the brief, `check_chapter3_consistency.py` for Chapter 3, which an examiner actually reads, and the interview **deck**, whose text frames and speaker notes `check_brief_consistency.py` now scans as a fourth document. The deck was added because an injection test caught the same claim in the brief and in PROJECT_STATUS and missed it in the deck — a document is not guarded until something has been injected into it and the guard has failed. `check_status_consistency.py` guards the value side of this document, and its own first version had exactly the bug it exists to prevent: it scanned a whole line for any explanatory word, so a stale figure passed because an unrelated clause elsewhere in the same line said "no longer". It now searches a 90-character window around the value. **The sentence list in `check_brief_consistency.py` covers this document too** — restricting it to the brief is how two contradictions survived here while the guard reported clean.

---

## 15. Note on authorship

Substantial parts of this pipeline were written with AI assistance rather than typed from scratch — particularly the evaluation and verification layer (`generate_validation_spatial_fields.py`, `compute_spatial_verification.py`, `compute_uncertainty_decomposition.py`, `compute_information_content.py`, `compute_feature_importance.py`, `hpo_pixelwise.py`, `generate_future_projections_rf.py`) and major rewrites of `unet_model.py`, `ml_dataset_common.py`, `compute_finegrid_clearsky_ghi.py`, `apply_edcm_bias_correction.py` and the training/evaluation scripts. The analytical decisions, diagnoses and interpretations recorded above should be understood and defensible independently of how the code was produced.
