# Machine learning downscaling of GCM output for solar energy suitability over Zimbabwe

Analysis code for an MSc dissertation (University of Zimbabwe, Department of Space
Science and Applied Physics). CMIP6 surface solar radiation is downscaled to a 0.1°
grid over Zimbabwe, four machine learning architectures are benchmarked against a
withheld reanalysis record, the deployed model is projected to 2100 under two SSP
pathways, and the resulting irradiance fields are carried into a multi-criteria
suitability assessment for utility-scale photovoltaic development.

**Headline results are in `PROJECT_STATUS.md`**, which is the authoritative record of
what was found, what was corrected, and what remains a limitation. This file covers
how to run the analysis and how to check it.

## Why this repository exists

Making code available is the low bar. The point of this repository is that the claims
in the dissertation should be **checkable rather than merely stated**, and two design
decisions carry that.

**The results chapters are generated, not typed.** `build_chapter4.py` and
`build_chapter5.py` write Chapters 4 and 5 from the analysis outputs, so a number
quoted in the prose cannot drift from the analysis behind it. Prose drifting from
tables was the most frequent defect encountered during this work; generating the
chapters removes the possibility rather than guarding against it.

**The written documents are guarded by tests.** `pytest` runs invariant checks over
the pipeline plus three consistency guards that compare the written chapters cell by
cell against the canonical result files, reject figures known to have been
superseded, and verify that Zotero citation fields remain structurally intact. These
run before any set of results is accepted.

The audit history is not hidden. `PROJECT_STATUS.md` records every defect found,
including two forms of information leakage between the training and evaluation
records, a set of figures whose plotted titles had fallen out of step with the
analysis, and a validation statistic derived from the answer it claimed to validate.
Each entry states what the number was, what it is, and why.

## Environment

```bash
conda env create -f environment.yml -n climate_stack
conda activate climate_stack
```

`environment.yml` is a full pinned export. The repo-local `env/` directory is **not**
the analysis environment and must not be used: it has no torch, scikit-learn or
xgboost. Verify with `python -c "import torch, xgboost, sklearn, xarray"`.

### The OpenMP workaround

On macOS, torch and the OpenMP-linked libraries collide in one process. Scripts that
spawn training set `KMP_DUPLICATE_LIB_OK=TRUE`; if you import torch alongside xarray
in your own session, set it too:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Data

**Raw inputs are not redistributed.** Every dataset is obtained under its own licence
and several require registration or accepting terms, so the repository documents
acquisition and ships the download scripts instead of the files.

| Dataset | Used for | Access |
|---|---|---|
| CMIP6 (ACCESS-CM2, CNRM-CM6-1, MPI-ESM1-2-HR) | predictors | ESGF, open data licence — `download_allcmip6_data.py` |
| ERA5 | training target and predictors | Copernicus CDS, free for research — `download_era5_predictors2.py` |
| CM SAF SARAH | independent comparison, present-day suitability layer | EUMETSAT CM SAF order, CC BY 4.0 — `fetch_sarah_order.py` |
| SRTM 90 m DEM | slope, elevation, sky-view factor | CGIAR-CSI, public domain — `download_srtm_dem.py` |
| ESA CCI Land Cover | exclusion mask, land-cover score | CDS, licence acceptance required |
| WorldPop | population criterion | open — `download_suitability_data.py` |
| OpenStreetMap | roads, transmission grid, settlements | ODbL — `download_suitability_data.py` |
| WDPA / ZimParks | protected-area exclusions | protectedplanet.net, terms accepted in a browser |
| HydroSHEDS / HydroRIVERS | riparian exclusions | open — `download_suitability_data.py` |

Expected layout after acquisition:

```
data/raw/{cmip6,era5,sarah,srtm,osm,worldpop,protected areas,hydrosheds}/
data/processed/                # everything below is generated
```

## Running the analysis

Stages are ordered; each depends on the ones above it. Runtimes are for an Apple M1
Air and are dominated by the per-cell models, which fit one estimator per grid cell.
Every script derives its paths from its own location, so run them from the
repository root.

```bash
# 1. acquisition                                     (each needs its own credentials)
python download_allcmip6_data.py                     # CMIP6 predictors, ESGF
python download_era5_predictors2.py                  # ERA5 training predictors + target
python download_era5_validation_predictors.py        # ERA5 2011-2024
python download_srtm_dem.py                          # SRTM 90 m
python download_suitability_data.py                  # WorldPop, OSM, land cover, WDPA
python fetch_sarah_order.py                          # CM SAF SARAH, after the order lands

# 2. preprocessing
python process_era5_predictors.py
python process_era5_validation_predictors.py
python apply_edcm_bias_correction.py                 # quantile mapping (EDCDFm)
python apply_qc_bounds.py                            # physical bounds, writes flag counts
python generate_topographic_features.py              # slope, elevation, sky-view factor
python compute_finegrid_clearsky_ghi.py              # PVLIB clear-sky ceiling
python merge_predictor_stack.py
python build_ml_features_and_targets.py
python build_ml_validation_dataset.py

# 3. hyperparameter selection, inside the training period only
python hpo_pixelwise.py                              # 5-fold temporal CV
python hpo_neural.py                                 # inner 2005-2010 split
python optimise_unet.py                              # controlled variant sweep

# 4. training                                        approximate runtime
python train_pixelwise_rf.py                         # 25 min
python train_pixelwise_xgb.py                        # 15 min
python train_cnn_downscaler.py                       #  5 min
python train_unet_downscaler.py                      # 30 min

# 5. evaluation on the withheld 2011-2024 record
python evaluate_cnn.py && python evaluate_unet.py
python generate_validation_spatial_fields.py
python compute_table33.py                            # headline metrics (Table 4.1)
python compute_spatial_verification.py               # Taylor statistics
python compute_baselines.py                          # OLS, ridge, climatology, interpolation
python compute_bootstrap_ci.py
python compute_bca_centred_rmse.py                   # percentile, basic and BCa intervals
python compute_power_spectra.py                      # spectra and the cut sensitivity
python compute_information_content.py                # round-trip test
python compute_feature_importance.py
python compute_rolling_origin.py                     # 2.5 h, refits every fold
python compute_rsds_ablation.py                      # predictor ablation (Table 3.4)
python compute_pixelwise_cv_config_check.py
python compute_dropout_variant_comparison.py
python test_perfect_prognosis.py                     # ERA5 -> CMIP6 transfer test
python validate_against_sarah.py
python compare_sarah_era5.py && python compare_sarah_clearsky.py

# 6. projection
python generate_future_projections_xgb.py            # the deployed model
python generate_future_projections_rf.py
python generate_future_projections_cnn.py
python generate_future_projections.py                # U-Net
PROJECTIONS_DIR=$PWD/data/processed/projections_xgb \
MME_OUTPUT_DIR=$PWD/data/processed/mme_aggregations_xgb \
  python compute_mme_aggregations.py
python compute_projection_baselines.py               # operational baseline, driver consistency
python compute_scenario_discrimination.py            # the deployment screen
python compute_uncertainty_decomposition.py          # four-component variance budget
python compute_pv_temperature_derating.py            # yield once modules heat

# 7. suitability
python build_suitability_layers.py
python compute_suitability.py
python compute_robustness_monte_carlo.py
python compute_robust_set_geography.py
python compute_elevation_irradiance_gradient.py

# 8. figures and the document
python make_study_area_map.py
python make_figures.py
python make_suitability_maps.py
./finalise_dissertation.sh                           # the only document entry point
```

`run_neural_cascade.sh` runs stages 4 to 8 for the neural models with per-step exit
checks, which exists because an earlier version of that cascade reported success
after every step had failed.

### The document chain

`finalise_dissertation.sh` is the single entry point for updating the dissertation,
and the order inside it matters. It regenerates Chapters 4 and 5 from the evaluation
outputs, splices them into the document while preserving the Zotero citation fields,
rewrites the abstract from the same CSVs, applies the presentation and factual
passes, rebuilds Appendix C, then the front-matter lists, then the table formatting,
and finally runs the figure-freshness guard and the verification pass. Running any
stage alone reintroduces something a later stage fixes.

Chapters 1 to 3 are not generated. They are edited in place, which is why the
corrections to them live in idempotent scripts the chain runs each time
(`fix_facts.py`, `fix_presentation.py`, `fix_unet_refit_recorded.py`,
`fix_methods_chapter_result_values.py`, `fix_method_citations_v7.py`,
`add_appendix_c_crossrefs.py`, `insert_missing_references.py`) rather than in a
generator. Each reports "nothing to change" once applied.

## Checking it

```bash
pytest -q                          # invariants and the three document guards
python check_status_consistency.py # PROJECT_STATUS against the canonical CSVs
python check_brief_consistency.py  # the interview brief and deck
CHAPTER3_PATH=/path/to/Chapter3.docx python check_chapter3_consistency.py
```

The guards need `python-docx` and `python-pptx`; run them in `climate_stack`. Under
the wrong interpreter they exit non-zero and say so rather than reporting a pass for
a document they never opened.

## Hyperparameter selection

`hpo_pixelwise.py` searches the tree models by 5-fold temporal cross-validation
within the training period. `hpo_neural.py` searches the CNN and U-Net by fitting on
1985–2004 and scoring on an inner 2005–2010 split, so no candidate is ever compared
on the withheld record. `optimise_unet.py` is a controlled sweep whose findings, and
whose limits, are recorded in `PROJECT_STATUS.md` §7.18.

## Repository layout

Flat by design: every script reads and writes under `data/` relative to its own
location, so the stage order lives in this README rather than in directory names.

| Path | Contents |
|---|---|
| `download_*.py`, `fetch_*.py` | data acquisition, one source each |
| `process_*.py`, `apply_*.py`, `build_ml_*.py`, `merge_predictor_stack.py` | preprocessing to the model-ready stack |
| `hpo_*.py`, `optimise_unet.py` | hyperparameter selection, inside the training period only |
| `train_*.py` | the four architectures; `pixelwise_common.py`, `unet_model.py`, `cnn_model.py` are their shared parts |
| `evaluate_*.py`, `compute_*.py`, `test_perfect_prognosis.py`, `validate_against_sarah.py` | evaluation, one result file each |
| `generate_future_projections*.py` | projections, one per architecture |
| `build_suitability_layers.py`, `compute_suitability.py` | the multi-criteria overlay |
| `make_*.py` | figures and maps |
| `build_chapter4.py`, `build_chapter5.py`, `build_appendix_c.py` | generate the results chapters and the supporting-table appendix |
| `splice_results_chapters.py` | swaps those chapters in, preserving the Zotero citation fields |
| `finalise_dissertation.sh` | the document chain; the only entry point for updating the dissertation |
| `fix_*.py`, `add_*.py`, `insert_missing_references.py` | idempotent corrections to the hand-edited chapters, run by the chain |
| `zimbabwe_mask.py` | the national mask every "over Zimbabwe" metric uses |
| `check_*.py` | document consistency guards |
| `tests/` | pytest invariants, including the three document guards |
| `assets/` | University of Zimbabwe logo used on the title page |
| `scripts/`, `deprecated/` | earlier variants of the download scripts, and a daily-resolution experiment that was abandoned; kept for the record and not part of any stage |
| `PROJECT_STATUS.md` | authoritative record of results, defects and limitations |

`data/`, `figures/` and `logs/` are generated and not tracked.

The repository holds only what produces the result. One-off scripts that patched
the document once and were consumed, the orchestration scripts written for
particular reruns, and an abandoned MERRA-2 predictor branch were removed once
spent; they remain in the git history, and `PROJECT_STATUS.md` records what each
round did.

**There is no script that rebuilds the dissertation from scratch.** One existed
and was removed: the merged document carries 77 Zotero citation fields that exist
only in it, and regenerating the file destroys them. Chapters 4 and 5 are updated
through `finalise_dissertation.sh`, which replaces only those chapters and
verifies the field count before and after.

## A limitation worth stating here

Training and validation both use ERA5, so the evaluation is out-of-sample in time but
not independent of the reference. `validate_against_sarah.py` quantifies why
re-scoring the product against the SARAH satellite record does not fix this: the
reference disagreement is larger than the whole spread across architectures by more
than an order of magnitude, so such a score measures the reanalysis rather than the
model. A genuine observational validation requires refitting against SARAH as the
target. The data is held for it; the work is not done.

## Use of AI tools

Prose in the dissertation was edited for clarity and grammar with the assistance of a
large language model (Claude, Anthropic), which was also used to draft analysis code
from specifications, to diagnose defects in it, and to summarise literature. No
research question, method choice, interpretation or reported number originates from
the tool. Appendix B of the dissertation states this in full.

The literature summaries proved to be the weak point. An audit against the sources
themselves, recorded in `PROJECT_STATUS.md`, found several attributions that the
cited papers do not support — among them the claim that Buster et al. (2024) used
ERA5 as a training target, when it used ERA5 as the low-resolution input and NSRDB
and WTK as targets. Those passages have been rewritten. Anyone reusing this work
should treat every citation as needing independent verification, and the audit
file records which have had it.

## Citation

See `CITATION.cff`. Please cite the dissertation rather than this repository alone.

## Licence

Code is released under the MIT Licence (`LICENSE`). The datasets are **not** covered
by it and remain under the terms of their respective providers.
