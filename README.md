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

```bash
# 1. acquisition and preprocessing
python download_allcmip6_data.py
python download_era5_predictors2.py
python process_era5_predictors.py
python apply_edcm_bias_correction.py        # quantile mapping
python apply_qc_bounds.py                   # physical bounds, writes flag counts
python generate_topographic_features.py
python compute_finegrid_clearsky_ghi.py     # PVLIB clear-sky ceiling
python build_ml_features_and_targets.py
python build_ml_validation_dataset.py

# 2. training                                        approximate runtime
python train_pixelwise_rf.py                        # 25 min
python train_pixelwise_xgb.py                       # 15 min
python train_cnn_downscaler.py                      #  5 min
python train_unet_downscaler.py                     # 30 min

# 3. evaluation
python evaluate_cnn.py && python evaluate_unet.py
python generate_validation_spatial_fields.py
python compute_table33.py                           # headline metrics
python compute_spatial_verification.py              # Taylor statistics
python compute_bootstrap_ci.py
python compute_bca_centred_rmse.py                  # BCa intervals
python compute_power_spectra.py                     # effective resolution
python compute_information_content.py               # round-trip test
python compute_feature_importance.py
python compute_rolling_origin.py                    # 2.5 h, refits every fold
python validate_against_sarah.py

# 4. projection
python generate_future_projections_xgb.py
PROJECTIONS_DIR=$PWD/data/processed/projections_xgb \
MME_OUTPUT_DIR=$PWD/data/processed/mme_aggregations_xgb \
  python compute_mme_aggregations.py
python compute_scenario_discrimination.py           # the deployment screen
python compute_uncertainty_decomposition.py

# 5. suitability
python build_suitability_layers.py
python compute_suitability.py

# 6. figures and documents
python make_figures.py
python make_suitability_maps.py
python build_chapter4.py
python build_chapter5.py
python build_thesis.py                              # assembles the dissertation
```

`run_neural_cascade.sh` runs stages 2 to 6 for the neural models with per-step exit
checks, which exists because an earlier version of that cascade reported success
after every step had failed.

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

| Path | Contents |
|---|---|
| `*.py` | analysis scripts, one stage each |
| `zimbabwe_mask.py` | the national mask every "over Zimbabwe" metric uses |
| `compute_baselines.py` | linear, ridge, climatology and interpolation baselines |
| `compute_projection_baselines.py` | the quantile-mapped operational baseline, and the driver-consistency check |
| `compute_robust_set_geography.py` | provinces and true areas of the robust set |
| `build_chapter4.py`, `build_chapter5.py` | generate the results chapters |
| `splice_results_chapters.py` | swaps those chapters into the dissertation, preserving citation fields |
| `normalise_tables.py` | puts every table on one format |
| `build_thesis.py` | assembles a full dissertation from scratch (see the warning below) |
| `check_*.py` | document consistency guards |
| `tests/` | pytest invariants |
| `brief/` | viva brief and slide deck, generated from `brief/viva-brief.html` |
| `assets/` | University of Zimbabwe logo used on the title page |
| `scripts/` | helper utilities |
| `PROJECT_STATUS.md` | authoritative record of results, defects and limitations |

`data/`, `figures/` and `logs/` are generated and not tracked.

**`build_thesis.py` rebuilds the whole document and will destroy the citation
fields inserted by hand into the merged file.** Once reference management has
begun, update the results chapters with `splice_results_chapters.py`, which
replaces only Chapters 4 and 5 and checks the field count before and after.

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
themselves, recorded in `CITATION_AUDIT.md`, found several attributions that the
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
