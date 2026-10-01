# Extending the domain to cover all of Zimbabwe

## The problem

Section 3.2.1 specifies 15.0–22.5 S and 25.0–33.5 E. The ERA5 download actually
requested `[-15, 25, -22, 33]`, half a degree short at the south and east, so the
analysis grid stops **at** 22.0 S. Zimbabwe does not: Beitbridge is at 22.217 S.

Evidence the truncation bites, from `zimbabwe_mask.py`:

```
southernmost row with land: -22.0
cells on the southern edge row (-22.0): 27   <- non-zero means the grid cuts the country
```

27 cells of Zimbabwe sit on the boundary row with more country beyond it. About
51 cells, roughly 6,000 km², are missing.

`ml_dataset_common.py` already records the discrepancy, and `CORE_LAT_MIN` is set
to −22.0 to match the data that exists rather than the data that was specified.

## Why this is not a quick fix

The domain is upstream of everything. Changing it invalidates the target, every
trained model, every projection and the suitability layers. The download scripts
now request the correct box, but **the constants in `ml_dataset_common.py` have
deliberately not been changed**, because they describe the data on disk and moving
them first would break the pipeline rather than extend it.

## Order of operations

Run from the project root with the `climate_stack` environment and
`KMP_DUPLICATE_LIB_OK=TRUE` set.

1. **Download.** `download_era5_predictors2.py` and
   `download_era5_validation_predictors.py` already carry the corrected box.
   Needs `~/.cdsapirc`. Expect hours, and the CDS queue is the bottleneck.
2. **Re-request the auxiliary layers** over the same box: SRTM topography, ESA
   CCI land cover, WorldPop, WDPA, HydroSHEDS, OpenStreetMap lines, GADM. The
   suitability criteria are built on the fine grid and will not extend themselves.
3. **Update the constants** in `ml_dataset_common.py`: `CORE_LAT_MIN` to −22.5 and
   the matching longitude bound to 33.5. Do this only once the files are present.
4. **Rebuild** the clear-sky climatology, the CSI target and the ML-ready datasets.
5. **Retrain** all four architectures. The per-cell models scale with cell count,
   so expect roughly a 10 per cent increase in fitting time.
6. **Re-project** and regenerate the MME aggregations for each architecture.
7. **Re-run** `compute_table33.py`, `compute_spatial_verification.py`,
   `compute_baselines.py`, `compute_uncertainty_decomposition.py`,
   `compute_projection_baselines.py`, `compute_suitability.py`,
   `compute_robust_set_geography.py`, `compute_robustness_monte_carlo.py`.
8. **Regenerate** figures and chapters, then `splice_results_chapters.py`.

## What to expect

Most headline numbers should move little: the added cells are a 1.5 per cent
increase in national area at the dry southern margin. Two things will change and
should be reported — the robust set's province breakdown, since Matabeleland
South and Masvingo gain area, and any statement about the southern lowveld, which
is currently made from a grid that excludes part of it.

## If the rerun is not possible before submission

Say so plainly in Section 3.2.1 and Section 4.9 rather than leaving the stated
domain and the computed domain in disagreement: give the actual box, state that
it excludes roughly 51 cells of southern Zimbabwe including Beitbridge, and note
that every "over Zimbabwe" figure is therefore over 98.5 per cent of the country.
An examiner who checks the coordinates will find this; it reads very differently
as a disclosed limitation than as an unnoticed error.
