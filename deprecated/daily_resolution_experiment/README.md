# Deprecated: daily-resolution experiment

These files are kept for the record and are **not part of the deployed pipeline**.
Do not run them expecting results consistent with the current products.

## What the experiment was

The monthly training record has only 312 samples, which is small for the neural
architectures. To test whether sample size was the binding constraint, the
Random Forest and U-Net were retrained on daily-aggregated ERA5 data (9,496
training / 5,114 validation samples), which required a 365-day clear-sky
climatology in place of the 12-month one.

## Why it is not deployed

Both daily models reached much higher correlation than their monthly
counterparts (RF 0.9794, U-Net 0.9977) and both scored **worse than doing
nothing**: bilinearly interpolating their own `rsds` predictor gives RMSE
0.65 W m-2, against 13.39 and 4.33 respectively.

The cause is that `rsds` was populated by reusing the ERA5 `ssrd` field that the
clear-sky index target is itself derived from, so at daily resolution a naive
interpolation of the predictor already reproduces the target almost exactly. The
high correlation measured the model recovering its own input.

That finding is now understood to apply at **monthly** resolution too — it was
masked there by a one-month misalignment in the validation predictor file. The
project's response was to drop `rsds` from the model input set entirely (see
`MODEL_PREDICTOR_VARS` in `ml_dataset_common.py`), which is why this experiment
is superseded rather than merely paused.

CMIP6 also publishes only monthly output for these variables, so there is no
daily analogue on the projection side in any case.

## Why the files are quarantined rather than fixed

`compute_finegrid_clearsky_daily.py` still uses `altitude=1000.0` and
`linke_turbidity=3.0` as domain-wide constants. The monthly script was corrected
to per-cell SRTM elevation and per-cell/per-month Linke turbidity, so the two
clear-sky definitions disagree, and `clearsky_ghi_finegrid_climatology_daily.nc`
in this directory was produced under the old constants.

Leaving an inconsistent second clear-sky definition in the live tree is a trap:
anything that picked it up would silently produce results incomparable with the
deployed products. If the experiment is ever revisited, apply the same
elevation and turbidity treatment as `compute_finegrid_clearsky_ghi.py` first,
and regenerate the climatology before use.

## Contents

- `compute_finegrid_clearsky_daily.py` — 365-day clear-sky climatology (stale constants)
- `build_ml_daily_dataset.py` — daily training/validation dataset builder
- `clearsky_ghi_finegrid_climatology_daily.nc` — output of the above, stale
