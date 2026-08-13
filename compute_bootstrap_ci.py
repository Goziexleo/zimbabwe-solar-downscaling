"""Confidence intervals on Table 3.3, and on the differences between models.

Every metric in this study is a point estimate. The deployment argument in
Section 3.8.4 turns on differences - a 12 per cent RMSE gap, a fivefold
difference in mean bias - with nothing said about whether those gaps are
distinguishable from sampling noise. This script supplies that.

Design:

  Resampling unit is the CALENDAR YEAR, not the individual month or grid cell.
  Residuals are strongly correlated in space (neighbouring cells share weather)
  and in time (consecutive months share a season), so resampling cells or months
  independently would treat correlated errors as independent and produce
  intervals far too narrow. Resampling whole years keeps all 5,751 cells and all
  12 months of a year together, preserving both structures within each unit.

  The bootstrap is PAIRED: every model is scored on the same resampled years in
  each replicate. Differences between models are therefore computed within
  replicate, which removes the year-to-year variation common to all models and
  is what makes the interval on a difference much tighter - and more
  informative - than the intervals on the two individual metrics.

A difference whose 95 per cent interval excludes zero is one that survives
resampling; one whose interval spans zero is not distinguishable from noise at
this sample size, whatever the point estimates suggest.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from ml_dataset_common import evaluation_baselines

ROOT = os.path.dirname(os.path.abspath(__file__))
FIELDS = os.path.join(ROOT, "data/processed/evaluation/validation_spatial_fields.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/bootstrap_ci.csv")
OUT_DIFF = os.path.join(ROOT, "data/processed/evaluation/bootstrap_differences.csv")

N_BOOT = int(os.environ.get("N_BOOT", "2000"))
SEED = 42

MODELS = [("Random Forest", "ghi_rf"), ("XGBoost", "ghi_xgb"),
          ("CNN", "ghi_cnn"), ("U-Net", "ghi_unet")]


def main():
    ds_f = xr.open_dataset(FIELDS)
    ds_val = xr.open_dataset(VAL)
    ds_cs = xr.open_dataset(CLEARSKY)
    truth = ds_f["ghi_true"].values
    times = pd.DatetimeIndex(ds_val.time.values)
    years = times.year.values
    uniq_years = np.unique(years)
    _, clim, _ = evaluation_baselines(ds_val, truth, ds_val.time.values, ds_cs)

    preds = {label: ds_f[var].values for label, var in MODELS}
    # index of the months belonging to each year, so a replicate is assembled
    # by concatenating whole years
    year_idx = {y: np.where(years == y)[0] for y in uniq_years}

    print(f"{len(uniq_years)} years, {len(times)} months, {truth.shape[1] * truth.shape[2]} cells")
    print(f"Resampling {N_BOOT} times over whole years (paired across models)...")

    rng = np.random.default_rng(SEED)

    def stats(idx):
        out = {}
        t = truth[idx]
        c = clim[idx]
        rmse_clim = np.sqrt(((c - t) ** 2).mean())
        for label in preds:
            p = preds[label][idx]
            err = p - t
            rmse = np.sqrt((err ** 2).mean())
            out[label] = {
                "RMSE": rmse,
                "MBE": err.mean(),
                "Pearson R": np.corrcoef(p.ravel(), t.ravel())[0, 1],
                "SS vs climatology": 1.0 - rmse / rmse_clim,
            }
        return out

    point = stats(np.arange(len(times)))

    boot = {label: {k: [] for k in ["RMSE", "MBE", "Pearson R", "SS vs climatology"]}
            for label in preds}
    diffs = {}
    for _ in range(N_BOOT):
        draw = rng.choice(uniq_years, size=len(uniq_years), replace=True)
        idx = np.concatenate([year_idx[y] for y in draw])
        s = stats(idx)
        for label in preds:
            for k in boot[label]:
                boot[label][k].append(s[label][k])
        for a, _ in MODELS:
            for b, _ in MODELS:
                if a >= b:
                    continue
                diffs.setdefault((a, b, "RMSE"), []).append(s[a]["RMSE"] - s[b]["RMSE"])
                diffs.setdefault((a, b, "MBE"), []).append(abs(s[a]["MBE"]) - abs(s[b]["MBE"]))

    rows = []
    for label in preds:
        for k in ["RMSE", "MBE", "Pearson R", "SS vs climatology"]:
            arr = np.array(boot[label][k])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rows.append({"model": label, "metric": k, "estimate": point[label][k],
                         "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 88)
    print(f" TABLE 3.3 WITH 95% CONFIDENCE INTERVALS ({N_BOOT} year-block bootstrap replicates)")
    print("=" * 88)
    for k in ["RMSE", "Pearson R", "MBE", "SS vs climatology"]:
        print(f"\n{k}:")
        sub = df[df["metric"] == k].sort_values("estimate")
        for _, r in sub.iterrows():
            print(f"  {r['model']:<15} {r['estimate']:8.4f}   95% CI [{r['ci_lo']:7.4f}, {r['ci_hi']:7.4f}]")

    drows = []
    print("\n" + "=" * 88)
    print(" PAIRED DIFFERENCES — does the gap survive resampling?")
    print("=" * 88)
    for (a, b, metric), vals in sorted(diffs.items()):
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        drows.append({"model_a": a, "model_b": b, "metric": metric,
                      "mean_difference": arr.mean(), "ci_lo": lo, "ci_hi": hi,
                      "excludes_zero": excludes_zero})
        if metric == "RMSE":
            verdict = "distinguishable" if excludes_zero else "NOT distinguishable"
            print(f"  {a:<15} − {b:<15} RMSE  {arr.mean():+7.4f}  "
                  f"95% CI [{lo:+7.4f}, {hi:+7.4f}]  {verdict}")
    pd.DataFrame(drows).to_csv(OUT_DIFF, index=False)

    key = [d for d in drows if {d["model_a"], d["model_b"]} == {"XGBoost", "Random Forest"}
           and d["metric"] == "RMSE"][0]
    print(f"""
The comparison Section 3.8.4 turns on:
  XGBoost against Random Forest, RMSE difference {key['mean_difference']:+.4f} W m-2,
  95% CI [{key['ci_lo']:+.4f}, {key['ci_hi']:+.4f}] -> {'excludes zero' if key['excludes_zero'] else 'SPANS ZERO'}.
""")
    print(f"Saved: {OUT_CSV}\nSaved: {OUT_DIFF}")


if __name__ == "__main__":
    main()
