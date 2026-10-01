"""Score the downscaled product against SARAH, and against ERA5, over 2011-2024.

Section 4.9 states that the study validates against a withheld ERA5 record rather
than against observations, and that this is its central limitation. The SARAH
record is held (479 usable monthly fields at 0.05 degrees, 1985-2024), so the
obvious question is why the product is not simply scored against it.

This script answers that quantitatively instead of by assertion. It reports, over
the same 168-month withheld period, each model's error against the ERA5-derived
target it was fitted to and against the independent SARAH retrieval, together
with the difference between the two references measured on the same months.

The point it establishes: the ERA5-SARAH reference difference is several times
larger than the differences between the models. A score against SARAH is
therefore dominated by which reanalysis the target came from rather than by how
well any model downscales, so it cannot be used to rank architectures or to
report skill without separating the two, and the honest route to an observational
validation is to retrain against SARAH as the target rather than to re-score an
ERA5-trained product against it.

    python validate_against_sarah.py

Writes data/processed/evaluation/sarah_product_validation.csv.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "data/processed/evaluation")
CACHE = os.path.join(EVAL, "_sarah_regrid_cache")
FIELDS = os.path.join(EVAL, "validation_spatial_fields.nc")
OUT = os.path.join(EVAL, "sarah_product_validation.csv")

MODELS = [("XGBoost", "ghi_xgb"), ("Random Forest", "ghi_rf"),
          ("CNN", "ghi_cnn"), ("U-Net", "ghi_unet"),
          ("Baseline (bilinear)", "ghi_baseline")]


def rmse(a, b, m):
    d = (a - b)[m]
    return float(np.sqrt(np.nanmean(d ** 2)))


def bias(a, b, m):
    return float(np.nanmean((a - b)[m]))


def main():
    from zimbabwe_mask import describe as mask_describe
    from zimbabwe_mask import zimbabwe_mask
    ds = xr.open_dataset(FIELDS)
    times = pd.DatetimeIndex(ds.time.values)
    truth = ds["ghi_true"].values

    sarah = np.full_like(truth, np.nan)
    missing = []
    for k, t in enumerate(times):
        f = os.path.join(CACHE, "%04d-%02d.npy" % (t.year, t.month))
        if os.path.exists(f):
            sarah[k] = np.load(f)
        else:
            missing.append(t.strftime("%Y-%m"))
    if missing:
        print("SARAH months absent from the cache: %s" % ", ".join(missing))

    # Compare only where both references exist, so every figure below is over
    # exactly the same cells and months.
    both = np.isfinite(sarah) & np.isfinite(truth)
    print("months %d | cells per month %d | usable values %d of %d (%.1f%%)"
          % (len(times), truth[0].size, both.sum(), both.size,
             100 * both.sum() / both.size))

    # Restricted to Zimbabwe as well. Two fifths of the analysis box is in
    # neighbouring countries, and a satellite comparison described as a check
    # over Zimbabwe should be one.
    mask = zimbabwe_mask(truth.shape)
    both_zw = both & mask[None, :, :]
    print("mask: %s -> %d usable values inside" % (mask_describe(), both_zw.sum()))

    rows = []
    for name, var in MODELS:
        if var not in ds:
            continue
        p = ds[var].values
        rows.append(dict(
            model=name,
            rmse_vs_era5=rmse(p, truth, both),
            rmse_vs_sarah=rmse(p, sarah, both),
            bias_vs_era5=bias(p, truth, both),
            bias_vs_sarah=bias(p, sarah, both),
            rmse_vs_era5_zw=rmse(p, truth, both_zw),
            rmse_vs_sarah_zw=rmse(p, sarah, both_zw),
            bias_vs_era5_zw=bias(p, truth, both_zw),
            bias_vs_sarah_zw=bias(p, sarah, both_zw)))

    ref = dict(model="ERA5 target itself",
               rmse_vs_era5=0.0,
               rmse_vs_sarah=rmse(truth, sarah, both),
               bias_vs_era5=0.0,
               bias_vs_sarah=bias(truth, sarah, both),
               rmse_vs_era5_zw=0.0,
               rmse_vs_sarah_zw=rmse(truth, sarah, both_zw),
               bias_vs_era5_zw=0.0,
               bias_vs_sarah_zw=bias(truth, sarah, both_zw))
    rows.append(ref)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("\n%-22s %12s %12s %12s %12s"
          % ("", "RMSE v ERA5", "RMSE v SARAH", "bias v ERA5", "bias v SARAH"))
    for r in rows:
        print("%-22s %12.3f %12.3f %+12.3f %+12.3f"
              % (r["model"], r["rmse_vs_era5"], r["rmse_vs_sarah"],
                 r["bias_vs_era5"], r["bias_vs_sarah"]))

    models_only = [r for r in rows if r["model"] != "ERA5 target itself"
                   and r["model"] != "Baseline (bilinear)"]
    spread = (max(r["rmse_vs_era5"] for r in models_only)
              - min(r["rmse_vs_era5"] for r in models_only))
    print("\nspread across the four models, against ERA5   %.3f W m-2" % spread)
    print("the ERA5 target's own distance from SARAH      %.3f W m-2"
          % ref["rmse_vs_sarah"])
    print("ratio                                          %.1fx" % (ref["rmse_vs_sarah"] / spread))
    print("\nSaved %s" % OUT)


if __name__ == "__main__":
    main()
