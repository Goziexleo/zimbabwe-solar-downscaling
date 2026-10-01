"""Baselines the ML models must beat, including the linear one Chapter 1 assumes.

Chapter 1 justifies machine learning by the linearity of classical statistical
downscaling: the relationship between predictors and irradiance is argued to be
nonlinear, so linear methods are said to be inadequate. No linear model was ever
fitted, so that premise was asserted and never tested. An examiner identified
this, together with the absence of any traditional baseline in Table 4.1.

Four references are scored here on exactly the data the ML models are scored on,
through the same build_fine_pixel_dataset call, the same clear-sky-index target
and the same csi_to_ghi conversion, so the comparison is controlled:

  linear (OLS)   per-cell ordinary least squares on the identical predictors.
                 This is the direct test of the nonlinearity premise: if the
                 trees beat it by little, Chapter 1's argument does not hold.
  linear (ridge) the same with L2 regularisation, because several predictors are
                 collinear over 312 months and unregularised OLS can be unstable
                 where a cell's cloud and humidity series nearly coincide.
  climatology    per-cell, per-calendar-month mean of the TRAINING record. The
                 only admissible skill reference (Section 3.7.3).
  bilinear       interpolation of the coarse rsds field. A circularity
                 diagnostic, not a skill reference: rsds is a reuse of the same
                 ssrd the target is built from.

Every metric is reported over the full analysis box and over Zimbabwe, as in
compute_table33.py, because the box is 42.8 per cent outside the country.
"""

import os
import time

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge

from ml_dataset_common import csi_to_ghi, evaluation_baselines
from pixelwise_common import build_fine_pixel_dataset
from zimbabwe_mask import describe as mask_describe
from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
VAL = os.path.join(ROOT, "data/processed/ml_ready/ml_validation_dataset.nc")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
OUT_CSV = os.path.join(ROOT, "data/processed/evaluation/baselines.csv")
OUT_NC = os.path.join(ROOT, "data/processed/evaluation/baseline_fields.nc")

RIDGE_ALPHA = float(os.environ.get("RIDGE_ALPHA", "1.0"))


def metrics(pred, truth, mask=None):
    if mask is not None:
        pred, truth = pred[:, mask], truth[:, mask]
    err = pred - truth
    return {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(np.abs(err).mean()),
        "Pearson R": float(np.corrcoef(pred.ravel(), truth.ravel())[0, 1]),
        "MBE": float(err.mean()),
    }


def fit_cell(Xtr, ytr, Xva, kind):
    m = (LinearRegression() if kind == "ols"
         else Ridge(alpha=RIDGE_ALPHA, random_state=0))
    m.fit(Xtr, ytr)
    return m.predict(Xva)


def fit_pixelwise(X_train, y_train, X_val, kind):
    n_lat, n_lon = X_train.shape[1], X_train.shape[2]
    out = np.zeros((X_val.shape[0], n_lat, n_lon), dtype=np.float64)
    res = Parallel(n_jobs=-1, verbose=0)(
        delayed(fit_cell)(X_train[:, i, j, :], y_train[:, i, j], X_val[:, i, j, :], kind)
        for i in range(n_lat) for j in range(n_lon))
    k = 0
    for i in range(n_lat):
        for j in range(n_lon):
            out[:, i, j] = res[k]
            k += 1
    return out


def main():
    ds_tr, ds_va = xr.open_dataset(TRAIN), xr.open_dataset(VAL)
    ds_cs = xr.open_dataset(CLEARSKY)
    X_train, y_train, feats = build_fine_pixel_dataset(ds_tr, TOPO)
    X_val, y_val, _ = build_fine_pixel_dataset(ds_va, TOPO)
    print("predictors (%d): %s" % (len(feats), ", ".join(feats)))
    print("cells %d x %d, train %d months, validation %d months"
          % (X_train.shape[1], X_train.shape[2], X_train.shape[0], X_val.shape[0]))

    times = ds_va.time.values
    truth = csi_to_ghi(y_val, times, ds_cs)
    interp, clim, _delta = evaluation_baselines(ds_va, truth, times, ds_cs)

    preds, varname = {}, {}
    for kind, label in (("ols", "Linear regression (OLS)"),
                        ("ridge", "Linear regression (ridge)")):
        varname[label] = "ghi_linear_" + kind
        t0 = time.time()
        csi = fit_pixelwise(X_train, y_train, X_val, kind)
        preds[label] = csi_to_ghi(csi, times, ds_cs)
        print("  %-28s fitted in %5.1fs" % (label, time.time() - t0))
    preds["Climatology (training record)"] = clim
    preds["Bilinear interpolation"] = interp
    varname["Climatology (training record)"] = "ghi_climatology"
    varname["Bilinear interpolation"] = "ghi_bilinear"

    mask = zimbabwe_mask(truth.shape)
    print("mask: " + mask_describe())

    rows = []
    for label, p in preds.items():
        m = metrics(p, truth)
        m.update({k + "_zw": v for k, v in metrics(p, truth, mask=mask).items()})
        m["baseline"] = label
        rows.append(m)
    cols = ["RMSE", "MAE", "Pearson R", "MBE"]
    df = pd.DataFrame(rows)[["baseline"] + cols + [c + "_zw" for c in cols]]
    df.to_csv(OUT_CSV, index=False)

    # Named from an explicit map. Deriving the variable name from the first word
    # of the label collapsed OLS and ridge into one array, because both labels
    # begin with "Linear" and the later write silently won.
    xr.Dataset(
        {varname[k]: (("time", "fine_lat", "fine_lon"), v) for k, v in preds.items()},
        coords={"time": times, "fine_lat": ds_va.fine_lat.values,
                "fine_lon": ds_va.fine_lon.values},
    ).to_netcdf(OUT_NC)

    print("\n" + "=" * 92)
    print(" BASELINES, validation 2011-2024 (W m-2), over Zimbabwe")
    print("=" * 92)
    print(df[["baseline"] + [c + "_zw" for c in cols]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 92)

    # Is the linear model's margin established, or resampling noise? Same paired
    # year-block bootstrap Section 3.8.2 uses for the model differences, so the
    # answer is comparable with Table 4.4 rather than a separate convention.
    v = xr.open_dataset(os.path.join(ROOT,
        "data/processed/evaluation/validation_spatial_fields.nc"))
    tt = pd.DatetimeIndex(times)
    yrs = np.unique(tt.year)
    ols_f = preds["Linear regression (OLS)"]
    rng = np.random.default_rng(42)
    B = int(os.environ.get("BASELINE_BOOT", "2000"))
    rm = lambda p, sel: float(np.sqrt((((p - truth)[sel][:, mask]) ** 2).mean()))
    boot = []
    for name, var in (("XGBoost", "ghi_xgb"), ("CNN", "ghi_cnn"),
                      ("U-Net", "ghi_unet"), ("Random Forest", "ghi_rf")):
        p_ = v[var].values
        d = np.empty(B)
        for k in range(B):
            yy = rng.choice(yrs, size=len(yrs), replace=True)
            sel = np.concatenate([np.where(tt.year == y)[0] for y in yy])
            d[k] = rm(p_, sel) - rm(ols_f, sel)
        lo, hi = np.percentile(d, [2.5, 97.5])
        boot.append(dict(model=name, diff_vs_linear=rm(p_, slice(None)) - rm(ols_f, slice(None)),
                         ci_lo=lo, ci_hi=hi,
                         verdict=("linear established better" if lo > 0 else
                                  "indistinguishable" if hi > 0 else "ML established better")))
    bdf = pd.DataFrame(boot)
    bdf.to_csv(os.path.join(ROOT, "data/processed/evaluation/baseline_vs_models.csv"),
               index=False)
    print("\npaired year-block bootstrap, model minus linear OLS (positive = linear better)")
    print(bdf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    t33 = pd.read_csv(os.path.join(ROOT, "data/processed/evaluation/table_3_3.csv"))
    best = t33.sort_values("RMSE_zw").iloc[0]
    ols = df[df.baseline == "Linear regression (OLS)"].iloc[0]
    gain = 100 * (ols["RMSE_zw"] - best["RMSE_zw"]) / ols["RMSE_zw"]
    print("\nTHE TEST OF CHAPTER 1's PREMISE")
    print("  best ML model (%s)      RMSE %.4f" % (best["model"], best["RMSE_zw"]))
    print("  per-cell linear regression on the same predictors  RMSE %.4f" % ols["RMSE_zw"])
    print("  the nonlinear models are worth %.1f per cent of error" % gain)
    if gain <= 0:
        print("  NEGATIVE: the linear model is at least as good, so Chapter 1's")
        print("  justification for machine learning is not supported on this target.")
    print("\nSaved %s\nSaved %s" % (OUT_CSV, OUT_NC))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
