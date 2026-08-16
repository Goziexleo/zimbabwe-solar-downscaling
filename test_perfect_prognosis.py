"""C1: does the ERA5-learned relationship transfer to bias-corrected CMIP6?

The whole projection chain rests on an untested assumption. The models are
fitted on ERA5 predictors and then applied to bias-corrected CMIP6 predictors
for 2026-2100. Nothing so far establishes that the learned predictor-to-
irradiance mapping survives the change of predictor source.

The test is standard and cheap: take bias-corrected CMIP6 output for the
HISTORICAL period, feed it to the deployed model, and score the result
against the same ERA5 truth used for validation. Comparing that against the
ERA5-driven figures isolates the cost of swapping predictor source, holding the
model, the target and the period fixed.

Note on interpretation: CMIP6 is a free-running climate model, so it reproduces
the STATISTICS of the historical period, not its specific months. A month-by-
month comparison would therefore be unfair by construction and would understate
transfer badly. The comparison here is climatological - per-cell, per-calendar-
month means - which is the quantity the projection products actually report and
the quantity that can legitimately be compared.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from ml_dataset_common import (
    MODEL_PREDICTOR_VARS, compute_astronomical_features, csi_to_ghi, safe_nan_to_num,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(ROOT, "data/processed/ml_ready/ml_training_dataset.nc")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
CMIP6_DIR = os.path.join(ROOT, "data/processed/cmip6_bias_corrected")
# Which pixel-wise family to test. The transfer assumption belongs to whichever
# model is deployed, so this defaults to XGBoost (Section 3.8.4) and can be
# pointed at the Random Forest with PP_MODEL=rf for the comparison.
PP_MODEL = os.environ.get("PP_MODEL", "xgb").lower()
_FAMILY = {"xgb": ("pixelwise_xgb", "xgb", "XGBoost"),
           "rf": ("pixelwise_rf", "rf", "Random Forest")}[PP_MODEL]
MODEL_DIR = os.path.join(ROOT, "data/processed/models", _FAMILY[0])
CELL_PREFIX = _FAMILY[1]
MODEL_LABEL = _FAMILY[2]
OUT_CSV = os.path.join(
    ROOT, f"data/processed/evaluation/perfect_prognosis_transfer_{PP_MODEL}.csv")

GCMS = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
HIST_SLICE = slice("1985-01-01", "2010-12-31")


def monthly_climatology(field, times):
    months = pd.DatetimeIndex(times).month
    return np.stack([field[months == m].mean(axis=0) for m in range(1, 13)], axis=0)


def build_features(ds_src, fine_lat, fine_lon, feature_vars, topo_fields, times):
    """Collocate coarse predictors onto the fine grid, exactly as pixelwise_common does."""
    coarse_lat, coarse_lon = ds_src["lat"].values, ds_src["lon"].values
    ilat = np.array([np.abs(coarse_lat - c).argmin() for c in fine_lat])
    ilon = np.array([np.abs(coarse_lon - c).argmin() for c in fine_lon])

    n_t, n_la, n_lo = len(times), len(fine_lat), len(fine_lon)
    X = np.zeros((n_t, n_la, n_lo, len(feature_vars)), dtype=np.float32)
    for k, var in enumerate(MODEL_PREDICTOR_VARS):
        arr = safe_nan_to_num(ds_src[var].values, f"perfect_prognosis:{var}")
        X[:, :, :, k] = arr[:, ilat, :][:, :, ilon]
    base = len(MODEL_PREDICTOR_VARS)
    for off, var in enumerate(["elevation", "slope", "svf"]):
        X[:, :, :, base + off] = topo_fields[var][np.newaxis, :, :]
    sza, sin_doy, cos_doy = compute_astronomical_features(fine_lat, fine_lon, times)
    X[:, :, :, base + 3] = sza
    X[:, :, :, base + 4] = sin_doy
    X[:, :, :, base + 5] = cos_doy
    return X


def main():
    ds_train = xr.open_dataset(TRAIN)
    ds_cs = xr.open_dataset(CLEARSKY)
    fine_lat = ds_train["fine_lat"].values
    fine_lon = ds_train["fine_lon"].values
    train_times = ds_train.time.values

    with open(os.path.join(MODEL_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    feature_vars = manifest["feature_vars"]
    n_la, n_lo = manifest["n_lat"], manifest["n_lon"]

    ds_topo = xr.open_dataset(TOPO)
    lat_n = "lat" if "lat" in ds_topo.coords else "y"
    lon_n = "lon" if "lon" in ds_topo.coords else "x"
    ds_topo = ds_topo.rename({lat_n: "lat", lon_n: "lon"}).sortby("lat")
    ds_topo = ds_topo.interp(lat=fine_lat, lon=fine_lon, method="linear")
    topo_fields = {v: safe_nan_to_num(ds_topo[v].values, f"perfect_prognosis:{v}")
                   for v in ["elevation", "slope", "svf"]}

    # --- truth: ERA5 climatology over the training period ---
    truth_ghi = csi_to_ghi(
        safe_nan_to_num(ds_train["clear_sky_index"].values, "perfect_prognosis:train_csi"),
        train_times, ds_cs)
    truth_clim = monthly_climatology(truth_ghi, train_times)

    # --- reference: the model driven by ERA5 predictors, same period ---
    print(f"Running the deployed {MODEL_LABEL} on ERA5 predictors (historical)...")
    X_era5 = build_features(ds_train, fine_lat, fine_lon, feature_vars, topo_fields, train_times)

    # --- transfer: the model driven by bias-corrected CMIP6, same period ---
    gcm_X, gcm_times = {}, {}
    for gcm in GCMS:
        path = os.path.join(CMIP6_DIR, f"{gcm}_historical_EDCDFm_corrected.nc")
        if not os.path.exists(path):
            path = os.path.join(CMIP6_DIR, f"{gcm}_ssp245_EDCDFm_corrected.nc")
            if not os.path.exists(path):
                print(f"  no bias-corrected historical file for {gcm}, skipping")
                continue
        ds_g = xr.open_dataset(path).sortby("lat").sel(time=HIST_SLICE)
        if ds_g.sizes.get("time", 0) == 0:
            print(f"  {gcm}: no overlap with {HIST_SLICE.start}-{HIST_SLICE.stop}, skipping")
            continue
        missing = set(MODEL_PREDICTOR_VARS) - set(ds_g.data_vars)
        if missing:
            print(f"  {gcm}: missing {sorted(missing)}, skipping")
            continue
        t = pd.to_datetime(ds_g.time.values)
        gcm_times[gcm] = t
        gcm_X[gcm] = build_features(ds_g, fine_lat, fine_lon, feature_vars, topo_fields, t)
        print(f"  {gcm}: {len(t)} historical months")

    if not gcm_X:
        raise SystemExit(
            "No bias-corrected CMIP6 HISTORICAL files found. apply_edcm_bias_correction.py "
            "currently corrects the SSP scenarios only, so the historical period must be "
            "bias-corrected before this test can run."
        )

    # --- inference, loading each persisted forest once ---
    print(f"Predicting across {n_la * n_lo} cells for ERA5 and {len(gcm_X)} GCMs...")
    pred_era5 = np.zeros((len(train_times), n_la, n_lo), dtype=np.float32)
    pred_gcm = {g: np.zeros((len(gcm_times[g]), n_la, n_lo), dtype=np.float32) for g in gcm_X}
    for i in range(n_la):
        for j in range(n_lo):
            p = os.path.join(MODEL_DIR, f"{CELL_PREFIX}_cell_{i:03d}_{j:03d}.joblib")
            if not os.path.exists(p):
                continue
            model = joblib.load(p)
            pred_era5[:, i, j] = model.predict(X_era5[:, i, j, :])
            for g in gcm_X:
                pred_gcm[g][:, i, j] = model.predict(gcm_X[g][:, i, j, :])
        if (i + 1) % 20 == 0 or i == n_la - 1:
            print(f"  row {i + 1}/{n_la}")

    era5_clim = monthly_climatology(
        csi_to_ghi(np.clip(pred_era5, 0, 1.1), train_times, ds_cs), train_times)

    rows = [{
        "predictor source": "ERA5 (as trained)",
        "RMSE": float(np.sqrt(((era5_clim - truth_clim) ** 2).mean())),
        "MBE": float((era5_clim - truth_clim).mean()),
        "spatial R": float(np.corrcoef(era5_clim.ravel(), truth_clim.ravel())[0, 1]),
    }]
    ens = []
    for g in gcm_X:
        c = monthly_climatology(
            csi_to_ghi(np.clip(pred_gcm[g], 0, 1.1), gcm_times[g].values, ds_cs), gcm_times[g])
        ens.append(c)
        rows.append({
            "predictor source": f"CMIP6 {g}",
            "RMSE": float(np.sqrt(((c - truth_clim) ** 2).mean())),
            "MBE": float((c - truth_clim).mean()),
            "spatial R": float(np.corrcoef(c.ravel(), truth_clim.ravel())[0, 1]),
        })
    ens_mean = np.mean(ens, axis=0)
    rows.append({
        "predictor source": "CMIP6 ensemble mean",
        "RMSE": float(np.sqrt(((ens_mean - truth_clim) ** 2).mean())),
        "MBE": float((ens_mean - truth_clim).mean()),
        "spatial R": float(np.corrcoef(ens_mean.ravel(), truth_clim.ravel())[0, 1]),
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n" + "=" * 84)
    print(f" PERFECT-PROGNOSIS TRANSFER TEST — historical climatology, deployed {MODEL_LABEL}")
    print("=" * 84)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 84)
    base = df.iloc[0]["RMSE"]
    ensr = df.iloc[-1]["RMSE"]
    print(f"\nDegradation from swapping predictor source: {base:.3f} -> {ensr:.3f} W m-2 "
          f"({ensr / base:.2f}x)")
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
