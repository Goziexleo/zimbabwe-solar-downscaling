"""The operational baseline, and changes measured against the model's own history.

Three faults in the projection chain, all identified by the examiner:

1. No operational baseline. The standard way to get fine-resolution CMIP6
   irradiance is to bias-correct the GCM's own rsds and interpolate it - the
   BCSD family behind NEX-GDDP-CMIP6. Without it nobody can tell whether the ML
   chain improves a projection at all. It is computed here by interpolating the
   EDCDFm-corrected rsds the project already produced.

2. Change measured against ERA5. Section 4.7 differences the downscaled future
   against the ERA5 present. That puts the model's own historical bias inside
   the change signal. The deployed per-cell XGBoost models are persisted, so the
   bias-corrected HISTORICAL predictors are pushed through them here and the
   change is taken against the model's own 1985-2010 run, which cancels it.

3. No driver-consistency check. A downscaled change should keep the sign, and
   roughly the size, of the change in the GCM's own bias-corrected rsds. That
   comparison is the cheapest guard against the ML chain inventing a signal, and
   it was never made.

Writes projection_baselines.csv (domain means over Zimbabwe, per GCM, scenario
and horizon) and xgb_historical_run.nc (the model's own historical field).
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from ml_dataset_common import (
    MODEL_PREDICTOR_VARS, compute_astronomical_features, csi_to_ghi, safe_nan_to_num,
)
from zimbabwe_mask import describe as mask_describe
from zimbabwe_mask import zimbabwe_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "data/processed/models/pixelwise_xgb")
CLEARSKY = os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
TOPO = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CMIP6 = os.path.join(ROOT, "data/processed/cmip6_bias_corrected")
PROJ = os.path.join(ROOT, "data/processed/projections_xgb")
EVAL = os.path.join(ROOT, "data/processed/evaluation")

GCMS = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
SCENARIOS = ["ssp245", "ssp585"]
HORIZONS = {"near_term_2026_2050": (2026, 2050),
            "mid_term_2051_2075": (2051, 2075),
            "long_term_2076_2100": (2076, 2100)}
HIST = (1985, 2010)


def nearest_index(coarse, fine):
    return np.array([np.abs(coarse - c).argmin() for c in fine])


def build_features(ds_g, fine_lat, fine_lon, topo_fields, n_feat):
    times = pd.to_datetime(ds_g.time.values)
    ilat = nearest_index(ds_g["lat"].values, fine_lat)
    ilon = nearest_index(ds_g["lon"].values, fine_lon)
    X = np.zeros((len(times), len(fine_lat), len(fine_lon), n_feat), dtype=np.float32)
    for k, var in enumerate(MODEL_PREDICTOR_VARS):
        arr = safe_nan_to_num(ds_g[var].values, "compute_projection_baselines:" + var)
        X[:, :, :, k] = arr[:, ilat, :][:, :, ilon]
    base = len(MODEL_PREDICTOR_VARS)
    for off, var in enumerate(["elevation", "slope", "svf"]):
        X[:, :, :, base + off] = topo_fields[var][np.newaxis, :, :]
    sza, sin_doy, cos_doy = compute_astronomical_features(fine_lat, fine_lon, times)
    X[:, :, :, base + 3], X[:, :, :, base + 4], X[:, :, :, base + 5] = sza, sin_doy, cos_doy
    return X, times


def main():
    with open(os.path.join(MODEL_DIR, "manifest.json")) as f:
        man = json.load(f)
    fine_lat, fine_lon = np.array(man["fine_lat"]), np.array(man["fine_lon"])
    n_lat, n_lon, n_feat = man["n_lat"], man["n_lon"], len(man["feature_vars"])
    mask = zimbabwe_mask((n_lat, n_lon))
    print("mask: " + mask_describe())

    ds_topo = xr.open_dataset(TOPO)
    ln = "lat" if "lat" in ds_topo.coords else "y"
    gn = "lon" if "lon" in ds_topo.coords else "x"
    ds_topo = ds_topo.rename({ln: "lat", gn: "lon"}).sortby("lat").interp(
        lat=fine_lat, lon=fine_lon, method="linear")
    topo = {v: safe_nan_to_num(ds_topo[v].values, "topo:" + v)
            for v in ["elevation", "slope", "svf"]}
    ds_cs = xr.open_dataset(CLEARSKY)

    # ---- 1. the model's own historical run -------------------------------
    hist_path = os.path.join(EVAL, "xgb_historical_run.nc")
    if os.path.exists(hist_path):
        print("historical run already present, reusing %s" % hist_path)
        hist_ds = xr.open_dataset(hist_path)
    else:
        feats, hist_times = {}, {}
        for gcm in GCMS:
            p = os.path.join(CMIP6, "%s_historical_EDCDFm_corrected.nc" % gcm)
            ds_g = xr.open_dataset(p).sortby("lat")
            X, t = build_features(ds_g, fine_lat, fine_lon, topo, n_feat)
            feats[gcm], hist_times[gcm] = X, t
            print("  %s historical: %d months" % (gcm, len(t)))

        preds = {g: np.zeros((len(hist_times[g]), n_lat, n_lon), dtype=np.float32)
                 for g in GCMS}
        print("inference over %d persisted models (each loaded once)..." % (n_lat * n_lon))
        t0 = time.time()
        for i in range(n_lat):
            for j in range(n_lon):
                fp = os.path.join(MODEL_DIR, "xgb_cell_%03d_%03d.joblib" % (i, j))
                if not os.path.exists(fp):
                    continue
                m = joblib.load(fp)
                for g in GCMS:
                    preds[g][:, i, j] = m.predict(feats[g][:, i, j, :])
            if (i + 1) % 10 == 0 or i == n_lat - 1:
                print("  row %d/%d (%.0fs)" % (i + 1, n_lat, time.time() - t0))
        hist_ds = xr.Dataset(
            {("ghi_" + g.replace("-", "_")):
             (("time", "lat", "lon"),
              csi_to_ghi(np.clip(preds[g], 0.0, 1.1), hist_times[g].values, ds_cs))
             for g in GCMS},
            coords={"time": hist_times[GCMS[0]].values, "lat": fine_lat, "lon": fine_lon})
        hist_ds.to_netcdf(hist_path)
        print("saved %s" % hist_path)

    # ---- 2. the QDM/EDCDFm baseline, interpolated to the fine grid --------
    def interp_rsds(path):
        d = xr.open_dataset(path).sortby("lat")
        return d["rsds"].interp(lat=fine_lat, lon=fine_lon, method="linear")

    qdm_hist = {g: interp_rsds(os.path.join(CMIP6, "%s_historical_EDCDFm_corrected.nc" % g))
                for g in GCMS}

    # The ERA5 present layer, so all three bases can be stated side by side:
    # against ERA5 (as Section 4.7 reports it), against the model's own history
    # (correct, because it cancels the model's historical bias), and the QDM
    # baseline against its own history (the operational standard).
    era5_present = float(np.asarray(xr.open_dataset(
        os.path.join(ROOT, "data/processed/suitability/criterion_layers.nc")
    )["ghi_present_era5"].values)[mask].mean())
    print("ERA5 present over Zimbabwe: %.3f W m-2" % era5_present)
    rows = []
    dm = lambda a: float(np.asarray(a)[mask].mean())

    for gcm in GCMS:
        gv = "ghi_" + gcm.replace("-", "_")
        ml_h = dm(hist_ds[gv].mean(dim="time").values)
        qd_h = dm(qdm_hist[gcm].mean(dim="time").values)
        for scen in SCENARIOS:
            ml = xr.open_dataset(os.path.join(
                PROJ, "downscaled_ghi_%s_%s_2026_2100.nc" % (gcm, scen)))["ghi"]
            qd = interp_rsds(os.path.join(CMIP6, "%s_%s_EDCDFm_corrected.nc" % (gcm, scen)))
            for hz, (y0, y1) in HORIZONS.items():
                mlp = ml.sel(time=slice("%d" % y0, "%d" % y1)).mean(dim="time").values
                qdp = qd.sel(time=slice("%d" % y0, "%d" % y1)).mean(dim="time").values
                rows.append(dict(
                    gcm=gcm, scenario=scen, horizon=hz,
                    ml_mean=dm(mlp), ml_hist=ml_h,
                    ml_change_own_history=dm(mlp) - ml_h,
                    qdm_mean=dm(qdp), qdm_hist=qd_h,
                    qdm_change_own_history=dm(qdp) - qd_h,
                    era5_present=era5_present,
                    ml_change_vs_era5=dm(mlp) - era5_present))

    df = pd.DataFrame(rows)
    df["ml_minus_qdm_change"] = df.ml_change_own_history - df.qdm_change_own_history
    df["same_sign"] = np.sign(df.ml_change_own_history) == np.sign(df.qdm_change_own_history)
    df["ratio_ml_to_qdm"] = df.ml_change_own_history / df.qdm_change_own_history
    df.to_csv(os.path.join(EVAL, "projection_baselines.csv"), index=False)

    print("\n" + "=" * 104)
    print(" PROJECTED CHANGE over Zimbabwe (W m-2), each against its OWN historical run")
    print("=" * 104)
    print(df[["gcm", "scenario", "horizon", "ml_change_own_history",
              "qdm_change_own_history", "ml_minus_qdm_change", "same_sign",
              "ratio_ml_to_qdm"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("=" * 104)
    agree = df.same_sign.mean() * 100
    print("\ndriver consistency: the downscaled change agrees in sign with the GCM's own")
    print("bias-corrected rsds change in %.0f per cent of the %d combinations."
          % (agree, len(df)))
    print("median ratio of downscaled to driver change: %.2f" % df.ratio_ml_to_qdm.median())
    print("\nthe same change on three bases (mean over the three GCMs):")
    print("  %-30s %10s %10s %10s" % ("", "vs ERA5", "vs own hist", "QDM base"))
    for scen in SCENARIOS:
        for hz in HORIZONS:
            sub = df[(df.horizon == hz) & (df.scenario == scen)]
            print("  %-30s %+10.3f %+10.3f %+10.3f"
                  % (scen + " " + hz.split("_")[0],
                     sub.ml_change_vs_era5.mean(),
                     sub.ml_change_own_history.mean(),
                     sub.qdm_change_own_history.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
