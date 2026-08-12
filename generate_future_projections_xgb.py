"""Apply the persisted pixel-wise XGBoost models to bias-corrected CMIP6 (2026-2100).

Exists to complete the sigma_arch ensemble at n = 4. XGBoost was previously the
only benchmarked architecture that could not be projected, because it was never
persisted; at 200 fixed boosting rounds the per-cell models total ~400 MB, an
order of magnitude below the Random Forest's ~4 GB.

Mirrors generate_future_projections_rf.py: each cell's model is loaded once and
applied across all six GCM/scenario combinations.
"""

import glob
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

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("XGB_MODEL_DIR", os.path.join(ROOT, "data/processed/models/pixelwise_xgb"))
CLEARSKY_PATH = os.environ.get(
    "FINEGRID_CLEARSKY_PATH",
    os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"))
TOPO_PATH = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CMIP6_DIR = os.path.join(ROOT, "data/processed/cmip6_bias_corrected")
OUTPUT_DIR = os.environ.get(
    "XGB_PROJECTIONS_OUTPUT_DIR", os.path.join(ROOT, "data/processed/projections_xgb"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

GCMS = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
SCENARIOS = ["ssp245", "ssp585"]

print("Loading XGBoost manifest...")
with open(os.path.join(MODEL_DIR, "manifest.json")) as f:
    manifest = json.load(f)
feature_vars = manifest["feature_vars"]
fine_lat = np.array(manifest["fine_lat"])
fine_lon = np.array(manifest["fine_lon"])
n_lat, n_lon = manifest["n_lat"], manifest["n_lon"]
print(f"  {n_lat} x {n_lon} = {n_lat * n_lon} cells, {len(feature_vars)} features")

ds_topo = xr.open_dataset(TOPO_PATH)
lat_n = "lat" if "lat" in ds_topo.coords else "y"
lon_n = "lon" if "lon" in ds_topo.coords else "x"
ds_topo = ds_topo.rename({lat_n: "lat", lon_n: "lon"}).sortby("lat")
ds_topo = ds_topo.interp(lat=fine_lat, lon=fine_lon, method="linear")
topo_fields = {v: safe_nan_to_num(ds_topo[v].values, f"generate_future_projections_xgb:{v}")
               for v in ["elevation", "slope", "svf"]}

ds_cs = xr.open_dataset(CLEARSKY_PATH)


def nearest_index(coarse, fine):
    return np.array([np.abs(coarse - c).argmin() for c in fine])


print("Assembling fine-grid feature tensors per GCM/scenario...")
combo = {}
for gcm in GCMS:
    for scenario in SCENARIOS:
        path = os.path.join(CMIP6_DIR, f"{gcm}_{scenario}_EDCDFm_corrected.nc")
        if not os.path.exists(path):
            print(f"  {gcm}/{scenario}: missing, skipping")
            continue
        ds_g = xr.open_dataset(path).sortby("lat")
        if set(MODEL_PREDICTOR_VARS) - set(ds_g.data_vars):
            print(f"  {gcm}/{scenario}: missing predictors, skipping")
            continue
        times = pd.to_datetime(ds_g.time.values)
        ilat = nearest_index(ds_g["lat"].values, fine_lat)
        ilon = nearest_index(ds_g["lon"].values, fine_lon)

        X = np.zeros((len(times), n_lat, n_lon, len(feature_vars)), dtype=np.float32)
        for k, var in enumerate(MODEL_PREDICTOR_VARS):
            arr = safe_nan_to_num(ds_g[var].values, f"generate_future_projections_xgb:{var}")
            X[:, :, :, k] = arr[:, ilat, :][:, :, ilon]
        base = len(MODEL_PREDICTOR_VARS)
        for off, var in enumerate(["elevation", "slope", "svf"]):
            X[:, :, :, base + off] = topo_fields[var][np.newaxis, :, :]
        sza, sin_doy, cos_doy = compute_astronomical_features(fine_lat, fine_lon, times)
        X[:, :, :, base + 3] = sza
        X[:, :, :, base + 4] = sin_doy
        X[:, :, :, base + 5] = cos_doy

        combo[(gcm, scenario)] = {"X": X, "times": times}
        print(f"  {gcm}/{scenario}: {len(times)} months")

if not combo:
    raise RuntimeError("No bias-corrected CMIP6 data available.")

preds = {k: np.zeros((len(v["times"]), n_lat, n_lon), dtype=np.float32) for k, v in combo.items()}

print(f"\nInference across {n_lat * n_lon} persisted models (each loaded once)...")
start = time.time()
for i in range(n_lat):
    for j in range(n_lon):
        p = os.path.join(MODEL_DIR, f"xgb_cell_{i:03d}_{j:03d}.joblib")
        if not os.path.exists(p):
            continue
        model = joblib.load(p)
        for key, data in combo.items():
            preds[key][:, i, j] = model.predict(data["X"][:, i, j, :])
    if (i + 1) % 10 == 0 or i == n_lat - 1:
        print(f"  Row {i + 1}/{n_lat} ({time.time() - start:.0f}s)")

print(f"Inference complete in {time.time() - start:.0f}s. Saving...")
for (gcm, scenario), data in combo.items():
    csi = np.clip(preds[(gcm, scenario)], 0.0, 1.1)
    ghi = csi_to_ghi(csi, data["times"].values, ds_cs)
    out = xr.Dataset(
        {"clear_sky_index": (["time", "lat", "lon"], csi),
         "ghi": (["time", "lat", "lon"], ghi)},
        coords={"time": data["times"].values, "lat": fine_lat, "lon": fine_lon},
    )
    out["ghi"].attrs["units"] = "W m-2"
    f_out = os.path.join(OUTPUT_DIR, f"downscaled_ghi_{gcm}_{scenario}_2026_2100.nc")
    out.to_netcdf(f_out)
    print(f"  -> {f_out}")

print("\nSuccess: XGBoost future climate projection generation complete.")
