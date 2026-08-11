"""Apply the persisted pixel-wise Random Forest models (Section 3.6.3) to the
bias-corrected CMIP6 monthly GCM data (2026-2100).

Defaults to the monthly-trained RF (data/processed/models/pixelwise_rf),
which has a genuine positive Skill Score against the naive interpolation
baseline. A daily-resolution variant was also tested (retrained on ~9,500
daily ERA5 samples instead of 312 monthly ones, hoping more samples would
raise validation R past 0.90) - it did cross that threshold, but a
diagnostic check showed both the daily RF and daily U-Net actually score
*worse* than trivially bilinear-interpolating their own rsds input field at
daily resolution (Skill Score deeply negative). That's because rsds is a
reused copy of the same ssrd field the CSI target is built from, and at
daily resolution a naive interpolation of it already reproduces the target
almost exactly - so a high R there mostly reflects the model weakly
reproducing its own input, not genuine atmospheric-to-irradiance skill.
The daily variant is kept only as a documented limitations finding, not
deployed here; override RF_MODEL_DIR/FINEGRID_CLEARSKY_PATH if you want to
regenerate it anyway.

RF has no daily-CMIP6 analogue in any case (CMIP6 archives only provide
monthly output), so each monthly GCM field is treated as a single
representative sample using that month's own CMIP6 timestamp (CMIP6 monthly
files are conventionally stamped mid-month, e.g. "2026-01-16"), the same
convention the monthly training/validation datasets themselves use.
"""

import os
import json
import glob
import numpy as np
import xarray as xr
import pandas as pd
import joblib

from ml_dataset_common import MODEL_PREDICTOR_VARS, compute_astronomical_features, csi_to_ghi, safe_nan_to_num

RF_MODEL_DIR = os.environ.get(
    "RF_MODEL_DIR", os.path.abspath("./data/processed/models/pixelwise_rf")
)
FINEGRID_CLEARSKY_PATH = os.environ.get(
    "FINEGRID_CLEARSKY_PATH",
    os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"),
)
TOPO_PATH = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CMIP6_DIR = os.path.abspath("./data/processed/cmip6_bias_corrected")
OUTPUT_DIR = os.environ.get("RF_PROJECTIONS_OUTPUT_DIR", os.path.abspath("./data/processed/projections_rf"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

GCMS = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
SCENARIOS = ["ssp245", "ssp585"]

print("Loading RF model manifest...")
with open(os.path.join(RF_MODEL_DIR, "manifest.json")) as f:
    manifest = json.load(f)
feature_vars = manifest["feature_vars"]
fine_lat = np.array(manifest["fine_lat"])
fine_lon = np.array(manifest["fine_lon"])
n_lat, n_lon = manifest["n_lat"], manifest["n_lon"]
print(f"  {n_lat} x {n_lon} = {n_lat * n_lon} cells, features: {feature_vars}")

print("Loading native fine-resolution topography and fine-grid clear-sky climatology...")
ds_topo = xr.open_dataset(TOPO_PATH)
topo_lat_name = "lat" if "lat" in ds_topo.coords else "y"
topo_lon_name = "lon" if "lon" in ds_topo.coords else "x"
ds_topo = ds_topo.rename({topo_lat_name: "lat", topo_lon_name: "lon"}).sortby("lat")
ds_topo = ds_topo.interp(lat=fine_lat, lon=fine_lon, method="linear")
topo_fields = {v: safe_nan_to_num(ds_topo[v].values, "generate_future_projections_rf:ds_topo_v_.values") for v in ["elevation", "slope", "svf"]}

ds_clearsky_fine = xr.open_dataset(FINEGRID_CLEARSKY_PATH)  # month- or day-indexed, auto-detected by csi_to_ghi


def nearest_index(coarse_coords, fine_coords):
    return np.array([np.abs(coarse_coords - c).argmin() for c in fine_coords])


print("Loading bias-corrected CMIP6 data and assembling fine-grid feature tensors per GCM/scenario...")
combo_data = {}
for gcm in GCMS:
    for scenario in SCENARIOS:
        gcm_path = os.path.join(CMIP6_DIR, f"{gcm}_{scenario}_EDCDFm_corrected.nc")
        if not os.path.exists(gcm_path):
            print(f"  Warning: {gcm_path} not found, skipping.")
            continue
        ds_gcm = xr.open_dataset(gcm_path).sortby("lat")
        missing = set(MODEL_PREDICTOR_VARS) - set(ds_gcm.data_vars)
        if missing:
            print(f"  Warning: {gcm}/{scenario} missing {sorted(missing)}, skipping.")
            continue

        coarse_lat, coarse_lon = ds_gcm["lat"].values, ds_gcm["lon"].values
        near_lat_idx = nearest_index(coarse_lat, fine_lat)
        near_lon_idx = nearest_index(coarse_lon, fine_lon)
        times = pd.to_datetime(ds_gcm.time.values)
        n_time = len(times)

        X = np.zeros((n_time, n_lat, n_lon, len(feature_vars)), dtype=np.float32)
        for k, var in enumerate(MODEL_PREDICTOR_VARS):
            coarse_arr = safe_nan_to_num(ds_gcm[var].values, "generate_future_projections_rf:ds_gcm_var_.values")
            X[:, :, :, k] = coarse_arr[:, near_lat_idx, :][:, :, near_lon_idx]

        base = len(MODEL_PREDICTOR_VARS)
        for offset, var in enumerate(["elevation", "slope", "svf"]):
            X[:, :, :, base + offset] = topo_fields[var][np.newaxis, :, :]

        sza_deg, sin_doy, cos_doy = compute_astronomical_features(fine_lat, fine_lon, times)
        X[:, :, :, base + 3] = sza_deg
        X[:, :, :, base + 4] = sin_doy
        X[:, :, :, base + 5] = cos_doy

        combo_data[(gcm, scenario)] = {"X": X, "times": times}
        print(f"  {gcm}/{scenario}: {n_time} months")

if not combo_data:
    raise RuntimeError("No bias-corrected CMIP6 data available - run apply_edcm_bias_correction.py first.")

# Pre-allocate CSI prediction arrays for each combo
csi_preds = {key: np.zeros((len(v["times"]), n_lat, n_lon), dtype=np.float32) for key, v in combo_data.items()}

print(f"\nRunning inference cell-by-cell across {n_lat * n_lon} persisted RF models "
      f"(each loaded once, applied to all {len(combo_data)} GCM/scenario combos)...")
import time
start = time.time()
for i in range(n_lat):
    for j in range(n_lon):
        model_path = os.path.join(RF_MODEL_DIR, f"rf_cell_{i:03d}_{j:03d}.joblib")
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        for key, data in combo_data.items():
            pred = model.predict(data["X"][:, i, j, :])
            csi_preds[key][:, i, j] = pred
    if (i + 1) % 10 == 0 or i == n_lat - 1:
        print(f"  Row {i + 1}/{n_lat} done ({time.time() - start:.0f}s elapsed)")

print(f"Inference complete in {time.time() - start:.0f}s. Converting CSI to GHI and saving...")

for (gcm, scenario), data in combo_data.items():
    csi = np.clip(csi_preds[(gcm, scenario)], 0.0, 1.1)
    ghi = csi_to_ghi(csi, data["times"].values, ds_clearsky_fine)

    ds_out = xr.Dataset(
        {
            "clear_sky_index": (["time", "lat", "lon"], csi),
            "ghi": (["time", "lat", "lon"], ghi),
        },
        coords={"time": data["times"].values, "lat": fine_lat, "lon": fine_lon},
    )
    ds_out["ghi"].attrs["units"] = "W m-2"
    out_file = os.path.join(OUTPUT_DIR, f"downscaled_ghi_{gcm}_{scenario}_2026_2100.nc")
    ds_out.to_netcdf(out_file)
    print(f"  -> Saved: {out_file}")

print("\nSuccess: RF future climate projection generation complete.")
