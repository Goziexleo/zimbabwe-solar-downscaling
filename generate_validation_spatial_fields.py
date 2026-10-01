"""Section 3.7.2 support: run inference for all 4 finalized models across the
full 2011-2024 validation period and save the complete (time, fine_lat,
fine_lon) predicted GHI fields - plus the true reference and the naive
bilinear-interpolation baseline - into one combined NetCDF file. This is the
shared input for the Taylor-diagram statistics and bias/RMSE maps.
"""

import os
import json
import numpy as np
import xarray as xr
import torch
import joblib
import warnings

from unet_model import load_unet_checkpoint, predict_csi_field
from cnn_model import load_cnn_checkpoint
from pixelwise_common import build_fine_pixel_dataset
from ml_dataset_common import csi_to_ghi, compute_astronomical_features, safe_nan_to_num

warnings.filterwarnings("ignore")

VAL_PATH = "./data/processed/ml_ready/ml_validation_dataset.nc"
TOPO_PATH = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
FINEGRID_CLEARSKY_PATH = os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
UNET_MODEL_PATH = os.environ.get(
    "UNET_MODEL_PATH",
    os.path.abspath("./data/processed/models/unet/unet_csi_downscaler.pth"))
CNN_MODEL_PATH = os.path.abspath("./data/processed/models/cnn/cnn_csi_downscaler.pth")
RF_MODEL_DIR = os.path.abspath("./data/processed/models/pixelwise_rf")
OUTPUT_PATH = os.environ.get(
    "VALIDATION_FIELDS_OUTPUT",
    os.path.abspath("./data/processed/evaluation/validation_spatial_fields.nc"))
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

print("Loading validation dataset and clear-sky climatology...")
ds_val = xr.open_dataset(VAL_PATH)
ds_clearsky_fine = xr.open_dataset(FINEGRID_CLEARSKY_PATH)
times = ds_val.time.values
fine_lats, fine_lons = ds_val["fine_lat"].values, ds_val["fine_lon"].values
n_time = len(times)

y_true_csi = safe_nan_to_num(ds_val["clear_sky_index"].values, "generate_validation_spatial_fields:ds_val_clear_sky_index_.values")
true_ghi = csi_to_ghi(y_true_csi, times, ds_clearsky_fine)

baseline_da = ds_val["rsds"].interp(lat=fine_lats, lon=fine_lons, method="linear")
baseline_ghi = safe_nan_to_num(baseline_da.values, "generate_validation_spatial_fields:baseline_da.values")

# ---------------------------------------------------------------- U-Net ----
print("\n[1/4] Running ClimateU-Net inference...")
ds_topo = xr.open_dataset(TOPO_PATH)
topo_lat_name = "lat" if "lat" in ds_topo.coords else "y"
topo_lon_name = "lon" if "lon" in ds_topo.coords else "x"
ds_topo_r = ds_topo.rename({topo_lat_name: "lat", topo_lon_name: "lon"}).sortby("lat")
ds_topo_r = ds_topo_r.interp(lat=fine_lats, lon=fine_lons, method="linear")

unet_model, checkpoint = load_unet_checkpoint(UNET_MODEL_PATH, device)
coarse_vars = checkpoint["coarse_vars"]
fine_static_vars = checkpoint["fine_static_vars"]
fine_dynamic_vars = checkpoint["fine_dynamic_vars"]
means, stds = checkpoint["feature_means"], checkpoint["feature_stds"]
climatology_mean, anomaly_std = checkpoint["climatology_mean"], checkpoint["anomaly_std"]

fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lons, fine_lats)

coarse_arrays = []
for f in coarse_vars:
    arr = safe_nan_to_num(ds_val[f].values, "generate_validation_spatial_fields:ds_val_f_.values")
    coarse_arrays.append((arr - means[f]) / stds[f])
X_coarse = np.stack(coarse_arrays, axis=1).astype(np.float32)

fine_arrays = []
for f in fine_static_vars:
    arr = safe_nan_to_num(ds_topo_r[f].values, "generate_validation_spatial_fields:ds_topo_r_f_.values")
    arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
    fine_arrays.append((arr - means[f]) / stds[f])

sza_deg, sin_doy, cos_doy = compute_astronomical_features(fine_lats, fine_lons, times)
for f, arr in zip(fine_dynamic_vars, [sza_deg, sin_doy, cos_doy]):
    fine_arrays.append((arr - means[f]) / stds[f])

fine_lat_b = np.tile(fine_lat_grid[np.newaxis, :, :], (n_time, 1, 1))
fine_lon_b = np.tile(fine_lon_grid[np.newaxis, :, :], (n_time, 1, 1))
fine_arrays.append((fine_lat_b - means["fine_lat"]) / stds["fine_lat"])
fine_arrays.append((fine_lon_b - means["fine_lon"]) / stds["fine_lon"])
X_fine = np.stack(fine_arrays, axis=1).astype(np.float32)

unet_preds = []
with torch.no_grad():
    for i in range(n_time):
        xc = torch.from_numpy(X_coarse[i:i + 1]).to(device)
        xf = torch.from_numpy(X_fine[i:i + 1]).to(device)
        unet_preds.append(predict_csi_field(unet_model, xc, xf, climatology_mean, anomaly_std))
unet_csi = np.stack(unet_preds, axis=0)
unet_ghi = csi_to_ghi(unet_csi, times, ds_clearsky_fine)

# ------------------------------------------------------------------ CNN ----
print("[2/4] Running CNN inference...")
cnn_model, cnn_ckpt = load_cnn_checkpoint(CNN_MODEL_PATH, device)
cnn_feature_vars = cnn_ckpt["feature_vars"]
cnn_means, cnn_stds = cnn_ckpt["feature_means"], cnn_ckpt["feature_stds"]

cnn_feature_arrays = []
for f in cnn_feature_vars:
    arr = ds_val[f].values
    if arr.ndim == 2:
        arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
    arr_clean = safe_nan_to_num(arr, "generate_validation_spatial_fields:arr")
    cnn_feature_arrays.append((arr_clean - cnn_means[f]) / cnn_stds[f])
X_cnn = np.stack(cnn_feature_arrays, axis=1).astype(np.float32)

cnn_preds = []
with torch.no_grad():
    for i in range(n_time):
        x_tensor = torch.tensor(X_cnn[i:i + 1], dtype=torch.float32).to(device)
        cnn_preds.append(cnn_model(x_tensor).cpu().numpy()[0, 0])
cnn_csi = np.stack(cnn_preds, axis=0)
cnn_ghi = csi_to_ghi(cnn_csi, times, ds_clearsky_fine)

# ------------------------------------------------------------------- RF ----
print("[3/4] Running pixel-wise Random Forest inference (loading 5,751 persisted models)...")
with open(os.path.join(RF_MODEL_DIR, "manifest.json")) as f:
    rf_manifest = json.load(f)
rf_feature_vars = rf_manifest["feature_vars"]

X_val_px, y_val_px, px_feature_vars = build_fine_pixel_dataset(ds_val, TOPO_PATH)
assert px_feature_vars == rf_feature_vars, "Feature order mismatch between pixelwise builder and RF manifest"
n_lat, n_lon = rf_manifest["n_lat"], rf_manifest["n_lon"]

rf_csi = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
for i in range(n_lat):
    for j in range(n_lon):
        model_path = os.path.join(RF_MODEL_DIR, f"rf_cell_{i:03d}_{j:03d}.joblib")
        if not os.path.exists(model_path):
            continue
        rf_cell = joblib.load(model_path)
        rf_csi[:, i, j] = rf_cell.predict(X_val_px[:, i, j, :])
    if (i + 1) % 20 == 0 or i == n_lat - 1:
        print(f"  Row {i + 1}/{n_lat} done")
rf_ghi = csi_to_ghi(rf_csi, times, ds_clearsky_fine)

# -------------------------------------------------------------- XGBoost ----
# Loaded from persisted models rather than refitted here. The previous version
# refitted with early_stopping_rounds against the validation set, which chose
# each cell's capacity by watching the data it was then scored on.
print("[4/4] Running pixel-wise XGBoost inference (loading persisted models)...")
XGB_MODEL_DIR = os.path.abspath("./data/processed/models/pixelwise_xgb")
with open(os.path.join(XGB_MODEL_DIR, "manifest.json")) as f:
    xgb_manifest = json.load(f)
assert xgb_manifest["feature_vars"] == px_feature_vars, "XGBoost manifest feature order mismatch"

xgb_csi = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
for i in range(n_lat):
    for j in range(n_lon):
        p = os.path.join(XGB_MODEL_DIR, f"xgb_cell_{i:03d}_{j:03d}.joblib")
        if not os.path.exists(p):
            continue
        xgb_csi[:, i, j] = joblib.load(p).predict(X_val_px[:, i, j, :])
    if (i + 1) % 20 == 0 or i == n_lat - 1:
        print(f"  Row {i + 1}/{n_lat} done")
xgb_ghi = csi_to_ghi(xgb_csi, times, ds_clearsky_fine)

# ------------------------------------------------------------------ Save ----
print("\nSaving combined validation spatial-field dataset...")
ds_out = xr.Dataset(
    {
        "ghi_true": (["time", "fine_lat", "fine_lon"], true_ghi),
        "ghi_baseline": (["time", "fine_lat", "fine_lon"], baseline_ghi),
        "ghi_unet": (["time", "fine_lat", "fine_lon"], unet_ghi),
        "ghi_cnn": (["time", "fine_lat", "fine_lon"], cnn_ghi),
        "ghi_rf": (["time", "fine_lat", "fine_lon"], rf_ghi),
        "ghi_xgb": (["time", "fine_lat", "fine_lon"], xgb_ghi),
    },
    coords={"time": times, "fine_lat": fine_lats, "fine_lon": fine_lons},
)
for v in ds_out.data_vars:
    ds_out[v].attrs["units"] = "W m-2"
ds_out.to_netcdf(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")
