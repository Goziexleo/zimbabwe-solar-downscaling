import os
import numpy as np
import xarray as xr
import torch
import warnings

from unet_model import load_unet_checkpoint, predict_csi_field
from ml_dataset_common import csi_to_ghi, compute_astronomical_features, evaluation_baselines, skill_score, safe_nan_to_num

warnings.filterwarnings("ignore")

# --- Configuration (Section 3.7.1) ---
val_path = os.environ.get("ML_VAL_PATH", "./data/processed/ml_ready/ml_validation_dataset.nc")
topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
model_path = os.environ.get(
    "UNET_MODEL_PATH", os.path.abspath("./data/processed/models/unet/unet_csi_downscaler.pth")
)
finegrid_clearsky_path = os.environ.get(
    "FINEGRID_CLEARSKY_PATH",
    os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"),
)

print("Loading validation dataset and trained ClimateU-Net checkpoint...")
ds_val = xr.open_dataset(val_path)

ds_topo = xr.open_dataset(topo_path)
topo_lat_name = 'lat' if 'lat' in ds_topo.coords else 'y'
topo_lon_name = 'lon' if 'lon' in ds_topo.coords else 'x'
ds_topo = ds_topo.rename({topo_lat_name: 'lat', topo_lon_name: 'lon'}).sortby('lat')
# The topo file's native grid is the buffered 0.1 deg grid, wider than the
# core fine target grid - interpolate down rather than assuming equality.
ds_topo = ds_topo.interp(lat=ds_val['fine_lat'].values, lon=ds_val['fine_lon'].values, method="linear")

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model, checkpoint = load_unet_checkpoint(model_path, device)

coarse_vars = checkpoint['coarse_vars']
fine_static_vars = checkpoint['fine_static_vars']
fine_dynamic_vars = checkpoint['fine_dynamic_vars']
means = checkpoint['feature_means']
stds = checkpoint['feature_stds']
climatology_mean = checkpoint['climatology_mean']
anomaly_std = checkpoint['anomaly_std']

n_time = len(ds_val.time.values)
coarse_lats, coarse_lons = ds_val['lat'].values, ds_val['lon'].values
fine_lats, fine_lons = ds_val['fine_lat'].values, ds_val['fine_lon'].values
fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lons, fine_lats)

coarse_arrays = []
for f in coarse_vars:
    arr = safe_nan_to_num(ds_val[f].values, "evaluate_unet:ds_val_f_.values")
    coarse_arrays.append((arr - means[f]) / stds[f])
X_coarse = np.stack(coarse_arrays, axis=1).astype(np.float32)

fine_arrays = []
for f in fine_static_vars:
    arr = safe_nan_to_num(ds_topo[f].values, "evaluate_unet:ds_topo_f_.values")
    arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
    fine_arrays.append((arr - means[f]) / stds[f])

sza_deg, sin_doy, cos_doy = compute_astronomical_features(fine_lats, fine_lons, ds_val.time.values)
for f, arr in zip(fine_dynamic_vars, [sza_deg, sin_doy, cos_doy]):
    fine_arrays.append((arr - means[f]) / stds[f])

fine_lat_b = np.tile(fine_lat_grid[np.newaxis, :, :], (n_time, 1, 1))
fine_lon_b = np.tile(fine_lon_grid[np.newaxis, :, :], (n_time, 1, 1))
fine_arrays.append((fine_lat_b - means['fine_lat']) / stds['fine_lat'])
fine_arrays.append((fine_lon_b - means['fine_lon']) / stds['fine_lon'])
X_fine = np.stack(fine_arrays, axis=1).astype(np.float32)

y_val_true = safe_nan_to_num(ds_val['clear_sky_index'].values, "evaluate_unet:ds_val_clear_sky_index_.values")

print("Running ClimateU-Net inference across the 2011-2024 validation period...")
preds = []
with torch.no_grad():
    for i in range(len(X_coarse)):
        xc = torch.from_numpy(X_coarse[i:i + 1]).to(device)
        xf = torch.from_numpy(X_fine[i:i + 1]).to(device)
        pred_csi = predict_csi_field(model, xc, xf, climatology_mean, anomaly_std)
        preds.append(pred_csi)

preds = np.stack(preds, axis=0)

# --- Convert CSI to physical GHI units using the real fine-grid clear-sky
#     climatology (Section 3.5.4), not a single constant, for Table 3.3 metrics ---
ds_clearsky_fine = xr.open_dataset(finegrid_clearsky_path)
preds_ghi = csi_to_ghi(preds, ds_val.time.values, ds_clearsky_fine)
refs_ghi = csi_to_ghi(y_val_true, ds_val.time.values, ds_clearsky_fine)

# Section 3.7.3 references: naive interpolation and climatology
interp_ghi, clim_ghi, delta_ghi = evaluation_baselines(ds_val, refs_ghi, ds_val.time.values, ds_clearsky_fine)


def compute_metrics(y_pred, y_true, y_interp, y_clim, y_delta):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    flat_pred, flat_true = y_pred.flatten(), y_true.flatten()
    pearson_r = np.corrcoef(flat_pred, flat_true)[0, 1] if len(flat_pred) > 1 else 0.0
    mbe = np.mean(y_pred - y_true)
    rmse_interp = np.sqrt(np.mean((y_interp - y_true) ** 2))
    rmse_clim = np.sqrt(np.mean((y_clim - y_true) ** 2))
    rmse_delta = np.sqrt(np.mean((y_delta - y_true) ** 2))
    return (rmse, mae, pearson_r, mbe,
            skill_score(rmse, rmse_interp), skill_score(rmse, rmse_clim),
            skill_score(rmse, rmse_delta), rmse_interp, rmse_clim, rmse_delta)


rmse, mae, r, mbe, ss_interp, ss_clim, ss_delta, rmse_interp, rmse_clim, rmse_delta = compute_metrics(
    preds_ghi, refs_ghi, interp_ghi, clim_ghi, delta_ghi)

print("\n" + "=" * 70)
print(" TABLE 3.3 VALIDATION PERFORMANCE SUMMARY - ClimateU-Net (2011-2024)")
print("=" * 70)
print(f"{'Metric':<20} | {'Target / Threshold':<24} | {'Result'}")
print("-" * 70)
print(f"{'RMSE':<20} | {'Lower is better':<24} | {rmse:.4f} W m-2")
print(f"{'MAE':<20} | {'Lower is better':<24} | {mae:.4f} W m-2")
print(f"{'Pearson R':<20} | {'> 0.90 (Target)':<24} | {r:.4f}")
print(f"{'Mean Bias (MBE)':<20} | {'|MBE| < 5 W m-2':<24} | {mbe:.4f} W m-2")
print(f"{'SS vs interpolation':<20} | {'ref RMSE ' + format(rmse_interp, '.3f'):<24} | {ss_interp:.4f}")
print(f"{'SS vs climatology':<20} | {'ref RMSE ' + format(rmse_clim, '.3f'):<24} | {ss_clim:.4f}")
print(f"{'SS vs delta-mapping':<20} | {'ref RMSE ' + format(rmse_delta, '.3f'):<24} | {ss_delta:.4f}")
print("=" * 70)
