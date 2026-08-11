import os
import numpy as np
import xarray as xr
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from cnn_model import load_cnn_checkpoint
from ml_dataset_common import csi_to_ghi, evaluation_baselines, skill_score, safe_nan_to_num

# --- Configuration (Section 3.7.1) ---
val_path = "./data/processed/ml_ready/ml_validation_dataset.nc"
model_path = os.path.abspath("./data/processed/models/cnn/cnn_csi_downscaler.pth")
finegrid_clearsky_path = os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")

print("Loading validation dataset and trained CNN checkpoint...")
ds_val = xr.open_dataset(val_path)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model, checkpoint = load_cnn_checkpoint(model_path, device)
feature_vars = checkpoint['feature_vars']
train_means = checkpoint['feature_means']
train_stds = checkpoint['feature_stds']

n_time = len(ds_val.time.values)
feature_arrays = []
for f in feature_vars:
    arr = ds_val[f].values
    if arr.ndim == 2:
        arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
    arr_clean = safe_nan_to_num(arr, "evaluate_cnn:arr")
    feature_arrays.append((arr_clean - train_means[f]) / train_stds[f])

X_val = np.stack(feature_arrays, axis=1).astype(np.float32)
y_val_true = safe_nan_to_num(ds_val['clear_sky_index'].values, "evaluate_cnn:ds_val_clear_sky_index_.values")

print("Running CNN inference across the 2011-2024 validation period...")
y_pred_list = []
with torch.no_grad():
    for i in range(len(X_val)):
        x_tensor = torch.tensor(X_val[i:i + 1], dtype=torch.float32).to(device)
        pred_field = model(x_tensor).cpu().numpy()[0, 0]
        y_pred_list.append(pred_field)

y_pred_fields = np.stack(y_pred_list, axis=0)

# --- Convert CSI to physical GHI units using the real fine-grid clear-sky
#     climatology (Section 3.5.4), not a single constant, for Table 3.3 metrics ---
ds_clearsky_fine = xr.open_dataset(finegrid_clearsky_path)
y_pred_ghi = csi_to_ghi(y_pred_fields, ds_val.time.values, ds_clearsky_fine).flatten()
y_true_ghi = csi_to_ghi(y_val_true, ds_val.time.values, ds_clearsky_fine).flatten()

# Section 3.7.3 references: naive interpolation and climatology
y_true_field = csi_to_ghi(y_val_true, ds_val.time.values, ds_clearsky_fine)
interp_ghi, clim_ghi, delta_ghi = evaluation_baselines(ds_val, y_true_field, ds_val.time.values, ds_clearsky_fine)
interp_ghi, clim_ghi, delta_ghi = interp_ghi.flatten(), clim_ghi.flatten(), delta_ghi.flatten()

rmse = np.sqrt(mean_squared_error(y_true_ghi, y_pred_ghi))
mae = mean_absolute_error(y_true_ghi, y_pred_ghi)
pearson_r = np.corrcoef(y_pred_ghi, y_true_ghi)[0, 1] if len(np.unique(y_pred_ghi)) > 1 else 0.0
mbe = np.mean(y_pred_ghi - y_true_ghi)
r2 = r2_score(y_true_ghi, y_pred_ghi)
rmse_interp = np.sqrt(mean_squared_error(y_true_ghi, interp_ghi))
rmse_clim = np.sqrt(mean_squared_error(y_true_ghi, clim_ghi))
rmse_delta = np.sqrt(mean_squared_error(y_true_ghi, delta_ghi))
ss_interp = skill_score(rmse, rmse_interp)
ss_clim = skill_score(rmse, rmse_clim)
ss_delta = skill_score(rmse, rmse_delta)

print("\n" + "=" * 70)
print(" TABLE 3.3 VALIDATION PERFORMANCE SUMMARY - CNN (2011-2024)")
print("=" * 70)
print(f"  - RMSE: {rmse:.4f} W m-2")
print(f"  - MAE: {mae:.4f} W m-2")
print(f"  - Pearson R: {pearson_r:.4f} (Target > 0.90)")
print(f"  - Mean Bias Error (MBE): {mbe:.4f} W m-2 (Target |MBE| < 5 W m-2)")
print(f"  - Skill Score vs interpolation: {ss_interp:.4f} (ref RMSE {rmse_interp:.4f})")
print(f"  - Skill Score vs climatology:   {ss_clim:.4f} (ref RMSE {rmse_clim:.4f})")
print(f"  - Skill Score vs delta-mapping: {ss_delta:.4f} (ref RMSE {rmse_delta:.4f})")
print(f"  - R^2 Score: {r2:.4f}")
