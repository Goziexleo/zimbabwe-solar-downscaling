import os
import time
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pixelwise_common import build_fine_pixel_dataset
from ml_dataset_common import csi_to_ghi, evaluation_baselines, skill_score

# --- Configuration (Section 3.6.4) ---
train_path = os.environ.get("ML_TRAIN_PATH", "./data/processed/ml_ready/ml_training_dataset.nc")
val_path = os.environ.get("ML_VAL_PATH", "./data/processed/ml_ready/ml_validation_dataset.nc")
topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
finegrid_clearsky_path = os.environ.get(
    "FINEGRID_CLEARSKY_PATH",
    os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"),
)
model_output_dir = os.path.abspath("./data/processed/models/pixelwise_xgb")
os.makedirs(model_output_dir, exist_ok=True)

MAX_DEPTH = int(os.environ.get("XGB_MAX_DEPTH", "6"))
ETA = float(os.environ.get("XGB_ETA", "0.05"))
SUBSAMPLE = float(os.environ.get("XGB_SUBSAMPLE", "0.8"))
MIN_CHILD_WEIGHT = int(os.environ.get("XGB_MIN_CHILD_WEIGHT", "3"))

print("Loading training and validation datasets...")
ds_train = xr.open_dataset(train_path)
ds_val = xr.open_dataset(val_path)

print("Assembling fine-grid pixel-wise feature tensors "
      "(coarse atmospheric predictors collocated + native fine topo/astronomical features)...")
X_train, y_train, feature_vars = build_fine_pixel_dataset(ds_train, topo_path)
X_val, y_val, _ = build_fine_pixel_dataset(ds_val, topo_path)

n_time_train, n_lat, n_lon, n_features = X_train.shape
n_time_val = X_val.shape[0]
print(f"Fine output grid: {n_lat} x {n_lon} ({n_lat * n_lon} pixel-wise models)")
print(f"Training time steps: {n_time_train}, Validation time steps: {n_time_val}, Features: {feature_vars}")

print(f"\nTraining pixel-wise XGBoost models (Section 3.6.4, max_depth={MAX_DEPTH}, eta={ETA}, "
      f"subsample={SUBSAMPLE}, min_child_weight={MIN_CHILD_WEIGHT}, early stopping patience=50)...")


def fit_predict_cell(i, j):
    xgb_cell = xgb.XGBRegressor(
        max_depth=MAX_DEPTH,
        learning_rate=ETA,
        subsample=SUBSAMPLE,
        min_child_weight=MIN_CHILD_WEIGHT,
        n_estimators=1000,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=42,
        n_jobs=1,
    )
    xgb_cell.fit(X_train[:, i, j, :], y_train[:, i, j],
                 eval_set=[(X_val[:, i, j, :], y_val[:, i, j])], verbose=False)
    return i, j, xgb_cell.predict(X_val[:, i, j, :]), xgb_cell.best_iteration + 1


start = time.time()
cell_indices = [(i, j) for i in range(n_lat) for j in range(n_lon)]
results = Parallel(n_jobs=-1, verbose=5)(delayed(fit_predict_cell)(i, j) for i, j in cell_indices)

y_val_pred = np.zeros_like(y_val)
boosting_rounds_used = []
for i, j, pred, rounds in results:
    y_val_pred[:, i, j] = pred
    boosting_rounds_used.append(rounds)

print(f"Successfully trained {n_lat * n_lon} local pixel-wise XGBoost models "
      f"in {time.time() - start:.1f}s (mean boosting rounds used: {np.mean(boosting_rounds_used):.1f}).")
print("Note: per-cell models are not persisted to disk (would require several "
      f"GB for {n_lat * n_lon} cells); only the aggregate validation metrics "
      "below are retained, consistent with Section 3.7.1's evaluation "
      "requirement and the model-selection criterion in Section 3.8.1.")

# --- Evaluation Framework (Section 3.7.1), converted to physical GHI units ---
ds_clearsky_fine = xr.open_dataset(finegrid_clearsky_path)
y_pred_ghi = csi_to_ghi(y_val_pred, ds_val.time.values, ds_clearsky_fine).flatten()
y_true_ghi = csi_to_ghi(y_val, ds_val.time.values, ds_clearsky_fine).flatten()

y_true_field = csi_to_ghi(y_val, ds_val.time.values, ds_clearsky_fine)
interp_ghi, clim_ghi, delta_ghi = evaluation_baselines(ds_val, y_true_field, ds_val.time.values, ds_clearsky_fine)
interp_ghi, clim_ghi, delta_ghi = interp_ghi.flatten(), clim_ghi.flatten(), delta_ghi.flatten()

rmse = np.sqrt(mean_squared_error(y_true_ghi, y_pred_ghi))
mae = mean_absolute_error(y_true_ghi, y_pred_ghi)
pearson_r = np.corrcoef(y_true_ghi, y_pred_ghi)[0, 1] if len(np.unique(y_pred_ghi)) > 1 else 0.0
mbe = np.mean(y_pred_ghi - y_true_ghi)
r2 = r2_score(y_true_ghi, y_pred_ghi)
rmse_interp = np.sqrt(mean_squared_error(y_true_ghi, interp_ghi))
rmse_clim = np.sqrt(mean_squared_error(y_true_ghi, clim_ghi))
rmse_delta = np.sqrt(mean_squared_error(y_true_ghi, delta_ghi))
ss_interp = skill_score(rmse, rmse_interp)
ss_clim = skill_score(rmse, rmse_clim)
ss_delta = skill_score(rmse, rmse_delta)

print("\n" + "=" * 70)
print(" TABLE 3.3 VALIDATION PERFORMANCE SUMMARY - XGBoost (2011-2024)")
print("=" * 70)
print(f"  - RMSE: {rmse:.4f} W m-2")
print(f"  - MAE: {mae:.4f} W m-2")
print(f"  - Pearson R: {pearson_r:.4f} (Target > 0.90)")
print(f"  - Mean Bias Error (MBE): {mbe:.4f} W m-2 (Target |MBE| < 5 W m-2)")
print(f"  - Skill Score vs interpolation: {ss_interp:.4f} (ref RMSE {rmse_interp:.4f})")
print(f"  - Skill Score vs climatology:   {ss_clim:.4f} (ref RMSE {rmse_clim:.4f})")
print(f"  - Skill Score vs delta-mapping: {ss_delta:.4f} (ref RMSE {rmse_delta:.4f})")
print(f"  - R^2 Score: {r2:.4f}")
