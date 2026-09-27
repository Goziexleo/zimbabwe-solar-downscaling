import os
import numpy as np
import xarray as xr
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from ml_dataset_common import CNN_FEATURE_VARS, safe_nan_to_num
from cnn_model import SuperResolutionCNN

# --- Configuration (Section 3.6.5) ---
train_path = os.environ.get("ML_TRAIN_PATH", "./data/processed/ml_ready/ml_training_dataset.nc")
val_path = os.environ.get("ML_VAL_PATH", "./data/processed/ml_ready/ml_validation_dataset.nc")
model_output_dir = os.path.abspath("./data/processed/models/cnn")
os.makedirs(model_output_dir, exist_ok=True)
model_path = os.environ.get("CNN_MODEL_PATH", os.path.join(model_output_dir, "cnn_csi_downscaler.pth"))

# Section 3.6.7/Table 3.2 hyperparameters, overridable for the HPO grid search.
# These defaults are the DEPLOYED configuration selected by hpo_pixelwise.py's
# Table 3.2, re-searched honestly (Section 3.6.7, hpo_neural.py): lr 1e-3 and
# lambda_gp 1e-3 win on an inner 2005-2010 split at 0.000400 against the
# previous 5e-4 / 1e-2 at 0.000424. The earlier values came from a grid scored
# on the withheld record. Before that they read 0.001/0.001 -
# the pre-HPO values - and the deployed settings were supplied only as
# command-line environment overrides, so a plain rerun silently reproduced a
# different model from the one reported in Table 3.3.
LEARNING_RATE = float(os.environ.get("CNN_LEARNING_RATE", "0.001"))
LAMBDA_GP = float(os.environ.get("CNN_LAMBDA_GP", "0.001"))
BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", "16"))

print("Loading datasets for CNN spatial super-resolution...")
ds_train = xr.open_dataset(train_path)
ds_val = xr.open_dataset(val_path)

feature_vars = CNN_FEATURE_VARS  # (C+3): 6 atmospheric predictors + elevation/slope/svf
target_var = "clear_sky_index"   # lives on the fine (fine_lat, fine_lon) target grid
fine_shape = (ds_train.sizes['fine_lat'], ds_train.sizes['fine_lon'])

# --- Feature Standardization (Z-score Normalization) ---
print("Computing feature normalization statistics from training set...")
train_means = {}
train_stds = {}

n_time_train = len(ds_train.time.values)
for f in feature_vars:
    arr = ds_train[f].values
    if arr.ndim == 2:
        arr = np.tile(arr[np.newaxis, :, :], (n_time_train, 1, 1))
    arr_clean = safe_nan_to_num(arr, "train_cnn_downscaler:arr")
    mean_val = np.mean(arr_clean)
    std_val = np.std(arr_clean)
    if std_val == 0.0:
        std_val = 1.0
    train_means[f] = mean_val
    train_stds[f] = std_val


class StandardizedClimateDataset(Dataset):
    def __init__(self, ds, feature_vars, target_var, means, stds):
        self.times = ds.time.values
        n_time = len(self.times)

        feature_arrays = []
        for f in feature_vars:
            arr = ds[f].values
            if arr.ndim == 2:
                arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
            arr_clean = safe_nan_to_num(arr, "train_cnn_downscaler:arr")
            arr_norm = (arr_clean - means[f]) / stds[f]
            feature_arrays.append(arr_norm)

        self.features = np.stack(feature_arrays, axis=1).astype(np.float32)
        # Target lives on the fine (fine_lat, fine_lon) grid - genuinely
        # coarser input than output, matching Section 3.6.1's problem formulation.
        self.target = safe_nan_to_num(ds[target_var].values, "train_cnn_downscaler:ds_target_var_.values")[:, np.newaxis, :, :].astype(np.float32)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return x, y


# --- Honest model selection (see Section 6.13) -------------------------------
# This script previously saved the checkpoint that scored best on ds_val, the
# withheld 2011-2024 record. That is selection on the evaluation set: the same
# defect Section 6.12 found and fixed for XGBoost, which is fitted for a fixed
# number of rounds. It makes the reported figure the minimum of ~100 noisy draws
# rather than an estimate of generalisation.
#
# The fix keeps the evaluation record untouched and still uses all 312 training
# months, so the comparison with RF and XGBoost stays fair:
#   Phase A  fit on 1985-2004, score on 2005-2010, record the best epoch.
#   Phase B  refit from scratch on the FULL training period for that many epochs.
# ds_val is opened only to confirm it is never read during training.
import pandas as _pd

INNER_SPLIT_YEAR = int(os.environ.get("INNER_SPLIT_YEAR", "2005"))
_t = _pd.DatetimeIndex(ds_train.time.values)
_fit = _t.year < INNER_SPLIT_YEAR

ds_fit = ds_train.isel(time=np.where(_fit)[0])
ds_sel = ds_train.isel(time=np.where(~_fit)[0])
print(f"Inner split: fit {_fit.sum()} months (<{INNER_SPLIT_YEAR}), "
      f"select {(~_fit).sum()} months (>={INNER_SPLIT_YEAR}). "
      f"The {ds_val.sizes['time']}-month evaluation record is not used in training.")

train_dataset = StandardizedClimateDataset(ds_train, feature_vars, target_var, train_means, train_stds)
fit_dataset = StandardizedClimateDataset(ds_fit, feature_vars, target_var, train_means, train_stds)
sel_dataset = StandardizedClimateDataset(ds_sel, feature_vars, target_var, train_means, train_stds)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
fit_loader = DataLoader(fit_dataset, batch_size=BATCH_SIZE, shuffle=True)
sel_loader = DataLoader(sel_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- CNN Architecture (Section 3.6.5) imported from cnn_model.py so training
#     and evaluation always share one definition and one checkpoint schema. ---
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")
model = SuperResolutionCNN(in_channels=len(feature_vars), fine_shape=fine_shape).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


def spatial_gradient_penalty(y_pred):
    dx = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
    dy = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)


def composite_loss(y_pred, y_true, lambda_gp=LAMBDA_GP):
    mse = F.mse_loss(y_pred, y_true)
    gp = spatial_gradient_penalty(y_pred)
    return mse + lambda_gp * gp


EPOCHS = int(os.environ.get("EPOCHS", "100"))


def _build():
    m = SuperResolutionCNN(in_channels=len(feature_vars), fine_shape=fine_shape).to(device)
    o = optim.Adam(m.parameters(), lr=LEARNING_RATE)
    return m, o, optim.lr_scheduler.CosineAnnealingLR(o, T_max=EPOCHS)


def _run_epoch(m, o, loader):
    m.train(); tot = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        o.zero_grad()
        loss = composite_loss(m(xb), yb)
        loss.backward(); o.step()
        tot += loss.item() * xb.size(0)
    return tot / len(loader.dataset)


def _score(m, loader):
    m.eval(); tot = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            tot += F.mse_loss(m(xb), yb).item() * xb.size(0)
    return tot / len(loader.dataset)


# --- Phase A: choose the epoch count on the inner split ---
print(f"\nPhase A: fitting on {len(fit_dataset)} months, selecting on {len(sel_dataset)}...")
model, optimizer, scheduler = _build()
best_sel, best_epoch = float("inf"), EPOCHS
for epoch in range(EPOCHS):
    tr = _run_epoch(model, optimizer, fit_loader)
    scheduler.step()
    sel = _score(model, sel_loader)
    if sel < best_sel:
        best_sel, best_epoch = sel, epoch + 1
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  epoch {epoch + 1}/{EPOCHS} | fit {tr:.6f} | select {sel:.6f}")
print(f"Phase A: best inner-select MSE {best_sel:.6f} at epoch {best_epoch}.")

# Honest hyperparameter search (Section 3.6.7). With PHASE_A_ONLY=1 the script
# stops after Phase A and reports the inner-selection score, so a grid search can
# compare candidates without any of them ever touching the withheld record. The
# search that fixed Table 3.2 was scored on the validation MSE; this is the same
# search moved inside the training period.
if os.environ.get("PHASE_A_ONLY", "0") == "1":
    _out = os.environ.get("PHASE_A_OUT", "")
    if _out:
        import json
        json.dump({"inner_select_mse": float(best_sel),
                   "selected_epoch": int(best_epoch)}, open(_out, "w"))
    print("PHASE_A_ONLY: stopping before the refit.")
    raise SystemExit(0)

# --- Phase B: refit on the full training period for that many epochs ---
print(f"\nPhase B: refitting on all {len(train_dataset)} training months for {best_epoch} epochs...")
model, optimizer, scheduler = _build()
for epoch in range(best_epoch):
    tr = _run_epoch(model, optimizer, train_loader)
    scheduler.step()
    if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == best_epoch:
        print(f"  epoch {epoch + 1}/{best_epoch} | train {tr:.6f}")

torch.save({
    'schema_version': 1,
    'state_dict': model.state_dict(),
    'feature_vars': feature_vars,
    'feature_means': train_means,
    'feature_stds': train_stds,
    'fine_shape': fine_shape,
    'selected_epoch': best_epoch,
    'inner_select_mse': best_sel,
    'selection': 'inner split %d, evaluation record untouched' % INNER_SPLIT_YEAR,
}, model_path)

print(f"\nCNN saved to: {model_path} (epoch {best_epoch} chosen on the inner split, "
      f"inner MSE {best_sel:.6f}; the evaluation record was never used)")
