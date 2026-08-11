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
train_path = "./data/processed/ml_ready/ml_training_dataset.nc"
val_path = "./data/processed/ml_ready/ml_validation_dataset.nc"
model_output_dir = os.path.abspath("./data/processed/models/cnn")
os.makedirs(model_output_dir, exist_ok=True)
model_path = os.environ.get("CNN_MODEL_PATH", os.path.join(model_output_dir, "cnn_csi_downscaler.pth"))

# Section 3.6.7/Table 3.2 hyperparameters, overridable for the HPO grid search.
# These defaults are the DEPLOYED configuration selected by hpo_pixelwise.py's
# CNN grid search (lr 5e-4, lambda_gp 1e-2). They previously read 0.001/0.001 -
# the pre-HPO values - and the deployed settings were supplied only as
# command-line environment overrides, so a plain rerun silently reproduced a
# different model from the one reported in Table 3.3.
LEARNING_RATE = float(os.environ.get("CNN_LEARNING_RATE", "0.0005"))
LAMBDA_GP = float(os.environ.get("CNN_LAMBDA_GP", "0.01"))
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


train_dataset = StandardizedClimateDataset(ds_train, feature_vars, target_var, train_means, train_stds)
val_dataset = StandardizedClimateDataset(ds_val, feature_vars, target_var, train_means, train_stds)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


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


print("\nTraining Convolutional Neural Network (Section 3.6.5) over 100 epochs...")
epochs = 100
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = composite_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    scheduler.step()
    epoch_loss = train_loss / len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            val_loss += F.mse_loss(y_pred, y_batch).item() * x_batch.size(0)
    val_loss /= len(val_dataset)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.6f}, Val MSE: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'schema_version': 1,
            'state_dict': model.state_dict(),
            'feature_vars': feature_vars,
            'feature_means': train_means,
            'feature_stds': train_stds,
            'fine_shape': fine_shape,
        }, model_path)

print(f"\nCNN Model successfully saved to: {model_path} (best validation MSE: {best_val_loss:.6f})")
