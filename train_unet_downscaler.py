import os
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
import warnings

from oni_utils import classify_enso_phase
from ml_dataset_common import MODEL_PREDICTOR_VARS, compute_astronomical_features, safe_nan_to_num
from unet_model import ClimateUNet, SCHEMA_VERSION

warnings.filterwarnings("ignore")

# --- Configuration (Section 3.6.6: U-Net, the primary deep learning model) ---
# ML_TRAIN_PATH/ML_VAL_PATH/UNET_MODEL_PATH let the daily-resolution
# diagnostic experiment reuse this script unchanged, pointing at
# ml_training_dataset_daily.nc / ml_validation_dataset_daily.nc and a
# separate checkpoint file, without touching the monthly pipeline's defaults.
train_path = os.environ.get("ML_TRAIN_PATH", os.path.abspath("./data/processed/ml_ready/ml_training_dataset.nc"))
val_path = os.environ.get("ML_VAL_PATH", os.path.abspath("./data/processed/ml_ready/ml_validation_dataset.nc"))
topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
model_output_dir = os.path.abspath("./data/processed/models/unet")
os.makedirs(model_output_dir, exist_ok=True)
model_path = os.environ.get("UNET_MODEL_PATH", os.path.join(model_output_dir, "unet_csi_downscaler.pth"))

# Coarse (0.25 deg) channels: the 6 atmospheric predictors (Table 3.1) - these
# are genuinely coarse GCM/reanalysis fields, upsampled inside the network.
COARSE_VARS = MODEL_PREDICTOR_VARS
# Fine (0.1 deg) channels: topography (native 0.1 deg per Section 3.5.2),
# astronomical features (analytically exact at any resolution, Section
# 3.5.3), and lat/lon - all already known precisely at the target
# resolution, so unlike the CNN's explicit coarse-only (C+3) formula
# (Section 3.6.5), there is no reason to degrade them for the U-Net.
FINE_STATIC_VARS = ["elevation", "slope", "svf"]
FINE_DYNAMIC_VARS = ["solar_zenith_angle", "sin_doy", "cos_doy"]

target_var = "clear_sky_index"  # lives on the fine (fine_lat, fine_lon) target grid

# Deployed configuration (Section 3.6.7): the HPO grid search selected
# lr 2e-4; batch size, early-stop patience and weight decay were already at
# their deployed values, and dropout 0.3 is set as DROPOUT_P in unet_model.py.
# LEARNING_RATE previously defaulted to the pre-HPO 5e-4 and the deployed 2e-4
# was supplied only as a command-line environment override, so a plain rerun
# silently reproduced a different model from the one reported in Table 3.3.
BATCH_SIZE = int(os.environ.get("UNET_BATCH_SIZE", "16"))
EPOCHS = 100
EARLY_STOP_PATIENCE = int(os.environ.get("UNET_EARLY_STOP_PATIENCE", "20"))
LEARNING_RATE = float(os.environ.get("UNET_LEARNING_RATE", "2e-4"))
WEIGHT_DECAY = float(os.environ.get("UNET_WEIGHT_DECAY", "1e-4"))

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")


# --- 1. U-Net Architecture (Section 3.6.6) ---
# Imported from unet_model.py so training, evaluation, and future-projection
# inference always share one definition and one checkpoint schema.


# --- 2. Dataset: coarse atmospheric grid + fine static/astronomical grid ->
#     fine CSI target, with flip augmentation ---
class DownscalingDataset(Dataset):
    def __init__(self, ds, means, stds, clim_mean, anom_std, topo_ds, augment=False):
        self.times = ds.time.values
        n_time = len(self.times)

        coarse_lats, coarse_lons = ds['lat'].values, ds['lon'].values
        coarse_lon_grid, coarse_lat_grid = np.meshgrid(coarse_lons, coarse_lats)

        coarse_arrays = []
        for f in COARSE_VARS:
            arr = safe_nan_to_num(ds[f].values, "train_unet_downscaler:ds_f_.values")
            coarse_arrays.append((arr - means[f]) / stds[f])
        self.coarse_features = np.stack(coarse_arrays, axis=1).astype(np.float32)

        fine_lats, fine_lons = ds['fine_lat'].values, ds['fine_lon'].values
        fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lons, fine_lats)

        fine_arrays = []
        for f in FINE_STATIC_VARS:
            arr = safe_nan_to_num(topo_ds[f].values, "train_unet_downscaler:topo_ds_f_.values")  # native fine grid, static
            arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
            fine_arrays.append((arr - means[f]) / stds[f])

        sza_deg, sin_doy, cos_doy = compute_astronomical_features(fine_lats, fine_lons, self.times)
        for f, arr in zip(FINE_DYNAMIC_VARS, [sza_deg, sin_doy, cos_doy]):
            fine_arrays.append((arr - means[f]) / stds[f])

        fine_lat_b = np.tile(fine_lat_grid[np.newaxis, :, :], (n_time, 1, 1))
        fine_lon_b = np.tile(fine_lon_grid[np.newaxis, :, :], (n_time, 1, 1))
        fine_arrays.append((fine_lat_b - means['fine_lat']) / stds['fine_lat'])
        fine_arrays.append((fine_lon_b - means['fine_lon']) / stds['fine_lon'])

        self.fine_features = np.stack(fine_arrays, axis=1).astype(np.float32)

        target_raw = safe_nan_to_num(ds[target_var].values, "train_unet_downscaler:ds_target_var_.values")[:, np.newaxis, :, :]
        self.target = ((target_raw - clim_mean) / anom_std).astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        x_coarse = self.coarse_features[idx]
        x_fine = self.fine_features[idx]
        y = self.target[idx]
        if self.augment:
            if np.random.rand() < 0.5:
                x_coarse = np.flip(x_coarse, axis=1).copy()  # vertical flip (lat axis)
                x_fine = np.flip(x_fine, axis=1).copy()
                y = np.flip(y, axis=1).copy()
            if np.random.rand() < 0.5:
                x_coarse = np.flip(x_coarse, axis=2).copy()  # horizontal flip (lon axis)
                x_fine = np.flip(x_fine, axis=2).copy()
                y = np.flip(y, axis=2).copy()
        return torch.from_numpy(x_coarse), torch.from_numpy(x_fine), torch.from_numpy(y)


# --- 3. ENSO-stratified mini-batch sampler (Section 3.6.8) ---
class ENSOStratifiedBatchSampler(Sampler):
    """Guarantees a minimum representation of 20% El Nino and 20% La Nina
    months per mini-batch, oversampling ENSO extremes beyond their natural
    frequency in the training record."""
    def __init__(self, phases, batch_size, el_nino_frac=0.2, la_nina_frac=0.2):
        self.batch_size = batch_size
        self.n = len(phases)
        self.el_nino_idx = np.array([i for i, p in enumerate(phases) if p == "el_nino"])
        self.la_nina_idx = np.array([i for i, p in enumerate(phases) if p == "la_nina"])
        self.all_idx = np.arange(self.n)
        self.n_el = max(1, round(el_nino_frac * batch_size)) if len(self.el_nino_idx) else 0
        self.n_la = max(1, round(la_nina_frac * batch_size)) if len(self.la_nina_idx) else 0
        self.n_batches = int(np.ceil(self.n / batch_size))

    def __iter__(self):
        for _ in range(self.n_batches):
            n_rest = self.batch_size - self.n_el - self.n_la
            batch = []
            if self.n_el:
                batch.extend(np.random.choice(self.el_nino_idx, self.n_el, replace=len(self.el_nino_idx) < self.n_el))
            if self.n_la:
                batch.extend(np.random.choice(self.la_nina_idx, self.n_la, replace=len(self.la_nina_idx) < self.n_la))
            batch.extend(np.random.choice(self.all_idx, n_rest, replace=len(self.all_idx) < n_rest))
            np.random.shuffle(batch)
            yield [int(b) for b in batch]

    def __len__(self):
        return self.n_batches


# --- 4. Load data, compute normalisation statistics ---
print("Loading training and validation datasets...")
ds_train = xr.open_dataset(train_path)
ds_val = xr.open_dataset(val_path)

n_time_train = len(ds_train.time.values)
coarse_lats = ds_train['lat'].values
coarse_lons = ds_train['lon'].values
coarse_lon_grid, coarse_lat_grid = np.meshgrid(coarse_lons, coarse_lats)

fine_lats = ds_train['fine_lat'].values
fine_lons = ds_train['fine_lon'].values
fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lons, fine_lats)
fine_shape = (len(fine_lats), len(fine_lons))

print("Loading native fine-resolution topography (Section 3.5.2) and "
      "interpolating onto the core-domain fine grid...")
ds_topo = xr.open_dataset(topo_path)
topo_lat_name = 'lat' if 'lat' in ds_topo.coords else 'y'
topo_lon_name = 'lon' if 'lon' in ds_topo.coords else 'x'
ds_topo = ds_topo.rename({topo_lat_name: 'lat', topo_lon_name: 'lon'}).sortby('lat')
# The topo file's native grid is the buffered 0.1 deg grid (Section 3.2.1's
# buffer), which is wider than the core fine target grid - interpolate down
# to fine_lat/fine_lon rather than assuming direct equality.
ds_topo = ds_topo.interp(lat=fine_lats, lon=fine_lons, method="linear")

print("Computing feature normalisation statistics from the training set...")
train_means, train_stds = {}, {}

for f in COARSE_VARS:
    arr = safe_nan_to_num(ds_train[f].values, "train_unet_downscaler:ds_train_f_.values")
    train_means[f] = float(np.mean(arr))
    std_val = float(np.std(arr))
    train_stds[f] = std_val if std_val > 0 else 1.0

for f in FINE_STATIC_VARS:
    arr = safe_nan_to_num(ds_topo[f].values, "train_unet_downscaler:ds_topo_f_.values")
    train_means[f] = float(np.mean(arr))
    std_val = float(np.std(arr))
    train_stds[f] = std_val if std_val > 0 else 1.0

sza_train, sin_doy_train, cos_doy_train = compute_astronomical_features(
    fine_lats, fine_lons, ds_train.time.values)
for f, arr in zip(FINE_DYNAMIC_VARS, [sza_train, sin_doy_train, cos_doy_train]):
    train_means[f] = float(np.mean(arr))
    std_val = float(np.std(arr))
    train_stds[f] = std_val if std_val > 0 else 1.0

train_means['fine_lat'] = float(np.mean(fine_lat_grid))
train_stds['fine_lat'] = float(np.std(fine_lat_grid)) if np.std(fine_lat_grid) > 0 else 1.0
train_means['fine_lon'] = float(np.mean(fine_lon_grid))
train_stds['fine_lon'] = float(np.std(fine_lon_grid)) if np.std(fine_lon_grid) > 0 else 1.0

target_raw_train = safe_nan_to_num(ds_train[target_var].values, "train_unet_downscaler:ds_train_target_var_.values")
climatology_mean = float(np.mean(target_raw_train))
anomaly_std = float(np.std(target_raw_train - climatology_mean))
if anomaly_std == 0.0:
    anomaly_std = 1.0

print(f"Coarse input grid: {len(coarse_lats)} x {len(coarse_lons)} | Fine target grid: {fine_shape}")

train_dataset = DownscalingDataset(ds_train, train_means, train_stds, climatology_mean, anomaly_std,
                                    ds_topo, augment=True)
val_dataset = DownscalingDataset(ds_val, train_means, train_stds, climatology_mean, anomaly_std,
                                  ds_topo, augment=False)

print("Classifying training months into ENSO phases (NOAA CPC ONI)...")
phases = classify_enso_phase(ds_train.time.values)
sampler = ENSOStratifiedBatchSampler(phases, BATCH_SIZE)
print(f"  El Nino months: {len(sampler.el_nino_idx)} | La Nina months: {len(sampler.la_nina_idx)} "
      f"| batches/epoch: {len(sampler)} (>= {sampler.n_el} El Nino + {sampler.n_la} La Nina per batch)")

train_loader = DataLoader(train_dataset, batch_sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Model, optimiser, composite loss (dual MSE + spatial gradient penalty,
#     Section 3.6.5's dual-loss rationale applied consistently to the U-Net) ---
coarse_channels = len(COARSE_VARS)
fine_channels = len(FINE_STATIC_VARS) + len(FINE_DYNAMIC_VARS) + 2  # + fine lat, lon
model = ClimateUNet(coarse_channels=coarse_channels, fine_channels=fine_channels,
                     out_channels=1, fine_shape=fine_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


def spatial_gradient_penalty(y_pred):
    dx = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])
    dy = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)


def composite_loss(y_pred, y_true, lambda_gp=0.001):
    mse = F.mse_loss(y_pred, y_true)
    gp = spatial_gradient_penalty(y_pred)
    return mse + lambda_gp * gp


best_val_loss = float('inf')
patience_counter = 0

print(f"\nTraining ClimateU-Net (max {EPOCHS} epochs, early-stopping patience {EARLY_STOP_PATIENCE})...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for xc_batch, xf_batch, y_batch in train_loader:
        xc_batch, xf_batch, y_batch = xc_batch.to(device), xf_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(xc_batch, xf_batch)
        loss = composite_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xc_batch.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xc_batch, xf_batch, y_batch in val_loader:
            xc_batch, xf_batch, y_batch = xc_batch.to(device), xf_batch.to(device), y_batch.to(device)
            y_pred = model(xc_batch, xf_batch)
            val_loss += F.mse_loss(y_pred, y_batch).item() * xc_batch.size(0)
    val_loss /= len(val_dataset)

    scheduler.step(val_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'schema_version': SCHEMA_VERSION,
            'state_dict': model.state_dict(),
            'coarse_vars': COARSE_VARS,
            'fine_static_vars': FINE_STATIC_VARS,
            'fine_dynamic_vars': FINE_DYNAMIC_VARS,
            'feature_means': train_means,
            'feature_stds': train_stds,
            'climatology_mean': climatology_mean,
            'anomaly_std': anomaly_std,
            'fine_shape': fine_shape,
            'coarse_channels': coarse_channels,
            'fine_channels': fine_channels,
        }, model_path)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1} "
                  f"(no improvement in validation MSE for {EARLY_STOP_PATIENCE} epochs).")
            break

print(f"\nSuccess! ClimateU-Net saved to: {model_path} (best validation MSE: {best_val_loss:.4f})")
