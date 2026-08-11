import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Configuration ---
val_path = "./data/processed/ml_ready/ml_validation_dataset.nc"
model_path = os.path.abspath("./data/processed/models/cnn/cnn_csi_downscaler.pth")

print("Loading validation dataset and trained CNN model for diagnostics...")
ds_val = xr.open_dataset(val_path)

feature_vars = [
    "clt", "tas", "ps", "huss", 
    "elevation", "slope", "svf", 
    "solar_zenith_angle", "sin_doy", "cos_doy"
]
target_var = "clear_sky_index"

class SuperResolutionCNN(nn.Module):
    def __init__(self, in_channels=10):
        super(SuperResolutionCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        _, _, h, w = x.shape
        target_h, target_w = int(h * 2.5), int(w * 2.5)
        x_upsampled = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        out = self.block1(x_upsampled)
        out = self.block2(out)
        out = self.block3(out)
        out = self.out_conv(out)
        out = torch.clamp(out, 0.0, 1.1)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SuperResolutionCNN(in_channels=len(feature_vars)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

n_time = len(ds_val.time.values)
feature_arrays = []
for f in feature_vars:
    arr = ds_val[f].values
    if arr.ndim == 2:
        arr = np.tile(arr[np.newaxis, :, :], (n_time, 1, 1))
    feature_arrays.append(np.nan_to_num(arr, nan=0.0))

X_val = np.stack(feature_arrays, axis=1)
y_val_true = np.nan_to_num(ds_val[target_var].values, nan=0.0)

print("\nRunning diagnostic inference on first 5 validation time steps...")
with torch.no_grad():
    for i in range(5):
        x_tensor = torch.tensor(X_val[i:i+1], dtype=torch.float32).to(device)
        pred_tensor = model(x_tensor)
        
        pred_field = pred_tensor.cpu().numpy()[0, 0]
        true_field = y_val_true[i]
        
        print(f"\n--- Time Step {i} ---")
        print(f"Prediction -> Min: {pred_field.min():.6f}, Max: {pred_field.max():.6f}, Mean: {pred_field.mean():.6f}, Std: {pred_field.std():.6f}")
        print(f"True Target -> Min: {true_field.min():.6f}, Max: {true_field.max():.6f}, Mean: {true_field.mean():.6f}, Std: {true_field.std():.6f}")