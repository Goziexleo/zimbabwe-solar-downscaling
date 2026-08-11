import xarray as xr
import numpy as np

# Load a sample training file and its corresponding CSI/clearsky file
ds_train = xr.open_dataset("./data/processed/era5/monthly/training/monthly_era5_training_1985.nc")
ds_csi = xr.open_dataset("./data/processed/era5/csi/training/csi_monthly_era5_training_1985.nc")

ssrd = ds_train['ssrd'].values
ghi_clear = ds_csi['clearsky_ghi'].values

print("--- Raw Data Inspection (1985 Sample) ---")
print(f"SSRD (Target) - Min: {ssrd.min():.2f}, Max: {ssrd.max():.2f}, Mean: {ssrd.mean():.2f}")
print(f"Clearsky GHI  - Min: {ghi_clear.min():.2f}, Max: {ghi_clear.max():.2f}, Mean: {ghi_clear.mean():.2f}")

# Check unclipped ratio
ghi_safe = np.where(ghi_clear < 1.0, 1.0, ghi_clear)
ratio = ssrd / ghi_safe
print(f"Unclipped Ratio - Min: {ratio.min():.4f}, Max: {ratio.max():.4f}, Mean: {ratio.mean():.4f}")