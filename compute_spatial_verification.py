"""Section 3.7.2 spatial verification: Taylor diagram statistics (spatial
correlation, normalised std ratio, centered RMSE of the time-mean
climatological pattern) plus per-pixel time-mean bias and RMSE maps, for
each of the 4 finalized models against the true reference field.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd

INPUT_PATH = os.path.abspath("./data/processed/evaluation/validation_spatial_fields.nc")
OUTPUT_PATH = os.path.abspath("./data/processed/evaluation/spatial_verification_maps.nc")
TAYLOR_CSV_PATH = os.path.abspath("./data/processed/evaluation/taylor_diagram_stats.csv")

ds = xr.open_dataset(INPUT_PATH)
ref = ds["ghi_true"].values  # (time, fine_lat, fine_lon)

models = {
    "Baseline (bilinear)": ds["ghi_baseline"].values,
    "Random Forest": ds["ghi_rf"].values,
    "XGBoost": ds["ghi_xgb"].values,
    "CNN": ds["ghi_cnn"].values,
    "U-Net": ds["ghi_unet"].values,
}

ref_mean_map = ref.mean(axis=0)
ref_std_spatial = ref_mean_map.std()

rows = []
bias_maps = {}
rmse_maps = {}

for name, arr in models.items():
    model_mean_map = arr.mean(axis=0)

    # --- Taylor diagram stats, computed on the spatial pattern of the
    #     time-mean climatology (Section 3.7.2) ---
    flat_m, flat_r = model_mean_map.flatten(), ref_mean_map.flatten()
    spatial_corr = np.corrcoef(flat_m, flat_r)[0, 1]
    std_model = model_mean_map.std()
    std_ratio = std_model / ref_std_spatial if ref_std_spatial > 0 else np.nan
    crmse = np.sqrt(std_model**2 + ref_std_spatial**2 - 2 * std_model * ref_std_spatial * spatial_corr)

    # --- Bias & RMSE maps: full temporal comparison at every pixel ---
    bias_map = model_mean_map - ref_mean_map
    rmse_map = np.sqrt(((arr - ref) ** 2).mean(axis=0))

    bias_maps[name] = bias_map
    rmse_maps[name] = rmse_map

    rows.append({
        "model": name,
        "spatial_correlation": spatial_corr,
        "std_ratio": std_ratio,
        "centered_rmse": crmse,
        "domain_mean_bias": bias_map.mean(),
        "domain_mean_rmse": rmse_map.mean(),
    })

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(TAYLOR_CSV_PATH), exist_ok=True)
df.to_csv(TAYLOR_CSV_PATH, index=False)

print("=" * 100)
print(" TAYLOR DIAGRAM STATISTICS (Section 3.7.2) - time-mean climatological pattern vs. reference")
print(" Reference spatial std (true GHI time-mean map): {:.4f} W m-2".format(ref_std_spatial))
print("=" * 100)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("=" * 100)

safe_names = {
    "Baseline (bilinear)": "baseline",
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "CNN": "cnn",
    "U-Net": "unet",
}

data_vars = {"ghi_true_mean_map": (["fine_lat", "fine_lon"], ref_mean_map)}
for name, key in safe_names.items():
    data_vars[f"bias_map_{key}"] = (["fine_lat", "fine_lon"], bias_maps[name])
    data_vars[f"rmse_map_{key}"] = (["fine_lat", "fine_lon"], rmse_maps[name])

ds_out = xr.Dataset(
    data_vars,
    coords={"fine_lat": ds["fine_lat"].values, "fine_lon": ds["fine_lon"].values},
)
for v in ds_out.data_vars:
    ds_out[v].attrs["units"] = "W m-2"
ds_out.to_netcdf(OUTPUT_PATH)
print(f"\nSaved bias/RMSE maps: {OUTPUT_PATH}")
print(f"Saved Taylor diagram stats table: {TAYLOR_CSV_PATH}")
