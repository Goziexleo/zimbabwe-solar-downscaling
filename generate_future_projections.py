import os
import numpy as np
import xarray as xr
import pandas as pd
import torch

from ml_dataset_common import MODEL_PREDICTOR_VARS, compute_astronomical_features, safe_nan_to_num
from unet_model import load_unet_checkpoint

# --- Configuration (Sections 3.8.1) ---
model_path = os.environ.get(
    "UNET_MODEL_PATH", os.path.abspath("./data/processed/models/unet/unet_csi_downscaler.pth")
)
train_path = os.path.abspath("./data/processed/ml_ready/ml_training_dataset.nc")
topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
finegrid_clearsky_path = os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")
cmip6_dir = os.path.abspath("./data/processed/cmip6_bias_corrected")
output_dir = os.environ.get("UNET_PROJECTIONS_OUTPUT_DIR",
                            os.path.abspath("./data/processed/projections"))
os.makedirs(output_dir, exist_ok=True)

gcms = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
scenarios = ["ssp245", "ssp585"]
INFERENCE_BATCH = 32

print("Loading trained ClimateU-Net checkpoint...")
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model, checkpoint = load_unet_checkpoint(model_path, device)

coarse_vars = checkpoint['coarse_vars']
fine_static_vars = checkpoint['fine_static_vars']
fine_dynamic_vars = checkpoint['fine_dynamic_vars']
if coarse_vars != MODEL_PREDICTOR_VARS:
    raise RuntimeError(
        f"Checkpoint coarse_vars {coarse_vars} does not match MODEL_PREDICTOR_VARS {MODEL_PREDICTOR_VARS}"
    )

means = checkpoint['feature_means']
stds = checkpoint['feature_stds']
climatology_mean = checkpoint['climatology_mean']
anomaly_std = checkpoint['anomaly_std']
fine_shape = checkpoint['fine_shape']

print("Loading native fine-resolution topography and fine-grid clear-sky climatology...")
ds_train = xr.open_dataset(train_path)
fine_lat = ds_train['fine_lat'].values
fine_lon = ds_train['fine_lon'].values
fine_lon_grid, fine_lat_grid = np.meshgrid(fine_lon, fine_lat)

ds_topo = xr.open_dataset(topo_path)
topo_lat_name = 'lat' if 'lat' in ds_topo.coords else 'y'
topo_lon_name = 'lon' if 'lon' in ds_topo.coords else 'x'
ds_topo = ds_topo.rename({topo_lat_name: 'lat', topo_lon_name: 'lon'}).sortby('lat')
# The topo file's native grid is the buffered 0.1 deg grid, wider than the
# core fine target grid - interpolate down rather than assuming equality.
ds_topo = ds_topo.interp(lat=fine_lat, lon=fine_lon, method="linear")

# svf in particular has NaN at grid edges from the fine-DEM interpolation;
# a single NaN poisons every BatchNorm channel across the whole field.
topo_fields = {f: safe_nan_to_num(ds_topo[f].values, "generate_future_projections:ds_topo_f_.values") for f in fine_static_vars}

ds_clearsky_fine = xr.open_dataset(finegrid_clearsky_path)
clearsky_by_month = ds_clearsky_fine['clearsky_ghi'].values  # (12, fine_lat, fine_lon)

fine_lat_norm = (fine_lat_grid - means['fine_lat']) / stds['fine_lat']
fine_lon_norm = (fine_lon_grid - means['fine_lon']) / stds['fine_lon']

print("Starting batch inference across GCMs and SSP emission scenarios (2026-2100)...")

with torch.no_grad():
    for gcm in gcms:
        for scenario in scenarios:
            gcm_path = os.path.join(cmip6_dir, f"{gcm}_{scenario}_EDCDFm_corrected.nc")
            if not os.path.exists(gcm_path):
                print(f"Warning: Projection file not found: {gcm_path}. Skipping...")
                continue

            ds_gcm = xr.open_dataset(gcm_path).sortby('lat')
            missing = set(MODEL_PREDICTOR_VARS) - set(ds_gcm.data_vars)
            if missing:
                print(f"Warning: {gcm_path} missing predictors {sorted(missing)}. Skipping...")
                continue

            times = pd.to_datetime(ds_gcm.time.values)
            n_times = len(times)
            print(f"Processing GCM: {gcm} | Scenario: {scenario} ({n_times} time steps)...")

            sza_deg, sin_doy, cos_doy = compute_astronomical_features(fine_lat, fine_lon, times)
            months = times.month.values

            projected_csi = np.zeros((n_times,) + tuple(fine_shape), dtype=np.float32)
            projected_ghi = np.zeros_like(projected_csi)

            for start in range(0, n_times, INFERENCE_BATCH):
                end = min(start + INFERENCE_BATCH, n_times)
                batch_n = end - start

                coarse_arrays = []
                for f in MODEL_PREDICTOR_VARS:
                    arr = np.nan_to_num(ds_gcm[f].isel(time=slice(start, end)).values, nan=0.0)
                    coarse_arrays.append((arr - means[f]) / stds[f])
                x_coarse = np.stack(coarse_arrays, axis=1).astype(np.float32)

                fine_arrays = []
                for f in fine_static_vars:
                    arr = np.tile(topo_fields[f][np.newaxis, :, :], (batch_n, 1, 1))
                    fine_arrays.append((arr - means[f]) / stds[f])
                for f, arr_full in zip(fine_dynamic_vars, [sza_deg, sin_doy, cos_doy]):
                    fine_arrays.append((arr_full[start:end] - means[f]) / stds[f])
                fine_arrays.append(np.tile(fine_lat_norm[np.newaxis, :, :], (batch_n, 1, 1)))
                fine_arrays.append(np.tile(fine_lon_norm[np.newaxis, :, :], (batch_n, 1, 1)))
                x_fine = np.stack(fine_arrays, axis=1).astype(np.float32)

                xc_tensor = torch.from_numpy(x_coarse).to(device)
                xf_tensor = torch.from_numpy(x_fine).to(device)

                pred_norm = model(xc_tensor, xf_tensor)
                pred_anomaly = pred_norm.detach().cpu().numpy()[:, 0] * anomaly_std
                pred_csi_batch = np.clip(pred_anomaly + climatology_mean, 0.0, 1.1)

                clearsky_batch = clearsky_by_month[months[start:end] - 1]
                ghi_batch = pred_csi_batch * clearsky_batch

                projected_csi[start:end] = pred_csi_batch
                projected_ghi[start:end] = ghi_batch

            ds_out = xr.Dataset(
                {
                    "clear_sky_index": (["time", "lat", "lon"], projected_csi),
                    "ghi": (["time", "lat", "lon"], projected_ghi),
                },
                coords={"time": times.values, "lat": fine_lat, "lon": fine_lon},
            )
            ds_out["ghi"].attrs["units"] = "W m-2"

            out_file = os.path.join(output_dir, f"downscaled_ghi_{gcm}_{scenario}_2026_2100.nc")
            ds_out.to_netcdf(out_file)
            print(f"  -> Successfully saved: {out_file}")

print("\nSuccess: Future climate projection generation complete.")
