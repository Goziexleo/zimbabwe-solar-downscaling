"""Apply the trained CNN (Section 3.6.5) to bias-corrected CMIP6 (2026-2100).

Exists so sigma_arch in the uncertainty decomposition can be estimated from
three architectures rather than two. A spread computed from n = 2 is |a - b| / 2
dressed up as a standard deviation, and it carries the headline result that
architecture choice overtakes GCM choice by the long-term horizon. Three members
matches sigma_GCM's own n = 3 and makes the comparison like-for-like.

The CNN is the simplest of the three to project: unlike the U-Net it takes a
single coarse-grid tensor of (C+3) channels - 5 atmospheric predictors plus
elevation, slope and sky-view factor interpolated onto the coarse grid - and
upsamples internally.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import torch

from cnn_model import load_cnn_checkpoint
from ml_dataset_common import (
    MODEL_PREDICTOR_VARS, FINE_LAT, FINE_LON, csi_to_ghi, safe_nan_to_num,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.environ.get(
    "CNN_MODEL_PATH", os.path.join(ROOT, "data/processed/models/cnn/cnn_csi_downscaler.pth"))
TOPO_PATH = os.path.join(ROOT, "data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
CLEARSKY_PATH = os.environ.get(
    "FINEGRID_CLEARSKY_PATH",
    os.path.join(ROOT, "data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc"))
CMIP6_DIR = os.path.join(ROOT, "data/processed/cmip6_bias_corrected")
OUTPUT_DIR = os.environ.get(
    "CNN_PROJECTIONS_OUTPUT_DIR", os.path.join(ROOT, "data/processed/projections_cnn"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

GCMS = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
SCENARIOS = ["ssp245", "ssp585"]

device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

model, ckpt = load_cnn_checkpoint(CNN_MODEL_PATH, device)
feature_vars = ckpt["feature_vars"]
means, stds = ckpt["feature_means"], ckpt["feature_stds"]
expected = MODEL_PREDICTOR_VARS + ["elevation", "slope", "svf"]
if feature_vars != expected:
    raise RuntimeError(f"Checkpoint feature_vars {feature_vars} != expected {expected}")

ds_cs = xr.open_dataset(CLEARSKY_PATH)

print("Loading topography...")
ds_topo = xr.open_dataset(TOPO_PATH)
lat_n = "lat" if "lat" in ds_topo.coords else "y"
lon_n = "lon" if "lon" in ds_topo.coords else "x"
ds_topo = ds_topo.rename({lat_n: "lat", lon_n: "lon"}).sortby("lat")

print("Projecting across GCMs and scenarios (2026-2100)...")
for gcm in GCMS:
    for scenario in SCENARIOS:
        path = os.path.join(CMIP6_DIR, f"{gcm}_{scenario}_EDCDFm_corrected.nc")
        if not os.path.exists(path):
            print(f"  {gcm}/{scenario}: missing, skipping")
            continue
        ds_g = xr.open_dataset(path).sortby("lat")
        missing = set(MODEL_PREDICTOR_VARS) - set(ds_g.data_vars)
        if missing:
            print(f"  {gcm}/{scenario}: missing {sorted(missing)}, skipping")
            continue

        times = pd.to_datetime(ds_g.time.values)
        coarse_lat, coarse_lon = ds_g["lat"].values, ds_g["lon"].values
        topo_coarse = ds_topo.interp(lat=coarse_lat, lon=coarse_lon, method="linear")

        chans = []
        for f in MODEL_PREDICTOR_VARS:
            arr = safe_nan_to_num(ds_g[f].values, f"generate_future_projections_cnn:{f}")
            chans.append((arr - means[f]) / stds[f])
        for f in ["elevation", "slope", "svf"]:
            arr = safe_nan_to_num(topo_coarse[f].values, f"generate_future_projections_cnn:{f}")
            arr = np.tile(arr[np.newaxis, :, :], (len(times), 1, 1))
            chans.append((arr - means[f]) / stds[f])
        X = np.stack(chans, axis=1).astype(np.float32)

        preds = []
        with torch.no_grad():
            for i in range(0, len(times), 32):
                xb = torch.from_numpy(X[i:i + 32]).to(device)
                preds.append(model(xb).cpu().numpy()[:, 0])
        csi = np.clip(np.concatenate(preds, axis=0), 0.0, 1.1)
        ghi = csi_to_ghi(csi, times.values, ds_cs)

        out = xr.Dataset(
            {"clear_sky_index": (["time", "lat", "lon"], csi),
             "ghi": (["time", "lat", "lon"], ghi)},
            coords={"time": times.values, "lat": FINE_LAT, "lon": FINE_LON},
        )
        out["ghi"].attrs["units"] = "W m-2"
        f_out = os.path.join(OUTPUT_DIR, f"downscaled_ghi_{gcm}_{scenario}_2026_2100.nc")
        out.to_netcdf(f_out)
        print(f"  -> {f_out}")

print("\nSuccess: CNN future projection generation complete.")
