import os
import numpy as np
import xarray as xr
import pandas as pd

from ml_dataset_common import (
    COARSE_PREDICTOR_VARS, compute_astronomical_features, build_fine_grid_csi, FINE_LAT, FINE_LON,
)

# --- Configuration (Section 3.4.1: two-stage regridding) ---
output_dir = os.path.abspath("./data/processed/ml_ready")
os.makedirs(output_dir, exist_ok=True)
output_val = os.path.join(output_dir, "ml_validation_dataset.nc")

topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
predictors_path = os.path.abspath("./data/processed/era5/predictors/era5_monthly_predictors_2011_2024.nc")
finegrid_clearsky_path = os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")

print("Loading topographic features and interpolating to the coarse ERA5 grid...")
ds_topo = xr.open_dataset(topo_path)
topo_lat_name = 'lat' if 'lat' in ds_topo.coords else 'y'
topo_lon_name = 'lon' if 'lon' in ds_topo.coords else 'x'
ds_topo = ds_topo.rename({topo_lat_name: 'lat', topo_lon_name: 'lon'})

# Core (non-buffered) domain matching the coarse ERA5 grid's actual coverage
# exactly - NOT the buffered topography grid (see ml_dataset_common.py).
fine_lat = FINE_LAT
fine_lon = FINE_LON

if not os.path.exists(predictors_path):
    raise FileNotFoundError(
        "Validation predictors are required but were not found at "
        f"{predictors_path}. Build this file (and run merge_predictor_stack.py) "
        "before creating the validation ML dataset."
    )
ds_predictors = xr.open_dataset(predictors_path).sortby('lat')
missing_predictors = set(COARSE_PREDICTOR_VARS) - set(ds_predictors.data_vars)
if missing_predictors:
    raise ValueError(
        "Validation predictor file is missing required variables: "
        + ", ".join(sorted(missing_predictors))
    )

print("Loading fine-grid (0.1 deg) clear-sky GHI climatology...")
ds_clearsky_fine = xr.open_dataset(finegrid_clearsky_path)

start_year, end_year = 2011, 2024
print(f"\nCompiling validation dataset ({start_year}-{end_year})...")

ssrd_list, sza_list, sin_doy_list, cos_doy_list = [], [], [], []
csi_fine_list = []
predictor_lists = {var: [] for var in COARSE_PREDICTOR_VARS}
time_list = []

lats = None
lons = None

for year in range(start_year, end_year + 1):
    target_file = f"./data/processed/era5/monthly/validation/monthly_era5_validation_{year}.nc"

    if not os.path.exists(target_file):
        print(f"Missing file for {year}, skipping...")
        continue

    ds_t = xr.open_dataset(target_file).sortby('lat')

    if lats is None:
        lats = ds_t.lat.values if 'lat' in ds_t.coords else ds_t.latitude.values
        lons = ds_t.lon.values if 'lon' in ds_t.coords else ds_t.longitude.values
        ds_topo_coarse = ds_topo.interp(lat=lats, lon=lons, method="linear")

    ssrd_coarse = ds_t['ssrd']
    ssrd_list.append(ssrd_coarse.values)

    times = pd.to_datetime(ds_t.time.values)
    time_list.extend(times)

    # Predictors are stored separately from the radiation target. Selecting by
    # timestamp keeps each predictor field aligned with its CSI target.
    target_months = pd.PeriodIndex(times, freq="M")
    predictor_months = pd.PeriodIndex(pd.to_datetime(ds_predictors.time.values), freq="M")
    predictor_indices = predictor_months.get_indexer(target_months)
    if np.any(predictor_indices < 0):
        raise ValueError(f"Predictor timestamps do not cover validation year {year}.")
    predictor_slice = ds_predictors.isel(time=predictor_indices)
    for var in COARSE_PREDICTOR_VARS:
        predictor_lists[var].append(predictor_slice[var].values)

    # --- Fine-grid target: bicubic interpolation + fine-grid clear-sky normalisation ---
    csi_fine, _ = build_fine_grid_csi(ssrd_coarse, times, fine_lat, fine_lon, ds_clearsky_fine)
    csi_fine_list.append(csi_fine)

    sza_deg, sin_doy, cos_doy = compute_astronomical_features(ds_t.lat.values, ds_t.lon.values, times)
    sza_list.append(sza_deg)
    sin_doy_list.append(sin_doy)
    cos_doy_list.append(cos_doy)

all_ssrd = np.concatenate(ssrd_list, axis=0)
all_csi_fine = np.concatenate(csi_fine_list, axis=0)
all_sza = np.concatenate(sza_list, axis=0)
all_sin_doy = np.concatenate(sin_doy_list, axis=0)
all_cos_doy = np.concatenate(cos_doy_list, axis=0)

data_vars = {
    "ssrd": (("time", "lat", "lon"), all_ssrd),
    "solar_zenith_angle": (("time", "lat", "lon"), all_sza),
    "sin_doy": (("time", "lat", "lon"), all_sin_doy),
    "cos_doy": (("time", "lat", "lon"), all_cos_doy),
    "elevation": (("lat", "lon"), ds_topo_coarse['elevation'].values),
    "slope": (("lat", "lon"), ds_topo_coarse['slope'].values),
    "svf": (("lat", "lon"), ds_topo_coarse['svf'].values),
    "clear_sky_index": (("time", "fine_lat", "fine_lon"), all_csi_fine),
}
for var in COARSE_PREDICTOR_VARS:
    data_vars[var] = (("time", "lat", "lon"), np.concatenate(predictor_lists[var], axis=0))

ds_ml_val = xr.Dataset(
    data_vars,
    coords={
        "time": pd.to_datetime(time_list),
        "lat": lats,
        "lon": lons,
        "fine_lat": fine_lat,
        "fine_lon": fine_lon,
    },
)

ds_ml_val.to_netcdf(output_val)
print(f"Success! ML Validation dataset saved to: {output_val}")
print(f"  Coarse input grid: {len(lats)} x {len(lons)} (0.25 deg)")
print(f"  Fine target grid:  {len(fine_lat)} x {len(fine_lon)} (0.1 deg)")
print(f"  Predictors: {COARSE_PREDICTOR_VARS}")
