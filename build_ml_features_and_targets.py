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
output_train = os.path.join(output_dir, "ml_training_dataset.nc")

topo_path = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
predictors_path = os.path.abspath("./data/processed/era5/predictors/era5_monthly_predictors_1985_2010.nc")
finegrid_clearsky_path = os.path.abspath("./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology.nc")

print("Loading ERA5 atmospheric predictors (1985-2010, coarse 0.25 deg grid)...")
# ERA5/CDS returns latitude descending (north-to-south); standardise to ascending
# so the coarse grid orientation matches the ascending fine (topography) grid.
ds_preds = xr.open_dataset(predictors_path).sortby('lat')
missing = set(COARSE_PREDICTOR_VARS) - set(ds_preds.data_vars)
if missing:
    raise ValueError(
        f"era5_monthly_predictors_1985_2010.nc is missing {sorted(missing)}. "
        "Run merge_predictor_stack.py first to assemble all 6 Table 3.1 predictors."
    )

lats = ds_preds.lat.values if 'lat' in ds_preds.coords else ds_preds.latitude.values
lons = ds_preds.lon.values if 'lon' in ds_preds.coords else ds_preds.longitude.values

print("Loading topographic features and interpolating to the coarse ERA5 grid "
      "(Section 3.6.5: topography enters as coarse-grid input channels)...")
ds_topo = xr.open_dataset(topo_path)
topo_lat_name = 'lat' if 'lat' in ds_topo.coords else 'y'
topo_lon_name = 'lon' if 'lon' in ds_topo.coords else 'x'
ds_topo = ds_topo.rename({topo_lat_name: 'lat', topo_lon_name: 'lon'})
ds_topo_coarse = ds_topo.interp(lat=lats, lon=lons, method="linear")

# Core (non-buffered) Zimbabwe domain per Section 3.2.1 - NOT the buffered
# topography grid, which extends beyond the coarse ERA5 grid's actual
# coverage and would fabricate a hard-zero CSI band at the interpolated edges.
fine_lat = FINE_LAT
fine_lon = FINE_LON

print("Loading fine-grid (0.1 deg) clear-sky GHI climatology...")
ds_clearsky_fine = xr.open_dataset(finegrid_clearsky_path)

start_year, end_year = 1985, 2010
print(f"\nCompiling training dataset ({start_year}-{end_year})...")

ssrd_list, sza_list, sin_doy_list, cos_doy_list = [], [], [], []
csi_fine_list = []
time_list = []

for year in range(start_year, end_year + 1):
    target_file = f"./data/processed/era5/monthly/training/monthly_era5_training_{year}.nc"
    if not os.path.exists(target_file):
        print(f"Missing file for {year}, skipping...")
        continue

    ds_t = xr.open_dataset(target_file).sortby('lat')
    ssrd_coarse = ds_t['ssrd']  # (time, lat, lon) coarse grid, W m-2 monthly mean, ascending lat
    ssrd_list.append(ssrd_coarse.values)

    times = pd.to_datetime(ds_t.time.values)
    time_list.extend(times)

    # --- Fine-grid target: bicubic interpolation + fine-grid clear-sky normalisation ---
    csi_fine, _ = build_fine_grid_csi(ssrd_coarse, times, fine_lat, fine_lon, ds_clearsky_fine)
    csi_fine_list.append(csi_fine)

    # --- Coarse-grid astronomical features (Section 3.5.3) ---
    sza_deg, sin_doy, cos_doy = compute_astronomical_features(ds_t.lat.values, ds_t.lon.values, times)
    sza_list.append(sza_deg)
    sin_doy_list.append(sin_doy)
    cos_doy_list.append(cos_doy)

all_ssrd = np.concatenate(ssrd_list, axis=0)
all_csi_fine = np.concatenate(csi_fine_list, axis=0)
all_sza = np.concatenate(sza_list, axis=0)
all_sin_doy = np.concatenate(sin_doy_list, axis=0)
all_cos_doy = np.concatenate(cos_doy_list, axis=0)
all_times = pd.to_datetime(time_list)

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
# Match predictors to targets on the calendar month, not by nearest timestamp.
# The predictor file stamps months at their first day and the ssrd target files
# at their last, so a nearest-neighbour match sits one day from the WRONG month
# and silently shifts every predictor by one month (see merge_predictor_stack).
predictor_months = pd.PeriodIndex(pd.to_datetime(ds_preds.time.values), freq="M")
target_months = pd.PeriodIndex(all_times, freq="M")
predictor_indices = predictor_months.get_indexer(target_months)
if np.any(predictor_indices < 0):
    raise ValueError("Predictor timestamps do not cover every training month.")
predictor_slice = ds_preds.isel(time=predictor_indices)
for var in COARSE_PREDICTOR_VARS:
    data_vars[var] = (("time", "lat", "lon"), predictor_slice[var].values)

ds_ml = xr.Dataset(
    data_vars,
    coords={
        "time": all_times,
        "lat": lats,
        "lon": lons,
        "fine_lat": fine_lat,
        "fine_lon": fine_lon,
    },
)

ds_ml.to_netcdf(output_train)
print(f"Success! ML Training dataset saved to: {output_train}")
print(f"  Coarse input grid: {len(lats)} x {len(lons)} (0.25 deg)")
print(f"  Fine target grid:  {len(fine_lat)} x {len(fine_lon)} (0.1 deg)")
print(f"  Predictors: {COARSE_PREDICTOR_VARS}")
