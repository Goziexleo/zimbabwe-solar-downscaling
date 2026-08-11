"""Shared feature assembly for the pixel-wise RF/XGBoost models (Sections
3.6.3, 3.6.4).

Section 3.6.1: "for RF and XGBoost, the spatial dimensions are flattened, and
each output grid cell is predicted independently as a function of the
collocated input features." The output grid is the fine 0.1 deg target grid.
Atmospheric predictors (Table 3.1) are collocated by nearest-neighbour
lookup from their native coarse 0.25 deg grid; topographic and astronomical
features (Section 3.5) are used at their own native fine 0.1 deg resolution,
per Section 3.6.3/3.6.4's "together with the topographic and astronomical
features described in Section 3.5" (Section 3.5.2 explicitly describes
topography aggregated to the 0.1 deg grid, not degraded further).
"""

import numpy as np
import xarray as xr

from ml_dataset_common import MODEL_PREDICTOR_VARS, compute_astronomical_features, safe_nan_to_num

FEATURE_VARS = MODEL_PREDICTOR_VARS + [
    "elevation", "slope", "svf",
    "solar_zenith_angle", "sin_doy", "cos_doy",
]


def _nearest_index(coarse_coords, fine_coords):
    return np.array([np.abs(coarse_coords - c).argmin() for c in fine_coords])


def build_fine_pixel_dataset(ds, topo_fine_path):
    """Returns (X, y, feature_vars) where X has shape
    (n_time, n_fine_lat, n_fine_lon, n_features) and y has shape
    (n_time, n_fine_lat, n_fine_lon)."""
    times = ds.time.values
    n_time = len(times)

    coarse_lat, coarse_lon = ds['lat'].values, ds['lon'].values
    fine_lat, fine_lon = ds['fine_lat'].values, ds['fine_lon'].values
    near_lat_idx = _nearest_index(coarse_lat, fine_lat)
    near_lon_idx = _nearest_index(coarse_lon, fine_lon)

    n_fine_lat, n_fine_lon = len(fine_lat), len(fine_lon)
    n_features = len(FEATURE_VARS)
    X = np.zeros((n_time, n_fine_lat, n_fine_lon, n_features), dtype=np.float32)

    # --- Atmospheric predictors: nearest-neighbour collocation from the
    #     coarse grid onto each fine output cell ---
    for k, var in enumerate(MODEL_PREDICTOR_VARS):
        coarse_arr = safe_nan_to_num(ds[var].values, "pixelwise_common:ds_var_.values")  # (time, coarse_lat, coarse_lon)
        collocated = coarse_arr[:, near_lat_idx, :][:, :, near_lon_idx]  # (time, fine_lat, fine_lon)
        X[:, :, :, k] = collocated

    # --- Topographic covariates at native fine resolution (Section 3.5.2) ---
    ds_topo = xr.open_dataset(topo_fine_path)
    topo_lat_name = 'lat' if 'lat' in ds_topo.coords else 'y'
    topo_lon_name = 'lon' if 'lon' in ds_topo.coords else 'x'
    ds_topo = ds_topo.rename({topo_lat_name: 'lat', topo_lon_name: 'lon'}).sortby('lat')
    ds_topo = ds_topo.interp(lat=fine_lat, lon=fine_lon, method="linear")
    for offset, var in enumerate(["elevation", "slope", "svf"]):
        arr = safe_nan_to_num(ds_topo[var].values, "pixelwise_common:ds_topo_var_.values")
        X[:, :, :, len(MODEL_PREDICTOR_VARS) + offset] = arr[np.newaxis, :, :]

    # --- Astronomical features at native fine resolution (Section 3.5.3) ---
    sza_deg, sin_doy, cos_doy = compute_astronomical_features(fine_lat, fine_lon, times)
    base = len(MODEL_PREDICTOR_VARS) + 3
    X[:, :, :, base + 0] = sza_deg
    X[:, :, :, base + 1] = sin_doy
    X[:, :, :, base + 2] = cos_doy

    y = safe_nan_to_num(ds['clear_sky_index'].values, "pixelwise_common:ds_clear_sky_index_.values")  # (time, fine_lat, fine_lon)
    return X, y, FEATURE_VARS
