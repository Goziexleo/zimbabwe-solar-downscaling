"""Build daily-resolution ml_ready training/validation datasets.

Diagnostic experiment testing whether the monthly pipeline's small sample
count (312 training / 168 validation months) is limiting validation R
(Table 3.3): this aggregates the same raw hourly ERA5 files to DAILY means
instead of monthly, giving ~9,490 training / ~5,110 validation daily
samples. Mirrors build_ml_features_and_targets.py / build_ml_validation_dataset.py
exactly, except:
  - Coarse atmospheric predictors (clt, tas, ps, huss, rsds) are aggregated
    from the raw hourly files directly to daily means (not monthly).
  - od550aer has no daily MERRA-2 product available; its already-processed
    monthly value is held constant across every day of that month - a
    reasonable simplification since aerosol optical depth varies far more
    slowly than cloud fraction or humidity.
  - CSI target uses the day-of-year fine-grid clear-sky climatology
    (compute_finegrid_clearsky_daily.py) instead of the monthly one.
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd

from ml_dataset_common import (
    COARSE_PREDICTOR_VARS, compute_astronomical_features, build_fine_grid_csi, FINE_LAT, FINE_LON,
)

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "training"
assert SPLIT in ("training", "validation")

YEAR_RANGE = range(1985, 2011) if SPLIT == "training" else range(2011, 2025)
HOURLY_TARGET_DIR = f"./data/processed/../raw/era5/hourly/{SPLIT}"
HOURLY_PRED_DIR = "./data/raw/era5/predictors" if SPLIT == "training" else "./data/raw/era5/predictors/validation"
MONTHLY_PREDICTORS_PATH = os.path.abspath(
    f"./data/processed/era5/predictors/era5_monthly_predictors_{'1985_2010' if SPLIT == 'training' else '2011_2024'}.nc"
)
TOPO_PATH = os.path.abspath("./data/processed/topography/zimbabwe_topographic_features_0.1deg.nc")
FINEGRID_CLEARSKY_DAILY_PATH = os.path.abspath(
    "./data/processed/era5/csi_finegrid/clearsky_ghi_finegrid_climatology_daily.nc"
)
OUTPUT_DIR = os.path.abspath("./data/processed/ml_ready")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"ml_{SPLIT}_dataset_daily.nc")

R_DRY, R_VAP = 287.0597, 461.5250
A1, A3, A4, T0 = 611.21, 17.502, 32.19, 273.16


def clean(ds):
    rename_map = {}
    if "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    return ds.rename(rename_map).drop_vars(["expver", "number"], errors="ignore")


print(f"=== Building daily {SPLIT} dataset ({YEAR_RANGE.start}-{YEAR_RANGE.stop - 1}) ===")

print("Loading topography and fine-grid daily clear-sky climatology...")
ds_topo = xr.open_dataset(TOPO_PATH)
topo_lat_name = "lat" if "lat" in ds_topo.coords else "y"
topo_lon_name = "lon" if "lon" in ds_topo.coords else "x"
ds_topo = ds_topo.rename({topo_lat_name: "lat", topo_lon_name: "lon"}).sortby("lat")

fine_lat, fine_lon = FINE_LAT, FINE_LON
ds_clearsky_daily = xr.open_dataset(FINEGRID_CLEARSKY_DAILY_PATH)

print("Loading already-merged monthly predictor file (for od550aer month->day broadcast)...")
ds_monthly_preds = xr.open_dataset(MONTHLY_PREDICTORS_PATH).sortby("lat")
od550aer_by_month = ds_monthly_preds["od550aer"]  # (time=monthly, lat, lon)

ssrd_daily_list, clt_list, tas_list, ps_list, huss_list, od550aer_list = [], [], [], [], [], []
sza_list, sin_doy_list, cos_doy_list = [], [], []
csi_fine_list, time_list = [], []

coarse_lats = coarse_lons = None
ds_topo_coarse = None

for year in YEAR_RANGE:
    print(f"  Processing {year}...")
    target_glob = glob.glob(f"./data/raw/era5/hourly/{SPLIT}/era5_{SPLIT}_{year}.nc")
    if not target_glob:
        print(f"    Missing hourly target file for {year}, skipping.")
        continue
    ds_t = clean(xr.open_dataset(target_glob[0])).sortby("lat")

    pred_dir = HOURLY_PRED_DIR
    paths = {
        "tcc": f"{pred_dir}/era5_total_cloud_cover_{year}.nc",
        "t2m": f"{pred_dir}/era5_2m_temperature_{year}.nc",
        "sp": f"{pred_dir}/era5_surface_pressure_{year}.nc",
        "d2m": f"{pred_dir}/era5_2m_dewpoint_temperature_{year}.nc",
    }
    if not all(os.path.exists(p) for p in paths.values()):
        print(f"    Missing hourly predictor file(s) for {year}, skipping.")
        continue
    ds_preds = {name: clean(xr.open_dataset(p)).sortby("lat") for name, p in paths.items()}

    if coarse_lats is None:
        coarse_lats = ds_t.lat.values
        coarse_lons = ds_t.lon.values
        ds_topo_coarse = ds_topo.interp(lat=coarse_lats, lon=coarse_lons, method="linear")

    # --- Daily-mean atmospheric predictors (Section 3.4.2's hourly->daily
    #     aggregation, extended here to serve as the primary training
    #     resolution for this diagnostic instead of daily->monthly) ---
    d2m, sp = ds_preds["d2m"]["d2m"], ds_preds["sp"]["sp"]
    vapour_pressure = A1 * np.exp(A3 * (d2m - T0) / (d2m - A4))
    huss = (R_DRY / R_VAP) * vapour_pressure / (sp - (1 - R_DRY / R_VAP) * vapour_pressure)

    clt_daily = (ds_preds["tcc"]["tcc"] * 100).resample(time="1D").mean()
    tas_daily = ds_preds["t2m"]["t2m"].resample(time="1D").mean()
    ps_daily = sp.resample(time="1D").mean()
    huss_daily = huss.resample(time="1D").mean()

    # --- Daily-mean ssrd target: accumulated J/m2 per hour -> sum over the
    #     day -> divide by 86400s for a true daily-mean W/m2 field ---
    ssrd_daily = ds_t["ssrd"].resample(time="1D").sum() / 86400.0

    times = pd.to_datetime(ssrd_daily.time.values)
    time_list.extend(times)

    ssrd_daily_list.append(ssrd_daily.values)
    clt_list.append(clt_daily.values)
    tas_list.append(tas_daily.values)
    ps_list.append(ps_daily.values)
    huss_list.append(huss_daily.values)

    # --- od550aer: broadcast the already-processed monthly value across
    #     every day of its month (no daily MERRA-2 product available) ---
    year_months = pd.PeriodIndex(times, freq="M")
    monthly_periods = pd.PeriodIndex(pd.to_datetime(od550aer_by_month.time.values), freq="M")
    month_indices = monthly_periods.get_indexer(year_months)
    if np.any(month_indices < 0):
        raise ValueError(f"od550aer monthly file does not cover {year}.")
    od550aer_list.append(od550aer_by_month.isel(time=month_indices).values)

    # --- Fine-grid CSI target: bicubic interpolation + daily clear-sky norm ---
    csi_fine, _ = build_fine_grid_csi(ssrd_daily, times, fine_lat, fine_lon, ds_clearsky_daily)
    csi_fine_list.append(csi_fine)

    sza_deg, sin_doy, cos_doy = compute_astronomical_features(coarse_lats, coarse_lons, times)
    sza_list.append(sza_deg)
    sin_doy_list.append(sin_doy)
    cos_doy_list.append(cos_doy)

all_times = pd.to_datetime(time_list)
data_vars = {
    "ssrd": (("time", "lat", "lon"), np.concatenate(ssrd_daily_list, axis=0)),
    "clt": (("time", "lat", "lon"), np.concatenate(clt_list, axis=0)),
    "tas": (("time", "lat", "lon"), np.concatenate(tas_list, axis=0)),
    "ps": (("time", "lat", "lon"), np.concatenate(ps_list, axis=0)),
    "huss": (("time", "lat", "lon"), np.concatenate(huss_list, axis=0)),
    "rsds": (("time", "lat", "lon"), np.concatenate(ssrd_daily_list, axis=0)),  # reused, same field as ssrd
    "od550aer": (("time", "lat", "lon"), np.concatenate(od550aer_list, axis=0)),
    "solar_zenith_angle": (("time", "lat", "lon"), np.concatenate(sza_list, axis=0)),
    "sin_doy": (("time", "lat", "lon"), np.concatenate(sin_doy_list, axis=0)),
    "cos_doy": (("time", "lat", "lon"), np.concatenate(cos_doy_list, axis=0)),
    "elevation": (("lat", "lon"), ds_topo_coarse["elevation"].values),
    "slope": (("lat", "lon"), ds_topo_coarse["slope"].values),
    "svf": (("lat", "lon"), ds_topo_coarse["svf"].values),
    "clear_sky_index": (("time", "fine_lat", "fine_lon"), np.concatenate(csi_fine_list, axis=0)),
}

ds_ml = xr.Dataset(
    data_vars,
    coords={"time": all_times, "lat": coarse_lats, "lon": coarse_lons, "fine_lat": fine_lat, "fine_lon": fine_lon},
)

ds_ml.to_netcdf(OUTPUT_PATH)
print(f"\nSuccess! Daily {SPLIT} ML dataset saved to: {OUTPUT_PATH}")
print(f"  Samples: {len(all_times)} (vs monthly equivalent: {312 if SPLIT == 'training' else 168})")
print(f"  Coarse input grid: {len(coarse_lats)} x {len(coarse_lons)} (0.25 deg)")
print(f"  Fine target grid:  {len(fine_lat)} x {len(fine_lon)} (0.1 deg)")
