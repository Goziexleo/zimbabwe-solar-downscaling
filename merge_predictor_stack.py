"""Assemble the complete 6-variable CMIP6-analogue predictor stack (Table 3.1).

era5_monthly_predictors_{period}.nc currently only carries clt, tas, ps, huss.
This script adds the two missing predictors so every downstream ML dataset,
bias-correction, and model script draws from one consistent 6-variable file:

  - rsds: reused from the already-computed monthly-mean ERA5 ssrd field
    (same physical quantity as CMIP6 rsds, W m-2), rather than a fresh
    download - ssrd is already downloaded/aggregated for the CSI target.
  - od550aer: MERRA-2 TOTEXTTAU, bilinearly interpolated from its native
    0.5x0.625 deg grid onto the ERA5 0.25 deg grid.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# Set FORCE_REMERGE=1 to recompute rsds/od550aer even when they are already
# present. Needed because the original nearest-timestamp alignment (below)
# wrote a one-month-lagged rsds into the validation predictor file.
FORCE_REMERGE = os.environ.get("FORCE_REMERGE", "0") == "1"


def align_to_months(da, target_times, label="", allow_carry_forward=False):
    """Align a monthly field onto target_times by calendar month.

    The previous implementation used .reindex(time=..., method="nearest"),
    which is unsafe here: the ERA5 predictor files stamp each month at its
    FIRST day (2011-02-01) while the monthly ssrd files stamp it at its LAST
    (2011-01-31). Nearest-neighbour then matched 2011-02-01 to 2011-01-31,
    one day away, and so wrote January's irradiance into February's slot -
    lagging rsds by exactly one month through the whole validation period.
    Matching on the calendar month itself is stamp-convention independent.
    """
    src_months = pd.PeriodIndex(pd.to_datetime(da.time.values), freq="M")
    tgt_months = pd.PeriodIndex(pd.to_datetime(target_times), freq="M")
    idx = src_months.get_indexer(tgt_months)
    if np.any(idx < 0):
        missing = list(tgt_months[idx < 0])
        if not allow_carry_forward:
            raise ValueError(f"{label}: source does not cover months {missing[:5]}")
        # Carry the nearest available month forward, but say so explicitly.
        # Silently filling gaps is exactly how the rsds lag went unnoticed.
        print(f"  WARNING [{label}]: source is missing {len(missing)} of "
              f"{len(tgt_months)} months ({missing[:3]}{'...' if len(missing) > 3 else ''}). "
              f"Carrying the nearest available month forward for these.")
        idx = np.where(idx < 0, src_months.get_indexer(tgt_months, method="nearest"), idx)
    return da.isel(time=idx).assign_coords(time=target_times)

PERIODS = {
    "training": {
        "era5_predictors": "./data/processed/era5/predictors/era5_monthly_predictors_1985_2010.nc",
        "monthly_ssrd_glob": "./data/processed/era5/monthly/training/monthly_era5_training_*.nc",
        "merra2_od550aer": "./data/processed/era5/predictors/merra2_monthly_od550aer_1985_2010.nc",
    },
    "validation": {
        "era5_predictors": "./data/processed/era5/predictors/era5_monthly_predictors_2011_2024.nc",
        "monthly_ssrd_glob": "./data/processed/era5/monthly/validation/monthly_era5_validation_*.nc",
        "merra2_od550aer": "./data/processed/era5/predictors/merra2_monthly_od550aer_2011_2024.nc",
    },
}


def load_lat_lon(ds):
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"
    return ds.rename({lat_key: "lat", lon_key: "lon"}) if (lat_key, lon_key) != ("lat", "lon") else ds


for period, paths in PERIODS.items():
    print(f"\n=== {period.upper()} ===")

    if not os.path.exists(paths["era5_predictors"]):
        print(f"  Missing {paths['era5_predictors']}, skipping.")
        continue
    if not os.path.exists(paths["merra2_od550aer"]):
        print(f"  Missing {paths['merra2_od550aer']}, skipping (run the MERRA-2 download/process step first).")
        continue

    ds_preds = load_lat_lon(xr.open_dataset(paths["era5_predictors"]))
    lats, lons = ds_preds.lat.values, ds_preds.lon.values

    if "rsds" in ds_preds and "od550aer" in ds_preds and not FORCE_REMERGE:
        print("  Already has all 6 predictors, skipping (set FORCE_REMERGE=1 to recompute).")
        continue

    # --- rsds: reuse monthly ssrd (already computed for the CSI target) ---
    print("  Merging rsds from monthly ssrd fields...")
    ssrd_files = sorted(glob.glob(paths["monthly_ssrd_glob"]))
    ds_ssrd = xr.open_mfdataset(ssrd_files, combine="by_coords", data_vars="minimal", coords="minimal")
    ds_ssrd = load_lat_lon(ds_ssrd)
    rsds = ds_ssrd["ssrd"].rename("rsds").interp(lat=lats, lon=lons, method="linear")
    rsds = align_to_months(rsds, ds_preds.time.values, label=f"{period}/rsds")

    # --- od550aer: MERRA-2, bilinearly regridded to the ERA5 0.25 deg grid ---
    print("  Merging od550aer from MERRA-2 (regridded to ERA5 grid)...")
    ds_aod = load_lat_lon(xr.open_dataset(paths["merra2_od550aer"]))
    od550aer = ds_aod["od550aer"].interp(lat=lats, lon=lons, method="linear")
    od550aer = align_to_months(od550aer, ds_preds.time.values,
                               label=f"{period}/od550aer", allow_carry_forward=True)

    ds_preds = ds_preds.drop_vars([v for v in ("rsds", "od550aer") if v in ds_preds])
    ds_out = xr.merge([ds_preds, rsds.compute(), od550aer.compute()], compat="override")
    ds_out.to_netcdf(paths["era5_predictors"] + ".tmp")
    ds_preds.close()
    os.replace(paths["era5_predictors"] + ".tmp", paths["era5_predictors"])
    print(f"  Success! {paths['era5_predictors']} now has vars: {list(ds_out.data_vars)}")

print("\nDone. era5_monthly_predictors files now carry all 6 Table 3.1 predictors.")
