import os
import numpy as np
import xarray as xr
import pandas as pd
import pvlib

# --- Configuration ---
years_train = range(1985, 2011)
years_val = range(2011, 2025)

def process_csi_files(years, split_type):
    print(f"\nRecalculating daylight-averaged {split_type} CSI files ({years[0]}-{years[-1]}):")
    folder_path = f"./data/processed/era5/csi/{split_type}"
    os.makedirs(folder_path, exist_ok=True)
    
    for year in years:
        target_file = f"./data/processed/era5/monthly/{split_type}/monthly_era5_{split_type}_{year}.nc"
        csi_out_file = os.path.join(folder_path, f"csi_monthly_era5_{split_type}_{year}.nc")
        
        if not os.path.exists(target_file):
            print(f"  - Missing target file for {year}, skipping.")
            continue
            
        ds_t = xr.open_dataset(target_file)
        ssrd = ds_t['ssrd'].values # shape: (time, lat, lon)
        lats = ds_t['lat'].values
        lons = ds_t['lon'].values
        times = pd.to_datetime(ds_t.time.values)
        
        lon_25d, lat_25d = np.meshgrid(lons, lats)
        cs_ghi_array = np.zeros_like(ssrd)
        
        lat_flat = lat_25d.flatten()
        lon_flat = lon_25d.flatten()
        
        for t_idx, dt in enumerate(times):
            # Evaluate across representative daytime hours (e.g., 6 AM to 6 PM local solar time) 
            # to compute a true monthly mean daytime clear-sky GHI baseline
            daytime_hours = np.linspace(6.0, 18.0, 13)
            accumulated_cs_ghi = np.zeros(lat_flat.shape)
            
            for hour in daytime_hours:
                # Construct datetime for each representative daytime hour on the 15th of the month
                hour_dt = pd.Timestamp(year=dt.year, month=dt.month, day=15, hour=int(hour), minute=int((hour % 1) * 60))
                time_repeated = pd.DatetimeIndex([hour_dt] * len(lat_flat))
                
                solpos = pvlib.solarposition.get_solarposition(
                    time_repeated, 
                    lat_flat, 
                    lon_flat
                )
                sza_flat = solpos['apparent_zenith'].values
                sza_flat = np.clip(sza_flat, 0.0, 89.9)
                
                rel_airmass = pvlib.atmosphere.get_relative_airmass(sza_flat)
                pressure = pvlib.atmosphere.alt2pres(1000.0)
                abs_airmass = pvlib.atmosphere.get_absolute_airmass(rel_airmass, pressure)
                
                dni_extra = pvlib.irradiance.get_extra_radiation(hour_dt)
                clearsky = pvlib.clearsky.ineichen(
                    sza_flat, 
                    abs_airmass,
                    linke_turbidity=3.0, 
                    altitude=1000.0, 
                    dni_extra=dni_extra
                )
                
                if isinstance(clearsky, dict):
                    ghi_hr = clearsky['ghi']
                else:
                    ghi_hr = clearsky['ghi'].values if hasattr(clearsky['ghi'], 'values') else clearsky['ghi']
                
                accumulated_cs_ghi += np.nan_to_num(ghi_hr, nan=0.0)
                
            # Average across daylight hours to get monthly mean daytime clear-sky GHI
            mean_cs_ghi = accumulated_cs_ghi / len(daytime_hours)
            mean_cs_ghi = np.where(mean_cs_ghi < 1.0, 1.0, mean_cs_ghi)
            cs_ghi_array[t_idx] = mean_cs_ghi.reshape(lat_25d.shape)
            
        ds_csi = xr.Dataset(
            {
                "clearsky_ghi": (("time", "lat", "lon"), cs_ghi_array),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons
            }
        )
        ds_csi.to_netcdf(csi_out_file)
        ds_t.close()
    print(f"Successfully regenerated daylight-averaged {split_type} CSI files.")

process_csi_files(years_train, "training")
process_csi_files(years_validation if 'years_validation' in globals() else range(2011, 2025), "validation")
print("\nAll CSI files successfully recomputed with correct daytime averaging.")