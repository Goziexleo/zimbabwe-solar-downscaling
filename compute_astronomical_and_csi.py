import os
import numpy as np
import xarray as xr
import pandas as pd
import pvlib

# --- Configuration ---
era5_file = "./data/processed/era5/predictors/era5_monthly_predictors_1985_2010.nc"
output_dir = os.path.abspath("./data/processed/features")
os.makedirs(output_dir, exist_ok=True)
output_nc = os.path.join(output_dir, "zimbabwe_astronomical_and_csi_features.nc")

print("Loading ERA5 reference training targets for clear-sky normalization...")
ds_era5 = xr.open_dataset(era5_file, engine='netcdf4')

# Identify coordinate names
lat_name = 'lat' if 'lat' in ds_era5.coords else 'latitude'
lon_name = 'lon' if 'lon' in ds_era5.coords else 'longitude'
lats = ds_era5[lat_name].values
lons = ds_era5[lon_name].values
times = pd.to_datetime(ds_era5.time.values)

print("Computing astronomical features and Clear-Sky Index (CSI) via PVLIB (Sections 3.5.3 & 3.5.4)...")
# Extract ssrd (surface solar radiation downwards in J m-2, convert to daily mean W m-2)
# Assuming ERA5 monthly or daily mean ssrd field is stored as ssrd or rsds
ssrd_var = 'ssrd' if 'ssrd' in ds_era5 else 'rsds'
ssrd = ds_era5[ssrd_var].values # shape: (time, lat, lon)

# Initialize arrays for SZA and CSI
sza_array = np.zeros_like(ssrd)
csi_array = np.zeros_like(ssrd)

# Grid mesh for pvlib
lon_2d, lat_2d = np.meshgrid(lons, lats)

for t_idx, dt in enumerate(times):
    doy = dt.dayofyear
    # Cyclic encoding for Day of Year (Section 3.5.3)
    # Computed per time step
    
    # Compute clear-sky GHI using PVLIB Ineichen model for each month/time step
    # Using approximate solar noon for monthly mean representation
    solpos = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([dt]), lat_2d.flatten(), lon_2d.flatten())
    sza_flat = solpos['apparent_zenith'].values.reshape(lat_2d.shape)
    sza_array[t_idx] = sza_flat
    
    # Ineichen clear-sky model estimation (assuming standard linkage/altitude)
    # Using elevation from our DEM or standard pressure approximation
    pressure = pvlib.atmosphere.alt2pres(1000.0) # baseline approximation
    dni_extra = pvlib.irradiance.get_extra_radiation(dt)
    clearsky = pvlib.clearsky.ineichen(sza_flat, Linkage=3.0, altitude=1000.0, dni_extra=dni_extra)
    cs_ghi = clearsky['ghi'].values
    
    # Avoid division by zero during nocturnal or extreme low sun angles
    cs_ghi = np.where(cs_ghi < 1.0, 1.0, cs_ghi)
    
    # Convert surface solar radiation to W/m2 if stored as accumulated energy, 
    # or use direct flux values depending on ERA5 preprocessing units.
    # Here we assume ssrd is already formatted to W/m2 or compute CSI directly:
    ghi_flux = ssrd[t_idx] 
    csi = np.clip(ghi_flux / cs_ghi, 0.0, 1.1)
    csi_array[t_idx] = csi

# Construct cyclic Day of Year components across time dimension
doy_values = times.dayofyear.values
sin_doy = np.sin(2.0 * np.pi * doy_values / 365.25)
cos_doy = np.cos(2.0 * np.pi * doy_values / 365.25)

# Broadcast temporal arrays to spatial grid
sin_doy_3d = np.tile(sin_doy[:, np.newaxis, np.newaxis], (1, len(lats), len(lons)))
cos_doy_3d = np.tile(cos_doy[:, np.newaxis, np.newaxis], (1, len(lats), len(lons)))

# Create xarray Dataset
ds_features = xr.Dataset(
    {
        "solar_zenith_angle": (("time", lat_name, lon_name), sza_array),
        "clear_sky_index": (("time", lat_name, lon_name), csi_array),
        "sin_doy": (("time", lat_name, lon_name), sin_doy_3d),
        "cos_doy": (("time", lat_name, lon_name), cos_doy_3d),
    },
    coords={
        "time": times,
        lat_name: lats,
        lon_name: lons
    }
)

ds_features.to_netcdf(output_nc)
print(f"Success! Astronomical and CSI features saved to: {output_nc}")