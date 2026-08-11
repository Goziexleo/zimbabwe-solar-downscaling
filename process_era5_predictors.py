import xarray as xr
import numpy as np
import os

# --- Configuration ---
input_dir = './data/raw/era5/predictors'
output_dir = './data/processed/era5/predictors'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'era5_monthly_predictors_1985_2010.nc')

start_year = 1985
end_year = 2010

monthly_datasets = []

print("Initializing Predictor Processing and Aggregation...")

def clean_dataset(ds):
    """Renames the new CDS valid_time dimension and drops server artifacts."""
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    # Drop experimental version and ensemble number if they exist
    ds = ds.drop_vars(['expver', 'number'], errors='ignore')
    return ds

for year in range(start_year, end_year + 1):
    print(f"Processing Year: {year}")
    
    # Load and clean the 4 hourly files for the year
    ds_tcc = clean_dataset(xr.open_dataset(os.path.join(input_dir, f'era5_total_cloud_cover_{year}.nc')))
    ds_t2m = clean_dataset(xr.open_dataset(os.path.join(input_dir, f'era5_2m_temperature_{year}.nc')))
    ds_sp = clean_dataset(xr.open_dataset(os.path.join(input_dir, f'era5_surface_pressure_{year}.nc')))
    ds_d2m = clean_dataset(xr.open_dataset(os.path.join(input_dir, f'era5_2m_dewpoint_temperature_{year}.nc')))
    
    # 1. Calculate Specific Humidity (huss) using ECMWF Tetens formula
    R_dry = 287.0597
    R_vap = 461.5250
    a1 = 611.21  # Pa
    a3 = 17.502
    a4 = 32.19
    T0 = 273.16
    
    d2m = ds_d2m['d2m']
    sp = ds_sp['sp']
    
    # Vapor pressure (E)
    E = a1 * np.exp(a3 * (d2m - T0) / (d2m - a4))
    # Specific humidity
    huss = (R_dry / R_vap) * E / (sp - ((1 - R_dry / R_vap) * E))
    huss.name = 'huss'
    
    # 2. Merge and rename to match CMIP6 methodology
    ds_year = xr.merge([
        (ds_tcc['tcc'] * 100).rename('clt'),  # Convert fraction (0-1) to percentage (0-100)
        ds_t2m['t2m'].rename('tas'),
        ds_sp['sp'].rename('ps'),
        huss
    ])
    
    # 3. Resample to monthly means
    ds_monthly = ds_year.resample(time='1MS').mean()
    
    monthly_datasets.append(ds_monthly)
    
    # Close datasets to free up memory
    ds_tcc.close()
    ds_t2m.close()
    ds_sp.close()
    ds_d2m.close()

# 4. Concatenate all years into a continuous dataset
print("\nConcatenating all years into a single file...")
final_ds = xr.concat(monthly_datasets, dim='time')

# 5. Save to disk
print("Saving final predictor dataset...")
final_ds.to_netcdf(output_file)
print(f"Successfully saved to: {output_file}")