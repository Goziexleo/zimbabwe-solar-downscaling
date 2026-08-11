import xarray as xr
import os

# --- Configuration ---
input_dir = './data/raw/merra2'
output_dir = './data/processed/era5/predictors'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'merra2_monthly_od550aer_1985_2010.nc')

print("Loading and spatially clipping global MERRA-2 data...")

try:
    # 1. Load all downloaded monthly files simultaneously
    # combine='by_coords' handles the time dimension automatically
    ds = xr.open_mfdataset(os.path.join(input_dir, '*.nc4'), combine='by_coords')
    
    # 2. Extract only the AOD variable and rename to match CMIP6 methodology
    # TOTEXTTAU is the standard variable name for Total Aerosol Extinction AOD
    if 'TOTEXTTAU' in ds:
        da = ds['TOTEXTTAU']
    else:
        # Fallback if variable is named differently in specific file versions
        raise KeyError("Variable 'TOTEXTTAU' not found in files. Check file content.")

    da = da.rename('od550aer')
    
    # 3. Handle coordinate orientation
    # MERRA-2 lats are usually decreasing (90 to -90). 
    # Sorting by lat ensures the slice works correctly for any domain.
    da = da.sortby('lat')
    
    # 4. Clip spatially to the Zimbabwe domain
    # Latitudes: -22.5 to -15.0 | Longitudes: 25.0 to 33.5
    ds_clipped = da.sel(
        lat=slice(-22.5, -15.0), 
        lon=slice(25.0, 33.5)
    )
    
    # 5. Save the finalized, lightweight file
    print("Saving clipped Zimbabwe domain data...")
    ds_clipped.to_netcdf(output_file)
    print(f"Success! Saved to: {output_file}")
    
except Exception as e:
    print(f"An error occurred: {e}")
    