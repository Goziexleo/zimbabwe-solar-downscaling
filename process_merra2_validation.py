import xarray as xr
import os

# --- Configuration ---
input_dir = './data/raw/merra2_validation'
output_dir = './data/processed/era5/predictors'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'merra2_monthly_od550aer_2011_2024.nc')

print("Loading and spatially clipping global MERRA-2 data (2011-2024 validation period)...")

try:
    ds = xr.open_mfdataset(os.path.join(input_dir, '*.nc4'), combine='by_coords')

    if 'TOTEXTTAU' in ds:
        da = ds['TOTEXTTAU']
    else:
        raise KeyError("Variable 'TOTEXTTAU' not found in files. Check file content.")

    da = da.rename('od550aer')
    da = da.sortby('lat')

    # Clip spatially to the Zimbabwe domain (Section 3.2.1)
    ds_clipped = da.sel(
        lat=slice(-22.5, -15.0),
        lon=slice(25.0, 33.5)
    )

    print("Saving clipped Zimbabwe domain data...")
    ds_clipped.to_netcdf(output_file)
    print(f"Success! Saved to: {output_file}")

except Exception as e:
    print(f"An error occurred: {e}")
