import xarray as xr
import pandas as pd
import numpy as np
import pvlib
import os
import glob
from pathlib import Path

# --- Configuration ---
input_dir = Path("./data/processed/era5/monthly")
output_dir = Path("./data/processed/era5/csi")
os.makedirs(output_dir / "training", exist_ok=True)
os.makedirs(output_dir / "validation", exist_ok=True)

# Zimbabwe elevation (approximate)
ALTITUDE_M = 1000 

def calculate_csi_for_file(file_path, output_path):
    print(f"Calculating CSI for: {file_path.name}")
    
    with xr.open_dataset(file_path) as ds:
        # Extract dims
        lat_arr = ds.lat.values
        lon_arr = ds.lon.values
        times = ds.time.values
        
        # Grid for pvlib
        lons_2d, lats_2d = np.meshgrid(lon_arr, lat_arr)
        n_pixels = lats_2d.size
        
        # Initialize output
        clearsky_ghi = np.zeros_like(ds['ssrd'].values)
        
        # Calculate site pressure for absolute airmass
        pressure = pvlib.atmosphere.alt2pres(ALTITUDE_M)
        
        # Iterate over time
        for i, t in enumerate(times):
            # Broadcast timestamp
            times_broadcasted = pd.DatetimeIndex([t] * n_pixels)
            
            # 1. Solar Position
            solpos = pvlib.solarposition.get_solarposition(
                time=times_broadcasted,
                latitude=lats_2d.ravel(),
                longitude=lons_2d.ravel()
            )
            
            # 2. Relative Airmass
            rel_am = pvlib.atmosphere.get_relative_airmass(
                solpos['apparent_zenith']
            )
            
            # 3. Absolute Airmass (Corrected for pressure)
            abs_am = pvlib.atmosphere.get_absolute_airmass(
                rel_am, 
                pressure=pressure
            )
            
            # 4. Ineichen Clear Sky
            cs = pvlib.clearsky.ineichen(
                apparent_zenith=solpos['apparent_zenith'],
                airmass_absolute=abs_am,
                linke_turbidity=3.0, 
                altitude=ALTITUDE_M
            )
            
            # 5. Reshape to grid
            clearsky_ghi[i, :, :] = cs['ghi'].values.reshape(lats_2d.shape)
        
        # Save CSI
        cs_da = xr.DataArray(
            ds['ssrd'].values / clearsky_ghi,
            coords=ds['ssrd'].coords,
            dims=ds['ssrd'].dims,
            name='clearsky_ghi'
        ).clip(0, 1.2)
        
        cs_da.to_netcdf(output_path)
        print(f"Saved CSI: {output_path.name}")

# --- Execution ---
files = sorted(glob.glob(str(input_dir / "**/*.nc"), recursive=True))

for f in files:
    f_path = Path(f)
    split = 'training' if 'training' in f_path.parts else 'validation'
    out_path = output_dir / split / f"csi_{f_path.name}"
    
    try:
        calculate_csi_for_file(f_path, out_path)
    except Exception as e:
        print(f"Error calculating CSI for {f_path.name}: {e}")