import cdsapi
import os
import time

c = cdsapi.Client()

print("Initializing Year-by-Year Training Target Extraction (1985-2010)...")

start_year = 1985
end_year = 2010

# Loop through each year individually to satisfy server conversion limits
for year in range(start_year, end_year + 1):
    str_year = str(year)
    output_filename = f'era5_training_{str_year}.nc'
    
    # Safety Check: Skip if the file already downloaded from a previous run
    if os.path.exists(output_filename):
        print(f"[{str_year}] File already exists locally. Skipping...")
        continue
        
    print(f"\n---> Requesting data for Year: {str_year}")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': 'surface_solar_radiation_downwards',  # GHI target (ssrd)
                'year': str_year,
                'month': [f'{m:02d}' for m in range(1, 13)],     # All 12 months
                'day': [f'{d:02d}' for d in range(1, 32)],       # All 31 days
                'time': [f'{h:02d}:00' for h in range(0, 24)],   # Full diurnal cycle
                'area': [-15, 25, -22, 33],                       # Zimbabwe bounding box
            },
            output_filename
        )
        print(f"[{str_year}] Successfully downloaded and saved as {output_filename}")
        
        # Short pause to allow local disk writing to settle
        time.sleep(2)
        
    except Exception as e:
        print(f"Error downloading data for year {str_year}: {e}")
        print("Stopping execution loop to inspect the issue.")
        break

print("\nTraining target extraction loop completed!")
