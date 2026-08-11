import cdsapi
import os
import time

c = cdsapi.Client()

print("Initializing Variable-by-Variable Predictor Extraction (1985-2010)...")

start_year = 1985
end_year = 2010

# We separate the variables to bypass the new CDS API payload cost limits
variables = [
    'total_cloud_cover',
    '2m_temperature',
    'surface_pressure',
    '2m_dewpoint_temperature'
]

output_dir = './data/raw/era5/predictors'
os.makedirs(output_dir, exist_ok=True)

for var in variables:
    print(f"\n==================================================")
    print(f"  Starting downloads for: {var.upper()}")
    print(f"==================================================")
    
    for year in range(start_year, end_year + 1):
        str_year = str(year)
        output_filename = os.path.join(output_dir, f'era5_{var}_{str_year}.nc')
        
        # Safety Check: Skip if the file already downloaded
        if os.path.exists(output_filename):
            print(f"[{str_year}] File exists locally. Skipping...")
            continue
            
        print(f"---> Requesting data for Year: {str_year}")
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': var,
                    'year': str_year,
                    'month': [f'{m:02d}' for m in range(1, 13)],     
                    'day': [f'{d:02d}' for d in range(1, 32)],       
                    'time': [f'{h:02d}:00' for h in range(0, 24)],   
                    'area': [-15, 25, -22, 33],  # Zimbabwe bounding box
                },
                output_filename
            )
            print(f"[{str_year}] Successfully downloaded and saved as {output_filename}")
            
            # Short pause to allow local disk writing to settle
            time.sleep(2)
            
        except Exception as e:
            print(f"Error downloading {var} for year {str_year}: {e}")
            print("Stopping execution loop to inspect the issue.")
            exit()

print("\nPredictor extraction loop completed!")