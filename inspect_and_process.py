import xarray as xr
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
input_base = Path("./data/raw/era5/hourly")
output_base = Path("./data/processed/era5/monthly")
plot_dir = Path("./data/processed/plots")

os.makedirs(output_base / "training", exist_ok=True)
os.makedirs(output_base / "validation", exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

def auto_detect_dims(ds):
    """Dynamically finds and renames dimension labels to 'time', 'lat', and 'lon'."""
    mapping = {}
    
    # Map any variation of time to 'time'
    for dim in ds.dims:
        if dim.lower() in ['time', 'valid_time', 'date', 't']:
            mapping[dim] = 'time'
        elif 'latitude' in dim.lower() or dim.lower() == 'lat':
            mapping[dim] = 'lat'
        elif 'longitude' in dim.lower() or dim.lower() == 'lon':
            mapping[dim] = 'lon'
            
    return ds.rename(mapping)

def process_and_inspect(file_path, output_path, is_sample=False):
    print(f"Processing: {file_path.name}")
    
    with xr.open_dataset(file_path) as ds:
        # Normalize dimension names
        ds = auto_detect_dims(ds)
        
        # Clip to Domain (Zimbabwe)
        ds_clipped = ds.sel(lat=slice(-15.0, -22.5), lon=slice(25.0, 33.5))
        
        # Unit Conversion: J/m2 (hourly) -> W/m2 (daily mean)
        # Note: 'ME' (Month End) replaces '1M'
        ds_daily = ds_clipped['ssrd'].resample(time='1D').sum() / 86400.0
        
        # Monthly Mean using 'ME' frequency
        ds_monthly = ds_daily.resample(time='ME').mean()
        
        # Save
        ds_monthly.to_netcdf(output_path)
        
        # Inspection Plot
        if is_sample:
            plt.figure(figsize=(8, 6))
            ds_monthly.isel(time=0).plot(cmap='viridis')
            plt.title(f"Sample Processed Data: {file_path.name}")
            plot_file = plot_dir / f"check_{file_path.stem}.png"
            plt.savefig(plot_file)
            print(f"Inspection plot saved to: {plot_file}")
            plt.close()

# --- Execution ---
files = sorted(glob.glob(str(input_base / "**/*.nc"), recursive=True))

for i, f in enumerate(files):
    f_path = Path(f)
    split = 'training' if 'training' in f_path.parts else 'validation'
    out_path = output_base / split / f"monthly_{f_path.name}"
    
    try:
        process_and_inspect(f_path, out_path, is_sample=(i == 0))
    except Exception as e:
        print(f"Error processing {f_path.name}: {e}")

print("\nProcessing complete. Please check 'data/processed/plots' for the sample image.")