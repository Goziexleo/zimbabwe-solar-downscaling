import os
import xarray as xr

# --- Configuration ---
training_dir = "./data/processed/era5/csi/training"
validation_dir = "./data/processed/era5/csi/validation"

def inspect_folder(folder_path, split_name):
    print(f"\n" + "="*50)
    print(f" Inspecting {split_name.upper()} CSI Directory")
    print(f"="*50)
    
    if not os.path.exists(folder_path):
        print(f"Directory not found: {folder_path}")
        return
        
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])
    if not files:
        print(f"No NetCDF files found in {folder_path}")
        return
        
    print(f"Total files found: {len(files)}")
    sample_file = os.path.join(folder_path, files[0])
    print(f"Inspecting sample file: {files[0]}\n")
    
    try:
        ds = xr.open_dataset(sample_file)
        print("Dataset Summary:")
        print(ds)
        
        print("\nData Variables:")
        for var_name, var in ds.data_vars.items():
            print(f"  * **{var_name}** | Dimensions: {var.dims} | Shape: {var.shape} | Units: {var.attrs.get('units', 'N/A')}")
            
        print("\nCoordinates:")
        for coord_name, coord in ds.coords.items():
            print(f"  * **{coord_name}** | Shape: {coord.shape} | Range: {coord.values.min()} to {coord.values.max() if coord.dtype.kind in 'iuf' else 'Categorical'}")
            
        ds.close()
    except Exception as e:
        print(f"Error reading NetCDF file: {e}")

# Run inspections
inspect_folder(training_dir, "training")
inspect_folder(validation_dir, "validation")