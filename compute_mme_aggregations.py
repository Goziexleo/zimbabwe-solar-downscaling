import os
import xarray as xr
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
proj_dir = os.environ.get("PROJECTIONS_DIR", os.path.abspath("./data/processed/projections"))
out_dir = os.environ.get("MME_OUTPUT_DIR", os.path.abspath("./data/processed/mme_aggregations"))
os.makedirs(out_dir, exist_ok=True)

gcms = ["CNRM-CM6-1", "MPI-ESM1-2-HR", "ACCESS-CM2"]
scenarios = ["ssp245", "ssp585"]

# Define temporal horizons for the suitability analysis
periods = {
    "near_term_2026_2050": slice("2026", "2050"),
    "mid_term_2051_2075": slice("2051", "2075"),
    "long_term_2076_2100": slice("2076", "2100")
}

print("Starting Multi-Model Ensemble (MME) computation...")

for scenario in scenarios:
    print(f"\n--- Processing Scenario: {scenario} ---")
    
    datasets = []
    for gcm in gcms:
        filepath = os.path.join(proj_dir, f"downscaled_ghi_{gcm}_{scenario}_2026_2100.nc")
        if os.path.exists(filepath):
            ds = xr.open_dataset(filepath)
            # Assign a new coordinate to track the model
            ds = ds.assign_coords(model=gcm)
            datasets.append(ds)
        else:
            print(f"Warning: Missing file for {gcm} {scenario}")
            
    if not datasets:
        print(f"No datasets found for {scenario}. Skipping.")
        continue
        
    # Concatenate along the new 'model' dimension
    ds_ensemble = xr.concat(datasets, dim="model")
    
    # Process each planning horizon
    for period_name, time_slice in periods.items():
        print(f"  -> Aggregating {period_name}...")
        
        # 1. Slice the 25-year period
        ds_period = ds_ensemble.sel(time=time_slice)
        
        # 2. Period Mean (Average across the time dimension first, for each individual model)
        period_model_means = ds_period.mean(dim="time", keep_attrs=True)
        
        # 3. Ensemble Mean (Average across the 3 models)
        mme_mean = period_model_means.mean(dim="model", keep_attrs=True)
        
        # 4. Structural Uncertainty (Standard Deviation across the 3 models)
        mme_std = period_model_means.std(dim="model", keep_attrs=True)
        
        # Package the outputs into a clean 2D spatial dataset
        ds_out = xr.Dataset(
            {
                "ghi_mean": mme_mean["ghi"],
                "ghi_uncertainty": mme_std["ghi"],
                "csi_mean": mme_mean["clear_sky_index"]
            },
            coords=mme_mean.coords
        )
        
        # Remove the leftover 'model' coordinate from the final output
        if 'model' in ds_out.coords:
            ds_out = ds_out.drop_vars('model')
            
        ds_out["ghi_mean"].attrs["description"] = f"Ensemble Mean GHI ({period_name.replace('_', ' ')})"
        ds_out["ghi_mean"].attrs["units"] = "W/m^2"
        ds_out["ghi_uncertainty"].attrs["description"] = f"GCM Structural Uncertainty (Std Dev) for GHI"
        ds_out["ghi_uncertainty"].attrs["units"] = "W/m^2"
        
        out_filename = f"mme_suitability_{scenario}_{period_name}.nc"
        out_filepath = os.path.join(out_dir, out_filename)
        ds_out.to_netcdf(out_filepath)
        
        print(f"     Saved: {out_filename}")

print("\nSuccess: MME and temporal aggregations completed. Files are ready for Multi-Criteria Evaluation.")